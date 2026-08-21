"""Tests for Llama-family decoder IG adapter."""

from __future__ import annotations

import pytest

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tiny_llama_adapter():
    pytest.importorskip("torch")
    from lig.adapters.decoder_ig import LlamaIGAdapter

    adapter = LlamaIGAdapter(TINY_LLAMA, device="cpu")
    inputs = adapter.encode("hello world")
    adapter.cache(inputs)
    return adapter


def test_llama_encoder_layers_resolved():
    pytest.importorskip("torch")
    from transformers import AutoModel

    from lig.encoder_access import get_encoder_layers

    model = AutoModel.from_pretrained(TINY_LLAMA)
    assert len(get_encoder_layers(model)) == 2


def test_llama_mlp_ig_shape_and_completeness(tiny_llama_adapter):
    result = tiny_llama_adapter.compute_mlp_ig(
        layer_idx=0,
        target_token_idx=0,
        baseline="zero",
        num_steps=4,
    )
    assert result.contributions.shape == (tiny_llama_adapter.hidden_size,)
    assert result.verification["relative_error"] < 0.25


def test_llama_attention_output_is_causal(tiny_llama_adapter):
    """Perturbing future z rows must leave the target's attention output alone.

    ``self_attn`` applies a mask only when one is passed, so this is the test that
    fails if a hand-rolled forward forgets it.  Asserting zeros on the returned
    attribution instead would pass even without a mask, as long as the future
    entries are zeroed after the fact.
    """
    import torch

    adapter = tiny_llama_adapter
    adapter.cache(adapter.encode("one two three four"))
    z = adapter.get_z(0)
    target_token_idx = 1
    for head_idx in (None, 0):
        clean = adapter.attention_output(0, z, target_token_idx, head_idx)
        perturbed_z = z.clone()
        perturbed_z[:, target_token_idx + 1 :, :] += 100.0
        perturbed = adapter.attention_output(0, perturbed_z, target_token_idx, head_idx)
        torch.testing.assert_close(clean, perturbed)

        # The other half of causality: a past source must actually move the target.
        # Invariance alone also holds for an output that ignores its input entirely.
        past_z = z.clone()
        past_z[:, target_token_idx - 1, :] += 100.0
        moved = adapter.attention_output(0, past_z, target_token_idx, head_idx)
        assert not torch.allclose(clean, moved, atol=1e-3), (
            f"head {head_idx}: perturbing a past source left the target unchanged"
        )


def test_llama_cache_matches_model_hidden_states(tiny_llama_adapter):
    """The hand-rolled block loop must reproduce the model's own forward."""
    import torch

    adapter = tiny_llama_adapter
    inputs = adapter.encode("one two three four")
    adapter.cache(inputs)
    with torch.no_grad():
        hidden_states = adapter.model(**inputs, output_hidden_states=True).hidden_states
    for layer_idx in range(adapter.num_layers):
        torch.testing.assert_close(
            adapter.get_z(layer_idx), hidden_states[layer_idx], msg=f"layer {layer_idx}"
        )
    # The model's last hidden state is post final RMSNorm; the cache stores it before.
    torch.testing.assert_close(
        adapter.decoder.norm(adapter.cache_data["z_next"][adapter.num_layers - 1]),
        hidden_states[adapter.num_layers],
    )


def test_llama_attention_ig_completeness_and_future_zeros(tiny_llama_adapter):
    """Completeness holds without post-hoc zeroing, and causality gives the zeros."""
    adapter = tiny_llama_adapter
    adapter.cache(adapter.encode("one two three four"))
    target_token_idx = 2
    result = adapter.compute_attention_ig(
        layer_idx=0,
        target_token_idx=target_token_idx,
        baseline="zero",
        num_steps=32,
    )
    assert result.verification["relative_error"] < 0.02
    assert result.verification["is_valid"]
    # Not zeroed by hand: the causal mask makes these gradients exactly zero.
    assert all(v == 0.0 for v in result.values[target_token_idx + 1 :])


def test_llama_attention_head_output_is_pre_projection(tiny_llama_adapter):
    """A selected head is the weighted Value before o_proj mixes heads."""
    import torch

    adapter = tiny_llama_adapter
    z = adapter.get_z(0)
    target = z.shape[1] - 1
    captured = {}

    def save_pre_projection(_module, inputs):
        captured["heads"] = inputs[0].view(
            z.shape[0], z.shape[1], adapter.num_heads, adapter.head_dim
        ).detach()

    handle = adapter.decoder.layers[0].self_attn.o_proj.register_forward_pre_hook(
        save_pre_projection
    )
    try:
        with torch.no_grad():
            adapter.decoder.layers[0].self_attn(
                adapter.decoder.layers[0].input_layernorm(z),
                position_embeddings=adapter._position_embeddings(z),
                attention_mask=adapter._causal_mask(z),
            )
    finally:
        handle.remove()

    for head in (0, adapter.num_heads - 1):
        actual = adapter.attention_output(0, z, target, head)
        expected = captured["heads"][:, target, head, :]
        torch.testing.assert_close(actual, expected)


def test_load_decoder_ig_factory_llama():
    pytest.importorskip("torch")
    from lig.adapters.decoder_ig import LlamaIGAdapter, load_decoder_ig_adapter

    adapter = load_decoder_ig_adapter(TINY_LLAMA, device="cpu")
    assert isinstance(adapter, LlamaIGAdapter)


def test_llama_probe_direction_ig_completeness(tiny_llama_adapter):
    import numpy as np

    w = np.ones(tiny_llama_adapter.hidden_size, dtype=np.float64)
    result = tiny_llama_adapter.compute_mlp_ig(
        layer_idx=0,
        target_token_idx=0,
        baseline="zero",
        num_steps=6,
        probe_w=w,
    )
    assert result.verification["target_mode"] == "probe_direction"
    assert result.verification["relative_error"] < 0.25


def test_llama_attention_probe_direction_ig_completeness(tiny_llama_adapter):
    """A signed linear read-out makes ATT-IG completeness exact, not approximate."""
    import numpy as np

    adapter = tiny_llama_adapter
    adapter.cache(adapter.encode("one two three four"))
    w = np.ones(adapter.hidden_size, dtype=np.float64)
    result = adapter.compute_attention_ig(
        layer_idx=0,
        target_token_idx=2,
        baseline="zero",
        num_steps=32,
        probe_w=w,
    )
    assert result.verification["target_mode"] == "probe_direction"
    assert result.verification["relative_error"] < 0.02


@pytest.mark.slow
def test_explain_llama_tiny_mlp():
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "The sum of 12 and 34",
        model=TINY_LLAMA,
        granularity=["mlp"],
        layers=[0],
        target_tokens=[0],
        num_steps=4,
        device="cpu",
    )
    assert result["model_type"] == "llama"
    u2z = result["layers"]["0"]["targets"]["0"]["u2z"]
    assert len(u2z["contributions"]) == 16
    assert u2z["l2_total"] >= 0.0


# --- regression tests for the 2026-08-19 causal-mask / per-head audit ---------
#
# Three bugs survived for months because the only guard, the causal-mask test
# above, asserted on values that ``compute_attention_ig`` had explicitly zeroed
# just before returning.  It tested the workaround, not the property.  These
# tests assert the properties directly.


def test_llama_cache_matches_model_hidden_states_longer_sequence(tiny_llama_adapter):
    """Same check as above on a longer sequence, where drift compounds.

    Without the causal mask the loop runs bidirectional attention and drifts
    layer by layer (rel. distance 0.296 at block 3 on Llama-2-7B).  Kept under a
    distinct name: two `def`s of one name leave only the second one running.
    """
    import torch

    ad = tiny_llama_adapter
    inputs = ad.encode("one two three four five six")
    ad.cache(inputs)
    with torch.no_grad():
        hidden_states = ad.model(**inputs, output_hidden_states=True).hidden_states
    for layer_idx in range(ad.num_layers):
        assert torch.allclose(
            ad.get_z(layer_idx), hidden_states[layer_idx], atol=1e-4
        ), f"block {layer_idx} diverges from the model's own hidden states"


def test_llama_attention_output_ignores_future_tokens(tiny_llama_adapter):
    """Perturbing future positions must not move the target's attention output."""
    import torch

    ad = tiny_llama_adapter
    ad.cache(ad.encode("one two three four five six"))
    z = ad.get_z(0)
    target = 2
    perturbed = z.clone()
    perturbed[:, target + 1 :, :] += 100.0
    assert torch.allclose(
        ad.attention_output(0, z, target), ad.attention_output(0, perturbed, target), atol=1e-4
    )
    # ... and it must still depend on the past, or the test above is vacuous.
    past = z.clone()
    past[:, target - 1, :] += 100.0
    assert not torch.allclose(
        ad.attention_output(0, z, target), ad.attention_output(0, past, target), atol=1e-3
    )


def test_llama_per_head_outputs_reconstruct_attention(tiny_llama_adapter):
    """Per-head outputs live before o_proj, so o_proj(concat) == the full output.

    Slicing the *post*-o_proj vector by ``head_dim`` (the previous behaviour)
    splits mixed residual coordinates and passes no such identity.
    """
    import torch

    ad = tiny_llama_adapter
    ad.cache(ad.encode("one two three four five six"))
    z = ad.get_z(0)
    target = 3
    full = ad.attention_output(0, z, target)
    heads = torch.cat(
        [ad.attention_output(0, z, target, head_idx=h) for h in range(ad.num_heads)], dim=-1
    )
    o_proj = ad.decoder.layers[0].self_attn.o_proj
    assert torch.allclose(o_proj(heads), full, atol=1e-4)


def test_llama_attention_ig_completeness_and_zero_future_attribution(tiny_llama_adapter):
    """No post-hoc zeroing: future attributions must be zero on their own."""
    import numpy as np

    ad = tiny_llama_adapter
    ad.cache(ad.encode("one two three four five six"))
    target = 3
    for probe_w in (None, np.ones(ad.hidden_size)):
        result = ad.compute_attention_ig(
            0, target_token_idx=target, baseline="zero", num_steps=32, probe_w=probe_w
        )
        assert result.verification["relative_error"] < 0.02
        assert np.abs(result.values[target + 1 :]).max() == 0.0


def test_completeness_guard_warns_when_unconverged():
    """An unconverged attribution must announce itself, not just set a dict key.

    This is the trap the Qwen2.5 sweep hit: at the default step count the
    completeness error was 1.005 while the ranking scored best of any method.
    """
    import warnings

    from lig.adapters.decoder_ig.base import COMPLETENESS_TOLERANCE, check_completeness

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert check_completeness(COMPLETENESS_TOLERANCE / 2, where="unit") is True
    assert not caught

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert check_completeness(1.005, where="unit") is False
    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert "num_steps" in str(caught[0].message)
