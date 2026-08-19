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
