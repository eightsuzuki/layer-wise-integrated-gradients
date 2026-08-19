"""Gemma3 LIG helpers vs the reference Transformers forward (tiny random model)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch


@pytest.fixture(scope="module")
def tiny_gemma3():
    """4-layer random Gemma3 text model with mixed sliding / full attention."""
    from transformers import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=3,
        layer_types=["sliding_attention", "full_attention"] * 2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return Gemma3TextModel(config).to(torch.float32).eval()


def _hidden_states(model, input_ids):
    from utils.calculations.ig.gemma3.block_forward import embed_tokens

    with torch.no_grad():
        embeddings = embed_tokens(model, input_ids)
        out = model(inputs_embeds=embeddings, output_hidden_states=True, use_cache=False)
    return embeddings, out.hidden_states


def test_block_forward_matches_transformers(tiny_gemma3):
    """Masks, dual RoPE and the u boundary must reproduce the reference block."""
    from utils.calculations.ig.gemma3.block_forward import (
        attn_pre_oproj_output,
        forward_gemma3_block,
        layer_attention_mask,
        mlp_branch,
        rope_position_embeddings,
    )

    input_ids = torch.tensor([[1, 5, 9, 12, 20, 3]])
    _, hidden_states = _hidden_states(tiny_gemma3, input_ids)
    seq_len = input_ids.shape[1]

    # hidden_states[-1] has the final RMSNorm applied, so stop one block earlier.
    for layer_idx in range(tiny_gemma3.config.num_hidden_layers - 1):
        z = hidden_states[layer_idx]
        z_next = hidden_states[layer_idx + 1]
        block = tiny_gemma3.layers[layer_idx]
        mask = layer_attention_mask(tiny_gemma3, layer_idx, seq_len, z.device, z.dtype)
        global_pe, local_pe = rope_position_embeddings(tiny_gemma3, z)

        assert torch.allclose(
            forward_gemma3_block(block, z, mask, global_pe, local_pe), z_next, atol=1e-5
        )

        # u (concatenated heads) -> o_proj -> post-attention RMSNorm -> +z -> FFN
        pe = local_pe if block.self_attn.is_sliding else global_pe
        with torch.no_grad():
            u = attn_pre_oproj_output(block, block.input_layernorm(z), mask, pe)
            post_attn = z + block.post_attention_layernorm(block.self_attn.o_proj(u))
            rebuilt = post_attn + mlp_branch(block, post_attn)
        assert torch.allclose(rebuilt, z_next, atol=1e-5)


def test_pre_oproj_u_is_scale_invariant_in_z(tiny_gemma3):
    """Why ATT IG interpolates embeddings instead of z^(l).

    Gemma3's u is taken before ``o_proj`` and carries no residual term, and
    RMSNorm is scale invariant, so ``u(a * z) == u(z)`` for every ``a > 0``.
    Interpolating z^(l) against a zero baseline would therefore give a constant
    path and ~zero attributions, breaking the ``zero`` / ``itb_zero_ratio``
    baselines. ``Gemma3AttentionModel`` propagates interpolated token embeddings
    through blocks ``0 .. l-1`` instead.
    """
    from utils.calculations.ig.gemma3.block_forward import (
        attn_pre_oproj_output,
        layer_attention_mask,
        rope_position_embeddings,
    )

    input_ids = torch.tensor([[1, 5, 9, 12]])
    _, hidden_states = _hidden_states(tiny_gemma3, input_ids)
    layer_idx = 2
    z = hidden_states[layer_idx]
    block = tiny_gemma3.layers[layer_idx]
    mask = layer_attention_mask(tiny_gemma3, layer_idx, z.shape[1], z.device, z.dtype)
    global_pe, local_pe = rope_position_embeddings(tiny_gemma3, z)
    pe = local_pe if block.self_attn.is_sliding else global_pe
    with torch.no_grad():
        u = attn_pre_oproj_output(block, block.input_layernorm(z), mask, pe)
        u_scaled = attn_pre_oproj_output(block, block.input_layernorm(4.0 * z), mask, pe)
    assert torch.allclose(u, u_scaled, atol=1e-5)


def test_attention_ig_is_causal_and_complete(tiny_gemma3):
    """ATT IG over token embeddings: causal, and complete for the L2 scalarizer."""
    from captum.attr import IntegratedGradients

    from lig.api import _gemma3_baseline_embeddings
    from utils.calculations.ig.attention.gemma3_attention_models import (
        create_gemma3_attention_model,
    )

    input_ids = torch.tensor([[1, 5, 9, 12]])
    embeddings, _ = _hidden_states(tiny_gemma3, input_ids)
    layer_idx, target_token_idx = 2, 2
    baseline = _gemma3_baseline_embeddings(embeddings, "self_input_token", target_token_idx)

    ig_model = create_gemma3_attention_model(
        text_model=tiny_gemma3,
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
        target_head_idx=0,
    )
    ig_model.eval()
    attributions = IntegratedGradients(ig_model).attribute(
        inputs=embeddings, baselines=baseline, n_steps=32, method="riemann_trapezoid"
    )
    values = attributions.sum(dim=-1).squeeze(0).detach()

    assert values.shape[0] == input_ids.shape[1]
    # Tokens after the target cannot reach it through a causal stack.
    future = values[target_token_idx + 1 :]
    assert torch.allclose(future, torch.zeros_like(future), atol=1e-4)

    with torch.no_grad():
        gap = float(ig_model(embeddings).item()) - float(ig_model(baseline).item())
    assert abs(float(values.sum()) - gap) / abs(gap) < 0.25


def test_mlp_ig_completeness(tiny_gemma3):
    from utils.calculations.ig.gemma3.block_forward import (
        attn_pre_oproj_output,
        layer_attention_mask,
        rope_position_embeddings,
    )
    from utils.calculations.ig.mlp.gemma3_mlp_lig_ig import compute_gemma3_mlp_lig_single_token

    input_ids = torch.tensor([[1, 5, 9, 12]])
    _, hidden_states = _hidden_states(tiny_gemma3, input_ids)
    layer_idx, target_token_idx = 2, 2
    z = hidden_states[layer_idx]
    block = tiny_gemma3.layers[layer_idx]
    mask = layer_attention_mask(tiny_gemma3, layer_idx, z.shape[1], z.device, z.dtype)
    global_pe, local_pe = rope_position_embeddings(tiny_gemma3, z)
    pe = local_pe if block.self_attn.is_sliding else global_pe
    with torch.no_grad():
        u = attn_pre_oproj_output(block, block.input_layernorm(z), mask, pe)[0, target_token_idx, :]

    result = compute_gemma3_mlp_lig_single_token(
        tiny_gemma3,
        layer_idx=layer_idx,
        z_layer=z,
        target_mlp_input=u.unsqueeze(0),
        baseline_mlp_input=torch.zeros_like(u).unsqueeze(0),
        target_token_idx=target_token_idx,
        num_steps=32,
    )
    config = tiny_gemma3.config
    assert result["input_width"] == config.num_attention_heads * config.head_dim
    assert abs(result["completeness_error"]) / abs(result["l2_total"]) < 0.25


def test_layout_detection_gemma3_vs_gemma2(tiny_gemma3):
    """Gemma2 shares the pre+post-LN wiring but must not take the Gemma3 path."""
    from lig.boundaries import BlockLayout, _inspect_layer_layout

    assert _inspect_layer_layout(tiny_gemma3.layers[0]) == BlockLayout.PRE_POST_LN_DECODER

    from transformers import Gemma2Config
    from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

    gemma2_layer = Gemma2DecoderLayer(
        Gemma2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        ),
        layer_idx=0,
    )
    assert _inspect_layer_layout(gemma2_layer) == BlockLayout.PRE_LN_DECODER


def test_torchvision_guard_restores_flag():
    import transformers.utils.import_utils as import_utils

    from lig.api import _torchvision_disabled

    original = import_utils._torchvision_available
    with _torchvision_disabled():
        assert import_utils._torchvision_available is False
    assert import_utils._torchvision_available is original
