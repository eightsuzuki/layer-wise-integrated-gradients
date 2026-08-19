"""Tests for dense Qwen3 decoder IG with post-attention-residual u."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch


class _DummyTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

    def convert_ids_to_tokens(self, ids):
        return [str(token_id) for token_id in ids]


@pytest.fixture()
def tiny_qwen3_adapter():
    from transformers import Qwen3Config, Qwen3Model

    from lig.adapters.decoder_ig.qwen3 import Qwen3IGAdapter

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3Model(config).eval()
    adapter = Qwen3IGAdapter(
        "synthetic-qwen3",
        model=model,
        tokenizer=_DummyTokenizer(),
        device="cpu",
    )
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    adapter.cache(inputs)
    return adapter


def test_qwen3_u_is_post_attention_residual(tiny_qwen3_adapter):
    adapter = tiny_qwen3_adapter
    z = adapter.get_z(0)
    computed_u = adapter.attention_output(0, z, target_token_idx=1, head_idx=None)
    captured_u = adapter.get_mlp_input(0)[:, 1, :]

    assert computed_u.shape == (1, adapter.hidden_size)
    assert torch.allclose(computed_u, captured_u, atol=1e-6, rtol=1e-5)


def test_qwen3_projected_heads_reconstruct_attention_branch(tiny_qwen3_adapter):
    from utils.calculations.ig.qwen3.block_forward import project_qwen3_head

    adapter = tiny_qwen3_adapter
    layer = adapter.decoder.layers[0]
    heads, _ = adapter.attention_components(0, adapter.get_z(0))
    projected_heads = [
        project_qwen3_head(layer, heads, head_idx)
        for head_idx in range(adapter.num_heads)
    ]

    expected = layer.self_attn.o_proj(heads)
    assert torch.allclose(sum(projected_heads), expected, atol=1e-6, rtol=1e-5)
    assert projected_heads[0].shape[-1] == adapter.hidden_size


def test_qwen3_attention_forward_is_causal_before_postprocessing(tiny_qwen3_adapter):
    adapter = tiny_qwen3_adapter
    z = adapter.get_z(0)
    perturbed = z.clone()
    perturbed[:, 3, :] += torch.randn_like(perturbed[:, 3, :]) * 5

    for head_idx in (0, None):
        original = adapter.attention_output(0, z, target_token_idx=1, head_idx=head_idx)
        changed = adapter.attention_output(
            0, perturbed, target_token_idx=1, head_idx=head_idx
        )
        assert torch.allclose(original, changed, atol=1e-7, rtol=1e-6)


def test_qwen3_attention_and_mlp_ig(tiny_qwen3_adapter):
    adapter = tiny_qwen3_adapter
    att = adapter.compute_attention_ig(
        layer_idx=0,
        target_token_idx=1,
        head_idx=0,
        baseline="zero",
        num_steps=4,
    )
    mlp = adapter.compute_mlp_ig(
        layer_idx=0,
        target_token_idx=1,
        baseline="zero",
        num_steps=4,
    )

    assert att.values.shape == (4,)
    assert att.values[2] == pytest.approx(0.0, abs=1e-9)
    assert att.values[3] == pytest.approx(0.0, abs=1e-9)
    assert att.verification["output_space"] == "residual_delta"
    assert mlp.contributions.shape == (adapter.hidden_size,)
    assert mlp.per_head.size == 0
    assert mlp.verification["per_head_available"] is False
    assert mlp.verification["relative_error"] < 0.2


def test_qwen3_layer_ig_shape(tiny_qwen3_adapter):
    from utils.calculations.ig.z2z.qwen3_layer_direct_ig import (
        compute_qwen3_layer_direct_ig_single_target,
    )

    adapter = tiny_qwen3_adapter
    values = compute_qwen3_layer_direct_ig_single_target(
        qwen3=adapter.model,
        z_layer=adapter.get_z(0),
        layer_idx=0,
        target_token_idx=1,
        position_ids=adapter.cache_data["position_ids"],
        attention_mask=adapter.cache_data["attention_mask"],
        num_steps=4,
        baseline_method="zero",
    )
    assert values.shape == (4,)
    assert values[2] == pytest.approx(0.0, abs=1e-9)
    assert values[3] == pytest.approx(0.0, abs=1e-9)


def test_qwen3_factory_selects_dedicated_adapter(tiny_qwen3_adapter):
    from lig.adapters.decoder_ig import Qwen3IGAdapter, load_decoder_ig_adapter

    source = tiny_qwen3_adapter
    with patch(
        "lig.adapters.decoder_ig.factory.AutoConfig.from_pretrained",
        return_value=source.model.config,
    ):
        loaded = load_decoder_ig_adapter(
            "synthetic-qwen3",
            model=source.model,
            tokenizer=_DummyTokenizer(),
            device="cpu",
        )
    assert isinstance(loaded, Qwen3IGAdapter)


def test_qwen3_shared_boundary_helpers_match_official_forward(tiny_qwen3_adapter):
    from lig.boundaries import detect_boundaries, forward_block, u_from_z

    adapter = tiny_qwen3_adapter
    boundaries = detect_boundaries(adapter.model)
    z = adapter.get_z(0)
    attention_mask = adapter.cache_data["attention_mask"]

    u = u_from_z(
        model=adapter.model,
        boundaries=boundaries,
        layer_idx=0,
        z_layer=z,
        attention_mask=attention_mask,
        target_token_idx=1,
    )
    z_next = forward_block(
        model=adapter.model,
        boundaries=boundaries,
        layer_idx=0,
        hidden_states=z,
        attention_mask=attention_mask,
    )

    assert torch.allclose(u, adapter.get_mlp_input(0)[0, 1], atol=1e-6, rtol=1e-5)
    assert torch.allclose(z_next, adapter.get_z(1), atol=1e-6, rtol=1e-5)


@pytest.mark.slow
def test_explain_qwen3_1_7b_smoke():
    from lig import explain

    result = explain(
        "Hello world",
        model="Qwen/Qwen3-1.7B",
        num_steps=2,
        granularity=["att", "mlp", "layer"],
        baseline_att="zero",
        baseline_mlp="zero",
        baseline_layer="zero",
        layers=[0],
        target_tokens=[1],
        target_head=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    assert result["model_type"] == "qwen3"
    assert result["boundaries"]["nodes"]["u"].startswith("Post-attention residual")
    layer = result["layers"]["0"]
    assert "z2z" in layer
    assert "z2u" in layer["targets"]["1"]
    assert "u2z" in layer["targets"]["1"]
