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


def test_llama_attention_ig_causal_mask(tiny_llama_adapter):
    inputs = tiny_llama_adapter.encode("one two three four")
    tiny_llama_adapter.cache(inputs)
    target_token_idx = 1
    result = tiny_llama_adapter.compute_attention_ig(
        layer_idx=0,
        target_token_idx=target_token_idx,
        head_idx=0,
        baseline="zero",
        num_steps=4,
    )
    # Future tokens must not receive attribution (causal).
    assert all(v == 0.0 for v in result.values[target_token_idx + 1 :])


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
