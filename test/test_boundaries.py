"""Tests for auto-detected Transformer block boundaries (z / u / z_next)."""

from __future__ import annotations

import pytest


def test_describe_boundaries_bert_config_only():
    from lig import describe_boundaries

    info = describe_boundaries("bert-base-uncased", load_weights=False)
    assert info["model_type"] == "bert"
    assert info["layout"] == "post_ln_encoder"
    assert info["supported_granularity"] == ["att", "layer", "mlp"]
    assert "attention.self" in (info["hook_points"]["att_core"] or "")


def test_describe_boundaries_gpt2_config_only():
    from lig import describe_boundaries

    info = describe_boundaries("gpt2", load_weights=False)
    assert info["model_type"] == "gpt2"
    assert info["layout"] == "pre_ln_decoder"
    assert info["architecture"] == "decoder"
    assert info["supported_granularity"] == ["att", "layer", "mlp"]


def test_describe_boundaries_distilbert_layer_only():
    from lig import describe_boundaries

    info = describe_boundaries("distilbert-base-uncased", load_weights=False)
    assert info["layout"] == "block_only"
    assert info["supported_granularity"] == ["layer"]


@pytest.mark.slow
def test_detect_boundaries_bert_weights():
    pytest.importorskip("torch")
    from lig import describe_boundaries

    info = describe_boundaries("bert-base-uncased", load_weights=True, device="cpu")
    assert info["layout"] == "post_ln_encoder"
    assert info["detection"].startswith("model_type")
    assert info["num_layers"] == 12


@pytest.mark.slow
def test_detect_boundaries_gpt2_weights():
    pytest.importorskip("torch")
    from lig import describe_boundaries

    info = describe_boundaries("gpt2", load_weights=True, device="cpu")
    assert info["layout"] == "pre_ln_decoder"
    assert info["ig_ready"] is True
    assert info["num_layers"] == 12


@pytest.mark.slow
def test_explain_includes_boundary_metadata():
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello",
        model="bert-base-uncased",
        num_steps=2,
        granularity="layer",
        layers=[0],
        device="cpu",
    )
    b = result["boundaries"]
    assert b["layout"] == "post_ln_encoder"
    assert "hook_points" in b
    assert b["nodes"]["z"]
