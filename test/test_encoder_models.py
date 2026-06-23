"""Regression tests for additional encoder families (MPNet, DistilBERT, MoE, Mamba)."""

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name,expected_type",
    [
        ("bert-base-uncased", "bert"),
    ],
)
def test_encoder_full_granularity_cpu(model_name: str, expected_type: str):
    """BERT-style encoders: z→u, u→z, z→z on one layer."""
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello world",
        model=model_name,
        num_steps=2,
        granularity=["att", "mlp", "layer"],
        layers=[0],
        target_tokens=[1],
        device="cpu",
    )
    assert result["model_type"] == expected_type
    layer0 = result["layers"]["0"]
    assert layer0["z2z"]["shape"][0] >= 3
    assert "z2u" in layer0["targets"]["1"]
    assert "u2z" in layer0["targets"]["1"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name,expected_type",
    [
        ("microsoft/mpnet-base", "mpnet"),
        ("distilbert-base-uncased", "distilbert"),
        ("google/switch-base-8", "switch_transformers"),
        ("state-spaces/mamba-130m", "mamba"),
    ],
)
def test_encoder_layer_only_cpu(model_name: str, expected_type: str):
    """Block-level z→z only (no BERT-style ATT/MLP split)."""
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello world",
        model=model_name,
        num_steps=2,
        granularity="layer",
        layers=[0],
        target_tokens=[1],
        device="cpu",
    )
    assert result["model_type"] == expected_type
    assert result["layers"]["0"]["z2z"]["shape"][0] >= 2
    assert "targets" not in result["layers"]["0"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name",
    [
        "microsoft/mpnet-base",
        "distilbert-base-uncased",
        "google/switch-base-8",
        "state-spaces/mamba-130m",
    ],
)
def test_layer_only_rejects_att_mlp(model_name: str):
    pytest.importorskip("torch")
    from lig import explain

    with pytest.raises(NotImplementedError, match="supports granularity"):
        explain(
            "Hi",
            model=model_name,
            num_steps=2,
            granularity="att",
            layers=[0],
            device="cpu",
        )
