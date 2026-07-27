"""Smoke tests for the public lig API."""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_explain_bert_single_layer_cpu():
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello world",
        model="bert-base-uncased",
        num_steps=2,
        granularity=["att", "mlp", "layer"],
        layers=[0],
        target_tokens=[1, 2],
        device="cpu",
    )
    assert result["model_type"] == "bert"
    assert "0" in result["layers"]
    layer0 = result["layers"]["0"]
    assert "z2z" in layer0
    assert "z2u" in layer0["targets"]["1"]
    assert "u2z" in layer0["targets"]["1"]


@pytest.mark.slow
def test_explain_roberta_single_layer_cpu():
    """Regression: RoBERTa encoder path (verified manually 2026-06)."""
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello world",
        model="roberta-base",
        num_steps=2,
        granularity=["att", "mlp", "layer"],
        layers=[0],
        target_tokens=[1, 2],
        device="cpu",
    )
    assert result["model_type"] == "roberta"
    assert result["layers"]["0"]["z2z"]["shape"][0] >= 3
    assert "heads" in result["layers"]["0"]["targets"]["1"]["z2u"]
    assert len(result["layers"]["0"]["targets"]["1"]["z2u"]["heads"]["0"]["contributions"]) >= 3


@pytest.mark.slow
def test_explain_gpt2_att_cpu():
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello world",
        model="gpt2",
        num_steps=2,
        granularity="att",
        baseline_att="zero",
        layers=[0],
        target_tokens=[1],
        target_head=0,
        device="cpu",
    )
    assert result["architecture"] == "decoder"
    assert result["layers"]["0"]["targets"]["1"]["z2u"]["heads"]["0"]["contributions"]


@pytest.mark.slow
def test_explain_gpt2_all_granularity_cpu():
    """GPT-2 decoder: z→u, u→z, z→z."""
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello world",
        model="gpt2",
        num_steps=2,
        granularity=["att", "mlp", "layer"],
        baseline_att="zero",
        baseline_mlp="zero",
        baseline_layer="zero",
        layers=[0],
        target_tokens=[1],
        target_head=0,
        device="cpu",
    )
    assert result["model_type"] == "gpt2"
    layer0 = result["layers"]["0"]
    assert "z2z" in layer0
    assert "z2u" in layer0["targets"]["1"]
    assert "u2z" in layer0["targets"]["1"]
    assert "contributions" in layer0["targets"]["1"]["u2z"]


@pytest.mark.slow
def test_explain_gpt2_all_granularity_att_itb_a0_cpu():
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello world",
        model="gpt2",
        num_steps=2,
        granularity=["att", "mlp", "layer"],
        baseline_att="self_input_token",
        baseline_mlp="att_itb_a0",
        baseline_layer="self_input_token",
        layers=[0],
        target_tokens=[1],
        target_head=0,
        device="cpu",
    )
    u2z = result["layers"]["0"]["targets"]["1"]["u2z"]
    assert u2z["baseline"] == "att_itb_a0"
    assert "contributions" in u2z
    assert len(u2z["contributions"]) > 0


@pytest.mark.slow
def test_explain_bert_itb_zero_ratio_att_cpu():
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello world",
        model="bert-base-uncased",
        num_steps=2,
        granularity="att",
        baseline_att="itb_zero_ratio",
        layers=[0],
        target_tokens=[1],
        target_head=0,
        device="cpu",
    )
    z2u = result["layers"]["0"]["targets"]["1"]["z2u"]
    assert z2u["baseline"] == "itb_zero_ratio"
    assert z2u["heads"]["0"]["contributions"]


@pytest.mark.slow
def test_explain_bert_itb_map_ratio_att_cpu():
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello world",
        model="bert-base-uncased",
        num_steps=2,
        granularity="att",
        baseline_att="itb_map_ratio",
        layers=[0],
        target_tokens=[1],
        target_head=0,
        device="cpu",
    )
    z2u = result["layers"]["0"]["targets"]["1"]["z2u"]
    assert z2u["baseline"] == "itb_map_ratio"
    assert z2u["heads"]["0"]["contributions"]


@pytest.mark.slow
def test_explain_bert_layer_itb_zero_ratio_cpu():
    pytest.importorskip("torch")
    from lig import explain

    result = explain(
        "Hello world",
        model="bert-base-uncased",
        num_steps=2,
        granularity="layer",
        baseline_layer="itb_zero_ratio",
        layers=[0],
        device="cpu",
    )
    assert result["layers"]["0"]["z2z"]["baseline"] == "itb_zero_ratio"


def test_release_rejects_unsupported_baseline():
    from lig import explain

    with pytest.raises(ValueError, match="self_output_token"):
        explain(
            "Hi",
            baseline_att="self_output_token",  # type: ignore[arg-type]
            granularity="att",
            layers=[0],
            device="cpu",
        )


def test_other_decoder_not_implemented_yet():
    from unittest.mock import MagicMock, patch

    from lig import explain

    mock_cfg = MagicMock()
    mock_cfg.model_type = "gpt_neox"  # DECODER_FAMILY_TYPES but not DECODER_IG_MODEL_TYPES yet

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg):
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            explain(
                "Hi",
                model="EleutherAI/pythia-70m",
                num_steps=2,
                granularity="layer",
                device="cpu",
            )


def test_describe_boundaries_gemma3_config_only():
    from unittest.mock import MagicMock, patch

    from lig import describe_boundaries

    text_cfg = MagicMock()
    text_cfg.num_hidden_layers = 34
    mock_cfg = MagicMock()
    mock_cfg.model_type = "gemma3"
    mock_cfg.num_hidden_layers = None
    mock_cfg.n_layer = None
    mock_cfg.num_layers = 0
    mock_cfg.text_config = text_cfg

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg):
        info = describe_boundaries("google/gemma-3-4b-it", load_weights=False)
    assert info["layout"] == "pre_post_ln_decoder"
    assert info["architecture"] == "decoder"
    assert info["num_layers"] == 34
    assert info["supported_granularity"] == ["att", "layer", "mlp"]


def test_decoder_load_stub():
    from lig.adapters import load_adapter

    adapter = load_adapter("gpt2", device="cpu", allow_decoder_stub=True)
    assert adapter.model_type == "gpt2"
    # Supported decoder families are IG-ready; ensure_ig_ready is a no-op.
    adapter.ensure_ig_ready()


@pytest.mark.slow
def test_explain_gemma3_4b_it_smoke():
    """Requires HF access to google/gemma-3-4b-it (gated) and GPU recommended."""
    import os

    from lig import explain

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token_file = os.path.expanduser("~/.cache/huggingface/token")
    if not token and not os.path.exists(token_file):
        pytest.skip("HF token required for google/gemma-3-4b-it")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    result = explain(
        "Hello world",
        model="google/gemma-3-4b-it",
        num_steps=4,
        granularity="all",
        layers=[0],
        target_tokens=[1],
        target_head=0,
        device=device,
    )
    assert result["architecture"] == "decoder"
    assert result["model_type"] == "gemma3"
    layer0 = result["layers"]["0"]
    assert "z2z" in layer0
    assert "targets" in layer0
    t1 = layer0["targets"]["1"]
    assert "z2u" in t1 and "0" in t1["z2u"]["heads"]
    assert "u2z" in t1
    assert len(t1["u2z"]["contributions"]) > 0
    assert len(t1["z2u"]["heads"]["0"]["contributions"]) == len(result["tokens"])


def test_unknown_model_type():
    from lig.adapters import load_adapter

    with pytest.raises(ValueError, match="Unsupported model_type"):
        load_adapter("t5-small", device="cpu")
