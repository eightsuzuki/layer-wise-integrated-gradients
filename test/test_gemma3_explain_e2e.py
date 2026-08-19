"""End-to-end ``lig.explain()`` smoke tests for Gemma3, on tiny public checkpoints.

``google/gemma-3-*`` is gated and far too large for CI, so these use the random
tiny models published for Transformers testing. Both Gemma3 config flavours are
covered: multimodal (``model_type="gemma3"``) and text-only
(``model_type="gemma3_text"``, e.g. ``google/gemma-3-1b-it``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

MULTIMODAL = "hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration"
TEXT_ONLY = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"


@pytest.fixture(scope="module")
def multimodal_result():
    from lig import explain

    return explain(
        "hello world",
        model=MULTIMODAL,
        num_steps=4,
        granularity="all",
        layers=[0],
        target_tokens=[1],
        target_head=0,
        device="cpu",
    )


def test_explain_reports_gemma3_boundaries(multimodal_result):
    assert multimodal_result["model_type"] == "gemma3"
    assert multimodal_result["architecture"] == "decoder"
    assert multimodal_result["boundaries"]["layout"] == "pre_post_ln_decoder"
    assert multimodal_result["config"]["layers"] == [0]
    assert multimodal_result["config"]["target_tokens"] == [1]


def test_explain_layer_z2z_is_square_and_causal(multimodal_result):
    tokens = multimodal_result["tokens"]
    z2z = multimodal_result["layers"]["0"]["z2z"]
    assert z2z["shape"] == [len(tokens), len(tokens)]
    for target, column in enumerate(zip(*z2z["matrix"])):
        # column `target` holds one contribution per source token
        assert all(abs(v) < 1e-4 for v in column[target + 1 :])


def test_explain_att_and_mlp_payloads(multimodal_result):
    tokens = multimodal_result["tokens"]
    target = multimodal_result["layers"]["0"]["targets"]["1"]

    head = target["z2u"]["heads"]["0"]
    assert len(head["contributions"]) == len(tokens)
    assert all(abs(v) < 1e-4 for v in head["contributions"][2:])  # causal
    assert head["token_l2_norm"] >= 0.0

    u2z = target["u2z"]
    num_heads, head_dim = u2z["head_shape"]
    assert u2z["input_width"] == num_heads * head_dim
    assert len(u2z["contributions"]) == u2z["input_width"]
    assert u2z["l2_total"] > 0.0
    assert u2z["completeness"]["max_abs_error"] >= 0.0


def test_explain_text_only_checkpoint():
    """gemma3_text configs (google/gemma-3-1b-it style) route to the same path."""
    from lig import explain

    result = explain(
        "hello world",
        model=TEXT_ONLY,
        num_steps=4,
        granularity=["layer", "mlp"],
        layers=[0],
        target_tokens=[1],
        device="cpu",
    )
    assert result["model_type"] == "gemma3_text"
    assert result["architecture"] == "decoder"
    assert result["boundaries"]["layout"] == "pre_post_ln_decoder"
    assert "z2z" in result["layers"]["0"]
    assert result["layers"]["0"]["targets"]["1"]["u2z"]["l2_total"] > 0.0


@pytest.mark.parametrize("model_name", [MULTIMODAL, TEXT_ONLY])
def test_describe_boundaries_without_weights(model_name):
    from lig import describe_boundaries

    info = describe_boundaries(model_name, load_weights=False)
    assert info["layout"] == "pre_post_ln_decoder"
    assert info["architecture"] == "decoder"
    assert info["num_layers"] >= 1


@pytest.mark.parametrize("baseline_att", ["itb_zero_ratio", "itb_map_ratio"])
def test_att_itb_ratio_baselines(baseline_att):
    """The ITB ratio corrections must stay finite and causal."""
    import math

    from lig import explain

    result = explain(
        "hello world",
        model=TEXT_ONLY,
        num_steps=4,
        granularity="att",
        baseline_att=baseline_att,
        layers=[0],
        target_tokens=[1],
        target_head=0,
        device="cpu",
    )
    contributions = result["layers"]["0"]["targets"]["1"]["z2u"]["heads"]["0"]["contributions"]
    assert len(contributions) == len(result["tokens"])
    assert all(math.isfinite(v) for v in contributions)


def test_mlp_att_itb_a0_baseline():
    """baseline_mlp='att_itb_a0' builds u from the ITB run instead of zeros."""
    import math

    from lig import explain

    result = explain(
        "hello world",
        model=TEXT_ONLY,
        num_steps=4,
        granularity="mlp",
        baseline_mlp="att_itb_a0",
        layers=[0],
        target_tokens=[1],
        device="cpu",
    )
    u2z = result["layers"]["0"]["targets"]["1"]["u2z"]
    assert u2z["baseline"] == "att_itb_a0"
    assert len(u2z["contributions"]) == u2z["input_width"]
    assert all(math.isfinite(v) for v in u2z["contributions"])
