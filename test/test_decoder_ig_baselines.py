"""Unit tests for decoder IG adapter baseline naming (LIG public names)."""

from __future__ import annotations

import pytest

from lig.adapters.decoder_ig.gpt2 import (
    ATT_ITB_MAP_RATIO,
    ATT_ITB_ZERO_RATIO,
    ATT_SELF_INPUT_TOKEN,
    ATT_ZERO,
    MLP_ATT_ITB_A0,
    MLP_ZERO,
    normalize_attention_baseline,
    normalize_mlp_baseline,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("zero", ATT_ZERO),
        ("self_input_token", ATT_SELF_INPUT_TOKEN),
        ("itb_raw", ATT_SELF_INPUT_TOKEN),
        ("itb_zero_ratio", ATT_ITB_ZERO_RATIO),
        ("itb_igzero", ATT_ITB_ZERO_RATIO),
        ("itb_map_ratio", ATT_ITB_MAP_RATIO),
        ("itb_attention_map", ATT_ITB_MAP_RATIO),
    ],
)
def test_normalize_attention_baseline_lig_names(raw: str, expected: str) -> None:
    assert normalize_attention_baseline(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("zero", MLP_ZERO),
        ("att_itb_a0", MLP_ATT_ITB_A0),
        ("attitba0", MLP_ATT_ITB_A0),
    ],
)
def test_normalize_mlp_baseline_lig_names(raw: str, expected: str) -> None:
    assert normalize_mlp_baseline(raw) == expected
