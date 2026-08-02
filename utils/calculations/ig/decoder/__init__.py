"""Decoder-model adapters and attribution helpers (re-export from lig)."""

from lig.adapters.decoder_ig import (
    ATT_BASELINES,
    DecoderIGAdapter,
    GPT2Adapter,
    MLP_BASELINES,
    normalize_attention_baseline,
    normalize_mlp_baseline,
)

# Backward-compatible alias used before LIG integration.
DecoderAdapter = DecoderIGAdapter

__all__ = [
    "DecoderAdapter",
    "DecoderIGAdapter",
    "GPT2Adapter",
    "ATT_BASELINES",
    "MLP_BASELINES",
    "normalize_attention_baseline",
    "normalize_mlp_baseline",
]
