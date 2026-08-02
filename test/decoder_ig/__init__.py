"""Decoder-model IG adapters (ATT/MLP boundary attribution).

Unified entry for decoder IG:
- :class:`GPT2Adapter` — GPT-2 z→u / u→z
- :func:`lig.explain` with ``model="gpt2"`` delegates here

Model load / block enumeration without IG: :mod:`lig.adapters.decoder`.
"""

from lig.adapters.decoder_ig.base import DecoderIGAdapter
from lig.adapters.decoder_ig.gpt2 import (
    ATT_BASELINES,
    GPT2Adapter,
    MLP_BASELINES,
    normalize_attention_baseline,
    normalize_mlp_baseline,
)

__all__ = [
    "DecoderIGAdapter",
    "GPT2Adapter",
    "ATT_BASELINES",
    "MLP_BASELINES",
    "normalize_attention_baseline",
    "normalize_mlp_baseline",
]
