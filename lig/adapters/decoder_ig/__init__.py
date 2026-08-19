"""Decoder-model IG adapters (ATT/MLP boundary attribution).

Unified entry for decoder IG:
- :class:`GPT2Adapter` — GPT-2 z→u / u→z
- :class:`LlamaIGAdapter` — Llama 2/3, Mistral, Qwen2, Gemma
- :class:`Qwen3IGAdapter` — dense Qwen3 with post-attention-residual u
- :func:`load_decoder_ig_adapter` — auto-select by ``model_type``
- :func:`lig.explain` with decoder model ids delegates here

Model load / block enumeration without IG: :mod:`lig.adapters.decoder`.
"""

from lig.adapters.decoder_ig.base import DecoderIGAdapter
from lig.adapters.decoder_ig.factory import (
    DECODER_IG_MODEL_TYPES,
    LLAMA_DECODER_TYPES,
    QWEN3_DECODER_TYPES,
    load_decoder_ig_adapter,
)
from lig.adapters.decoder_ig.gpt2 import (
    ATT_BASELINES,
    GPT2Adapter,
    MLP_BASELINES,
    normalize_attention_baseline,
    normalize_mlp_baseline,
)
from lig.adapters.decoder_ig.llama import LlamaIGAdapter
from lig.adapters.decoder_ig.qwen3 import Qwen3IGAdapter

__all__ = [
    "DecoderIGAdapter",
    "GPT2Adapter",
    "LlamaIGAdapter",
    "Qwen3IGAdapter",
    "load_decoder_ig_adapter",
    "DECODER_IG_MODEL_TYPES",
    "LLAMA_DECODER_TYPES",
    "QWEN3_DECODER_TYPES",
    "ATT_BASELINES",
    "MLP_BASELINES",
    "normalize_attention_baseline",
    "normalize_mlp_baseline",
]
