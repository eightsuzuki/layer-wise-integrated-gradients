"""Backward-compatible re-exports. Prefer ``lig.adapters.load_adapter``."""

from lig.adapters.decoder import DECODER_FAMILY_TYPES, DecoderAdapter
from lig.adapters.encoder import (
    BERT_FAMILY_TYPES,
    ENCODER_TYPES,
    LAYER_ONLY_ENCODER_TYPES,
    EncoderAdapter,
)
from lig.adapters.factory import ModelAdapter, load_adapter

__all__ = [
    "BERT_FAMILY_TYPES",
    "ENCODER_TYPES",
    "LAYER_ONLY_ENCODER_TYPES",
    "DECODER_FAMILY_TYPES",
    "DecoderAdapter",
    "EncoderAdapter",
    "ModelAdapter",
    "load_adapter",
]
