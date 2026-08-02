"""Backward-compatible re-export. Prefer ``lig.adapters.decoder_ig.base``."""

from lig.adapters.decoder_ig.base import DecoderIGAdapter as DecoderAdapter

__all__ = ["DecoderAdapter"]
