"""Load the correct adapter for a Hugging Face model id."""

from __future__ import annotations

from typing import Optional, Union

import torch

from lig.adapters.decoder import DECODER_FAMILY_TYPES, DecoderAdapter
from lig.adapters.encoder import BERT_FAMILY_TYPES, ENCODER_TYPES, EncoderAdapter
from transformers import AutoConfig

ModelAdapter = Union[EncoderAdapter, DecoderAdapter]


def load_adapter(
    model_name: str,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    *,
    allow_decoder_stub: bool = False,
) -> ModelAdapter:
    """
    Pick encoder or decoder adapter from ``config.model_type``.

    By default, decoder models raise ``NotImplementedError`` at explain-time.
    Pass ``allow_decoder_stub=True`` only for adapter introspection / tests.
    """
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "unknown")

    if model_type in ENCODER_TYPES:
        return EncoderAdapter.from_pretrained(model_name, device=device, torch_dtype=torch_dtype)

    if model_type in DECODER_FAMILY_TYPES:
        adapter = DecoderAdapter.from_pretrained(model_name, device=device, torch_dtype=torch_dtype)
        if not allow_decoder_stub:
            adapter.ensure_ig_ready()
        return adapter

    raise ValueError(
        f"Unsupported model_type '{model_type}' for '{model_name}'. "
        f"Encoders: {sorted(ENCODER_TYPES)}. "
        f"Decoders (planned): {sorted(DECODER_FAMILY_TYPES)}. "
        "See docs/DECODER_DESIGN.md."
    )
