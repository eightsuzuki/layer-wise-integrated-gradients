"""Factory for decoder IG adapters (GPT-2, Llama family, …)."""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoConfig

from lig.adapters.decoder_ig.base import DecoderIGAdapter
from lig.adapters.decoder_ig.gpt2 import GPT2Adapter
from lig.adapters.decoder_ig.llama import LlamaIGAdapter

LLAMA_DECODER_TYPES = frozenset({"llama", "mistral", "qwen2", "gemma"})
DECODER_IG_MODEL_TYPES = frozenset({"gpt2"}) | LLAMA_DECODER_TYPES


def load_decoder_ig_adapter(
    model_name: str,
    *,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> DecoderIGAdapter:
    """Load a :class:`DecoderIGAdapter` for the given Hugging Face model id."""
    model_type = getattr(AutoConfig.from_pretrained(model_name), "model_type", "unknown")
    if model_type == "gpt2":
        return GPT2Adapter(
            model_name=model_name,
            device=device,
            model=model,
            tokenizer=tokenizer,
        )
    if model_type in LLAMA_DECODER_TYPES:
        return LlamaIGAdapter(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            model=model,
            tokenizer=tokenizer,
        )
    raise NotImplementedError(
        f"Decoder LIG is not implemented for model_type={model_type!r} ({model_name}). "
        f"Supported: {sorted(DECODER_IG_MODEL_TYPES)}."
    )
