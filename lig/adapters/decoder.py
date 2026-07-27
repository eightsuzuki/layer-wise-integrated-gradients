"""
Decoder-only adapter — model load and boundary introspection.

For GPT-2 ATT/MLP IG, use :mod:`lig.adapters.decoder_ig` (:class:`GPT2Adapter`)
and :func:`lig.explain` with ``model="gpt2"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

# Planned first wave (GPT-2 block layout) + Gemma3
DECODER_FAMILY_TYPES = frozenset(
    {
        "gpt2",
        "gpt_neox",
        "llama",
        "mistral",
        "qwen2",
        "gemma",
        "gemma3",
    }
)

_IG_READY_TYPES = frozenset({"gpt2", "llama", "mistral", "qwen2", "gemma", "gemma3"})

@dataclass
class DecoderAdapter:
    """
    Decoder-only causal LM adapter.

    Block mapping::
        z^(l) = hidden before block l self-attention
        u^(l) = GPT-2 post-attention residual; Gemma3 concatenated attention
                 heads before the attention output projection
        z^(l+1) = hidden after MLP + residual
    """

    model: nn.Module
    tokenizer: Any
    model_name: str
    model_type: str
    device: torch.device
    architecture: str = "decoder"

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "DecoderAdapter":
        config = AutoConfig.from_pretrained(model_name)
        model_type = getattr(config, "model_type", "unknown")
        if model_type not in DECODER_FAMILY_TYPES:
            raise ValueError(
                f"Unsupported decoder model_type '{model_type}' for '{model_name}'."
            )

        load_kwargs: Dict[str, Any] = {
            "output_hidden_states": True,
            "attn_implementation": "eager",
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        if model_type == "gemma3":
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        model = model.to(dev)
        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            model_type=model_type,
            device=dev,
        )

    def get_blocks(self) -> nn.ModuleList:
        """Return transformer blocks (model-specific)."""
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h  # GPT-2
        # Multimodal Gemma3: model.language_model.layers or model.model.language_model.layers
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "layers"):
            return self.model.language_model.layers
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
                return inner.language_model.layers
            if hasattr(inner, "layers"):
                return inner.layers  # Llama-style / Gemma3ForCausalLM
        if hasattr(self.model, "layers"):
            return self.model.layers  # Gemma3TextModel
        raise NotImplementedError(
            f"Block enumeration not implemented for {self.model_type}. "
            "See docs/DECODER_DESIGN.md"
        )

    @property
    def num_layers(self) -> int:
        return len(self.get_blocks())

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def tokens_as_strings(self, text: str) -> List[str]:
        encoded = self.tokenizer(text, truncation=True, max_length=512)
        return self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    def forward_hidden_states(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states
        if hidden is None and hasattr(outputs, "language_model_outputs"):
            # Multimodal ConditionalGeneration may nest LM outputs.
            hidden = outputs.language_model_outputs.hidden_states
        if hidden is None:
            raise RuntimeError("Model did not return hidden_states")
        return hidden

    def ensure_ig_ready(self) -> None:
        if self.model_type in _IG_READY_TYPES:
            return
        raise NotImplementedError(
            f"Decoder LIG is not implemented yet for {self.model_type} ({self.model_name}). "
            f"Supported: {sorted(_IG_READY_TYPES)}. "
            "See docs/DECODER_DESIGN.md for the planned API."
        )
