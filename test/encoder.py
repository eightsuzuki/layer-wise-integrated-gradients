"""
Hugging Face encoder adapter — auto-detect z (ATT input) / u (MLP input) boundaries.

Verified: BERT, RoBERTa, ELECTRA (see test/test_encoder_models.py).
Full ATT/MLP/Layer: BERT-family encoders + GPT-2 decoder.
Layer-only (z→z): MPNet, DistilBERT, ModernBERT, Switch MoE encoder, Mamba SSM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from lig.boundaries import (
    BlockBoundaries,
    detect_boundaries,
    forward_block,
    u_from_z as boundaries_u_from_z,
    z_at_layer as boundaries_z_at_layer,
)
from lig.encoder_access import get_encoder_layers

# Bidirectional encoders with BERT-style encoder.layer[i].attention + intermediate
BERT_FAMILY_TYPES = frozenset(
    {
        "bert",
        "roberta",
        "xlm-roberta",
        "electra",
        "deberta",
        "deberta-v2",
        "camembert",
        "albert",
    }
)

# Layer-whole IG only (block layout differs from BERT ATT/MLP split)
LAYER_ONLY_ENCODER_TYPES = frozenset(
    {
        "mpnet",
        "distilbert",
        "modernbert",
        "switch_transformers",
        "mamba",
    }
)

ENCODER_TYPES = BERT_FAMILY_TYPES | LAYER_ONLY_ENCODER_TYPES

FULL_GRANULARITY: FrozenSet[str] = frozenset({"att", "mlp", "layer"})
LAYER_GRANULARITY: FrozenSet[str] = frozenset({"layer"})


@dataclass
class EncoderAdapter:
    """Wraps a Hugging Face encoder and exposes ATT/MLP module boundaries."""

    model: nn.Module
    tokenizer: Any
    model_name: str
    model_type: str
    device: torch.device
    boundaries: BlockBoundaries
    architecture: str = "encoder"

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "EncoderAdapter":
        config = AutoConfig.from_pretrained(model_name)
        model_type = getattr(config, "model_type", "unknown")
        if model_type not in ENCODER_TYPES:
            raise ValueError(
                f"Unsupported encoder model_type '{model_type}' for '{model_name}'. "
                f"Supported: {sorted(ENCODER_TYPES)}"
            )

        load_kwargs: Dict[str, Any] = {
            "output_hidden_states": True,
            "output_attentions": False,
            "attn_implementation": "eager",
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        if model_type == "switch_transformers":
            from transformers import SwitchTransformersEncoderModel

            model = SwitchTransformersEncoderModel.from_pretrained(model_name, **load_kwargs)
        elif model_type == "mamba":
            from transformers import MambaModel

            model = MambaModel.from_pretrained(model_name, **load_kwargs)
        else:
            model = AutoModel.from_pretrained(model_name, **load_kwargs)

        if model_type == "mamba":
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        model = model.to(dev)
        model.eval()

        boundaries = detect_boundaries(model)
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            model_type=model_type,
            device=dev,
            boundaries=boundaries,
            architecture=boundaries.architecture,
        )

    @property
    def supported_granularity(self) -> FrozenSet[str]:
        return self.boundaries.supported_granularity

    @property
    def num_layers(self) -> int:
        return len(self.get_encoder_layers())

    @property
    def hidden_size(self) -> int:
        config = self.model.config
        return int(getattr(config, "hidden_size", None) or getattr(config, "d_model"))

    @property
    def num_attention_heads(self) -> int:
        return int(getattr(self.model.config, "num_attention_heads", 1))

    def get_encoder_layers(self) -> nn.ModuleList:
        return get_encoder_layers(self.model)

    def get_layer_module(self, layer_idx: int) -> nn.Module:
        return self.get_encoder_layers()[layer_idx]

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
            outputs = self.model(**inputs)
        hidden = outputs.hidden_states
        if hidden is None:
            raise RuntimeError("Model did not return hidden_states")
        return hidden

    def z_at_layer(
        self, hidden_states: Tuple[torch.Tensor, ...], layer_idx: int
    ) -> torch.Tensor:
        """z^(l): ATT input. Shape [1, seq, hidden]."""
        return boundaries_z_at_layer(hidden_states, layer_idx)

    def u_from_z(
        self,
        layer_idx: int,
        z_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        target_token_idx: int,
    ) -> torch.Tensor:
        """u_j: ATT output = MLP input. Shape [hidden]."""
        return boundaries_u_from_z(
            model=self.model,
            boundaries=self.boundaries,
            layer_idx=layer_idx,
            z_layer=z_layer,
            attention_mask=attention_mask,
            target_token_idx=target_token_idx,
        )

    def run_layer_forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return forward_block(
            model=self.model,
            boundaries=self.boundaries,
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    def inputs_for_ig(self, tokenized: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(tokenized)
        if "token_type_ids" not in out:
            out["token_type_ids"] = torch.zeros_like(out["input_ids"])
        return out

    def attention_mask_for_layer(
        self, tokenized: Dict[str, torch.Tensor], z_layer: torch.Tensor
    ) -> torch.Tensor:
        """Mask dtype matching z for SDPA."""
        if "attention_mask" not in tokenized:
            seq_len = z_layer.shape[1]
            return torch.ones(
                1, seq_len, device=z_layer.device, dtype=z_layer.dtype
            )
        return tokenized["attention_mask"].to(dtype=z_layer.dtype)
