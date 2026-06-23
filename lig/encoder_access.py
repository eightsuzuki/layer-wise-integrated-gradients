"""
Resolve encoder / block layout across Hugging Face model families for LIG.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def get_model_type(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    return str(getattr(config, "model_type", "unknown"))


def get_encoder_stack(model: nn.Module) -> nn.Module:
    """Return the module that owns ``.layer`` or ``.block`` (or Mamba ``.layers`` parent)."""
    model_type = get_model_type(model)

    if model_type == "mamba":
        return model
    if model_type == "modernbert":
        return model
    if model_type == "switch_transformers":
        return model.encoder
    if model_type == "distilbert":
        return model.transformer

    # Decoder-only causal LMs (GPT-2, Llama, …)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "layers"):
        return model.model
    if hasattr(model, "h"):
        return model

    for attr in ("bert", "roberta", "deberta", "electra", "mpnet"):
        if hasattr(model, attr):
            sub = getattr(model, attr)
            if hasattr(sub, "encoder"):
                return sub.encoder

    if hasattr(model, "encoder"):
        return model.encoder

    raise AttributeError(
        f"Cannot locate encoder stack on {type(model).__name__} (model_type={model_type})"
    )


def get_encoder_layers(model: nn.Module) -> nn.ModuleList:
    stack = get_encoder_stack(model)
    if hasattr(stack, "layers"):
        return stack.layers
    if hasattr(stack, "layer"):
        return stack.layer
    if hasattr(stack, "block"):
        return stack.block
    if hasattr(stack, "h"):
        return stack.h
    raise AttributeError(
        f"No layer/block list on {type(stack).__name__} (model_type={get_model_type(model)})"
    )


def forward_encoder_layer(
    model: nn.Module,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    layer_idx: int = 0,
) -> torch.Tensor:
    """
    Run one encoder block and return hidden states ``[batch, seq, hidden]``.
    """
    model_type = get_model_type(model)

    if model_type == "distilbert":
        mask = attention_mask.to(dtype=hidden_states.dtype)
        if mask.dim() == 2 and mask.shape[0] == 1 and hidden_states.shape[0] > 1:
            mask = mask.expand(hidden_states.shape[0], -1)
        out = layer(hidden_states, mask)
    elif model_type == "switch_transformers":
        seq_len = hidden_states.shape[1]
        cache_position = torch.arange(seq_len, device=hidden_states.device)
        out = layer(
            hidden_states,
            attention_mask=None,
            cache_position=cache_position,
        )
    elif model_type == "mamba":
        out = layer(hidden_states)
    elif model_type == "modernbert":
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        mask = attention_mask.to(dtype=hidden_states.dtype)
        out = layer(hidden_states, attention_mask=mask, position_ids=position_ids)
    else:
        mask = attention_mask.to(dtype=hidden_states.dtype)
        out = layer(hidden_states, attention_mask=mask)

    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "hidden_states"):
        out = out.hidden_states
    return out


def get_embeddings_module(model: nn.Module) -> Any:
    model_type = get_model_type(model)
    if model_type in {"mamba", "modernbert"}:
        return model.embeddings
    if model_type == "distilbert":
        return model.embeddings
    if model_type == "switch_transformers":
        return model.shared
    if hasattr(model, "bert"):
        return model.bert.embeddings
    if hasattr(model, "roberta"):
        return model.roberta.embeddings
    if hasattr(model, "embeddings"):
        return model.embeddings
    raise AttributeError(f"Cannot locate embeddings on {type(model).__name__}")
