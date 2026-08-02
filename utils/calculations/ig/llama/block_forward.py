"""Llama / Mistral / Qwen2 decoder block forward helpers for LIG."""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn


def make_position_embeddings(
    model: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE (cos, sin) for the current hidden states."""
    rotary = getattr(model, "rotary_emb", None)
    if rotary is None:
        raise AttributeError("Model has no rotary_emb; not a Llama-family decoder?")
    return rotary(hidden_states, position_ids)


def forward_llama_block(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One Llama Pre-LN block: z^(l) -> z^(l+1)."""
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    attn_out, _ = layer.self_attn(
        normed,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )
    hidden = residual + attn_out
    residual = hidden
    mlp_out = layer.mlp(layer.post_attention_layernorm(hidden))
    return residual + mlp_out


def hidden_after_attn_residual(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Post-attention residual stream (MLP input boundary u)."""
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    attn_out, _ = layer.self_attn(
        normed,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )
    return residual + attn_out
