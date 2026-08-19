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


def make_causal_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    """Additive causal mask ``(B, 1, T, T)`` for a decoder block.

    ``eager_attention_forward`` only applies a mask when one is passed, so a
    hand-rolled block forward has to build it; ``diagonal=1`` keeps self-attention.
    ``finfo.min`` (not ``-inf``) keeps the backward pass free of ``0 * inf = NaN``.
    """
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    mask = torch.full(
        (seq_len, seq_len),
        torch.finfo(hidden_states.dtype).min,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None].expand(batch_size, 1, seq_len, seq_len)


def forward_llama_block(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One Llama Pre-LN block: z^(l) -> z^(l+1).

    ``attention_mask=None`` means "causal", not "unmasked": these decoders are
    causal, and leaving the mask out silently yields bidirectional attention.
    """
    if attention_mask is None:
        attention_mask = make_causal_mask(hidden_states)
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
    """Post-attention residual stream (MLP input boundary u).

    ``attention_mask=None`` defaults to the causal mask, as in ``forward_llama_block``.
    """
    if attention_mask is None:
        attention_mask = make_causal_mask(hidden_states)
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)
    attn_out, _ = layer.self_attn(
        normed,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )
    return residual + attn_out
