"""GPT-2 block forward helpers for LIG (causal decoder)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Model


def create_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Causal additive mask [1, 1, seq, seq] for GPT-2 attention."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    mask = mask.view(1, 1, seq_len, seq_len)
    return (1.0 - mask) * torch.finfo(dtype).min


def forward_gpt2_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """One GPT-2 Pre-LN block: z^(l) -> z^(l+1)."""
    attn_out = block.attn(
        block.ln_1(hidden_states),
        attention_mask=attention_mask,
    )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    hidden = hidden_states + attn_out
    mlp_out = block.mlp(block.ln_2(hidden))
    return hidden + mlp_out


def hidden_after_attn_residual(
    block: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Post-attention residual stream (MLP input boundary u)."""
    attn_out = block.attn(
        block.ln_1(hidden_states),
        attention_mask=attention_mask,
    )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    return hidden_states + attn_out


def run_blocks_up_to(
    gpt2: GPT2Model,
    hidden_states: torch.Tensor,
    layer_idx: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run blocks ``0 .. layer_idx-1``; return z^(layer_idx)."""
    h = hidden_states
    for i in range(layer_idx):
        h = forward_gpt2_block(gpt2.h[i], h, attention_mask)
    return h


def embeddings_with_positions(gpt2: GPT2Model, token_embeddings: torch.Tensor) -> torch.Tensor:
    """wte + wpe (no token type)."""
    seq_len = token_embeddings.shape[1]
    position_ids = torch.arange(seq_len, device=token_embeddings.device).unsqueeze(0)
    return token_embeddings + gpt2.wpe(position_ids)
