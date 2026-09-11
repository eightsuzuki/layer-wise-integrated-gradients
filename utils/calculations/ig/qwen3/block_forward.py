"""Qwen3 dense-decoder block helpers used by LIG.

The LIG boundaries are::

    z  = residual stream entering a decoder layer
    u  = z + o_proj(concatenated attention heads)
    z' = u + mlp(post_attention_layernorm(u))

Qwen3-1.7B uses full causal attention.  The helpers nevertheless preserve a
2-D tokenizer padding mask and keep position ids fixed along the IG path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def get_qwen3_model(model: nn.Module) -> nn.Module:
    """Resolve the inner ``Qwen3Model`` from a base or causal-LM model."""
    if hasattr(model, "layers") and hasattr(model, "embed_tokens") and hasattr(
        model, "rotary_emb"
    ):
        return model
    inner = getattr(model, "model", None)
    if inner is not None:
        return get_qwen3_model(inner)
    raise AttributeError(
        f"Cannot locate Qwen3Model on {type(model).__name__}; expected dense qwen3 weights."
    )


def create_qwen3_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return an additive causal/padding mask of shape ``[batch, 1, T, T]``."""
    min_value = torch.finfo(dtype).min
    allowed = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    additive = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    additive = additive.masked_fill(~allowed, min_value).view(1, 1, seq_len, seq_len)

    if attention_mask is None:
        return additive
    if attention_mask.dim() != 2 or attention_mask.shape[-1] != seq_len:
        raise ValueError(
            "Qwen3 attention_mask must have shape [batch, seq_len]; "
            f"got {tuple(attention_mask.shape)} for seq_len={seq_len}."
        )
    key_padding = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
    batch_mask = additive.expand(attention_mask.shape[0], -1, -1, -1).clone()
    return batch_mask.masked_fill(~key_padding, min_value)


def qwen3_position_embeddings(
    qwen3: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the shared Qwen3 RoPE cosine/sine tensors."""
    return qwen3.rotary_emb(hidden_states, position_ids)


def forward_qwen3_attention(
    layer: nn.Module,
    z: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return the projected attention branch and optional attention weights."""
    normed = layer.input_layernorm(z)
    out, weights = layer.self_attn(
        normed,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
        output_attentions=True,
    )
    return out, weights


def attention_pre_oproj_output(
    layer: nn.Module,
    z: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Capture concatenated Q-head output immediately before ``o_proj``.

    Calling the official attention module preserves Q/K RMSNorm, GQA, RoPE,
    and Transformers-version-specific details.  The captured tensor remains
    attached to autograd, which is required by Captum.
    """
    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured["heads"] = args[0]

    handle = layer.self_attn.o_proj.register_forward_pre_hook(capture_input)
    try:
        _, weights = forward_qwen3_attention(layer, z, position_embeddings, attention_mask)
    finally:
        handle.remove()
    if "heads" not in captured:
        raise RuntimeError("Qwen3 attention did not call o_proj; cannot capture head output")
    return captured["heads"], weights


def project_qwen3_head(
    layer: nn.Module, concatenated_heads: torch.Tensor, head_idx: int
) -> torch.Tensor:
    """Project one pre-``o_proj`` head into the residual-stream space.

    The returned ``delta_u_h`` has width ``hidden_size``.  Qwen3's attention
    projection is bias-free, so summing it over heads reconstructs the full
    attention branch exactly.
    """
    attn = layer.self_attn
    num_heads = int(attn.config.num_attention_heads)
    head_dim = int(getattr(attn, "head_dim", attn.config.head_dim))
    if not 0 <= head_idx < num_heads:
        raise IndexError(f"head_idx={head_idx} out of range for num_heads={num_heads}")
    if attn.o_proj.bias is not None:
        raise NotImplementedError("Qwen3 head projection decomposition assumes bias-free o_proj")
    start = head_idx * head_dim
    end = start + head_dim
    head = concatenated_heads[..., start:end]
    weight = attn.o_proj.weight[:, start:end]
    return torch.nn.functional.linear(head, weight)


def hidden_after_attn_residual(
    layer: nn.Module,
    z: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute Qwen3's ``u = z + attention_branch(z)`` boundary."""
    attn_out, _ = forward_qwen3_attention(layer, z, position_embeddings, attention_mask)
    return z + attn_out


def forward_qwen3_block(
    layer: nn.Module,
    z: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run one dense Qwen3 decoder block from ``z`` to ``z'``."""
    out = layer(
        z,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        past_key_values=None,
        use_cache=False,
    )
    if isinstance(out, tuple):
        return out[0]
    return out
