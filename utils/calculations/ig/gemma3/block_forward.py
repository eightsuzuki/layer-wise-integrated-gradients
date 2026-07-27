"""
Gemma 3 pre+post-LN block helpers for LIG.

Depends on transformers Gemma3 modules (tested against 4.57.x)::

    z  -> input_layernorm -> self_attn -> post_attention_layernorm -> +residual = u
    u  -> pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm -> +residual = z'

Per-head ATT IG uses the pre-``o_proj`` attention output
``[batch, seq, n_head, head_dim]`` (not residual-stream space).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

# transformers.models.gemma3 internals (version-sensitive; see docs/DECODER_DESIGN.md)
from transformers.models.gemma3.modeling_gemma3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def get_gemma3_text_model(model: nn.Module) -> nn.Module:
    """
    Resolve ``Gemma3TextModel`` from a loaded HF checkpoint.

    Supports:
    - ``Gemma3TextModel`` directly
    - ``Gemma3ForCausalLM`` (``.model``)
    - ``Gemma3ForConditionalGeneration`` / multimodal (``.model.language_model`` or ``.language_model``)
    """
    if hasattr(model, "layers") and hasattr(model, "embed_tokens") and hasattr(model, "rotary_emb"):
        return model
    if hasattr(model, "language_model"):
        return get_gemma3_text_model(model.language_model)
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model"):
            return get_gemma3_text_model(inner.language_model)
        return get_gemma3_text_model(inner)
    raise AttributeError(
        f"Cannot locate Gemma3TextModel on {type(model).__name__}. "
        "Expected gemma3 weights."
    )


def get_gemma3_layers(text_model: nn.Module) -> nn.ModuleList:
    return text_model.layers


def create_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive full causal mask ``[1, 1, seq, seq]`` (0 allowed, -inf blocked)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    mask = mask.view(1, 1, seq_len, seq_len)
    return (1.0 - mask) * torch.finfo(dtype).min


def create_sliding_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    sliding_window: int,
) -> torch.Tensor:
    """
    Additive sliding-window causal mask ``[1, 1, seq, seq]``.

    Position ``(i, j)`` is allowed iff ``0 <= i - j < sliding_window``.
    """
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    allowed = (cols <= rows) & ((rows - cols) < sliding_window)
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
    return mask.view(1, 1, seq_len, seq_len)


def layer_attention_mask(
    text_model: nn.Module,
    layer_idx: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-layer causal or sliding-window additive mask."""
    block = text_model.layers[layer_idx]
    if getattr(block.self_attn, "is_sliding", False):
        window = int(block.self_attn.sliding_window or text_model.config.sliding_window)
        return create_sliding_causal_mask(seq_len, device, dtype, window)
    return create_causal_mask(seq_len, device, dtype)


def rope_position_embeddings(
    text_model: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(position_embeddings_global, position_embeddings_local)`` as (cos, sin) pairs."""
    seq_len = hidden_states.shape[1]
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    global_pe = text_model.rotary_emb(hidden_states, position_ids)
    local_pe = text_model.rotary_emb_local(hidden_states, position_ids)
    return global_pe, local_pe


def _position_embeddings_for_block(
    block: nn.Module,
    position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
    position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if block.self_attn.is_sliding:
        return position_embeddings_local
    return position_embeddings_global


def attn_pre_oproj_output(
    block: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    *,
    target_head_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Attention core up to (but not including) ``o_proj``.

    Returns:
        - if ``target_head_idx`` is set: ``[batch, seq, head_dim]``
        - else: ``[batch, seq, n_head * head_dim]`` (concatenated heads)
    """
    attn = block.self_attn
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states = attn.q_norm(query_states)
    key_states = attn.k_norm(key_states)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    softcap = getattr(attn, "attn_logit_softcapping", None)
    attn_output, _ = eager_attention_forward(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=attn.scaling,
        softcap=softcap,
    )
    # attn_output: [batch, seq, n_head, head_dim]
    if target_head_idx is not None:
        return attn_output[:, :, target_head_idx, :]
    return attn_output.reshape(*input_shape, -1).contiguous()


def attn_branch_output(
    block: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
    position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Full attention branch in residual space (through ``o_proj`` + ``post_attention_layernorm``).

    Does **not** add the residual (caller adds ``z + branch`` for ``u``).
    """
    pe = _position_embeddings_for_block(
        block, position_embeddings_global, position_embeddings_local
    )
    normed = block.input_layernorm(hidden_states)
    attn_out, _ = block.self_attn(
        hidden_states=normed,
        position_embeddings=pe,
        attention_mask=attention_mask,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
    )
    return block.post_attention_layernorm(attn_out)


def hidden_after_attn_residual(
    block: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
    position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Actual post-attention residual stream (the architecture's FFN input).

    The Gemma 3 LIG boundary named ``u`` is earlier: concatenated attention
    heads before the output projection. This helper remains useful for block
    parity and full-forward calculations.
    """
    branch = attn_branch_output(
        block,
        hidden_states,
        attention_mask,
        position_embeddings_global,
        position_embeddings_local,
    )
    return hidden_states + branch


def mlp_branch(block: nn.Module, u: torch.Tensor) -> torch.Tensor:
    """MLP branch without residual: ``pre_ffn_ln -> mlp -> post_ffn_ln``."""
    h = block.pre_feedforward_layernorm(u)
    h = block.mlp(h)
    return block.post_feedforward_layernorm(h)


def forward_gemma3_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
    position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """One Gemma3 pre+post-LN block: ``z^(l) -> z^(l+1)``."""
    out = block(
        hidden_states,
        position_embeddings_global=position_embeddings_global,
        position_embeddings_local=position_embeddings_local,
        attention_mask=attention_mask,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
    )
    if isinstance(out, tuple):
        return out[0]
    return out


def run_blocks_up_to(
    text_model: nn.Module,
    hidden_states: torch.Tensor,
    layer_idx: int,
    *,
    position_ids: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """Run blocks ``0 .. layer_idx-1``; return ``z^(layer_idx)``."""
    seq_len = hidden_states.shape[1]
    device, dtype = hidden_states.device, hidden_states.dtype
    global_pe, local_pe = rope_position_embeddings(text_model, hidden_states, position_ids)
    h = hidden_states
    for i in range(layer_idx):
        mask = layer_attention_mask(text_model, i, seq_len, device, dtype)
        h = forward_gemma3_block(
            text_model.layers[i],
            h,
            mask,
            global_pe,
            local_pe,
        )
    return h


def embed_tokens(text_model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Token embeddings (scaled embedding already applied inside ``embed_tokens``)."""
    return text_model.embed_tokens(input_ids)
