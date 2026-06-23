"""Attention Map 対角 α_{j,j} の取得（ITB-mapRatio 用）。"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import GPT2Model


def get_encoder_self_attention_alpha(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    *,
    layer_idx: int,
    head_idx: int,
    token_idx: int,
) -> float:
    """Encoder の layer ``layer_idx``、head ``head_idx`` における α_{j,j}。"""
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return attentions")
    attn = attentions[layer_idx]
    return float(attn[0, head_idx, token_idx, token_idx].detach().cpu())


def get_gpt2_self_attention_alpha(
    gpt2: GPT2Model,
    *,
    inputs_embeds: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    token_idx: int,
) -> float:
    """GPT-2 causal attention の α_{j,j}。"""
    with torch.no_grad():
        outputs = gpt2(inputs_embeds=inputs_embeds, output_attentions=True)
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("GPT-2 did not return attentions")
    attn = attentions[layer_idx]
    return float(attn[0, head_idx, token_idx, token_idx].detach().cpu())
