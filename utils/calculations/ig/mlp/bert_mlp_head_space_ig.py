"""BERT MLP IG (u -> z) attributed in head space, before W_o.

``run_ptb_mlp_ig.py`` / ``mlp_ig.py`` slice the MLP input
``u = LayerNorm(z + W_o[a_1;...;a_H] + b_o)`` (the residual coordinates, after
``W_o`` has mixed the heads) into ``head_dim`` blocks and call those blocks
"per-head contributions". ``W_o`` has already mixed those coordinates, so the
blocks are not heads. Eq.~(mlp-o) differentiates with respect to
``u^{(l,h)}``, the head output *before* the projection -- exactly the
``context_layer`` that ``BertSelfAttention`` returns.

This module attributes to ``x = concat_h a_h`` (before ``W_o``) and only then
sums within each head's ``head_dim`` slice, mirroring the GPT-2 fix in
``layer-wise-integrated-gradients/lig/adapters/decoder_ig/mlp_ig.py``
(``run_mlp_head_space_ig``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients


def _get_encoder(bert_model):
    if hasattr(bert_model, "bert"):
        return bert_model.bert.encoder
    return bert_model.encoder


def head_context(bert_model, layer_idx: int, z_layer: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """``x = concat_h a_h``: BertSelfAttention の出力（W_o の前）。[1, seq, hidden]。"""
    layer = _get_encoder(bert_model).layer[layer_idx]
    out = layer.attention.self(z_layer, attention_mask)
    return out[0] if isinstance(out, tuple) else out


def compute_mlp_head_space_ig(
    bert_model,
    layer_idx: int,
    target_token_idx: int,
    z_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    mlp_baseline: str,
    num_steps: int = 32,
) -> Tuple[np.ndarray, float, dict]:
    """理論どおりの per-head MLP IG（頭空間で帰属する）。

    Args:
        mlp_baseline: "zero"（x=0、ヘッドを止めて z は残す）または
            "attitba0"（ATT の self_input_token（ITB）経路の a=0 終点を
            頭空間で渡す。§3.7.4 ATTITBa=0 と同じ考え方）。

    Returns:
        (per_head [num_heads], total, verification dict)
    """
    layer = _get_encoder(bert_model).layer[layer_idx]
    num_heads = bert_model.config.num_attention_heads
    head_dim = bert_model.config.hidden_size // num_heads

    with torch.no_grad():
        x = head_context(bert_model, layer_idx, z_layer, attention_mask)[:, target_token_idx, :].clone()
        z_tok = z_layer[:, target_token_idx, :].clone()

        if mlp_baseline == "zero":
            x_base = torch.zeros_like(x)
        elif mlp_baseline == "attitba0":
            _, seq_len, hidden = z_layer.shape
            z_j = z_layer[0, target_token_idx, :].clone()
            baseline_z = z_j.unsqueeze(0).unsqueeze(0).expand(1, seq_len, hidden)
            x_base = head_context(bert_model, layer_idx, baseline_z, attention_mask)[:, target_token_idx, :].clone()
        else:
            raise ValueError(f"未対応の mlp_baseline: {mlp_baseline!r}")

    def to_u(x_vec: torch.Tensor) -> torch.Tensor:
        z_b = z_tok.expand(x_vec.shape[0], -1)
        return layer.attention.output(x_vec, z_b)

    def mlp_final(x_vec: torch.Tensor) -> torch.Tensor:
        u = to_u(x_vec)
        inter = layer.intermediate(u)
        return layer.output(inter, u)

    with torch.no_grad():
        base_out = mlp_final(x_base).detach()

    def forward_fn(x_vec: torch.Tensor) -> torch.Tensor:
        return torch.norm(mlp_final(x_vec) - base_out, dim=-1)

    attr = IntegratedGradients(forward_fn).attribute(
        inputs=x.float(), baselines=x_base.float(), n_steps=num_steps
    )
    attr = attr.squeeze(0)
    per_head = attr.view(num_heads, head_dim).sum(-1)
    total = float(attr.sum().detach().cpu().item())

    with torch.no_grad():
        actual = float(forward_fn(x.float()).detach().cpu().item())
        base = float(forward_fn(x_base.float()).detach().cpu().item())
    theoretical_diff = actual - base
    relative_error = (
        abs(total - theoretical_diff) / abs(theoretical_diff)
        if abs(theoretical_diff) > 1e-8
        else abs(total - theoretical_diff)
    )
    verification = {
        "theoretical_diff": theoretical_diff,
        "ig_sum": total,
        "relative_error": float(relative_error),
        "per_head_boundary": "pre_proj",
        "mlp_baseline": mlp_baseline,
    }
    return per_head.detach().cpu().numpy().astype(np.float64), total, verification


def aggregate_subword_mlp_to_words(
    tokenizer,
    words,
    mlp_subword,
    max_sequence_length: int = 128,
):
    """サブワード単位の MLP per-head 値を、ATT キャッシュと同じ語空間に集約する。

    ATT 側 (``adapter._build_word_level_matrix``) は source/target とも語ごとに
    サブワードの寄与を **和** で集約し、[CLS]/[SEP] を語リストの末尾
    (``num_words``, ``num_words+1``) に置く。MLP 側の target 添字も同じ規約に
    そろえないと、合成時に語数の食い違いを先頭切り詰めで処理してしまい、
    分割が起きた語より後ろが丸ごとずれる（dev 1700 文中 1467 文で発生していた）。

    Args:
        mlp_subword: [n_layers][n_subword][n_heads]

    Returns:
        [n_layers][num_words + 2][n_heads]
    """
    encoding = tokenizer(
        list(words),
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_sequence_length,
    )
    word_ids = encoding.word_ids(0)
    num_words = len(words)
    cls_idx, sep_idx = num_words, num_words + 1
    total = num_words + 2

    sub_to_word = {}
    for sub_idx, wid in enumerate(word_ids):
        if wid is not None:
            sub_to_word[sub_idx] = wid
        elif sub_idx == 0:
            sub_to_word[sub_idx] = cls_idx
        elif sub_idx == len(word_ids) - 1:
            sub_to_word[sub_idx] = sep_idx

    out = []
    for layer in mlp_subword:
        num_heads = len(layer[0]) if layer else 0
        rows = [[0.0] * num_heads for _ in range(total)]
        for sub_idx, vec in enumerate(layer):
            w = sub_to_word.get(sub_idx)
            if w is None:
                continue
            for h in range(num_heads):
                rows[w][h] += float(vec[h])
        out.append(rows)
    return out
