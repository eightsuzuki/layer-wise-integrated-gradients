# §3.7.4: Attention の ITB 経路で積分パラメータ a=0 の ATT 出力を MLP のベースライン入力とする（表記 **ATTITBa=0**）。
"""
§3.7.4 対応ヘルパー（MLP 側ベースライン）。

ITB では z(0) = z_j を全位置に置いたときの ATT 出力が u(0)。
その u(0) のターゲット位置 j のベクトルを MLP の baseline_mlp_input として使う。

キャッシュディレクトリ名は `baseline_att_itb_attitba0` を正規表記とする。
"""

from __future__ import annotations

import lightning as L
import torch


def _get_encoder(bert_model: L.LightningModule):
    if hasattr(bert_model, "bert"):
        return bert_model.bert.encoder
    return bert_model.encoder


def get_mlp_baseline_att_itb_eq_zero(
    bert_model: L.LightningModule,
    z_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
) -> torch.Tensor:
    """
    ITB のベースライン入力 z(0) = z_j を全位置に置いたときの
    ATT 出力（＝ MLP 入力）u(0) の、ターゲット位置 j のベクトルを返す（ATTITBa=0）。

    Args:
        bert_model: BERT
        z_layer: 層 l の入力 z^{(l)} [1, seq_len, hidden]
        attention_mask: [1, seq_len]
        layer_idx: 層インデックス
        target_token_idx: 出力トークン j（ITB の基準トークン）

    Returns:
        u(0)_j [hidden]。MLP の baseline_mlp_input として使う。
    """
    _, seq_len, hidden = z_layer.shape
    device = z_layer.device
    dtype = z_layer.dtype
    z_j = z_layer[0, target_token_idx, :].clone()
    baseline_z = z_j.unsqueeze(0).unsqueeze(0).expand(1, seq_len, hidden)
    encoder = _get_encoder(bert_model)
    layer = encoder.layer[layer_idx]
    with torch.no_grad():
        attn_out = layer.attention(baseline_z, attention_mask)
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]
    u0_j = attn_out[0, target_token_idx, :].clone().to(device=device, dtype=dtype)
    return u0_j


get_mlp_baseline_from_att_itb = get_mlp_baseline_att_itb_eq_zero

__all__ = ["get_mlp_baseline_att_itb_eq_zero", "get_mlp_baseline_from_att_itb"]
