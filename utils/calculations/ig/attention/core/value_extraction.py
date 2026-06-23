"""
Valueベクトル抽出

理論文書の定義に基づく:
v_i^{(l,h)} = W_v^{(l,h)} z_i^{(l)}
"""

import logging
from typing import Optional

import lightning as L
import torch

logger = logging.getLogger(__name__)


def extract_value_vectors(
    bert_model: L.LightningModule,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    layer_idx: int,
    target_head_idx: Optional[int],
    debug: bool = False,
) -> torch.Tensor:
    """
    Valueベクトルを取得する

    理論文書の定義に基づく:
    v_i^{(l,h)} = W_v^{(l,h)} z_i^{(l)}

    Args:
        bert_model: BERTモデル
        input_embeddings: 入力埋め込み z^{(l)} [1, seq_len, hidden_size]
        attention_mask: アテンションマスク
        token_type_ids: トークンタイプID
        layer_idx: 対象レイヤーインデックス
        target_head_idx: 対象ヘッドインデックス（Noneの場合は全ヘッド）
        debug: デバッグフラグ

    Returns:
        torch.Tensor: Valueベクトル [1, seq_len, head_dim] (target_head_idx指定時) または [1, seq_len, hidden_size] (全ヘッド)
    """
    # モデルの種類を判定して適切にアクセス
    if hasattr(bert_model, "bert"):
        encoder_layers = bert_model.bert.encoder.layer
        embeddings_layer = bert_model.bert.embeddings
    else:
        encoder_layers = bert_model.encoder.layer
        embeddings_layer = bert_model.embeddings

    # attention_maskの型を修正
    if attention_mask.dtype != torch.float32:
        attention_mask = attention_mask.float()

    # 位置エンコーディングとトークンタイプ埋め込みを追加
    from ..attention_models import AttentionModel

    attention_model = AttentionModel(
        bert_model=bert_model,
        layer_idx=layer_idx,
        target_token_idx=0,  # ダミー値
        target_head_idx=target_head_idx,
        debug=debug,
    )
    embeddings = attention_model._add_positional_embeddings(
        input_embeddings, token_type_ids, embeddings_layer
    )

    # 指定された層まで順次計算
    hidden_states = embeddings
    for l_idx in range(layer_idx + 1):
        layer = encoder_layers[l_idx]

        if l_idx == layer_idx:
            # 対象層のValueベクトルを取得
            # Value projection: W_v^{(l,h)} z_i^{(l)}
            value_layer = layer.attention.self.value
            value_vectors = value_layer(hidden_states)  # [batch, seq_len, hidden_size]

            # ヘッドごとに分割
            config = (
                bert_model.config
                if hasattr(bert_model, "config")
                else bert_model.bert.config
            )
            num_heads = config.num_attention_heads
            head_dim = config.hidden_size // num_heads

            batch_size, seq_len, hidden_size = value_vectors.shape
            # [batch, seq_len, num_heads, head_dim]に変形
            value_vectors = value_vectors.view(batch_size, seq_len, num_heads, head_dim)

            if target_head_idx is not None:
                # 特定のヘッドのみを取得 [batch, seq_len, head_dim]
                value_vectors = value_vectors[:, :, target_head_idx, :]
            else:
                # 全ヘッドを結合 [batch, seq_len, hidden_size]
                value_vectors = value_vectors.view(batch_size, seq_len, hidden_size)

            if debug:
                logger.debug(f"Valueベクトル形状: {value_vectors.shape}")

            return value_vectors
        else:
            # 前の層を計算
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = (
                layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            )

    # ここには到達しないはず
    raise RuntimeError(f"Layer {layer_idx}のValueベクトルを取得できませんでした")

