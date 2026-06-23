"""
線形性を利用した直接計算によるAttention貢献度計算

理論文書「4.IGの経路の定義について.md」の5.3節に基づく実装。
入力$v$（Valueベクトル）で出力$u$（Attention出力）の場合、IGのような数値積分を行わずに
直接的に貢献度を計算できます。

理論:
- Zero Baseline: Contribution(v_j -> u_i) = w_{ij} v_j
- Self Input Token Baseline: Contribution(v_j -> u_i) = w_{ij} (v_j - v_i)

スカラー化:
各貢献度ベクトルのL2ノルムを計算してスカラー値（IG値）に変換します。
"""

import logging
from typing import Dict, List, Optional

import lightning as L
import torch

from .core.baseline_computation import compute_baseline_embeddings
from .core.value_extraction import extract_value_vectors

logger = logging.getLogger(__name__)


def compute_attention_contribution_direct(
    bert_model: L.LightningModule,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
    target_head_idx: Optional[int],
    baseline_method: str = "zero",
    debug: bool = False,
) -> List[float]:
    """
    線形性を利用した直接計算でAttention貢献度を計算

    理論文書「4.IGの経路の定義について.md」の5.3節に基づく。
    Attention機構が線形結合であるため、IGのような数値積分は不要で、
    一度の計算で直接貢献度を求められます。

    Args:
        bert_model: BERTモデル
        input_embeddings: 入力埋め込み [1, seq_len, hidden_size]
        attention_mask: アテンションマスク
        token_type_ids: トークンタイプID
        layer_idx: 対象レイヤーインデックス
        target_token_idx: 対象トークンインデックス（出力トークン i'）
        target_head_idx: 対象ヘッドインデックス
        baseline_method: ベースライン方法 ("zero", "self_input_token")
        debug: デバッグフラグ

    Returns:
        List[float]: 各トークン位置の貢献度（スカラー値、IG値相当）
    """
    device = input_embeddings.device
    seq_len = input_embeddings.shape[1]

    # Valueベクトルを取得
    value_vectors = extract_value_vectors(
        bert_model=bert_model,
        input_embeddings=input_embeddings,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        layer_idx=layer_idx,
        target_head_idx=target_head_idx,
        debug=debug,
    )
    # value_vectors: [1, seq_len, head_dim]

    # Query, Keyベクトルを取得してAttention重みを計算
    attention_weights = _compute_attention_weights(
        bert_model=bert_model,
        input_embeddings=input_embeddings,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
        target_head_idx=target_head_idx,
        debug=debug,
    )
    # attention_weights: [seq_len] (target_token_idxに対する各トークンのAttention重み)

    # ベースラインに応じて貢献度を計算
    if baseline_method == "zero":
        contributions = _compute_zero_baseline_contribution(
            value_vectors=value_vectors,
            attention_weights=attention_weights,
            target_token_idx=target_token_idx,
            debug=debug,
        )
    elif baseline_method == "self_input_token":
        contributions = _compute_self_input_token_baseline_contribution(
            value_vectors=value_vectors,
            attention_weights=attention_weights,
            target_token_idx=target_token_idx,
            debug=debug,
        )
    else:
        raise ValueError(f"未知のベースライン方法: {baseline_method}")

    # スカラー値（IG値相当）に変換: 各貢献度ベクトルのL2ノルム
    ig_values = []
    for pos_idx in range(seq_len):
        if baseline_method == "self_input_token" and pos_idx == target_token_idx:
            # Self Input Token Baselineの場合、自己トークンの寄与度は理論的に0
            ig_values.append(0.0)
        else:
            # 貢献度ベクトルのL2ノルムを計算
            contribution_vector = contributions[pos_idx]  # [head_dim]
            ig_value = torch.norm(contribution_vector, p=2).item()
            ig_values.append(ig_value)

    if debug:
        logger.debug(
            f"🔹 直接計算完了: layer={layer_idx}, head={target_head_idx}, "
            f"target_token={target_token_idx}, baseline={baseline_method}"
        )
        logger.debug(f"   貢献度の合計: {sum(ig_values):.6f}")

    return ig_values


def _compute_attention_weights(
    bert_model: L.LightningModule,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
    target_head_idx: Optional[int],
    debug: bool = False,
) -> torch.Tensor:
    """
    Attention重みを計算

    Args:
        bert_model: BERTモデル
        input_embeddings: 入力埋め込み
        attention_mask: アテンションマスク
        token_type_ids: トークンタイプID
        layer_idx: 対象レイヤーインデックス
        target_token_idx: 対象トークンインデックス（Queryトークン）
        target_head_idx: 対象ヘッドインデックス
        debug: デバッグフラグ

    Returns:
        torch.Tensor: Attention重み [seq_len] (target_token_idxに対する各トークンの重み)
    """
    # モデルの種類を判定
    if hasattr(bert_model, "bert"):
        encoder_layers = bert_model.bert.encoder.layer
        embeddings_layer = bert_model.bert.embeddings
    else:
        encoder_layers = bert_model.encoder.layer
        embeddings_layer = bert_model.embeddings

    # 位置エンコーディングとトークンタイプ埋め込みを追加
    from .attention_models import AttentionModel

    attention_model = AttentionModel(
        bert_model=bert_model,
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
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
            # Query, Keyベクトルを取得
            query_layer = layer.attention.self.query
            key_layer = layer.attention.self.key

            query_vectors = query_layer(hidden_states)  # [1, seq_len, hidden_size]
            key_vectors = key_layer(hidden_states)  # [1, seq_len, hidden_size]

            # ヘッドごとに分割
            config = (
                bert_model.config
                if hasattr(bert_model, "config")
                else bert_model.bert.config
            )
            num_heads = config.num_attention_heads
            head_dim = config.hidden_size // num_heads

            batch_size, seq_len, hidden_size = query_vectors.shape
            # [batch, seq_len, num_heads, head_dim]に変形
            query_vectors = query_vectors.view(batch_size, seq_len, num_heads, head_dim)
            key_vectors = key_vectors.view(batch_size, seq_len, num_heads, head_dim)

            if target_head_idx is not None:
                # 特定のヘッドのみを取得
                query_vectors = query_vectors[
                    :, :, target_head_idx, :
                ]  # [1, seq_len, head_dim]
                key_vectors = key_vectors[
                    :, :, target_head_idx, :
                ]  # [1, seq_len, head_dim]
            else:
                # 全ヘッドを平均（簡易実装）
                query_vectors = query_vectors.mean(dim=2)  # [1, seq_len, head_dim]
                key_vectors = key_vectors.mean(dim=2)  # [1, seq_len, head_dim]

            # target_token_idxのQueryベクトルを取得
            query_i = query_vectors[0, target_token_idx, :]  # [head_dim]

            # すべてのKeyベクトルとの内積を計算
            # attention_scores = query_i @ key_vectors[0].T  # [seq_len]
            attention_scores = torch.matmul(
                query_i.unsqueeze(0), key_vectors[0].T
            ).squeeze(
                0
            )  # [seq_len]

            # スケーリング
            attention_scores = attention_scores / (head_dim**0.5)

            # attention_maskを適用（マスクされた位置は-10000に設定）
            if attention_mask is not None:
                mask = attention_mask[0, :].float()  # [seq_len]
                attention_scores = attention_scores + (1.0 - mask) * (-10000.0)

            # Softmax正規化
            attention_weights = torch.softmax(attention_scores, dim=0)  # [seq_len]

            if debug:
                logger.debug(
                    f"Attention重み計算完了: target_token={target_token_idx}, "
                    f"重みの合計={attention_weights.sum().item():.6f}"
                )

            return attention_weights
        else:
            # 前の層を計算
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = (
                layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            )

    raise RuntimeError(f"Layer {layer_idx}のAttention重みを計算できませんでした")


def _compute_zero_baseline_contribution(
    value_vectors: torch.Tensor,
    attention_weights: torch.Tensor,
    target_token_idx: int,
    debug: bool = False,
) -> List[torch.Tensor]:
    """
    Zero Baselineでの貢献度を計算

    理論: Contribution(v_j -> u_i) = w_{ij} v_j

    Args:
        value_vectors: Valueベクトル [1, seq_len, head_dim]
        attention_weights: Attention重み [seq_len]
        target_token_idx: 対象トークンインデックス（使用しないが互換性のため）
        debug: デバッグフラグ

    Returns:
        List[torch.Tensor]: 各トークン位置の貢献度ベクトル [head_dim]
    """
    seq_len = value_vectors.shape[1]
    contributions = []

    for j in range(seq_len):
        # Contribution(v_j -> u_i) = w_{ij} v_j
        w_ij = attention_weights[j]
        v_j = value_vectors[0, j, :]  # [head_dim]
        contribution = w_ij * v_j  # [head_dim]
        contributions.append(contribution)

    if debug:
        logger.debug("Zero Baselineでの貢献度計算完了")

    return contributions


def _compute_self_input_token_baseline_contribution(
    value_vectors: torch.Tensor,
    attention_weights: torch.Tensor,
    target_token_idx: int,
    debug: bool = False,
) -> List[torch.Tensor]:
    """
    Self Input Token Baselineでの貢献度を計算

    理論: Contribution(v_j -> u_i) = w_{ij} (v_j - v_i)

    Args:
        value_vectors: Valueベクトル [1, seq_len, head_dim]
        attention_weights: Attention重み [seq_len]
        target_token_idx: 対象トークンインデックス（i'）
        debug: デバッグフラグ

    Returns:
        List[torch.Tensor]: 各トークン位置の貢献度ベクトル [head_dim]
    """
    seq_len = value_vectors.shape[1]
    contributions = []

    # 自己トークンのValueベクトルを取得
    v_i = value_vectors[0, target_token_idx, :]  # [head_dim]

    for j in range(seq_len):
        if j == target_token_idx:
            # 自己トークンの場合、理論的に貢献度は0
            contribution = torch.zeros_like(v_i)  # [head_dim]
        else:
            # Contribution(v_j -> u_i) = w_{ij} (v_j - v_i)
            w_ij = attention_weights[j]
            v_j = value_vectors[0, j, :]  # [head_dim]
            contribution = w_ij * (v_j - v_i)  # [head_dim]
        contributions.append(contribution)

    if debug:
        logger.debug(
            f"Self Input Token Baselineでの貢献度計算完了: "
            f"target_token={target_token_idx}"
        )

    return contributions


def compute_attention_all_tokens_direct_multi_layer_multi_token(
    bert_model: L.LightningModule,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    layer_indices: List[int],
    target_token_indices: List[int],
    target_head_idx: Optional[int],
    baseline_method: str = "zero",
    debug: bool = False,
) -> Dict[int, Dict[int, List[float]]]:
    """
    複数レイヤー・複数トークンに対して直接計算を実行

    Args:
        bert_model: BERTモデル
        input_embeddings: 入力埋め込み [1, seq_len, hidden_size]
        attention_mask: アテンションマスク
        token_type_ids: トークンタイプID
        layer_indices: 対象レイヤーインデックスリスト
        target_token_indices: 対象トークンインデックスリスト
        target_head_idx: 対象ヘッドインデックス
        baseline_method: ベースライン方法
        debug: デバッグフラグ

    Returns:
        Dict[layer_idx, Dict[token_idx, List[float]]]: 各レイヤー・各トークンのIG値リスト
    """
    results = {layer_idx: {} for layer_idx in layer_indices}

    for layer_idx in layer_indices:
        for token_idx in target_token_indices:
            try:
                ig_values = compute_attention_contribution_direct(
                    bert_model=bert_model,
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    layer_idx=layer_idx,
                    target_token_idx=token_idx,
                    target_head_idx=target_head_idx,
                    baseline_method=baseline_method,
                    debug=debug,
                )
                results[layer_idx][token_idx] = ig_values
            except Exception as e:
                logger.error(
                    f"❌ Layer {layer_idx}, Token {token_idx}の直接計算失敗: {e}"
                )
                # エラー時は空のリストを返す
                results[layer_idx][token_idx] = []

    return results
