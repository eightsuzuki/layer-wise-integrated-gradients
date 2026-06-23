"""
ベースライン埋め込み計算

理論文書「4.IGの経路の定義について.md」に基づく実装
"""

import logging
from typing import Optional

import lightning as L
import torch

from utils.calculations.ig.shared.release_scope import reject_otb_baseline

logger = logging.getLogger(__name__)


def compute_baseline_embeddings(
    baseline_method: str,
    input_embeddings: torch.Tensor,
    bert_model: L.LightningModule,
    layer_idx: int,
    target_token_idx: int,
    target_head_idx: Optional[int],
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    debug: bool = False,
    input_type: str = "z",  # "z": 入力埋め込み, "v": Valueベクトル
) -> torch.Tensor:
    """
    ベースライン埋め込みを計算する

    理論文書「4.IGの経路の定義について.md」に基づく:
    - 方法1 (zero): z_i^{base} = 0 （ゼロベクトル）- 4.1節参照
    - 方法2 (self_input_token): z_i^{base} = z_{i'} （すべてのトークンのベースラインを出力トークンi'の入力表現に設定）- 4.2節参照

    Args:
        baseline_method: ベースライン選択方法 ("zero", "self_input_token")
        input_embeddings: 入力埋め込み [1, seq_len, hidden]
        bert_model: BERTモデル
        layer_idx: 対象レイヤー
        target_token_idx: 対象トークンインデックス
        target_head_idx: 対象ヘッドインデックス
        attention_mask: アテンションマスク
        token_type_ids: トークンタイプID
        debug: デバッグフラグ
        input_type: 入力タイプ ("z": 入力埋め込み, "v": Valueベクトル)

    Returns:
        torch.Tensor: ベースライン埋め込み [1, seq_len, hidden] または [1, seq_len, head_dim] (input_type="v"の場合)
    """
    reject_otb_baseline(baseline_method)
    seq_len = input_embeddings.shape[1]
    hidden_size = input_embeddings.shape[2]
    device = input_embeddings.device
    dtype = input_embeddings.dtype

    if baseline_method == "zero":
        # 方法1: ゼロベースライン
        if input_type == "z":
            baseline_embeddings = torch.zeros(
                1, seq_len, hidden_size, device=device, dtype=dtype
            )
        elif input_type == "v":
            # input_embeddings は既に Value 空間（[1, seq_len, head_dim] など）
            baseline_embeddings = torch.zeros_like(input_embeddings)
        if debug:
            logger.debug(f"🔹 ベースライン方法: zero (ゼロベクトル)")

    elif baseline_method == "self_input_token":
        # 方法2: 自己入力トークンベースライン
        # z_i^{base} = z_{i'} （すべてのトークンのベースラインを出力トークンi'の入力表現に設定）
        if input_type == "z":
            target_embedding = input_embeddings[0, target_token_idx, :].clone()  # [hidden]
            # すべてのトークン位置に同じ埋め込みをコピー
            baseline_embeddings = (
                target_embedding.unsqueeze(0).unsqueeze(0).expand(1, seq_len, hidden_size)
            )
        elif input_type == "v":
            # input_embeddings は Value 空間。target token の Value を全位置へ複製する
            target_value_vector = input_embeddings[0, target_token_idx, :].clone()
            baseline_embeddings = target_value_vector.unsqueeze(0).unsqueeze(0).expand_as(
                input_embeddings
            )
        if debug:
            logger.debug(
                f"🔹 ベースライン方法: self_input_token (target_token_idx={target_token_idx})"
            )

    else:
        raise ValueError(f"未知のベースライン方法: {baseline_method}")

    return baseline_embeddings
