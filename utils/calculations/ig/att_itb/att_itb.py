# att_itb.py
"""
Attention Input Token Baseline (ATT ITB) の実装

Attention 機構に対する Self Input Token Baseline (ITB) に基づく
Integrated Gradients を計算するラッパーです。

定義（論文・理論メモに合わせる）:
- 寄与を見たい出力トークンを j と固定する。
- ITB では入力補間: z_k^{(l)}(a; j) = z_j^{(l)} + a(z_k^{(l)} - z_j^{(l)})
- ベースラインは全位置で z_j^{(l)}（出力トークン j の入力表現）。
- これにより自己トークン j の Attention 寄与は 0、他トークンは j からの相対寄与として IG が得られる。

内部では attention_ig の baseline_method="self_input_token", input_type="z" を固定して呼び出します。
"""

from typing import Dict, List, Optional, Tuple

import lightning as L
import torch

from utils.calculations.ig.attention.attention_ig import (
    compute_attention_ig_global_analysis_multi_layer,
    compute_attention_ig_global_analysis_multi_layer_multi_token,
)

# ATT ITB で使用するベースライン名（attention_ig の baseline_method と一致）
ATT_ITB_BASELINE_METHOD = "self_input_token"
ATT_ITB_INPUT_TYPE = "z"


def compute_att_itb_multi_layer(
    bert_model: L.LightningModule,
    inputs: Dict[str, torch.Tensor],
    layer_indices: List[int],
    target_token_idx: int,
    target_head_idx: Optional[int] = None,
    num_steps: int = 50,
    debug: bool = False,
    cached_hidden_states: Optional[Tuple] = None,
    use_direct_computation: bool = True,
) -> Dict[int, Dict]:
    """
    Attention ITB（Input Token Baseline）に基づく複数レイヤーの IG を計算する。

    出力トークン j を固定し、入力補間 z_k(a;j) = z_j + a(z_k - z_j) に対する
    Integrated Gradients を計算します。Attention 機構として ITB は最も理にかなった
    貢献度の測り方とされる（論文 04_method.tex 参照）。

    Args:
        bert_model: BERT モデル
        inputs: 入力テンソル（input_ids, attention_mask, token_type_ids 等）
        layer_indices: 対象レイヤーインデックスリスト
        target_token_idx: 寄与を見る出力トークン j のインデックス
        target_head_idx: 対象ヘッド（None の場合は全ヘッドは未対応のため指定推奨）
        num_steps: 積分ステップ数
        debug: デバッグフラグ
        cached_hidden_states: 事前計算済み hidden states
        use_direct_computation: v→u 用オプション（z→u では未使用）

    Returns:
        Dict[int, Dict]: 各レイヤーの IG 結果
        layer_idx -> {"ig_values": List[float], "verification": None}
    """
    return compute_attention_ig_global_analysis_multi_layer(
        bert_model=bert_model,
        inputs=inputs,
        layer_indices=layer_indices,
        target_token_idx=target_token_idx,
        target_head_idx=target_head_idx,
        num_steps=num_steps,
        debug=debug,
        cached_hidden_states=cached_hidden_states,
        baseline_method=ATT_ITB_BASELINE_METHOD,
        input_type=ATT_ITB_INPUT_TYPE,
        use_direct_computation=use_direct_computation,
    )


def compute_att_itb_multi_layer_multi_token(
    bert_model: L.LightningModule,
    inputs: Dict[str, torch.Tensor],
    layer_indices: List[int],
    target_token_indices: List[int],
    target_head_idx: Optional[int] = None,
    num_steps: int = 50,
    debug: bool = False,
    cached_hidden_states: Optional[Tuple] = None,
) -> Dict[int, Dict[int, Dict]]:
    """
    Attention ITB に基づく複数レイヤー×複数出力トークンの IG を一度に計算する。

    各出力トークン j ごとに ITB（ベースライン z_j）で IG を計算します。

    Args:
        bert_model: BERT モデル
        inputs: 入力テンソル
        layer_indices: 対象レイヤーインデックスリスト
        target_token_indices: 寄与を見る出力トークン j のインデックスリスト
        target_head_idx: 対象ヘッド
        num_steps: 積分ステップ数
        debug: デバッグフラグ
        cached_hidden_states: 事前計算済み hidden states

    Returns:
        Dict[int, Dict[int, Dict]]:
        layer_idx -> token_idx -> {"ig_values": List[float], "verification": None}
    """
    return compute_attention_ig_global_analysis_multi_layer_multi_token(
        bert_model=bert_model,
        inputs=inputs,
        layer_indices=layer_indices,
        target_token_indices=target_token_indices,
        target_head_idx=target_head_idx,
        num_steps=num_steps,
        debug=debug,
        cached_hidden_states=cached_hidden_states,
        baseline_method=ATT_ITB_BASELINE_METHOD,
        input_type=ATT_ITB_INPUT_TYPE,
    )
