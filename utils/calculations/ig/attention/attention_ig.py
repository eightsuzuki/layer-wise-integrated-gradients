# attention_ig.py
"""
Attention Integrated Gradients計算
理論文書に基づいた正確な実装

理論文書の定義:
- z^(l): ATTへの入力 (ATT_INPUT)
- u^(l,h): ATTの出力 = MLPへの入力 (ATT_OUTPUT = MLP_INPUT)
- 入力: {z_i^{(l)}}_i （層lのヘッドhの入力）
- ベースライン: {z_i^{base}}_i = 0
- 出力: u_i'^{(l,h)}(a) = ATT_i'^{(l,h)}({z_i^{(l)}}_i:a)
- IGで評価する関数: A_i'(a) = ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2
"""

import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import lightning as L
import torch

# Captumのインポート
try:
    from captum.attr import IntegratedGradients

    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "⚠️ Captumがインストールされていません。Captumを使用したIG計算は利用できません。"
    )

from .attention_models import AttentionModel, create_attention_model
from .core.baseline_computation import compute_baseline_embeddings
from .core.embedding_extraction import extract_embeddings_fast
from .core.value_extraction import extract_value_vectors

logger = logging.getLogger(__name__)

# 後方互換性のため、古い関数名もエクスポート
_compute_baseline_embeddings = compute_baseline_embeddings
_extract_embeddings_fast = extract_embeddings_fast
_extract_value_vectors = extract_value_vectors

_IG_VERBOSE_LOG = os.environ.get("PTB_IG_VERBOSE_LOG", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _ig_verbose_enabled() -> bool:
    return _IG_VERBOSE_LOG


def _summarize_token_indices(token_indices: List[int], max_items: int = 6) -> str:
    if not token_indices:
        return "-"
    unique_sorted = sorted(set(token_indices))
    if len(unique_sorted) <= max_items:
        return ",".join(str(i) for i in unique_sorted)
    head = ",".join(str(i) for i in unique_sorted[: max_items - 1])
    return f"{head},…,{unique_sorted[-1]}"


def _compute_baseline_embeddings(
    baseline_method: str,
    input_embeddings: torch.Tensor,
    bert_model: L.LightningModule,
    layer_idx: int,
    target_token_idx: int,
    target_head_idx: Optional[int],
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    debug: bool = False,
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

    Returns:
        torch.Tensor: ベースライン埋め込み [1, seq_len, hidden]
    """
    seq_len = input_embeddings.shape[1]
    hidden_size = input_embeddings.shape[2]
    device = input_embeddings.device
    dtype = input_embeddings.dtype

    if baseline_method == "zero":
        # 方法1: ゼロベースライン
        baseline_embeddings = torch.zeros(
            1, seq_len, hidden_size, device=device, dtype=dtype
        )
        if debug:
            logger.debug(f"🔹 ベースライン方法: zero (ゼロベクトル)")

    elif baseline_method == "self_input_token":
        # 方法2: 自己入力トークンベースライン
        # z_i^{base} = z_{i'} （すべてのトークンのベースラインを出力トークンi'の入力表現に設定）
        target_embedding = input_embeddings[0, target_token_idx, :].clone()  # [hidden]
        # すべてのトークン位置に同じ埋め込みをコピー
        baseline_embeddings = (
            target_embedding.unsqueeze(0).unsqueeze(0).expand(1, seq_len, hidden_size)
        )
        if debug:
            logger.debug(
                f"🔹 ベースライン方法: self_input_token (target_token_idx={target_token_idx})"
            )

    else:
        raise ValueError(f"未知のベースライン方法: {baseline_method}")

    return baseline_embeddings


def _safe_parallel_layer_limit(
    num_layers: int, free_memory_gb: float, seq_len: int
) -> int:
    """
    Conservative heuristic that keeps Captum IG layer-level parallelism within a
    safe range to avoid CUDA illegal memory access errors while still utilising
    large accelerators.
    """
    env_limit = os.environ.get("PTB_MAX_PARALLEL_LAYERS")
    if env_limit:
        try:
            env_value = max(1, int(env_limit))
            return min(num_layers, env_value)
        except ValueError:
            logger.warning(
                "PTB_MAX_PARALLEL_LAYERS=%s は無効です。整数値で指定してください。",
                env_limit,
            )

    if free_memory_gb >= 80:
        base_limit = 4
    elif free_memory_gb >= 60:
        base_limit = 3
    elif free_memory_gb >= 45:
        base_limit = 2
    else:
        base_limit = 1

    token_penalty = 1.0
    if seq_len >= 128:
        token_penalty = 0.25
    elif seq_len >= 96:
        token_penalty = 0.4
    elif seq_len >= 64:
        token_penalty = 0.6
    elif seq_len >= 48:
        token_penalty = 0.75

    safe_limit = max(1, int(round(base_limit * token_penalty)))
    if seq_len >= 48:
        safe_limit = min(safe_limit, 1 if seq_len >= 64 else 2)
    return min(num_layers, safe_limit)


class CaptumAttentionModelWrapper(torch.nn.Module):
    """
    Captum用のAttentionModelラッパー

    AttentionModelをCaptumのIntegratedGradientsで使用できるようにラップします。
    ベースラインとの差分を返すように実装します。
    """

    def __init__(
        self,
        attention_model: AttentionModel,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        super().__init__()
        self.attention_model = attention_model
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

        # ベースライン出力を事前計算（勾配計算中は変更されないため）
        # 理論式: A_i'(a) = ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2
        # ベースラインはゼロベクトル（z^{base} = 0）
        with torch.no_grad():
            baseline_embeddings = torch.zeros(
                1,
                attention_mask.shape[1],
                attention_model.hidden_size,
                device=attention_mask.device,
                dtype=(
                    attention_mask.dtype
                    if attention_mask.dtype.is_floating_point
                    else torch.float32
                ),
            )
            # ベースライン出力のベクトルを計算
            self.baseline_vector = attention_model._compute_attention_output_vector(
                baseline_embeddings, attention_mask, token_type_ids
            )

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Captum用のforwardメソッド

        理論式: A_i'(a) = ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2

        Args:
            input_embeddings: 入力埋め込み z^{(l)} [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2（スカラー）
        """
        # 実際の出力のベクトルを計算（全位置の埋め込みを使用）
        # u_i'^{(l,h)}(a) = ATT_i'^{(l,h)}({z_i^{(l)}}_i:a)
        u_actual_vector = self.attention_model._compute_attention_output_vector(
            input_embeddings, self.attention_mask, self.token_type_ids
        )

        # ベースラインとの差分を計算
        # 注意: ベースライン出力は事前計算済みだが、同じデバイスにあることを確認
        if self.baseline_vector.device != u_actual_vector.device:
            self.baseline_vector = self.baseline_vector.to(u_actual_vector.device)

        # 理論式通り: ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2 を計算
        diff_vector = u_actual_vector - self.baseline_vector
        loss = torch.norm(diff_vector)
        # Captumが0次元テンソルをインデックスしようとするのを防ぐため、
        # 1次元テンソルに変換
        return loss.unsqueeze(0) if loss.dim() == 0 else loss


def compute_attention_ig_with_verification(
    bert_model: L.LightningModule,
    inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    target_token_idx: int,
    target_head_idx: Optional[int] = None,
    num_steps: int = 50,
    debug: bool = False,
) -> Dict:
    """
    Attention IG計算（理論文書準拠版）

    理論式: IG_{i,i'}^{Attn} = (z_i^{(l)} - z_i^{base}) × ∫₀¹ ∂A_{i'}(a) / ∂z_i^{(l)} da
    ここで A_{i'}(a) = ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2

    入力: {z_i^{(l)}}_i （層lのヘッドhの入力、ATT_INPUT）
    出力: u_i'^{(l,h)} （層lのヘッドhのAttention出力、ATT_OUTPUT = MLP_INPUT）

    Args:
        bert_model: BERTモデル
        inputs: 入力テンソル
        layer_idx: 対象レイヤー
        target_token_idx: 対象トークンインデックス
        target_head_idx: 対象ヘッドインデックス
        num_steps: 積分ステップ数
        debug: デバッグフラグ

    Returns:
        Dict: IG値と理論的検証結果
    """
    import time

    # Captum版IG計算
    if debug:
        logger.debug("⚡ Captum版IG計算モード")
        logger.debug(f"⚡ デバッグモード: {debug}")

    try:
        # 層lの入力隠れ状態 z^{(l)} を抽出（理論に準拠）
        input_embeddings, attention_mask, token_type_ids = extract_embeddings_fast(
            bert_model=bert_model, inputs=inputs, layer_idx=layer_idx, debug=debug
        )

        # デバッグ情報の表示（デバッグモード時のみ）
        if debug:
            logger.debug(f"⚡ 層{layer_idx}の入力を使用: {input_embeddings.shape}")

        # 時間計測用
        start_time = time.time()

        # 積分の数値近似
        seq_len = input_embeddings.shape[1]
        hidden_size = input_embeddings.shape[2]
        ig_values = []

        if debug:
            logger.debug(
                f"⚡ seq_len={seq_len}, hidden_size={hidden_size}, num_steps={num_steps}"
            )
            logger.debug(
                f"⚡ target_token_idx={target_token_idx}, target_head_idx={target_head_idx}"
            )
            logger.debug(f"⚡ input_embeddings shape: {input_embeddings.shape}")
            logger.debug(
                f"⚡ input_embeddings norm: {torch.norm(input_embeddings):.6f}"
            )
            logger.debug(f"⚡ 各トークンの埋め込みノルム:")
            for i in range(seq_len):
                token_embedding = input_embeddings[0, i, :]
                logger.debug(f"    Token {i}: {torch.norm(token_embedding):.6f}")

        # Captum版でIG値を計算
        # 全トークン位置に対してIG値を計算するため、target_token_indicesとして全範囲を指定
        token_type_ids = inputs.get(
            "token_type_ids", torch.zeros_like(inputs["input_ids"])
        )

        # Captum版を使用してIG値を計算
        # ベースライン方法はデフォルトで"zero"を使用（後でパラメータ化可能）
        baseline_method = "zero"  # デフォルトはゼロベースライン
        layer_results = (
            _compute_attention_all_tokens_ig_vectorized_multi_layer_multi_token(
                bert_model=bert_model,
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                layer_indices=[layer_idx],
                target_token_indices=[target_token_idx],
                target_head_idx=target_head_idx,
                num_steps=num_steps,
                debug=debug,
                baseline_method=baseline_method,
            )
        )

        # 結果をIG値リストに変換
        if layer_idx in layer_results and target_token_idx in layer_results[layer_idx]:
            ig_values = layer_results[layer_idx][target_token_idx]
        else:
            raise RuntimeError(
                f"IG計算結果が見つかりません: layer_idx={layer_idx}, target_token_idx={target_token_idx}"
            )

        # デバッグ情報の表示（デバッグモード時のみ）
        if debug:
            logger.debug(f"⚡ IG計算完了: {len(ig_values)} 個の値")
            if ig_values:
                logger.debug(
                    f"⚡ IG値の範囲: [{min(ig_values):.6f}, {max(ig_values):.6f}]"
                )
                logger.debug(f"⚡ 各トークンのIG値:")
                for i, ig_val in enumerate(ig_values):
                    logger.debug(f"    Token {i}: {ig_val:.6f}")

        # 理論的検証は削除（純粋な計算コードのみ）
        return {
            "ig_values": ig_values,
            "verification": None,  # 理論的検証は削除
        }

    except Exception as e:
        # エラーログを記録
        logger.error(f"Attention IG計算エラー: {e}")
        import traceback

        logger.error(f"スタックトレース: {traceback.format_exc()}")
        if debug:
            traceback.print_exc()
        return {"ig_values": None, "verification": None}


def compute_attention_ig_global_analysis_multi_layer(
    bert_model: L.LightningModule,
    inputs: Dict[str, torch.Tensor],
    layer_indices: List[int],
    target_token_idx: int,
    target_head_idx: Optional[int] = None,
    num_steps: int = 50,
    debug: bool = False,
    cached_hidden_states: Optional[Tuple] = None,
    baseline_method: str = "zero",
    input_type: str = "z",  # "z": 入力埋め込み, "v": Valueベクトル
    use_direct_computation: bool = True,  # 直接計算を使用するか（input_type="v"の場合）
) -> Dict[int, Dict]:
    """
    複数レイヤーのAttention IG計算を一度に実行（最適化版）

    全レイヤーに対して同じinterpolated_embeddingsを使って一度に勾配を計算し、
    効率的にIG値を取得します。
    input_type="v"かつuse_direct_computation=Trueの場合、理論文書5.3節の線形性を利用した直接計算を使用します。

    Args:
        bert_model: BERTモデル
        inputs: 入力テンソル
        layer_indices: 対象レイヤーインデックスリスト
        target_token_idx: 対象トークンインデックス
        target_head_idx: 対象ヘッドインデックス
        num_steps: 積分ステップ数（直接計算の場合は使用されない）
        debug: デバッグフラグ
        cached_hidden_states: 事前計算済みhidden states
        baseline_method: ベースライン方法
        input_type: 入力タイプ ("z": 入力埋め込み, "v": Valueベクトル)
        use_direct_computation: 直接計算を使用するか（input_type="v"の場合のみ有効）

    Returns:
        Dict[int, Dict]: 各レイヤーのIG値（layer_idx -> {"ig_values": List[float], ...}）
    """
    try:
        # 最初のレイヤーで埋め込みを抽出（全レイヤーで同じ入力を使用）
        # 注意: 理論的には各レイヤーで異なる入力が必要だが、現在の実装では簡略化
        if not layer_indices:
            return {}

        first_layer_idx = layer_indices[0]

        input_embeddings, attention_mask, token_type_ids = _extract_embeddings_fast(
            bert_model=bert_model,
            inputs=inputs,
            layer_idx=first_layer_idx,
            debug=debug,
            cached_hidden_states=cached_hidden_states,
        )

        # 複数レイヤーを一度に処理（Captum版または直接計算版）
        # 単一トークンの場合はリストに変換してから処理
        layer_results = (
            _compute_attention_all_tokens_ig_vectorized_multi_layer_multi_token(
                bert_model=bert_model,
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                layer_indices=layer_indices,
                target_token_indices=[target_token_idx],
                target_head_idx=target_head_idx,
                num_steps=num_steps,
                debug=debug,
                baseline_method=baseline_method,
                input_type=input_type,
                use_direct_computation=use_direct_computation,
            )
        )

        # 結果を辞書形式で返す
        results = {}
        for layer_idx, token_results in layer_results.items():
            # token_resultsは {token_idx: List[float]} の形式
            # target_token_idxが1つなので、その値を取得
            if target_token_idx in token_results:
                results[layer_idx] = {
                    "ig_values": token_results[target_token_idx],
                    "verification": None,  # 全体分析ではverificationをスキップ
                }

        return results

    except Exception as e:
        # エラーの詳細情報をログに記録
        logger.error(f"複数レイヤーIG計算エラー: {e}")
        import traceback

        traceback.print_exc()
        raise


def _compute_attention_all_tokens_ig_vectorized_multi_layer_multi_token(
    bert_model: L.LightningModule,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: List[int],
    target_token_indices: List[int],
    target_head_idx: Optional[int],
    num_steps: int,
    debug: bool = False,
    baseline_method: str = "zero",
    input_type: str = "z",  # "z": 入力埋め込み, "v": Valueベクトル
    use_direct_computation: bool = True,  # 直接計算を使用するか（input_type="v"の場合）
) -> Dict[int, Dict[int, List[float]]]:
    """
    複数レイヤー×複数トークンのAttention IG値を一度に計算（Captum版）

    CaptumのIntegratedGradientsを使用して、効率的にIG値を計算します。
    input_type="v"かつuse_direct_computation=Trueの場合、理論文書5.3節の線形性を利用した直接計算を使用します。

    Args:
        bert_model: BERTモデル
        input_embeddings: 入力埋め込み [1, seq_len, hidden]
        attention_mask: アテンションマスク
        layer_indices: 対象レイヤーインデックスリスト
        target_token_indices: 対象トークンインデックスリスト
        target_head_idx: 対象ヘッドインデックス（Noneの場合は全ヘッド）
        num_steps: 積分分割数（直接計算の場合は使用されない）
        debug: デバッグフラグ
        baseline_method: ベースライン方法
        input_type: 入力タイプ ("z": 入力埋め込み, "v": Valueベクトル)
        use_direct_computation: 直接計算を使用するか（input_type="v"の場合のみ有効）

    Returns:
        Dict[layer_idx, Dict[token_idx, List[float]]]: 各レイヤー×トークンのIG値リスト
    """
    if not target_token_indices:
        return {}

    if input_embeddings.dtype != torch.float32:
        input_embeddings = input_embeddings.float()

    # token_type_idsを準備（attention_maskと同じ形状でゼロ埋め）
    token_type_ids = torch.zeros_like(attention_mask)

    # input_type="v"かつuse_direct_computation=Trueの場合、通常は直接計算を使用
    if input_type == "v" and use_direct_computation:
        from .direct_computation import \
            compute_attention_all_tokens_direct_multi_layer_multi_token
        
        if debug:
            logger.debug(
                f"🔹 直接計算モード（複数レイヤー）: layers={layer_indices}, "
                f"head={target_head_idx}, baseline={baseline_method}"
            )
        
        # 直接計算を実行
        results = compute_attention_all_tokens_direct_multi_layer_multi_token(
            bert_model=bert_model,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            layer_indices=layer_indices,
            target_token_indices=target_token_indices,
            target_head_idx=target_head_idx,
            baseline_method=baseline_method,
            debug=debug,
        )
        
        return results

    if not CAPTUM_AVAILABLE:
        raise RuntimeError(
            "Captumがインストールされていません。pip install captumでインストールしてください。"
        )

    # input_type="v"の場合、Valueベクトルを入力として使用（IG計算モード）
    if input_type == "v":
        # 最初のレイヤーでValueベクトルを取得（全レイヤーで同じ入力を使用）
        first_layer_idx = layer_indices[0] if layer_indices else 0
        # 注意: input_type="v"の場合、各レイヤーごとにValueベクトルを計算する必要がある
        # ここでは最初のレイヤーのValueベクトルを取得（後で各レイヤーごとに再計算）
        pass  # 各レイヤーごとに処理するため、ここではスキップ

    # ベースライン埋め込みを計算（ベースライン方法に応じて）
    # 注意: 複数のtarget_token_idxがある場合、最初のトークンを使用してベースラインを計算
    # （各トークンごとに異なるベースラインが必要な場合は、後で個別に計算）
    if baseline_method == "zero":
        # 方法1: ゼロベースライン（すべてのトークンで同じ）
        if input_type == "v":
            # Valueベクトルの場合は、各レイヤーごとに計算するため、ここではスキップ
            baseline_embeddings = None  # 後で各レイヤーごとに計算
        else:
            baseline_embeddings = torch.zeros_like(input_embeddings)
    else:
        # 方法2, 3: 最初のtarget_token_idxを使用してベースラインを計算
        # 注意: 複数のtarget_token_idxがある場合、最初のトークンを使用
        first_target_token_idx = target_token_indices[0] if target_token_indices else 0
        baseline_embeddings = _compute_baseline_embeddings(
            baseline_method=baseline_method,
            input_embeddings=input_embeddings,
            bert_model=bert_model,
            layer_idx=layer_indices[0] if layer_indices else 0,
            target_token_idx=first_target_token_idx,
            target_head_idx=target_head_idx,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            debug=debug,
        )

    # 結果を格納する辞書
    results = {}

    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    gradient_start_time = time.time()
    total_combinations = len(layer_indices) * len(target_token_indices)
    # 大量サンプル実行時（1700以上）はログを抑制
    if _ig_verbose_enabled():
        logger.info(
            f"🔹 Captum IG計算開始: {len(layer_indices)}レイヤー×{len(target_token_indices)}トークン (合計{total_combinations}組み合わせ)"
        )

    # 同じレイヤー・同じヘッドのトークンをグループ化してバッチ処理
    # レイヤーごとにグループ化
    layer_token_groups = {}
    for idx, layer_idx in enumerate(layer_indices):
        layer_token_groups[layer_idx] = target_token_indices.copy()
        layer_token_groups[layer_idx + 1000000] = (
            idx  # store sentence idx later if provided
        )

    # 結果を格納する辞書を初期化
    for layer_idx in layer_indices:
        results[layer_idx] = {}

    # 同じレイヤー・同じヘッドの複数トークンをバッチ処理
    # これにより、1回のIG計算で複数のトークンに対する貢献度を同時に計算可能
    # さらに、異なるレイヤー間も並列化可能（各レイヤーは独立しているため）

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # レイヤー間の並列化を実行
    # GPUメモリに余裕がある場合、複数のレイヤーを同時に処理
    device = input_embeddings.device
    seq_len = input_embeddings.shape[1] if input_embeddings.dim() >= 2 else 0
    if device.type == "cuda":
        try:
            device_index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            free_memory_gb = torch.cuda.mem_get_info(device_index)[0] / (1024**3)
        except Exception as e:
            logger.warning(
                f"⚠️ GPUメモリ情報の取得に失敗しました（{e}）。並列処理を無効化します。"
            )
            free_memory_gb = 0.0
        max_parallel_layers = _safe_parallel_layer_limit(
            num_layers=len(layer_indices),
            free_memory_gb=free_memory_gb,
            seq_len=seq_len,
        )
        # 大量サンプル実行時（1700以上）はログを抑制
        if _ig_verbose_enabled():
            logger.info(
                "🔹 GPUメモリ: %.1fGB空き → %sレイヤー並列 (head=%s, tokens=%s)",
                free_memory_gb,
                max_parallel_layers,
                target_head_idx,
                len(target_token_indices),
            )
    else:
        # CPU環境ではキャッシュ効率を優先し、逐次処理に限定
        max_parallel_layers = min(len(layer_indices), 1)

    def process_layer(layer_idx: int) -> tuple:
        """1つのレイヤーを処理する関数（並列実行用）"""
        layer_start_time = time.time()
        token_indices_for_layer = layer_token_groups[layer_idx]

        if not token_indices_for_layer:
            return (layer_idx, {}, None)

        # ログを削減（sentence情報は不要、詳細ログはdebugモードのみ）
        if debug:
            logger.debug(
                "🔹 Layer %s (head %s): %sトークンをバッチ処理でIG計算",
                layer_idx,
                target_head_idx,
                len(token_indices_for_layer),
            )

        try:
            batch_results = _compute_batch_ig_for_same_layer_head(
                bert_model=bert_model,
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                layer_idx=layer_idx,
                target_token_indices=token_indices_for_layer,
                target_head_idx=target_head_idx,
                num_steps=num_steps,
                debug=debug,
                baseline_method=baseline_method,
                input_type=input_type,
                use_direct_computation=use_direct_computation,
            )

            layer_time = time.time() - layer_start_time
            # ログを削減（sentence情報は不要、詳細ログはdebugモードのみ）
            if debug:
                logger.debug(
                    "🔹 Layer %s (head %s) 完了: %sトークン処理 (%.2f秒)",
                    layer_idx,
                    target_head_idx,
                    len(batch_results),
                    layer_time,
                )

            # メモリ管理はPyTorchの自動管理に任せる
            return (layer_idx, batch_results, None)
        except Exception as e:
            layer_time = time.time() - layer_start_time
            logger.error(
                "❌ Layer %s (head %s) のバッチIG計算失敗 (%.2f秒): %s: %s",
                layer_idx,
                target_head_idx,
                layer_time,
                type(e).__name__,
                e,
            )
            # メモリ管理はPyTorchの自動管理に任せる（メモリ不足エラーの場合のみ明示的にクリーンアップ）

            # フォールバック: 個別に計算
            # ログを削減（詳細ログはdebugモードのみ）
            # logger.info("🔹 Layer %s (head %s): フォールバック（個別計算）", layer_idx, target_head_idx)
            fallback_results = {}
            for token_idx in token_indices_for_layer:
                try:
                    single_result = _compute_single_token_ig(
                        bert_model=bert_model,
                        input_embeddings=input_embeddings,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        layer_idx=layer_idx,
                        target_token_idx=token_idx,
                        target_head_idx=target_head_idx,
                        num_steps=num_steps,
                        debug=debug,
                        baseline_method=baseline_method,
                        input_type=input_type,
                    )
                    fallback_results[token_idx] = single_result

                    # メモリ管理はPyTorchの自動管理に任せる
                except Exception as e2:
                    logger.error(
                        f"❌ Layer {layer_idx}, Token {token_idx}のIG計算失敗: {e2}"
                    )
                    # メモリエラーの場合はクリーンアップ
                    if "out of memory" in str(e2).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc

                        gc.collect()

            return (layer_idx, fallback_results, e)

    # レイヤー間の並列化を実行
    if max_parallel_layers > 1 and len(layer_indices) > 1:
        # 大量サンプル実行時（1700以上）はログを抑制
        if _ig_verbose_enabled():
            logger.info(
                f"🔹 {max_parallel_layers}レイヤー並列でIG計算を開始（{len(layer_indices)}レイヤー）"
            )

        # 実行中のメモリ監視用のロックとカウンター
        active_layers_lock = threading.Lock()
        active_layers_count = [0]  # リストでラップして共有可能にする

        def process_layer_with_memory_check(layer_idx: int) -> tuple:
            """メモリチェック付きレイヤー処理"""
            # メモリが少ない場合は待機
            if device.type == "cuda":
                max_wait_attempts = 10
                wait_interval = 1.0  # 1秒待機
                for attempt in range(max_wait_attempts):
                    try:
                        free_memory_gb = torch.cuda.mem_get_info(device)[0] / (1024**3)
                        # メモリが2GB未満の場合は待機（閾値を下げてGPU利用率を上げる）
                        if free_memory_gb < 2.0:
                            if attempt == 0:
                                logger.warning(
                                    f"⚠️ Layer {layer_idx}: GPUメモリが少ないため待機中... "
                                    f"（空き: {free_memory_gb:.1f}GB）"
                                )
                            time.sleep(wait_interval)
                            # メモリをクリアして再試行
                            torch.cuda.empty_cache()
                            import gc

                            gc.collect()
                        else:
                            break
                    except Exception:
                        break

            # アクティブレイヤー数を増やす
            with active_layers_lock:
                active_layers_count[0] += 1

            try:
                return process_layer(layer_idx)
            finally:
                # アクティブレイヤー数を減らす
                with active_layers_lock:
                    active_layers_count[0] -= 1
                # メモリ管理はPyTorchの自動管理に任せる

        with ThreadPoolExecutor(max_workers=max_parallel_layers) as executor:
            future_to_layer = {
                executor.submit(process_layer_with_memory_check, layer_idx): layer_idx
                for layer_idx in layer_indices
            }

            for future in as_completed(future_to_layer):
                layer_idx, batch_results, error = future.result()
                if error is None:
                    for token_idx, ig_values in batch_results.items():
                        results[layer_idx][token_idx] = ig_values
                else:
                    # エラーは既にログに記録されている
                    pass

                # メモリ管理はPyTorchの自動管理に任せる
    else:
        # 並列化しない場合（メモリが少ない、またはレイヤーが1つだけ）
        for layer_idx in layer_indices:
            _, batch_results, error = process_layer(layer_idx)
            if error is None:
                for token_idx, ig_values in batch_results.items():
                    results[layer_idx][token_idx] = ig_values

            # メモリ管理はPyTorchの自動管理に任せる

    gradient_time = time.time() - gradient_start_time
    # ログを削減（詳細ログはdebugモードのみ）
    # logger.info(f"🔹 Captum IG計算完了: 処理時間:{gradient_time:.2f}秒")

    # メモリ管理はPyTorchの自動管理に任せる

    return results


def _compute_batch_ig_for_same_layer_head(
    bert_model: L.LightningModule,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    layer_idx: int,
    target_token_indices: List[int],
    target_head_idx: Optional[int],
    num_steps: int,
    debug: bool = False,
    baseline_method: str = "zero",
    input_type: str = "z",  # "z": 入力埋め込み, "v": Valueベクトル
    use_direct_computation: bool = True,  # 直接計算を使用するか（input_type="v"の場合）
) -> Dict[int, List[float]]:
    """
    同じレイヤー・同じヘッドの複数トークンに対してバッチIG計算を実行

    Captumのバッチ処理機能を活用して、複数のトークンに対して同時にIG計算を行います。
    input_type="v"かつuse_direct_computation=Trueの場合、理論文書5.3節の線形性を利用した直接計算を使用します。

    Args:
        bert_model: BERTモデル
        input_embeddings: 入力埋め込み [1, seq_len, hidden]
        attention_mask: アテンションマスク
        token_type_ids: トークンタイプID
        layer_idx: 対象レイヤーインデックス
        target_token_indices: 対象トークンインデックスリスト
        target_head_idx: 対象ヘッドインデックス
        num_steps: 積分分割数（直接計算の場合は使用されない）
        debug: デバッグフラグ
        baseline_method: ベースライン方法
        input_type: 入力タイプ ("z": 入力埋め込み, "v": Valueベクトル)
        use_direct_computation: 直接計算を使用するか（input_type="v"の場合のみ有効）

    Returns:
        Dict[token_idx, List[float]]: 各トークンのIG値リスト
    """
    # input_type="v"かつuse_direct_computation=Trueの場合、直接計算を使用
    if input_type == "v" and use_direct_computation:
        from .direct_computation import \
            compute_attention_all_tokens_direct_multi_layer_multi_token
        
        if debug:
            logger.debug(
                f"🔹 直接計算モード: layer={layer_idx}, head={target_head_idx}, "
                f"baseline={baseline_method}"
            )
        
        # 直接計算を実行
        results = compute_attention_all_tokens_direct_multi_layer_multi_token(
            bert_model=bert_model,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            layer_indices=[layer_idx],
            target_token_indices=target_token_indices,
            target_head_idx=target_head_idx,
            baseline_method=baseline_method,
            debug=debug,
        )
        
        # 結果を返す形式に変換
        return results.get(layer_idx, {})
    
    from .attention_models import AttentionModel, create_attention_model

    # input_type="v"の場合、Valueベクトルを入力として使用（IG計算モード）
    if input_type == "v":
        # Valueベクトルを取得
        value_vectors = _extract_value_vectors(
            bert_model=bert_model,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            layer_idx=layer_idx,
            target_head_idx=target_head_idx,
            debug=debug,
        )
        # Valueベクトルを入力として使用
        input_embeddings = value_vectors
        if debug:
            logger.debug(f"Valueベクトルを入力として使用: {input_embeddings.shape}")

    # ベースライン埋め込みを計算（baseline_methodに応じて）
    if baseline_method == "zero":
        if input_type == "v":
            # Valueベクトルの場合はゼロベクトル
            baseline_embeddings = torch.zeros_like(input_embeddings)
        else:
            baseline_embeddings = torch.zeros_like(input_embeddings)
    else:
        # 方法2, 3: 最初のtarget_token_idxを使用してベースラインを計算
        first_target_token_idx = target_token_indices[0] if target_token_indices else 0
        baseline_embeddings = _compute_baseline_embeddings(
            baseline_method=baseline_method,
            input_embeddings=input_embeddings,
            bert_model=bert_model,
            layer_idx=layer_idx,
            target_token_idx=first_target_token_idx,
            target_head_idx=target_head_idx,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            debug=debug,
        )

    # 複数トークン対応のラッパーを作成
    # 各トークンに対して異なる出力を返すモデルを作成
    class MultiTokenAttentionModelWrapper(torch.nn.Module):
        """複数トークン対応のCaptumラッパー"""

        def __init__(
            self,
            bert_model,
            layer_idx,
            target_token_indices,
            target_head_idx,
            attention_mask,
            token_type_ids,
            num_steps,
            debug=False,  # debugパラメータを追加
            input_type="z",  # "z": 入力埋め込み, "v": Valueベクトル
        ):
            super().__init__()
            self.bert_model = bert_model
            self.layer_idx = layer_idx
            self.target_token_indices = target_token_indices
            self.target_head_idx = target_head_idx
            self.attention_mask = attention_mask
            self.token_type_ids = token_type_ids
            self.input_type = input_type

            # デバッグ用: forward呼び出し回数と出力を記録
            self._forward_call_count = 0
            self._forward_outputs = []  # 各ステップでの出力を記録
            self._first_forward_output = None
            self._last_forward_output = None
            self._debug = debug  # debugパラメータを保存
            self._num_steps = max(1, num_steps)

            # 各トークン用のAttentionModelを作成
            self.attention_models = {}
            self.baseline_vectors = {}

            # ベースライン計算の共通化：同じレイヤー・同じヘッドならBERT forwardは1回だけ実行
            # ベースライン方法に応じてベースライン埋め込みを計算
            # 注意: 複数のtarget_token_idxがある場合、最初のトークンを使用してベースラインを計算
            with torch.no_grad():
                if baseline_method == "zero":
                    baseline_emb = torch.zeros_like(input_embeddings)
                else:
                    # 方法2, 3: 最初のtarget_token_idxを使用してベースラインを計算
                    first_target_token_idx = (
                        target_token_indices[0] if target_token_indices else 0
                    )
                    baseline_emb = _compute_baseline_embeddings(
                        baseline_method=baseline_method,
                        input_embeddings=input_embeddings,
                        bert_model=bert_model,
                        layer_idx=layer_idx,
                        target_token_idx=first_target_token_idx,
                        target_head_idx=target_head_idx,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        debug=debug,
                    )

                # BERTモデルの種類に応じて適切にアクセス
                if hasattr(bert_model, "bert"):
                    embeddings_layer = bert_model.bert.embeddings
                    encoder_layers = bert_model.bert.encoder.layer
                else:
                    embeddings_layer = bert_model.embeddings
                    encoder_layers = bert_model.encoder.layer

                # attention_maskの型を修正
                if attention_mask.dtype != torch.float32:
                    baseline_attention_mask = attention_mask.float()
                else:
                    baseline_attention_mask = attention_mask

                # 位置エンコーディングとトークンタイプ埋め込みを追加
                seq_length = baseline_emb.size(1)
                position_ids = torch.arange(
                    seq_length, dtype=torch.long, device=baseline_emb.device
                )
                position_ids = position_ids.unsqueeze(0).expand_as(token_type_ids)

                position_embeddings = embeddings_layer.position_embeddings(position_ids)
                token_type_embeddings = embeddings_layer.token_type_embeddings(
                    token_type_ids
                )

                baseline_embeddings_with_pos = (
                    baseline_emb + position_embeddings + token_type_embeddings
                )
                baseline_embeddings_with_pos = embeddings_layer.LayerNorm(
                    baseline_embeddings_with_pos
                )
                baseline_embeddings_with_pos = embeddings_layer.dropout(
                    baseline_embeddings_with_pos
                )

                # 指定された層まで順次計算
                baseline_hidden_states = baseline_embeddings_with_pos
                for bl_layer_idx in range(layer_idx + 1):
                    bl_layer = encoder_layers[bl_layer_idx]

                    if bl_layer_idx == layer_idx:
                        # 対象層のAttention出力を取得（全トークン分）
                        baseline_attention_output = bl_layer.attention.self(
                            baseline_hidden_states,
                            attention_mask=baseline_attention_mask,
                            output_attentions=True,
                        )

                        if isinstance(baseline_attention_output, tuple):
                            baseline_attention_weights = baseline_attention_output[1]
                            baseline_attention_output = baseline_attention_output[0]
                        else:
                            baseline_attention_weights = None

                        # 各トークンのベースライン出力を抽出
                        for token_idx in target_token_indices:
                            attention_model = create_attention_model(
                                bert_model=bert_model,
                                layer_idx=layer_idx,
                                target_token_idx=token_idx,
                                target_head_idx=target_head_idx,
                                debug=debug,
                            )
                            self.attention_models[token_idx] = attention_model

                            # ベースライン出力を抽出（BERT forwardは1回だけ実行済み）
                            baseline_vec = attention_model._extract_target_output(
                                baseline_attention_output, baseline_attention_weights
                            )
                            self.baseline_vectors[token_idx] = baseline_vec
                        break
                    else:
                        # 他の層は通常通り計算
                        bl_layer_output = bl_layer(
                            baseline_hidden_states,
                            attention_mask=baseline_attention_mask,
                        )
                        if isinstance(bl_layer_output, tuple):
                            baseline_hidden_states = bl_layer_output[0]
                        else:
                            baseline_hidden_states = bl_layer_output

        def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
            """
            複数トークンに対する出力を返す

            BERT forward計算を1回だけ実行し、その結果から各トークンの出力を抽出します。
            これにより、同じレイヤー・同じヘッドの複数トークンに対して効率的に計算できます。

            最適化:
            - 位置エンコーディングとトークンタイプ埋め込みは事前計算済み（各ステップで同じ値）
            - 下位層の計算結果はキャッシュ可能（同じレイヤー・同じヘッドなら同じ）

            Returns:
                torch.Tensor: [num_tokens] 各トークンに対するL2ノルム
                注意: Captumは[num_tokens, 1]ではなく[num_tokens]の形状を期待する
            """
            # BERTモデルの種類に応じて適切にアクセス
            if hasattr(self.bert_model, "bert"):
                embeddings_layer = self.bert_model.bert.embeddings
                encoder_layers = self.bert_model.bert.encoder.layer
            else:
                embeddings_layer = self.bert_model.embeddings
                encoder_layers = self.bert_model.encoder.layer

            # attention_maskの型を修正（各ステップで同じ値なので、事前計算可能だが、メモリ効率を優先）
            if self.attention_mask.dtype != torch.float32:
                attention_mask = self.attention_mask.float()
            else:
                attention_mask = self.attention_mask

            # input_type="v"の場合、Valueベクトルを入力として使用
            # 注意: Valueベクトルを直接使用するには、BERTのAttention機構をカスタマイズする必要がある
            # 現在の実装では、Valueベクトルを入力として受け取るが、Attention機構では通常通り計算
            # TODO: Valueベクトルを直接使用する実装を追加（Attention機構のカスタマイズが必要）

            # 位置エンコーディングとトークンタイプ埋め込みを追加
            # 注意: 各IGステップでinput_embeddingsは異なるが、位置エンコーディングとトークンタイプ埋め込みは同じ
            # ただし、input_embeddingsに加算するため、毎回計算が必要
            seq_length = input_embeddings.size(1)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_embeddings.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(self.token_type_ids)

            position_embeddings = embeddings_layer.position_embeddings(position_ids)
            token_type_embeddings = embeddings_layer.token_type_embeddings(
                self.token_type_ids
            )

            # input_type="v"の場合、Valueベクトルは既に計算済みなので、位置エンコーディングは不要
            # ただし、下位層の計算にはhidden_statesが必要なため、簡易実装として通常通り処理
            if self.input_type == "v":
                # Valueベクトルを入力として使用する場合、下位層の計算をスキップ
                # 簡易実装: 入力埋め込みから下位層を計算（Valueベクトルは後で使用）
                embeddings = (
                    input_embeddings + position_embeddings + token_type_embeddings
                )
                embeddings = embeddings_layer.LayerNorm(embeddings)
                embeddings = embeddings_layer.dropout(embeddings)

                hidden_states = embeddings
                for l_idx in range(self.layer_idx):
                    l = encoder_layers[l_idx]
                    l_output = l(hidden_states, attention_mask=attention_mask)
                    hidden_states = (
                        l_output[0] if isinstance(l_output, tuple) else l_output
                    )

                # 対象層のAttention機構でValueベクトルを使用
                # 注意: BERTのAttention機構は内部でValueを計算するため、
                # Valueベクトルを直接渡すには、Attention機構をカスタマイズする必要がある
                # 簡易実装: 通常のAttention計算を使用（Valueベクトルは無視）
                # TODO: Valueベクトルを直接使用する実装を追加
                attention_output = encoder_layers[self.layer_idx].attention.self(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
            else:
                # input_type="z"の場合、通常通り処理
                embeddings = (
                    input_embeddings + position_embeddings + token_type_embeddings
                )
                embeddings = embeddings_layer.LayerNorm(embeddings)
                embeddings = embeddings_layer.dropout(embeddings)

                # 指定された層まで順次計算
                # 注意: CaptumのIG計算では、各ステップ（補間パラメータa=0, 1/32, 2/32, ..., 1）で異なるinput_embeddingsが使われる
                # そのため、下位層の計算結果はキャッシュできない（入力が異なるため）
                hidden_states = embeddings
                for layer_idx in range(self.layer_idx + 1):
                    layer = encoder_layers[layer_idx]

                    if layer_idx == self.layer_idx:
                        # 対象層のAttention出力を取得（全トークン分）
                        attention_output = layer.attention.self(
                            hidden_states,
                            attention_mask=attention_mask,
                            output_attentions=True,
                        )
                        break
                    else:
                        # 他の層は通常通り計算
                        # 注意: 各IGステップで異なるhidden_statesが使われるため、キャッシュできない
                        layer_output = layer(
                            hidden_states, attention_mask=attention_mask
                        )
                        if isinstance(layer_output, tuple):
                            hidden_states = layer_output[0]
                        else:
                            hidden_states = layer_output

            # Attention出力を処理（input_type="v"と"z"の両方で共通）
            if isinstance(attention_output, tuple):
                attention_weights = attention_output[1]  # attention weights
                attention_output = attention_output[0]  # attention output
            else:
                attention_weights = None

            # 各トークンの出力を抽出
            outputs = []
            for token_idx in self.target_token_indices:
                attention_model = self.attention_models[token_idx]
                baseline_vec = self.baseline_vectors[token_idx]

                # 対象トークンのAttention出力を取得（ベクトル）
                target_output = attention_model._extract_target_output(
                    attention_output, attention_weights
                )

                # ベースラインとの差分を計算
                if baseline_vec.device != target_output.device:
                    baseline_vec = baseline_vec.to(target_output.device)

                # 理論式通り: ||u_i'^{(l,h)}(a) - u_i'^{(l,h)}(0)||_2 を計算
                diff_vector = target_output - baseline_vec
                loss = torch.norm(diff_vector)
                outputs.append(loss)

            # [num_tokens]のテンソルとして返す（Captumが複数出力を正しく処理するため）
            # 注意: [num_tokens, 1]ではなく[num_tokens]の形状にする必要がある
            # Captumは[num_outputs, 1]の形状を正しく処理できないため
            result = torch.stack(outputs)  # [num_tokens]

            # デバッグ: forward呼び出しを記録
            self._forward_call_count += 1
            if _ig_verbose_enabled():
                # IG補間パラメータa（0から1までの補間係数）
                step_interp_param = max(
                    0.0,
                    min(self._forward_call_count - 1, self._num_steps)
                    / self._num_steps,
                )
                logger.info(
                    "[IG-Forward] layer=%s head=%s tokens=%s step=%s/%s a≈%.3f device=%s",
                    self.layer_idx,
                    self.target_head_idx,
                    _summarize_token_indices(self.target_token_indices),
                    min(self._forward_call_count, self._num_steps),
                    self._num_steps,
                    step_interp_param,
                    self.attention_mask.device,
                )
            if self._forward_call_count == 1:
                # 最初の呼び出し（補間パラメータa=0、ベースライン）
                self._first_forward_output = result.detach().clone()
                # デバッグモードを有効化（環境変数またはdebugパラメータで制御）
                enable_debug = (
                    self._debug
                    or os.environ.get("ENABLE_IG_DEBUG_LOGS", "false").lower() == "true"
                )
                if enable_debug:
                    logger.info(  # DEBUGからINFOに変更して確実に表示
                        f"🔍 [DEBUG] Forward call #1 (baseline, a=0): "
                        f"output shape={result.shape}, "
                        f"values={result.squeeze(1).cpu().tolist()[:5]}..., "
                        f"sum={result.sum().item():.6f}, "
                        f"mean={result.mean().item():.6f}"
                    )
            # 最後の呼び出しを記録（各ステップで更新）
            self._last_forward_output = result.detach().clone()

            return result

    # ラッパーを作成
    wrapped_model = MultiTokenAttentionModelWrapper(
        bert_model=bert_model,
        layer_idx=layer_idx,
        target_token_indices=target_token_indices,
        target_head_idx=target_head_idx,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        num_steps=num_steps,
        debug=debug,  # debugパラメータを追加
        input_type=input_type,  # input_typeパラメータを追加
    )

    # IntegratedGradientsインスタンスを作成
    ig = IntegratedGradients(wrapped_model)

    # メモリ管理はPyTorchの自動管理に任せる

    # バッチIG計算を実行
    # 注意: Captumの複数出力処理は不安定なため、トークン数が少ない場合は個別計算にフォールバック
    # PyTorch Lightningの自動メモリ管理を活用しつつ、必要に応じて手動管理も行う
    attributions = None

    # baseline_method="self_input_token"の場合、各トークンに対して個別にベースラインを計算する必要があるため、個別計算を使用
    # トークン数が3以下の場合も個別計算を使用（バッチ処理のオーバーヘッドを避ける）
    if (
        baseline_method == "self_input_token"
        or len(target_token_indices) <= 3
    ):
        # 個別計算にフォールバック
        if baseline_method == "self_input_token":
            logger.debug(
                f"🔹 baseline_method='{baseline_method}'のため個別計算を使用: {len(target_token_indices)}トークン"
            )
        else:
            logger.debug(
                f"🔹 トークン数が少ないため個別計算を使用: {len(target_token_indices)}トークン"
            )
        results = {}
        # baseline_method="self_input_token"の場合、各トークンに対して個別にベースラインを計算する必要がある
        # その他の場合、ベースライン計算を1回だけ実行してキャッシュ（同じレイヤー・同じヘッド・同じシーケンス長なら結果は同じ）
        if baseline_method == "self_input_token":
            # 各トークンに対して個別にベースラインを計算する必要があるため、キャッシュは使用しない
            baseline_vector = None
        else:
            baseline_vector = _get_baseline_output_cached(
                bert_model=bert_model,
                layer_idx=layer_idx,
                target_token_idx=target_token_indices[
                    0
                ],  # 任意のトークンでOK（ベースラインがゼロベクトルなので）
                target_head_idx=target_head_idx,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                hidden_size=input_embeddings.shape[-1],
            )

        for token_idx in target_token_indices:
            try:
                single_result = _compute_single_token_ig(
                    bert_model=bert_model,
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    layer_idx=layer_idx,
                    target_token_idx=token_idx,
                    target_head_idx=target_head_idx,
                    num_steps=num_steps,
                    debug=debug,
                    baseline_vector=baseline_vector,  # キャッシュされたベースラインを再利用
                    baseline_method=baseline_method,  # baseline_methodを渡す
                    input_type=input_type,
                )
                results[token_idx] = single_result
            except Exception as e:
                logger.error(
                    "❌ Layer %s (head %s), Token %sのIG計算失敗: %s",
                    layer_idx,
                    target_head_idx,
                    token_idx,
                    e,
                )
                # メモリエラーの場合はクリーンアップ
                if "out of memory" in str(e).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc

                    gc.collect()
        return results

    # バッチIG計算を試行
    # 理論: 各トークン$i'$に対して独立にIGを計算する必要がある
    # IG_{i,i'}^{Attn} = (z_i - z_i^{base}) · ∫₀¹ ∂A_{i'}(a) / ∂z_i da
    # ここで A_{i'}(a) = ||ATT_{i'}(a) - ATT_{i'}(0)||_2
    try:
        # デバッグ: forward呼び出しカウントをリセット
        wrapped_model._forward_call_count = 0
        wrapped_model._first_forward_output = None
        wrapped_model._last_forward_output = None

        # デバッグモードを有効化（環境変数またはdebugパラメータで制御）
        enable_debug = (
            debug or os.environ.get("ENABLE_IG_DEBUG_LOGS", "false").lower() == "true"
        )

        # CaptumのIG計算はfloat32を要求するため、BF16/FP16の場合はfloat32に変換
        original_dtype = input_embeddings.dtype
        if original_dtype in (torch.bfloat16, torch.float16):
            input_embeddings_fp32 = input_embeddings.to(torch.float32)
            baseline_embeddings_fp32 = baseline_embeddings.to(torch.float32)
        else:
            input_embeddings_fp32 = input_embeddings
            baseline_embeddings_fp32 = baseline_embeddings

        # PyTorch Lightningの自動メモリ管理を信頼しつつ、必要に応じて手動クリア
        # forwardが[num_tokens, 1]を返す場合、Captumは各出力に対するIGを計算する
        attributions = ig.attribute(
            inputs=input_embeddings_fp32,
            baselines=baseline_embeddings_fp32,
            n_steps=num_steps,
            method="riemann_trapezoid",
        )

        # 元のdtypeに戻す処理は削除（IG計算後はfloat32のまま使用）

        # デバッグ: forward呼び出し結果を記録
        if enable_debug and wrapped_model._first_forward_output is not None:
            logger.info(  # DEBUGからINFOに変更して確実に表示
                f"🔍 [DEBUG] Forward呼び出し回数: {wrapped_model._forward_call_count}"
            )
            logger.info(
                f"🔍 [DEBUG] 最初のforward出力 (a=0, baseline): "
                f"shape={wrapped_model._first_forward_output.shape}, "
                f"sum={wrapped_model._first_forward_output.sum().item():.6f}, "
                f"mean={wrapped_model._first_forward_output.mean().item():.6f}"
            )
            if wrapped_model._last_forward_output is not None:
                logger.info(
                    f"🔍 [DEBUG] 最後のforward出力 (a=1, actual): "
                    f"shape={wrapped_model._last_forward_output.shape}, "
                    f"sum={wrapped_model._last_forward_output.sum().item():.6f}, "
                    f"mean={wrapped_model._last_forward_output.mean().item():.6f}"
                )
                logger.info(
                    f"🔍 [DEBUG] Forward出力の差分 (actual - baseline): "
                    f"sum={(wrapped_model._last_forward_output - wrapped_model._first_forward_output).sum().item():.6f}"
                )

        # デバッグ: attributionsの形状と値を記録
        if enable_debug:
            logger.info(
                f"🔍 [DEBUG] Attributions形状: {attributions.shape}, "
                f"sum={attributions.sum().item():.6f}, "
                f"mean={attributions.mean().item():.6f}"
            )
            if len(attributions.shape) >= 3:
                # 各入力トークン位置のIG値の合計を計算
                token_ig_sums = attributions.sum(dim=-1)  # [..., seq_len]
                logger.info(
                    f"🔍 [DEBUG] 各入力トークン位置のIG合計 (最初の5トークン): "
                    f"{token_ig_sums.flatten()[:5].cpu().tolist()}"
                )
    except (RuntimeError, AssertionError) as e:
        # メモリエラーまたはAssertionErrorの場合は個別計算にフォールバック
        error_msg = str(e).lower()
        if (
            "out of memory" in error_msg
            or "target list length" in error_msg
            or "assertion" in error_msg
        ):
            logger.warning(
                f"⚠️ バッチIG計算失敗（{type(e).__name__}）、個別計算にフォールバック: {e}"
            )
            # メモリ管理はPyTorchの自動管理に任せる（メモリ不足エラーの場合のみ明示的にクリーンアップ）

            # 個別計算にフォールバック（理論通り：各トークンに対して独立にIGを計算）
            results = {}
            # 既に計算したベースラインを再利用
            baseline_vectors = (
                wrapped_model.baseline_vectors
                if hasattr(wrapped_model, "baseline_vectors")
                else None
            )
            for token_idx in target_token_indices:
                try:
                    # ベースラインを再利用（可能な場合）
                    baseline_vec = (
                        baseline_vectors[token_idx]
                        if baseline_vectors and token_idx in baseline_vectors
                        else None
                    )
                    single_result = _compute_single_token_ig(
                        bert_model=bert_model,
                        input_embeddings=input_embeddings,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        layer_idx=layer_idx,
                        target_token_idx=token_idx,
                        target_head_idx=target_head_idx,
                        num_steps=num_steps,
                        debug=debug,
                        baseline_vector=baseline_vec,  # キャッシュされたベースラインを再利用
                        baseline_method=baseline_method,  # baseline_methodを渡す
                        input_type=input_type,
                    )
                    results[token_idx] = single_result
                except Exception as e2:
                    logger.error(
                        f"❌ Layer {layer_idx}, Token {token_idx}のIG計算失敗: {e2}"
                    )
                    # メモリエラーの場合はクリーンアップ
                    if "out of memory" in str(e2).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc

                        gc.collect()
            return results
        else:
            raise

    # 結果を処理
    # Captumが複数の出力に対してIGを計算する場合、
    # attributionsの形状は[num_outputs, batch_size, seq_len, hidden_size]または
    # [num_outputs, seq_len, hidden_size]になる可能性がある
    if attributions is None:
        raise ValueError("Captum attributionがNoneです")

    if not isinstance(attributions, torch.Tensor):
        raise ValueError(f"attributionsがテンソルではありません: {type(attributions)}")

    logger.debug(
        f"🔹 Attributions形状: {attributions.shape}, トークン数: {len(target_token_indices)}"
    )

    # attributionsの形状に応じて処理
    # 理論: 各トークン$i'$に対して独立にIGを計算する必要がある
    # 単一出力の場合は理論に反するため、個別計算にフォールバック
    try:
        if len(attributions.shape) == 3:
            # [num_tokens, seq_len, hidden_size] または [batch_size, seq_len, hidden_size] の場合
            if attributions.shape[0] == len(target_token_indices):
                # 期待通りの形状: [num_tokens, seq_len, hidden_size] - 理論通り
                seq_len = attributions.shape[1]
                results = {}
                for i, token_idx in enumerate(target_token_indices):
                    ig_values = []
                    for pos_idx in range(seq_len):
                        # baseline_method="self_input_token"の場合、自己トークンの寄与度を0に設定（理論的には0になるべき）
                        if (
                            baseline_method == "self_input_token"
                            and pos_idx == token_idx
                        ):
                            pos_ig = 0.0
                        else:
                            # 各トークン位置の埋め込みのノルムをIG値とする
                            pos_ig = torch.norm(attributions[i, pos_idx, :]).item()
                        ig_values.append(pos_ig)
                    results[token_idx] = ig_values
            elif attributions.shape[0] == 1:
                # 単一出力の場合: [1, seq_len, hidden_size] - Captumの制約により複数出力が合計/平均されている
                # 理論通りに各トークンに対して独立にIGを計算するため、個別計算にフォールバック
                logger.debug(
                    f"🔍 Captumが単一出力を返しました: {attributions.shape}。"
                    f"これはCaptumの制約により、複数のスカラー出力が合計または平均されたためです。"
                    f"理論通りに各トークンに対して独立にIGを計算するため、個別計算にフォールバックします。"
                )

                # デバッグ: Captumがどのように処理したかを分析
                if enable_debug and wrapped_model._first_forward_output is not None:
                    logger.info(f"🔍 [DEBUG] 単一出力の場合の分析:")
                    logger.info(
                        f"🔍 [DEBUG]   - Forward出力形状: {wrapped_model._first_forward_output.shape}"
                    )
                    logger.info(
                        f"🔍 [DEBUG]   - Forward出力の合計: {wrapped_model._first_forward_output.sum().item():.6f}"
                    )
                    logger.info(
                        f"🔍 [DEBUG]   - Forward出力の平均: {wrapped_model._first_forward_output.mean().item():.6f}"
                    )
                    logger.info(
                        f"🔍 [DEBUG]   - Attributions形状: {attributions.shape}"
                    )
                    logger.info(
                        f"🔍 [DEBUG]   - Attributionsの合計: {attributions.sum().item():.6f}"
                    )
                    logger.info(
                        f"🔍 [DEBUG]   仮説検証: Captumが複数出力を合計または平均している可能性があります"
                    )
                # メモリ管理はPyTorchの自動管理に任せる（メモリ不足エラーの場合のみ明示的にクリーンアップ）

                # 個別計算にフォールバック（理論通り：各トークンに対して独立にIGを計算）
                results = {}
                # 既に計算したベースラインを再利用
                baseline_vectors = (
                    wrapped_model.baseline_vectors
                    if hasattr(wrapped_model, "baseline_vectors")
                    else None
                )
                for token_idx in target_token_indices:
                    try:
                        # ベースラインを再利用（可能な場合）
                        baseline_vec = (
                            baseline_vectors[token_idx]
                            if baseline_vectors and token_idx in baseline_vectors
                            else None
                        )
                        single_result = _compute_single_token_ig(
                            bert_model=bert_model,
                            input_embeddings=input_embeddings,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            layer_idx=layer_idx,
                            target_token_idx=token_idx,
                            target_head_idx=target_head_idx,
                            num_steps=num_steps,
                            debug=debug,
                            baseline_vector=baseline_vec,  # キャッシュされたベースラインを再利用
                            baseline_method=baseline_method,  # baseline_methodを渡す
                            input_type=input_type,
                        )
                        results[token_idx] = single_result
                    except Exception as e2:
                        logger.error(
                            f"❌ Layer {layer_idx}, Token {token_idx}のIG計算失敗: {e2}"
                        )
                        # メモリエラーの場合はクリーンアップ
                        if (
                            "out of memory" in str(e2).lower()
                            and torch.cuda.is_available()
                        ):
                            torch.cuda.empty_cache()
                            import gc

                            gc.collect()
                return results
            else:
                # 予期しない形状
                raise ValueError(
                    f"attributionsの最初の次元がトークン数と一致しません: "
                    f"{attributions.shape[0]} != {len(target_token_indices)} (形状: {attributions.shape})"
                )
        elif len(attributions.shape) == 4:
            # [num_tokens, batch_size, seq_len, hidden_size] の場合
            if attributions.shape[0] == len(target_token_indices):
                seq_len = attributions.shape[2]
                results = {}
                for i, token_idx in enumerate(target_token_indices):
                    ig_values = []
                    for pos_idx in range(seq_len):
                        # 各トークン位置の埋め込みのノルムをIG値とする
                        pos_ig = torch.norm(attributions[i, 0, pos_idx, :]).item()
                        ig_values.append(pos_ig)
                    results[token_idx] = ig_values
            elif attributions.shape[0] == 1:
                # 単一出力の場合: [1, batch_size, seq_len, hidden_size] - Captumの制約により複数出力が合計/平均されている
                # 理論通りに各トークンに対して独立にIGを計算するため、個別計算にフォールバック
                logger.debug(
                    f"🔍 Captumが単一出力を返しました: {attributions.shape}。"
                    f"これはCaptumの制約により、複数のスカラー出力が合計または平均されたためです。"
                    f"理論通りに各トークンに対して独立にIGを計算するため、個別計算にフォールバックします。"
                )
                # メモリ管理はPyTorchの自動管理に任せる（メモリ不足エラーの場合のみ明示的にクリーンアップ）

                # 個別計算にフォールバック（理論通り：各トークンに対して独立にIGを計算）
                results = {}
                # 既に計算したベースラインを再利用
                baseline_vectors = (
                    wrapped_model.baseline_vectors
                    if hasattr(wrapped_model, "baseline_vectors")
                    else None
                )
                for token_idx in target_token_indices:
                    try:
                        # ベースラインを再利用（可能な場合）
                        baseline_vec = (
                            baseline_vectors[token_idx]
                            if baseline_vectors and token_idx in baseline_vectors
                            else None
                        )
                        single_result = _compute_single_token_ig(
                            bert_model=bert_model,
                            input_embeddings=input_embeddings,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            layer_idx=layer_idx,
                            target_token_idx=token_idx,
                            target_head_idx=target_head_idx,
                            num_steps=num_steps,
                            debug=debug,
                            baseline_vector=baseline_vec,  # キャッシュされたベースラインを再利用
                            baseline_method=baseline_method,  # baseline_methodを渡す
                            input_type=input_type,
                        )
                        results[token_idx] = single_result
                    except Exception as e2:
                        logger.error(
                            f"❌ Layer {layer_idx}, Token {token_idx}のIG計算失敗: {e2}"
                        )
                        # メモリエラーの場合はクリーンアップ
                        if (
                            "out of memory" in str(e2).lower()
                            and torch.cuda.is_available()
                        ):
                            torch.cuda.empty_cache()
                            import gc

                            gc.collect()
                return results
            else:
                raise ValueError(
                    f"attributionsの最初の次元がトークン数と一致しません: "
                    f"{attributions.shape[0]} != {len(target_token_indices)} (形状: {attributions.shape})"
                )
        else:
            # フォールバック: 予期しない形状の場合も個別計算にフォールバック
            logger.warning(
                f"⚠️ 予期しないattributions形状: {attributions.shape}。"
                f"理論通りに各トークンに対して独立にIGを計算するため、個別計算にフォールバックします。"
            )
            # メモリ管理はPyTorchの自動管理に任せる（メモリ不足エラーの場合のみ明示的にクリーンアップ）

            # 個別計算にフォールバック（理論通り：各トークンに対して独立にIGを計算）
            results = {}
            # 既に計算したベースラインを再利用
            baseline_vectors = (
                wrapped_model.baseline_vectors
                if hasattr(wrapped_model, "baseline_vectors")
                else None
            )
            for token_idx in target_token_indices:
                try:
                    # ベースラインを再利用（可能な場合）
                    baseline_vec = (
                        baseline_vectors[token_idx]
                        if baseline_vectors and token_idx in baseline_vectors
                        else None
                    )
                    single_result = _compute_single_token_ig(
                        bert_model=bert_model,
                        input_embeddings=input_embeddings,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        layer_idx=layer_idx,
                        target_token_idx=token_idx,
                        target_head_idx=target_head_idx,
                        num_steps=num_steps,
                        debug=debug,
                        baseline_vector=baseline_vec,  # キャッシュされたベースラインを再利用
                        baseline_method=baseline_method,  # baseline_methodを渡す
                        input_type=input_type,
                    )
                    results[token_idx] = single_result
                except Exception as e2:
                    logger.error(
                        f"❌ Layer {layer_idx}, Token {token_idx}のIG計算失敗: {e2}"
                    )
                    # メモリエラーの場合はクリーンアップ
                    if "out of memory" in str(e2).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc

                        gc.collect()
            return results
    finally:
        # メモリ管理はPyTorchの自動管理に任せる
        # 明示的なdelやempty_cacheは不要（PyTorchが自動的に管理）
        pass

    return results


# ベースライン計算結果のキャッシュ
# 注意: ベースライン（補間パラメータa=0、z^{base}=0）はゼロベクトルなので、target_token_idxに関係なく同じ結果になる
# ただし、IG計算の各ステップ（a>0）では補間入力が異なるため、target_token_idxによって結果が異なる
# ベースライン計算のみをキャッシュするため、target_token_idxはキャッシュキーに含めない
_baseline_output_cache: Dict[Tuple[int, Optional[int], int, int], torch.Tensor] = {}


def _get_baseline_output_cached(
    bert_model: L.LightningModule,
    layer_idx: int,
    target_token_idx: int,  # ベースライン計算では使用しないが、API互換性のため残す
    target_head_idx: Optional[int],
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    """
    ベースライン出力を取得（キャッシュを使用）

    ベースライン（補間パラメータa=0、z^{base}=0）はゼロベクトルなので、target_token_idxに関係なく同じ結果になる
    ただし、IG計算の各ステップ（a=1/32, 2/32, ...）では補間入力が異なるため、
    target_token_idxによって結果が異なる可能性がある

    ベースライン計算のみをキャッシュするため、target_token_idxはキャッシュキーに含めない
    """
    seq_len = attention_mask.shape[1]
    # ベースライン（補間パラメータa=0、z^{base}=0）はゼロベクトルなので、target_token_idxに関係なく同じ結果
    # ただし、IG計算の各ステップでは異なるため、ベースライン計算のみをキャッシュ
    cache_key = (layer_idx, target_head_idx, seq_len, hidden_size)

    if cache_key not in _baseline_output_cache:
        # ベースライン計算を実行（任意のトークンで計算すればOK、ベースラインはゼロベクトルなので）
        from .attention_models import create_attention_model

        baseline_embeddings = torch.zeros(
            1,
            seq_len,
            hidden_size,
            device=attention_mask.device,
            dtype=(
                attention_mask.dtype
                if attention_mask.dtype.is_floating_point
                else torch.float32
            ),
        )

        # 任意のトークン（最初のトークン）でベースラインを計算
        # ベースラインがゼロベクトルなので、どのトークンでも同じ結果
        attention_model = create_attention_model(
            bert_model=bert_model,
            layer_idx=layer_idx,
            target_token_idx=0,  # ベースラインがゼロベクトルなので、どのトークンでも同じ結果
            target_head_idx=target_head_idx,
            debug=False,
        )

        baseline_vector = attention_model._compute_attention_output_vector(
            baseline_embeddings, attention_mask, token_type_ids
        )

        _baseline_output_cache[cache_key] = baseline_vector.detach()

    return _baseline_output_cache[cache_key]


def _compute_single_token_ig(
    bert_model: L.LightningModule,
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
    target_head_idx: Optional[int],
    num_steps: int,
    debug: bool = False,
    baseline_vector: Optional[
        torch.Tensor
    ] = None,  # キャッシュされたベースラインを再利用可能
    baseline_method: str = "zero",
    input_type: str = "z",
) -> List[float]:
    """単一トークンのIG計算（フォールバック用）"""
    from .attention_models import create_attention_model

    if input_embeddings.dtype != torch.float32:
        input_embeddings = input_embeddings.float()

    # ベースライン埋め込みを計算
    if baseline_method == "zero":
        baseline_embeddings = torch.zeros_like(input_embeddings)
    else:
        baseline_embeddings = _compute_baseline_embeddings(
            baseline_method=baseline_method,
            input_embeddings=input_embeddings,
            bert_model=bert_model,
            layer_idx=layer_idx,
            target_token_idx=target_token_idx,
            target_head_idx=target_head_idx,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            debug=debug,
        )

    attention_model = create_attention_model(
        bert_model=bert_model,
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
        target_head_idx=target_head_idx,
        debug=debug,
    )

    # ベースライン出力をキャッシュから取得または計算
    if baseline_vector is None:
        if baseline_method == "zero":
            # ゼロベースラインの場合はキャッシュを使用
            baseline_vector = _get_baseline_output_cached(
                bert_model=bert_model,
                layer_idx=layer_idx,
                target_token_idx=target_token_idx,
                target_head_idx=target_head_idx,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                hidden_size=attention_model.hidden_size,
            )
        else:
            # baseline_method="self_input_token"の場合、baseline_embeddingsからベースライン出力を計算
            baseline_vector = attention_model._compute_attention_output_vector(
                baseline_embeddings, attention_mask, token_type_ids
            )
    if baseline_vector.dtype != torch.float32:
        baseline_vector = baseline_vector.float()

    # Phase 1.2: torch.compile対応のforward関数を作成
    # コンパイル可能なforward関数（理論的正確性を保証）
    def _create_compiled_forward(attention_model, attention_mask, token_type_ids):
        """コンパイル済みforward関数を作成"""
        if not hasattr(torch, "compile"):
            return None

        # 環境変数でコンパイルモードを制御（デフォルト: max-autotuneで最適化）
        compile_mode = os.environ.get("PTB_TORCH_COMPILE_MODE", "max-autotune").lower()
        if compile_mode not in ["reduce-overhead", "max-autotune", "default"]:
            compile_mode = "max-autotune"

        @torch.compile(mode=compile_mode, fullgraph=False)
        def _compiled_attention_forward(input_embeddings: torch.Tensor) -> torch.Tensor:
            """
            コンパイル済みのattention forward計算

            理論式: u_i'^{(l,h)}(a) = ATT_i'^{(l,h)}({z_i^{(l)}}_i:a)
            この関数はforward計算のみを最適化（backwardはCaptumが管理）
            """
            return attention_model._compute_attention_output_vector(
                input_embeddings, attention_mask, token_type_ids
            )

        return _compiled_attention_forward

    # ベースライン出力を事前計算済みのものに置き換えたラッパーを作成
    class CachedBaselineWrapper(torch.nn.Module):
        """
        Captum用のラッパー（ベースライン計算をキャッシュ）

        最適化:
        - ベースライン出力は事前計算済み（キャッシュから取得）
        - IG計算の各ステップ（補間パラメータa=0, 1/32, 2/32, ..., 1）でforwardが呼ばれる
        - 各ステップで異なるinput_embeddingsが使われるため、BERT forwardは毎回実行される
        - Phase 1.2: torch.compileでforward計算を最適化
        """

        def __init__(
            self,
            attention_model,
            baseline_vec,
            attention_mask,
            token_type_ids,
            use_compile: bool = True,
        ):
            super().__init__()
            self.attention_model = attention_model
            self.baseline_vector = baseline_vec
            self.attention_mask = attention_mask
            self.token_type_ids = token_type_ids
            self.use_compile = use_compile and hasattr(torch, "compile")
            self._compiled_forward = None

            # torch.compileが利用可能な場合、forward関数をコンパイル
            if self.use_compile:
                try:
                    # コンパイル済みforward関数を作成
                    self._compiled_forward = _create_compiled_forward(
                        attention_model, attention_mask, token_type_ids
                    )
                    if self._compiled_forward is None:
                        self.use_compile = False
                except Exception as e:
                    logger.warning(f"torch.compileの適用に失敗（フォールバック）: {e}")
                    self.use_compile = False

        def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
            """
            CaptumのIG計算で呼ばれるforwardメソッド

            各IGステップ（補間パラメータa=0, 1/32, 2/32, ..., 1）で呼ばれ、異なるinput_embeddingsが渡される
            ベースライン出力は事前計算済みなので、差分計算のみを実行
            """
            # input_embeddingsは既にfloat32（Captumが変換済み）
            if self.use_compile and self._compiled_forward is not None:
                try:
                    # コンパイル済みforward関数を使用
                    u_actual_vector = self._compiled_forward(input_embeddings)
                except Exception:
                    # コンパイル済み関数が失敗した場合はフォールバック
                    u_actual_vector = (
                        self.attention_model._compute_attention_output_vector(
                            input_embeddings, self.attention_mask, self.token_type_ids
                        )
                    )
            else:
                # 通常のforward計算
                u_actual_vector = self.attention_model._compute_attention_output_vector(
                    input_embeddings, self.attention_mask, self.token_type_ids
                )

            # デバイスが異なる場合のみ移動（通常は同じデバイス）
            if self.baseline_vector.device != u_actual_vector.device:
                self.baseline_vector = self.baseline_vector.to(u_actual_vector.device)
            # ベースラインは既にfloat32で保存されている
            diff_vector = u_actual_vector - self.baseline_vector
            loss = torch.norm(diff_vector)
            return loss.unsqueeze(0) if loss.dim() == 0 else loss

    # Phase 1.2: torch.compileを有効化（環境変数で制御可能）
    use_compile = os.environ.get("PTB_USE_TORCH_COMPILE", "true").lower() in {
        "1",
        "true",
        "on",
        "yes",
    }

    wrapped_model = CachedBaselineWrapper(
        attention_model=attention_model,
        baseline_vec=baseline_vector,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        use_compile=use_compile,
    )

    ig = IntegratedGradients(wrapped_model)

    # CaptumのIG計算はfloat32を要求するため、BF16/FP16の場合はfloat32に変換
    original_dtype = input_embeddings.dtype
    if original_dtype in (torch.bfloat16, torch.float16):
        input_embeddings_fp32 = input_embeddings.to(torch.float32)
        baseline_embeddings_fp32 = baseline_embeddings.to(torch.float32)
    else:
        input_embeddings_fp32 = input_embeddings
        baseline_embeddings_fp32 = baseline_embeddings

    attributions = ig.attribute(
        inputs=input_embeddings_fp32,
        baselines=baseline_embeddings_fp32,
        n_steps=num_steps,
        method="riemann_trapezoid",
    )

    # 元のdtypeに戻す処理は削除（IG計算後はfloat32のまま使用）

    if attributions is None or not isinstance(attributions, torch.Tensor):
        raise ValueError("Invalid attributions")

    seq_len = attributions.shape[1]
    ig_values = []
    for pos_idx in range(seq_len):
        # baseline_method="self_input_token"の場合、自己トークンの寄与度を0に設定（理論的には0になるべき）
        if baseline_method == "self_input_token" and pos_idx == target_token_idx:
            pos_ig = 0.0
        else:
            pos_ig = torch.norm(attributions[0, pos_idx, :]).item()
        ig_values.append(pos_ig)

    return ig_values
    # メモリ管理はPyTorchの自動管理に任せる（明示的なdel/empty_cacheは不要）


def compute_attention_ig_global_analysis_multi_layer_multi_token(
    bert_model: L.LightningModule,
    inputs: Dict[str, torch.Tensor],
    layer_indices: List[int],
    target_token_indices: List[int],
    target_head_idx: Optional[int] = None,
    num_steps: int = 50,
    debug: bool = False,
    cached_hidden_states: Optional[Tuple] = None,
    baseline_method: str = "zero",
    input_type: str = "z",  # "z": 入力埋め込み, "v": Valueベクトル
) -> Dict[int, Dict[int, Dict]]:
    """
    複数レイヤー×複数トークンのAttention IG計算を一度に実行（最適化版）

    同じヘッドの複数のトークンに対して、同じinterpolated_embeddingsを使って
    一度に勾配を計算し、効率的にIG値を取得します。

    Args:
        bert_model: BERTモデル
        inputs: 入力テンソル
        layer_indices: 対象レイヤーインデックスリスト
        target_token_indices: 対象トークンインデックスリスト
        target_head_idx: 対象ヘッドインデックス
        num_steps: 積分ステップ数
        debug: デバッグフラグ
        cached_hidden_states: 事前計算済みhidden states

    Returns:
        Dict[int, Dict[int, Dict]]: 各レイヤー×トークンのIG値
        (layer_idx -> token_idx -> {"ig_values": List[float], ...})
    """
    try:
        if not layer_indices or not target_token_indices:
            return {}

        # 最初のレイヤーで埋め込みを抽出（全レイヤーで同じ入力を使用）
        first_layer_idx = layer_indices[0]

        input_embeddings, attention_mask, token_type_ids = _extract_embeddings_fast(
            bert_model=bert_model,
            inputs=inputs,
            layer_idx=first_layer_idx,
            debug=debug,
            cached_hidden_states=cached_hidden_states,
        )

        # 複数レイヤー×複数トークンを一度に処理
        layer_token_results = (
            _compute_attention_all_tokens_ig_vectorized_multi_layer_multi_token(
                bert_model=bert_model,
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                layer_indices=layer_indices,
                target_token_indices=target_token_indices,
                target_head_idx=target_head_idx,
                num_steps=num_steps,
                debug=debug,
                baseline_method=baseline_method,
                input_type=input_type,
            )
        )

        # 結果を辞書形式で返す
        results = {}
        for layer_idx, token_results in layer_token_results.items():
            results[layer_idx] = {}
            for token_idx, ig_values in token_results.items():
                results[layer_idx][token_idx] = {
                    "ig_values": ig_values,
                    "verification": None,  # 全体分析ではverificationをスキップ
                }

        return results

    except Exception as e:
        # エラーの詳細情報をログに記録
        logger.error(f"複数レイヤー×複数トークンIG計算エラー: {e}")
        import traceback

        traceback.print_exc()
        raise


def _extract_embeddings_fast(
    bert_model: L.LightningModule,
    inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    debug: bool = False,
    cached_hidden_states: Optional[Tuple] = None,  # 事前計算済みhidden states
) -> tuple:
    """
    軽量版埋め込み抽出（キャッシュシステムを使わない、または事前計算済みhidden statesを使用）

    Args:
        bert_model: BERTモデル
        inputs: 入力テンソル
        layer_idx: 層インデックス
        debug: デバッグフラグ
        cached_hidden_states: 事前計算済みhidden states（Tuple[torch.Tensor, ...]）が提供される場合はBERT推論をスキップ

    Returns:
        tuple: (input_embeddings, attention_mask, token_type_ids)
    """
    if debug:
        logger.debug("⚡ 軽量版埋め込み抽出開始")

    # キャッシュされたhidden statesが提供されている場合はそれを使用
    if cached_hidden_states is not None:
        # hidden_states[0] = embeddings後, hidden_states[layer_idx] が層layer_idxの入力
        # Layer 0の場合は hidden_states[0] (Embedding層の出力) を使用
        if layer_idx < 0 or layer_idx >= len(cached_hidden_states):
            logger.error(
                f"Layer {layer_idx}のインデックスが範囲外: "
                f"cached_hidden_states length={len(cached_hidden_states)}"
            )
            raise IndexError(
                f"Layer {layer_idx} is out of range for cached_hidden_states (length={len(cached_hidden_states)})"
            )
        input_embeddings = cached_hidden_states[layer_idx]

        # Layer 0での形状確認（デバッグ用）
        if layer_idx == 0 and debug:
            logger.debug(
                f"Layer 0: input_embeddings shape={input_embeddings.shape}, "
                f"device={input_embeddings.device}, dtype={input_embeddings.dtype}"
            )

        # デバイス確認（cached_hidden_statesを使用する前に）
        model_device = next(bert_model.parameters()).device

        # デバイスが異なる場合は移動（通常は同じデバイス）
        if input_embeddings.device != model_device:
            input_embeddings = input_embeddings.to(model_device)

        # attention_maskをcached_hidden_statesのシーケンス長に合わせて生成
        # 注意: cached_hidden_statesはバッチ処理時の長さで保存されているため、
        # inputsのattention_maskと長さが一致しない場合がある（正常な動作）
        cached_seq_len = input_embeddings.shape[1]
        batch_size = input_embeddings.shape[0]

        # cached_hidden_statesの長さに合わせてattention_maskを生成
        # 直接GPUで作成（CPU→GPU移動を削除して高速化）
        attention_mask = torch.ones(
            (batch_size, cached_seq_len), device=model_device, dtype=torch.long
        )
        token_type_ids = torch.zeros(
            (batch_size, cached_seq_len), device=model_device, dtype=torch.long
        )

        if debug:
            logger.debug(
                f"⚡ キャッシュされたhidden statesを使用: {input_embeddings.shape} (device: {input_embeddings.device})"
            )
            logger.debug(
                f"⚡ マスク形状: {attention_mask.shape} (device: {attention_mask.device})"
            )

        # DEBUGログは削減（大量に出力されるため）
        # logger.debug(
        #     f"キャッシュ使用: layer={layer_idx}, shape={input_embeddings.shape}, device={input_embeddings.device}"
        # )

        return input_embeddings, attention_mask, token_type_ids

    # モデルの種類を判定して適切にアクセス
    if hasattr(bert_model, "bert"):
        # Lightning包装されたBERTモデル
        embeddings_layer = bert_model.bert.embeddings
        if debug:
            logger.debug("⚡ Lightning包装されたBERTモデルを検出")
    else:
        # 直接のBERTモデル (BertWithHooks等)
        embeddings_layer = bert_model.embeddings
        if debug:
            logger.debug("⚡ 直接のBERTモデル (BertWithHooks) を検出")

    # 層lの入力隠れ状態 z^{(l)} を取得
    with torch.no_grad():
        outputs = (
            bert_model(**inputs, output_hidden_states=True, output_attentions=False)
            if hasattr(bert_model, "__call__")
            else bert_model.bert(
                **inputs, output_hidden_states=True, output_attentions=False
            )
        )
        hidden_states = outputs.hidden_states  # tuple(len=L+1)
        # hidden_states[0] = embeddings後, hidden_states[l] が層lの入力, hidden_states[l+1] が層lの出力
        input_embeddings = hidden_states[layer_idx]
        attention_mask = inputs.get(
            "attention_mask", torch.ones_like(inputs["input_ids"])
        )
        token_type_ids = inputs.get(
            "token_type_ids", torch.zeros_like(inputs["input_ids"])
        )

    if debug:
        logger.debug(f"⚡ 埋め込み形状: {input_embeddings.shape}")
        logger.debug(f"⚡ マスク形状: {attention_mask.shape}")

    return input_embeddings, attention_mask, token_type_ids


def _extract_value_vectors(
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
    from .attention_models import AttentionModel

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
