# optimized_ig.py
"""
Optimised Integrated Gradients runtime coordinated with modular components.

This refactor follows the theoretical breakdown documented in
`theory/1.transformerの記号体系の定義と計算の流れ.md` by separating
device placement, task scheduling and execution into dedicated modules.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple

import torch

from utils.cache.unified_cache import get_unified_cache
from utils.calculations.ig.adaptive_batch_size import get_adaptive_batch_calculator
from utils.calculations.ig.gpu_memory_monitor import get_gpu_memory_monitor

from .optimized_runtime import (
    ActiveComputationRegistry,
    AttentionExecutor,
    DevicePool,
    IGTask,
    MLPExecutor,
    generate_task_key,
)

logger = logging.getLogger(__name__)


class OptimizedIGCalculator:
    """
    Optimised IG calculator orchestrating attention/MLP relevance evaluation.

    Responsibilities are split across helper components:
        * DevicePool – owns model replicas, streams, and input placement.
        * AttentionExecutor / MLPExecutor – perform IG integrations with
          caching and duplicate-elimination.
        * ActiveComputationRegistry – serialises identical in-flight work.
    """

    def __init__(
        self,
        unified_model,
        tokenizer,
        use_lightning_trainer: bool = False,
        total_samples: int = 0,
    ):
        self.unified_model = unified_model
        self.tokenizer = tokenizer
        self.cache = get_unified_cache()
        self.use_lightning_trainer = use_lightning_trainer
        self._total_samples = total_samples  # 大量サンプル実行時のログ抑制用

        self.is_h100 = (
            torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0)
        )
        self.is_a100 = (
            torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0)
        )

        workers_per_gpu = self._determine_workers_per_gpu()
        self.device_pool = DevicePool(
            unified_model,
            workers_per_gpu=workers_per_gpu,
            use_lightning_trainer=use_lightning_trainer,
        )

        self.max_concurrent_tasks = self.device_pool.max_concurrent_tasks
        self.max_batch_size = self.device_pool.max_batch_size
        self.max_workers = max(workers_per_gpu * self.device_pool.gpu_count, 4)

        self.registry = ActiveComputationRegistry()
        # baseline_methodはデフォルトで"zero"を使用（後で設定可能）
        self.baseline_method = "zero"
        # input_typeはデフォルトで"z"を使用（後で設定可能）
        self.input_type = "z"
        # use_direct_computationはinput_type="v"の場合True（後で設定可能）
        self.use_direct_computation = False
        self.attention_executor = AttentionExecutor(
            self.cache,
            self.registry,
            is_h100=self.is_h100,
            baseline_method=self.baseline_method,
            input_type=self.input_type,
            use_direct_computation=self.use_direct_computation,
        )
        self.mlp_executor = MLPExecutor(
            self.cache,
            self.registry,
            is_h100=self.is_h100,
            baseline_method=self.baseline_method,
            input_type=self.input_type,
            use_direct_computation=self.use_direct_computation,
        )

        logger.info(
            "OptimizedIGCalculator initialised: gpu_count=%d, max_workers=%d, "
            "max_batch_size=%d",
            self.device_pool.gpu_count,
            self.max_workers,
            self.max_batch_size,
        )

        # 混合精度コンテキスト（デフォルトはNone）
        self.precision_context = None

    # ----------------------------------------------AA-------------------- #
    # Baseline method management
    # ------------------------------------------------------------------ #
    def set_baseline_method(self, baseline_method: str = "zero"):
        """
        ベースライン方法を設定

        Args:
            baseline_method: ベースライン選択方法（"zero", "self_input_token"）
        """
        self.baseline_method = baseline_method
        from utils.calculations.ig.shared.release_scope import reject_otb_baseline

        reject_otb_baseline(baseline_method)
        # Executorにも同じbaseline設定を伝播
        self.attention_executor.baseline_method = baseline_method
        self.mlp_executor.baseline_method = baseline_method
        self._refresh_direct_computation_mode()
        logger.info(f"ベースライン方法を設定: {baseline_method}")

    def set_input_type(self, input_type: str = "z"):
        """
        入力タイプを設定

        Args:
            input_type: 入力タイプ（"z": 入力埋め込み, "v": Valueベクトル）
        """
        self.input_type = input_type
        # Executorにも同じ入力タイプを伝播
        self.attention_executor.input_type = input_type
        self.mlp_executor.input_type = input_type
        self._refresh_direct_computation_mode()
        logger.info(
            "入力タイプを設定: %s, ベースライン: %s, 直接計算: %s",
            input_type,
            self.baseline_method,
            self.use_direct_computation,
        )

    def _refresh_direct_computation_mode(self) -> None:
        """現在の input_type / baseline_method から direct 計算可否を再評価する。"""
        self.use_direct_computation = (
            self.input_type == "v" 
        )
        self.attention_executor.use_direct_computation = self.use_direct_computation
        self.mlp_executor.use_direct_computation = self.use_direct_computation

    # ------------------------------------------------------------------ #
    # Precision management
    # ------------------------------------------------------------------ #
    def set_precision_context(self, precision_context):
        """
        混合精度コンテキストを設定

        Args:
            precision_context: torch.cuda.amp.autocast context manager
        """
        self.precision_context = precision_context
        # executorにも伝播
        self.attention_executor.set_precision_context(precision_context)
        self.mlp_executor.set_precision_context(precision_context)
        logger.info("Precision context set for IG calculations and executors")

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    def _determine_workers_per_gpu(self) -> int:
        if self.is_h100:
            return 128  # H100なら128に大幅増加（GPU使用率向上のため）
        if self.is_a100:
            return 64
        # V100: 4GPU環境では32に増加（従来の16から2倍）
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if gpu_count >= 4:
            return 32  # 4GPU以上なら32に増加
        return 24  # 1-3GPUなら24

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute_batch_ig(
        self,
        tasks: List[IGTask],
        text: str,
        progress_callback=None,
        cached_hidden_states: Optional[
            Tuple
        ] = None,  # 事前計算済みhidden states（オプション）
        tokenizer_lock: Optional[Any] = None,  # tokenizerのスレッド安全性のためのロック
        preferred_device_id: Optional[
            int
        ] = None,  # 優先的に使用するGPU ID（文レベル割り当て用）
        is_retry: bool = False,  # 再計算フラグ（並列処理を控えめに）
    ) -> Dict[str, Any]:
        """
        Execute deduplicated IG tasks in parallel across available devices.

        Args:
            tasks: IGタスクリスト
            text: 入力テキスト
            progress_callback: 進捗コールバック
            cached_hidden_states: 事前計算済みhidden states（Tuple[torch.Tensor, ...]）が提供される場合はBERT推論をスキップ
            tokenizer_lock: tokenizerのスレッド安全性のためのロック（Optional）
        """
        import threading

        # cached_hidden_statesがある場合はテキストのトークン化をスキップ
        # attention_maskとtoken_type_idsはcached_hidden_statesから取得した情報で構築
        if cached_hidden_states is not None and len(cached_hidden_states) > 0:
            # cached_hidden_statesから情報を取得してbase_inputsを構築
            # hidden_states[0]の形状からseq_lenを取得
            first_hidden_state = cached_hidden_states[0]
            batch_size = first_hidden_state.shape[0]
            seq_len = first_hidden_state.shape[1]
            device = first_hidden_state.device

            # attention_maskとtoken_type_idsを構築（input_idsは不要）
            attention_mask = torch.ones(
                (batch_size, seq_len), device=device, dtype=torch.long
            )
            token_type_ids = torch.zeros(
                (batch_size, seq_len), device=device, dtype=torch.long
            )

            base_inputs = {
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            logger.debug(
                f"cached_hidden_statesからbase_inputsを構築: seq_len={seq_len}, device={device}"
            )
        else:
            # tokenizerはスレッドセーフではないため、ロックを使用
            if tokenizer_lock is not None:
                with tokenizer_lock:
                    base_inputs = self.tokenizer(text, return_tensors="pt")
            else:
                base_inputs = self.tokenizer(text, return_tensors="pt")

        device_inputs_map = self.device_pool.prepare_inputs(base_inputs)
        device_ids = list(device_inputs_map.keys()) or [
            self.device_pool.primary_device_id
        ]

        # preferred_device_idが指定されている場合、そのGPUのみを使用
        if preferred_device_id is not None:
            # 指定されたGPUのみを使用するように変更
            if preferred_device_id < self.device_pool.gpu_count:
                device = torch.device(f"cuda:{preferred_device_id}")
                # 指定されたGPUの入力のみを準備
                device_inputs_map = {
                    preferred_device_id: {
                        k: v.to(device) for k, v in base_inputs.items()
                    }
                }
                device_ids = [preferred_device_id]
                # GPU分散のログは削除（パフォーマンス向上のため）
            else:
                logger.warning(
                    f"指定されたGPU {preferred_device_id}が存在しません。"
                    f"利用可能なGPU数: {self.device_pool.gpu_count}。"
                    f"デフォルトGPUを使用します。"
                )

        unique_tasks: Dict[str, IGTask] = {}
        for task in tasks:
            task_key = generate_task_key(task, text)
            unique_tasks.setdefault(task_key, task)

        # DEBUGログは削減（大量に出力されるため）
        # logger.debug(
        #     "Task deduplication: %d -> %d (removed %d duplicates)",
        #     len(tasks),
        #     len(unique_tasks),
        #     len(tasks) - len(unique_tasks),
        # )

        attention_tasks = [
            task for task in unique_tasks.values() if task.task_type == "attention"
        ]
        mlp_tasks = [
            task for task in unique_tasks.values() if task.task_type != "attention"
        ]

        # DEBUGログは削減（大量に出力されるため）
        # logger.debug(
        #     "Task classification: Attention=%d, MLP=%d",
        #     len(attention_tasks),
        #     len(mlp_tasks),
        # )

        total_tasks = len(unique_tasks)
        adaptive_calculator = get_adaptive_batch_calculator()
        memory_monitor = get_gpu_memory_monitor(0)

        # GPUメモリ状況を取得して動的に設定
        memory_status = memory_monitor.get_memory_status()
        free_memory_gb = memory_status["free_gb"]
        total_memory_gb = memory_status["total_gb"]
        utilization = memory_status["utilization_percent"] / 100.0

        # GPU情報を取得（V100判定用）
        try:
            gpu_name = (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
            )
            is_v100 = "V100" in gpu_name
            gpu_count = (
                self.device_pool.gpu_count
                if hasattr(self.device_pool, "gpu_count")
                else 1
            )
        except Exception:
            is_v100 = False
            gpu_count = 1

        # メモリ使用状況に応じて安全マージンとバッチサイズ制限を動的に設定
        if total_memory_gb > 0:
            if free_memory_gb < 5 or utilization > 0.85:
                # メモリが5GB未満または使用率85%以上の場合、非常に控えめに
                memory_safety_margin = 0.4  # 40%まで使用
                max_batch_limit = 32  # バッチサイズを32に制限
            elif free_memory_gb < 10 or utilization > 0.75:
                # メモリが10GB未満または使用率75%以上の場合、控えめに
                memory_safety_margin = 0.5  # 50%まで使用
                max_batch_limit = 64  # バッチサイズを64に制限
            elif free_memory_gb < 20:
                # メモリが20GB未満の場合
                memory_safety_margin = 0.6  # 60%まで使用
                max_batch_limit = 128  # バッチサイズを128に制限
            else:
                # メモリに余裕がある場合（計算速度優先、IG計算中にパラメータ取得があるため控えめに）
                memory_safety_margin = 0.70  # 70%まで使用（計算速度優先のため控えめに）
                if self.is_h100 and self.device_pool.gpu_count >= 2:
                    max_batch_limit = 4096  # H100 2枚なら4096まで
                elif self.is_h100:
                    max_batch_limit = 2048  # H100 1枚なら2048まで
                else:
                    max_batch_limit = 2048
        else:
            memory_safety_margin = 0.65  # 計算速度優先のため控えめに
            if is_v100 and gpu_count >= 4:
                max_batch_limit = 2048  # V100 4枚なら2048まで
            else:
                max_batch_limit = 512  # その他は512まで

        adaptive_batch = adaptive_calculator.calculate_optimal_batch_size(
            total_tasks=total_tasks,
            use_mixed_precision=self.is_h100,
            memory_safety_margin=memory_safety_margin,
        )

        # バッチサイズをメモリ状況に応じて制限
        if self.is_h100:
            optimal_batch_size = min(
                adaptive_batch, self.max_batch_size, max_batch_limit
            )
        else:
            optimal_batch_size = min(
                adaptive_batch, self.max_batch_size, max_batch_limit
            )

        # GPUメモリ監視器で最適化
        # H100でメモリに余裕がある場合、より積極的にバッチサイズを増やす
        # V100-4GPU環境でも同様に最適化
        if self.is_h100:
            # H100: GPU数に応じてバッチサイズを最適化
            # H100 2枚: 計算速度優先でバッチサイズを最大化
            if gpu_count >= 2:
                min_batch = 256
                max_batch = 4096  # 2枚なら最大4096
            else:
                min_batch = 256
                max_batch = 2048  # 1枚なら最大2048
        elif self.is_a100:
            # A100: 40GBメモリを活用してバッチサイズを大幅に増加
            min_batch = 128  # 最小128
            max_batch = 2048  # 最大2048（H100の半分、V100の2倍）
        elif is_v100 and gpu_count >= 4:
            # V100-4GPU環境での最適化
            min_batch = 128  # 最小128に増加
            max_batch = 2048  # 最大2048に増加
        else:
            min_batch = 16
            max_batch = 1024

        optimal_batch_size = memory_monitor.calculate_optimal_batch_size(
            optimal_batch_size,
            min_batch_size=min_batch,
            max_batch_size=max_batch,
            target_memory_utilization=0.75,  # 計算速度優先：メモリ使用率75%（IG計算中にパラメータ取得があるため控えめに）
        )

        # H100/A100の場合はより積極的にバッチサイズを増やす
        # メモリに大量の余裕がある場合、さらに積極的に増やす
        # 並列度は戻したが、バッチサイズは大きくしてGPU利用率を上げる
        if self.is_a100 and free_memory_gb > 30:
            # A100: メモリに余裕がある場合（30GB以上空き）、バッチサイズを大幅に増やす
            optimal_batch_size = min(
                optimal_batch_size * 4, 2048
            )  # 4倍に増加、上限2048
            logger.info(
                f"🚀 A100最適化（大量メモリ余裕）: バッチサイズを大幅に増やしました: {optimal_batch_size}"
            )
        elif self.is_a100 and free_memory_gb > 10:
            # A100: メモリに余裕がある場合、バッチサイズを増やす
            optimal_batch_size = min(
                optimal_batch_size * 2, 1024
            )  # 2倍に増加、上限1024
            logger.info(
                f"🚀 A100最適化: バッチサイズを増やしました: {optimal_batch_size}"
            )
        elif self.is_h100 and free_memory_gb > 70:
            # H100: GPU数に応じてバッチサイズを最適化（計算速度優先）
            # H100 2枚: メモリに大量の余裕がある場合、バッチサイズをさらに大きく
            if self.device_pool.gpu_count >= 2:
                optimal_batch_size = min(
                    optimal_batch_size * 8, 4096  # 8倍に増加、上限4096（2枚なら）
                )
            else:
                optimal_batch_size = min(
                    optimal_batch_size * 8, 2048  # 8倍に増加、上限2048（1枚なら）
                )
            logger.info(
                f"🚀 H100最適化（{self.device_pool.gpu_count}GPU、大量メモリ余裕）: バッチサイズを大幅に増やしました: {optimal_batch_size}"
            )
        elif self.is_h100 and free_memory_gb > 10:
            # H100: GPU数に応じてバッチサイズを最適化（計算速度優先）
            # H100 2枚: メモリに余裕がある場合は、バッチサイズをさらに大きく
            if self.device_pool.gpu_count >= 2:
                optimal_batch_size = min(
                    optimal_batch_size * 4, 4096  # 4倍に増加、上限4096（2枚なら）
                )
            else:
                optimal_batch_size = min(
                    optimal_batch_size * 4, 2048  # 4倍に増加、上限2048（1枚なら）
                )
            logger.info(
                f"🚀 H100最適化（{self.device_pool.gpu_count}GPU）: バッチサイズを増やしました: {optimal_batch_size}"
            )

        # ワーカー数も動的に調整（GPU利用率向上のため最適化）
        # A100/V100-4GPU環境の場合、GPU利用率を上げるため並列度を上げる
        # GPU数 × 適切な倍率（過剰な並列化を避けつつ、利用率を上げる）
        if self.is_a100:
            # A100の場合、H100の半分程度の並列度でGPU利用率を最大化
            base_workers = min(
                self.max_workers * 32,  # 32倍に増加
                self.device_pool.gpu_count * 256,  # GPU数 × 256
            )
            max_workers_limit = min(
                2048, self.device_pool.gpu_count * 256
            )  # 最大2048まで
            min_workers_limit = 128  # 最小128
            logger.info(
                f"🚀 A100最適化: ワーカー数を増やしました: {base_workers} (最大: {max_workers_limit}, 最小: {min_workers_limit})"
            )
        elif self.is_h100:
            # H100の場合、V100の2倍の並列度を目指してGPU利用率を最大化
            # GPU負荷分散の改善（タスクスコアの重み調整）は維持
            base_workers = min(
                self.max_workers * 64,  # 32倍→64倍に増加（V100の2倍）
                self.device_pool.gpu_count * 512,  # GPU数 × 512に増加（256→512）
            )
            max_workers_limit = min(
                4096, self.device_pool.gpu_count * 512
            )  # 2048→4096に増加
            min_workers_limit = 256  # 128→256に増加
        else:
            # V100-4GPU環境の場合、GPU利用率向上のため並列度を上げる
            try:
                gpu_name = (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
                )
                is_v100 = "V100" in gpu_name
            except Exception:
                is_v100 = False

            if is_v100 and self.device_pool.gpu_count >= 4:
                # V100-4GPU環境での最適化（計算速度優先）
                # メモリに余裕がある場合、バッチサイズとワーカー数をさらに増やす
                if (
                    free_memory_gb >= 20
                ):  # 20GB以上空きの場合（計算速度優先のため閾値を下げる）
                    optimal_batch_size = min(
                        optimal_batch_size * 2, 2048  # バッチサイズを2倍に増加
                    )
                    logger.info(
                        f"🚀 V100-4GPU最適化（メモリ余裕）: バッチサイズを増やしました: {optimal_batch_size}"
                    )

                # V100 4GPUの場合、GPU利用率向上のため並列度をさらに上げる
                base_workers = min(
                    self.max_workers * 8,  # 4倍→8倍に増加（計算速度優先）
                    self.device_pool.gpu_count * 128,  # GPU数 × 128に増加（4枚なら512）
                )
                max_workers_limit = min(
                    512,
                    self.device_pool.gpu_count * 128,  # 最大512まで増加（4枚なら512）
                )
                min_workers_limit = 64  # 最小64に増加（計算速度優先）
                logger.info(
                    f"🚀 V100-4GPU最適化: ワーカー数を増やしました: {base_workers} (最大: {max_workers_limit}, 最小: {min_workers_limit})"
                )
            elif self.device_pool.gpu_count >= 4:
                # その他の4GPU以上環境
                base_workers = min(
                    self.max_workers * 3,  # 2倍から3倍に増加
                    self.device_pool.gpu_count * 32,  # GPU数 × 32に制限
                )
                max_workers_limit = min(
                    128, self.device_pool.gpu_count * 32
                )  # 最大128まで増加
                min_workers_limit = 16  # 最小16に増加
            else:
                # 1-3GPUの場合
                base_workers = min(
                    self.max_workers * 2, self.device_pool.gpu_count * 32
                )
                max_workers_limit = 64
                min_workers_limit = 8

        optimal_workers = memory_monitor.calculate_optimal_workers(
            base_workers,
            min_workers=min_workers_limit,
            max_workers=max_workers_limit,
            target_memory_utilization=(
                0.75 if (self.is_h100 or self.is_a100) else 0.70
            ),  # 計算速度優先：H100/A100は75%、その他は70%（IG計算中にパラメータ取得があるため控えめに）
        )
        # max_workers must be greater than 0 を保証
        optimal_workers = max(1, optimal_workers)

        # H100/A100環境での最適化ログ出力
        if self.is_h100:
            logger.info(
                f"🚀 H100最適化（optimized_ig）: バッチサイズ={optimal_batch_size}, "
                f"ワーカー数={optimal_workers} (最大: {max_workers_limit}, 最小: {min_workers_limit}), "
                f"メモリ使用率目標=75%, GPUメモリ: {memory_monitor.get_memory_summary()}"
            )
        elif self.is_a100:
            logger.info(
                f"🚀 A100最適化（optimized_ig）: バッチサイズ={optimal_batch_size}, "
                f"ワーカー数={optimal_workers} (最大: {max_workers_limit}, 最小: {min_workers_limit}), "
                f"メモリ使用率目標=75%, GPUメモリ: {memory_monitor.get_memory_summary()}"
            )

        # V100-4GPU環境での最適化ログ出力
        if is_v100 and self.device_pool.gpu_count >= 4:
            logger.info(
                f"🚀 V100-4GPU最適化（optimized_ig）: バッチサイズ={optimal_batch_size}, "
                f"ワーカー数={optimal_workers} (最大: {max_workers_limit}), "
                f"GPUメモリ: {memory_monitor.get_memory_summary()}"
            )

        # 再計算時はより控えめな設定で実行（メモリエラーを避けるため）
        if is_retry:
            # バッチサイズを半分に減らす
            optimal_batch_size = max(optimal_batch_size // 2, 16)  # 最小16を保証
            # ワーカー数も半分に減らす
            optimal_workers = max(optimal_workers // 2, 4)  # 最小4を保証
            logger.info(
                f"🔄 再計算モード: バッチサイズ={optimal_batch_size}, "
                f"ワーカー数={optimal_workers}（控えめな設定で実行）"
            )

        # 実行時にワーカー数を調整できるように保存
        self._current_optimal_workers = optimal_workers

        logger.info(
            f"🚀 IG計算最適化設定: バッチサイズ={optimal_batch_size} (adaptive={adaptive_batch}, max={self.max_batch_size}), "
            f"ワーカー数={optimal_workers} (最大: {self.max_workers}), "
            f"GPUメモリ: {memory_monitor.get_memory_summary()}"
        )

        results: Dict[str, Dict[str, Any]] = {}
        completed_tasks = 0

        # @logs/ ディレクトリを使用
        logs_dir = os.path.abspath("logs")
        os.makedirs(logs_dir, exist_ok=True)
        # すべてのログを統一されたファイルに書き込む
        log_filename = os.path.join(logs_dir, "ig_calculation_debug.log")

        # 新しい実行時にログファイルをクリア（0から書き始める）
        if os.path.exists(log_filename):
            try:
                os.remove(log_filename)
                logger.info(f"🗑️ 既存のログファイルを削除しました: {log_filename}")
            except Exception as e:
                logger.warning(f"⚠️ ログファイルの削除に失敗しました: {e}")

        # Streamlit警告を最初から抑制
        streamlit_logger = logging.getLogger(
            "streamlit.runtime.scriptrunner.script_run_context"
        )
        streamlit_logger.setLevel(logging.ERROR)
        streamlit_logger = logging.getLogger("streamlit.runtime.caching")
        streamlit_logger.setLevel(logging.ERROR)
        streamlit_logger = logging.getLogger("streamlit")
        streamlit_logger.setLevel(logging.ERROR)

        # 統一ログ設定を使用（既にセットアップされている場合は追加設定をしない）
        root_logger = logging.getLogger()

        # 既に統一ログ設定がセットアップされているかチェック
        has_file_handler = any(
            isinstance(h, logging.FileHandler) for h in root_logger.handlers
        )

        # 統一ログ設定がセットアップされていない場合のみ、簡易設定を追加
        if not has_file_handler:
            try:
                from utils.common.logging_setup import setup_unified_logging

                setup_unified_logging(
                    log_file_path=log_filename,
                    log_level=logging.INFO,
                    enable_console=False,  # このモジュールではコンソール出力を無効化
                    enable_file=True,
                    redirect_stdout=False,  # このモジュールでは標準出力をリダイレクトしない
                )
            except ImportError:
                # 統一ログ設定モジュールが利用できない場合は従来の設定を使用
                file_handler = logging.FileHandler(
                    log_filename, mode="w", encoding="utf-8"
                )
                file_handler.setLevel(logging.INFO)
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(file_formatter)

                # 既存のハンドラーをクリア（重複を防ぐ）
                for handler in root_logger.handlers[:]:
                    if (
                        isinstance(handler, logging.FileHandler)
                        and handler.baseFilename == log_filename
                    ):
                        root_logger.removeHandler(handler)

                root_logger.addHandler(file_handler)

        logging.info("=" * 80)
        logging.info("IGバッチ計算開始: %dタスク", total_tasks)
        logging.info(
            "Attentionタスク数: %d, MLPタスク数: %d",
            len(attention_tasks),
            len(mlp_tasks),
        )
        logging.info("ユニークタスク数: %d (重複削除後)", len(unique_tasks))
        if progress_callback:
            progress_callback(0, max(total_tasks, 1), "IG計算を開始しました")

        import warnings

        try:
            # Streamlitロガーの警告レベルを下げる（より包括的に）
            streamlit_logger = logging.getLogger(
                "streamlit.runtime.scriptrunner.script_run_context"
            )
            streamlit_logger.setLevel(logging.ERROR)
            streamlit_logger = logging.getLogger("streamlit.runtime.caching")
            streamlit_logger.setLevel(logging.ERROR)
            streamlit_logger = logging.getLogger("streamlit")
            streamlit_logger.setLevel(logging.ERROR)

            # osは既にファイルの先頭でインポートされているため、再インポート不要
            os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "ERROR")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="streamlit"
                )
                warnings.filterwarnings(
                    "ignore", message=".*missing ScriptRunContext.*"
                )
                warnings.filterwarnings(
                    "ignore", message=".*Thread.*missing ScriptRunContext.*"
                )
                # すべてのStreamlit関連の警告を抑制
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="streamlit.*"
                )

                # 動的に調整されたワーカー数を使用
                optimal_workers = getattr(
                    self, "_current_optimal_workers", self.max_workers
                )
                # max_workers must be greater than 0 を保証
                optimal_workers = max(1, optimal_workers)

                # ATTとMLPを別々に並列実行（異なるToken/Head/Layerを跨いで並列化）
                with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                    # ATTタスクを並列実行（異なるToken/Head/Layerを跨いで）
                    attention_futures, remaining_attention_tasks = self._schedule_tasks(
                        executor,
                        attention_tasks,
                        text,
                        device_inputs_map,
                        device_ids,
                        optimal_batch_size,
                        optimal_workers,
                        is_attention=True,
                        cached_hidden_states=cached_hidden_states,
                        preferred_device_id=preferred_device_id,
                    )
                    logging.info(
                        f"Attentionタスクスケジュール完了: "
                        f"スケジュール済みfutures={len(attention_futures)}, "
                        f"残りタスク={len(remaining_attention_tasks)}"
                    )

                    # MLPタスクを並列実行（異なるToken/Layerを跨いで）
                    mlp_futures, remaining_mlp_tasks = self._schedule_tasks(
                        executor,
                        mlp_tasks,
                        text,
                        device_inputs_map,
                        device_ids,
                        optimal_batch_size,
                        optimal_workers,
                        is_attention=False,
                        cached_hidden_states=None,
                        preferred_device_id=preferred_device_id,
                    )
                    logging.info(
                        f"MLPタスクスケジュール完了: "
                        f"スケジュール済みfutures={len(mlp_futures)}, "
                        f"残りタスク={len(remaining_mlp_tasks)}"
                    )

                    combined_futures = {**attention_futures, **mlp_futures}
                    logging.info(
                        f"combined_futures作成完了: "
                        f"合計futures数={len(combined_futures)}, "
                        f"Attention={len(attention_futures)}, MLP={len(mlp_futures)}"
                    )

                    # 段階的スケジュール用のデバイスサイクル
                    device_cycle = cycle(device_ids)
                    # 初期値は控えめに設定（後で動的に調整される）
                    # 並列度は戻したが、同時実行タスク数は増やしてGPU利用率を上げる
                    # H100の場合はさらに積極的に増やす
                    if self.is_h100:
                        max_concurrent_futures = (
                            optimal_workers * 16
                        )  # H100ならワーカー数の16倍に増加（8倍→16倍、GPU利用率最大化）
                    else:
                        max_concurrent_futures = optimal_workers * 4  # その他は4倍

                    # メモリエラーが連続で発生した場合のカウンタ
                    consecutive_memory_errors = 0
                    max_consecutive_memory_errors = (
                        10  # 連続10回メモリエラーが発生したら一時停止
                    )

                    # 処理済みfutureを追跡（バッチ処理の重複を防ぐ）
                    processed_futures = set()

                    # デバッグ: 処理開始時のタスク数を記録
                    initial_futures_count = len(combined_futures)
                    logging.info("=" * 80)
                    logging.info(
                        f"compute_batch_ig: as_completedループ開始 "
                        f"(初期futures数: {initial_futures_count}, "
                        f"総タスク数: {total_tasks}, "
                        f"残りAttentionタスク={len(remaining_attention_tasks)}, "
                        f"残りMLPタスク={len(remaining_mlp_tasks)})"
                    )
                    # 各futureのタスク情報をログに記録
                    batch_futures_count = sum(
                        1
                        for info in combined_futures.values()
                        if isinstance(info, tuple)
                        and len(info) == 3
                        and info[2] is True
                    )
                    normal_futures_count = initial_futures_count - batch_futures_count
                    logging.info(
                        f"future分類: バッチ処理={batch_futures_count}, "
                        f"通常処理={normal_futures_count}"
                    )

                    processed_count = 0
                    skipped_count = 0
                    last_progress_update = 0
                    last_progress_time = time.time()
                    progress_update_interval = 2.0  # 2秒ごとに進捗を更新

                    # タスクの進捗速度を測定
                    task_start_time = time.time()
                    tasks_completed_at_start = completed_tasks
                    last_speed_check_time = task_start_time
                    last_speed_check_completed = completed_tasks

                    for future in as_completed(combined_futures):
                        # 定期的に進捗を更新（バッチ処理中でも進捗を表示）
                        # 出力頻度を制限（2秒ごと、または10%の進捗、または完了時）
                        current_time = time.time()
                        should_update_progress = (
                            current_time - last_progress_time
                            >= progress_update_interval
                            and completed_tasks % max(1, total_tasks // 20)
                            == 0  # 5%ごと
                        )

                        # タスクの進捗速度を測定
                        # 大量サンプル実行時（1700以上）は頻度を下げる（30秒ごと）、通常は5秒ごと
                        # IG計算の特性上、GPU利用率が0%になることもあるため、タスク処理速度で性能を評価
                        speed_check_interval = (
                            30.0
                            if (
                                hasattr(self, "_total_samples")
                                and getattr(self, "_total_samples", 0) >= 1700
                            )
                            else 5.0
                        )
                        if current_time - last_speed_check_time >= speed_check_interval:
                            elapsed = current_time - last_speed_check_time
                            tasks_done = completed_tasks - last_speed_check_completed

                            # GPU使用状況を取得
                            gpu_status_lines = []
                            if torch.cuda.is_available():
                                for device_id in device_ids:
                                    try:
                                        free_bytes, total_bytes = (
                                            torch.cuda.mem_get_info(device_id)
                                        )
                                        free_gb = free_bytes / (1024**3)
                                        total_gb = total_bytes / (1024**3)
                                        used_gb = total_gb - free_gb
                                        mem_util = (
                                            (used_gb / total_gb * 100)
                                            if total_gb > 0
                                            else 0.0
                                        )

                                        # 実行中タスク数
                                        task_count = (
                                            self._device_task_count.get(device_id, 0)
                                            if hasattr(self, "_device_task_count")
                                            else 0
                                        )

                                        gpu_status_lines.append(
                                            f"GPU{device_id}: {used_gb:.1f}GB/{total_gb:.1f}GB ({mem_util:.1f}%), "
                                            f"実行中タスク: {task_count}"
                                        )
                                    except Exception:
                                        pass

                            if elapsed > 0 and tasks_done > 0:
                                tasks_per_second = tasks_done / elapsed
                                # 残りタスク数の推定
                                remaining_tasks = total_tasks - completed_tasks
                                if tasks_per_second > 0:
                                    estimated_remaining_seconds = (
                                        remaining_tasks / tasks_per_second
                                    )
                                    estimated_remaining_minutes = (
                                        estimated_remaining_seconds / 60
                                    )

                                    # 全体の処理速度も計算
                                    total_elapsed = current_time - task_start_time
                                    if total_elapsed > 0:
                                        overall_tasks_per_second = (
                                            completed_tasks - tasks_completed_at_start
                                        ) / total_elapsed
                                    else:
                                        overall_tasks_per_second = 0.0

                                    # 分かりやすいログ形式
                                    gpu_status = (
                                        " | ".join(gpu_status_lines)
                                        if gpu_status_lines
                                        else "N/A"
                                    )
                                    logging.info(
                                        f"\n{'='*80}\n"
                                        f"📊 パフォーマンス状況\n"
                                        f"  進捗: {completed_tasks:,}/{total_tasks:,} ({completed_tasks*100//total_tasks if total_tasks > 0 else 0}%)\n"
                                        f"  処理速度: {tasks_per_second:.2f} タスク/秒 (直近) | {overall_tasks_per_second:.2f} タスク/秒 (全体平均)\n"
                                        f"  残り時間推定: {estimated_remaining_minutes:.1f}分\n"
                                        f"  実行中タスク: {len(combined_futures)}\n"
                                        f"  {gpu_status}\n"
                                        f"{'='*80}"
                                    )
                            last_speed_check_time = current_time
                            last_speed_check_completed = completed_tasks

                        if should_update_progress:
                            # 実行中のタスク数から進捗を推定
                            running_tasks = len(combined_futures)
                            estimated_progress = (
                                completed_tasks
                                + (total_tasks - completed_tasks - running_tasks) // 2
                            )

                            # タスク処理速度を計算
                            total_elapsed = current_time - task_start_time
                            if total_elapsed > 0:
                                overall_tasks_per_second = (
                                    completed_tasks - tasks_completed_at_start
                                ) / total_elapsed
                            else:
                                overall_tasks_per_second = 0.0

                            if progress_callback:
                                # 使用中のGPUを表示（複数GPU対応）
                                if not torch.cuda.is_available():
                                    device_label = "CPU"
                                elif len(device_ids) == 1:
                                    device_label = f"GPU{device_ids[0]}"
                                elif len(device_ids) <= 4:
                                    device_label = (
                                        f"GPU{'-'.join(map(str, sorted(device_ids)))}"
                                    )
                                else:
                                    device_label = (
                                        f"GPU{device_ids[0]}-{device_ids[-1]}"
                                    )
                                progress_callback(
                                    min(estimated_progress, total_tasks),
                                    total_tasks,
                                    f"{device_label}: {completed_tasks:,}/{total_tasks:,} ({completed_tasks*100//total_tasks if total_tasks > 0 else 0}%) "
                                    f"[{overall_tasks_per_second:.1f} tasks/s]",
                                )
                            last_progress_time = current_time
                        # 完了したタスクの情報を取得（削除前に取得）
                        task_info = None
                        if future in combined_futures:
                            task_info = combined_futures[future]
                            # 完了したタスクを削除
                            del combined_futures[future]

                            # 実行中のタスク数を減らす（GPU使用状況の追跡）
                            if isinstance(task_info, tuple) and len(task_info) >= 2:
                                device_id = task_info[1]
                                if (
                                    hasattr(self, "_device_task_count")
                                    and device_id in self._device_task_count
                                ):
                                    self._device_task_count[device_id] = max(
                                        0, self._device_task_count[device_id] - 1
                                    )
                        else:
                            # futureが既に削除されている場合はスキップ
                            # これは通常発生しないが、念のため
                            logging.warning(
                                f"futureがcombined_futuresに存在しません: {id(future)}"
                            )
                            # futureから結果を取得してエラー情報を記録
                            try:
                                result = future.result()
                                # task_infoが取得できないため、エラーとして記録
                                future_id = id(future)
                                error_key = f"unknown_task_{future_id}"
                                results[error_key] = {
                                    "task": None,
                                    "result": None,
                                    "success": False,
                                    "error": f"futureがcombined_futuresに存在しません: {id(future)}",
                                }
                                completed_tasks += 1
                            except Exception as exc:
                                # future.result()も失敗した場合
                                future_id = id(future)
                                error_key = f"unknown_task_{future_id}"
                                results[error_key] = {
                                    "task": None,
                                    "result": None,
                                    "success": False,
                                    "error": f"futureがcombined_futuresに存在しません: {id(future)}, future.result()エラー: {exc}",
                                }
                                completed_tasks += 1
                            continue

                        # task_infoがNoneの場合はスキップ
                        if task_info is None:
                            logging.warning(f"task_infoがNoneです: {id(future)}")
                            # futureから結果を取得してエラー情報を記録
                            try:
                                result = future.result()
                                # task_infoが取得できないため、エラーとして記録
                                future_id = id(future)
                                error_key = f"unknown_task_{future_id}"
                                results[error_key] = {
                                    "task": None,
                                    "result": None,
                                    "success": False,
                                    "error": f"task_infoがNoneです: {id(future)}",
                                }
                                completed_tasks += 1
                            except Exception as exc:
                                # future.result()も失敗した場合
                                future_id = id(future)
                                error_key = f"unknown_task_{future_id}"
                                results[error_key] = {
                                    "task": None,
                                    "result": None,
                                    "success": False,
                                    "error": f"task_infoがNoneです: {id(future)}, future.result()エラー: {exc}",
                                }
                                completed_tasks += 1
                            continue

                        # 残りのタスクがあれば、段階的にスケジュール
                        # GPU利用率を最大化するため、同時実行タスク数を大幅に増やす
                        # GPU使用状況を監視して、アイドルGPUに即座にタスクを割り当て
                        # H100などの高性能GPUでは、より多くのタスクを同時実行可能
                        # メモリに余裕がある限り、タスクを事前にメモリにロードして待機させる

                        # GPUメモリ使用状況を確認して、プリロード可能なタスク数を決定
                        # 頻繁に呼ばれるため、キャッシュしてパフォーマンスを向上
                        if (
                            not hasattr(self, "_last_preload_check_time")
                            or (current_time - self._last_preload_check_time) > 2.0
                        ):
                            max_preload_tasks = self._calculate_max_preload_tasks(
                                device_ids, optimal_workers
                            )
                            self._cached_max_preload_tasks = max_preload_tasks
                            self._last_preload_check_time = current_time
                        else:
                            max_preload_tasks = getattr(
                                self, "_cached_max_preload_tasks", optimal_workers * 4
                            )

                        max_concurrent_futures = max(
                            max_concurrent_futures, max_preload_tasks
                        )  # メモリに応じて動的に調整

                        # タスクを事前にメモリにロードして待機させる（パイプライン処理）
                        # より積極的にタスクをプリロード（一度に複数のタスクを投入）
                        # GPU利用率が下がらないように、常に十分なタスクを待機させる
                        available_slots = max_concurrent_futures - len(combined_futures)
                        # GPU利用率を最大化するため、より積極的にタスクをプリロード
                        # メモリに大量の余裕がある場合、さらに積極的にプリロード
                        # メモリ使用率を確認して動的に調整
                        if torch.cuda.is_available():
                            try:
                                free_bytes, total_bytes = torch.cuda.mem_get_info(
                                    device_ids[0] if device_ids else 0
                                )
                                free_ratio = (
                                    free_bytes / total_bytes if total_bytes > 0 else 0.5
                                )
                                if free_ratio >= 0.90:  # 90%以上空きがある場合
                                    preload_batch_size = min(
                                        200, available_slots
                                    )  # 一度に最大200タスクをプリロード（120→200に増加）
                                elif free_ratio >= 0.85:  # 85%以上空きがある場合
                                    preload_batch_size = min(
                                        150, available_slots
                                    )  # 一度に最大150タスクをプリロード（100→150に増加）
                                else:
                                    preload_batch_size = min(
                                        120, available_slots
                                    )  # 一度に最大120タスクをプリロード（80→120に増加）
                            except Exception:
                                preload_batch_size = min(
                                    120, available_slots
                                )  # エラー時はデフォルト値（80→120に増加）
                        else:
                            preload_batch_size = min(
                                120, available_slots
                            )  # CPUの場合はデフォルト値（80→120に増加）
                        preloaded_count = 0

                        # GPU利用率を最大化するため、常に十分なタスクを待機させる
                        while (
                            len(combined_futures) < max_concurrent_futures
                            and preloaded_count < preload_batch_size
                        ):
                            # 優先順位: Attentionタスク → MLPタスク
                            if remaining_attention_tasks:
                                task = remaining_attention_tasks.pop(0)
                                # GPU使用状況を考慮して最適なGPUを選択（動的スケジューリング）
                                device_id = self._get_optimal_device_id(device_ids)
                                model = self.device_pool.get_model(device_id)
                                inputs_for_device = device_inputs_map[device_id]
                                stream = self.device_pool.get_stream(device_id)

                                new_future = executor.submit(
                                    self.attention_executor.execute,
                                    task,
                                    text,
                                    inputs_for_device,
                                    fallback_model=self.unified_model,
                                    model=model,
                                    device_id=device_id,
                                    stream=stream,
                                    cached_hidden_states=cached_hidden_states,
                                )
                                combined_futures[new_future] = (task, device_id, False)
                            elif remaining_mlp_tasks:
                                # MLPタスクは通常、バッチ処理で既にスケジュールされているため、
                                # ここに来ることは少ない
                                task = remaining_mlp_tasks.pop(0)
                                # GPU使用状況を考慮して最適なGPUを選択（動的スケジューリング）
                                device_id = self._get_optimal_device_id(device_ids)
                                model = self.device_pool.get_model(device_id)
                                inputs_for_device = device_inputs_map[device_id]
                                stream = self.device_pool.get_stream(device_id)

                                # タスクを事前にメモリにロードして待機させる
                                new_future = executor.submit(
                                    self.mlp_executor.execute,
                                    task,
                                    text,
                                    inputs_for_device,
                                    fallback_model=self.unified_model,
                                    model=model,
                                    device_id=device_id,
                                    stream=stream,
                                )
                                combined_futures[new_future] = (task, device_id, False)
                                preloaded_count += 1
                            else:
                                # 残りのタスクがない場合は終了
                                break

                        # バッチ処理の場合と通常処理の場合を区別
                        if len(task_info) == 3 and task_info[2] is True:
                            # バッチ処理の場合: (batch_tasks, device_id, True)
                            if future in processed_futures:
                                skipped_count += 1
                                logging.warning(
                                    f"compute_batch_ig: futureが既に処理済みです: {id(future)}"
                                )
                                continue  # 既に処理済み
                            processed_futures.add(future)
                            processed_count += 1

                            batch_tasks, device_id, _ = task_info

                            try:
                                batch_results = future.result()
                                # バッチ処理の結果を処理
                                if isinstance(batch_results, dict):
                                    # すべてのタスクが結果に含まれているか確認
                                    batch_task_keys = {
                                        generate_task_key(task, text)
                                        for task in batch_tasks
                                    }
                                    missing_in_batch_results = batch_task_keys - set(
                                        batch_results.keys()
                                    )
                                    if missing_in_batch_results:
                                        logging.warning(
                                            f"バッチ結果に{len(missing_in_batch_results)}個のタスクが含まれていません: {missing_in_batch_results}"
                                        )

                                    for task in batch_tasks:
                                        task_key = generate_task_key(task, text)
                                        if task_key in batch_results:
                                            result = batch_results[task_key]
                                            consecutive_memory_errors = 0
                                            # execute_batchはDict[str, Optional[np.ndarray]]を返す
                                            # 辞書形式に変換
                                            results[task_key] = {
                                                "task": task,
                                                "result": result,
                                                "success": result is not None,
                                            }
                                            completed_tasks += 1
                                        else:
                                            logging.warning(
                                                "バッチ結果に見つかりません: %s",
                                                task_key,
                                            )
                                            results[task_key] = {
                                                "task": task,
                                                "result": None,
                                                "success": False,
                                                "error": "バッチ結果に存在しません",
                                            }
                                            completed_tasks += 1
                                else:
                                    # バッチ結果が辞書でない場合
                                    logging.warning(
                                        "バッチ結果の型が不正: %s", type(batch_results)
                                    )
                                    for task in batch_tasks:
                                        task_key = generate_task_key(task, text)
                                        if task_key not in results:
                                            results[task_key] = {
                                                "task": task,
                                                "result": None,
                                                "success": False,
                                                "error": f"バッチ結果の型が不正: {type(batch_results)}",
                                            }
                                            completed_tasks += 1

                                # 進捗更新（完了タスク数が増加した時のみ、かつ5%ごと）
                                current_time = time.time()
                                should_update = (
                                    completed_tasks > last_progress_update
                                    and (
                                        completed_tasks % max(1, total_tasks // 20)
                                        == 0  # 5%ごと
                                        or current_time - last_progress_time
                                        >= progress_update_interval
                                    )
                                )
                                if progress_callback and should_update:
                                    device_label = (
                                        "CPU"
                                        if device_id in (-1, None)
                                        else f"GPU{device_id}"
                                    )
                                    running_count = len(combined_futures)
                                    progress_callback(
                                        completed_tasks,
                                        total_tasks,
                                        f"{device_label}: {completed_tasks:,}/{total_tasks:,} ({completed_tasks*100//total_tasks if total_tasks > 0 else 0}%)",
                                    )
                                    last_progress_update = completed_tasks
                                    last_progress_time = current_time
                            except Exception as exc:
                                error_msg = str(exc)
                                # エラーが発生した場合は即座に計算を停止
                                logging.error(f"IGバッチ処理エラー: {error_msg}")
                                # 残りのfutureをキャンセル
                                for f in combined_futures:
                                    f.cancel()
                                # 計算を停止
                                raise RuntimeError(
                                    f"IGバッチ処理エラーにより計算を停止します。エラー: {error_msg}"
                                ) from exc
                        else:
                            # 通常処理の場合: (task, device_id, False)
                            processed_count += 1
                            if len(task_info) == 3:
                                task, device_id, _ = task_info
                            elif len(task_info) == 2:
                                # 後方互換性のため
                                task, device_id = task_info
                            else:
                                # task_infoが不正な場合は警告を記録してスキップ
                                logging.warning(
                                    f"不正なtask_info: {task_info} (長さ: {len(task_info) if hasattr(task_info, '__len__') else 'N/A'})"
                                )
                                # 不正なtask_infoの場合でも、futureから結果を取得してエラー情報を記録
                                try:
                                    result = future.result()
                                    # future.result()が成功した場合でも、task_infoが不正なのでエラーとして記録
                                    future_id = id(future)
                                    error_key = f"unknown_task_{future_id}"
                                    results[error_key] = {
                                        "task": None,
                                        "result": None,
                                        "success": False,
                                        "error": f"不正なtask_info: {task_info}",
                                    }
                                    completed_tasks += 1
                                except Exception as exc:
                                    # future.result()も失敗した場合
                                    future_id = id(future)
                                    error_key = f"unknown_task_{future_id}"
                                    results[error_key] = {
                                        "task": None,
                                        "result": None,
                                        "success": False,
                                        "error": f"不正なtask_info: {task_info}, future.result()エラー: {exc}",
                                    }
                                    completed_tasks += 1
                                continue

                            task_key = generate_task_key(task, text)
                            logging.debug(
                                f"通常処理開始: {task_key} "
                                f"(L{task.layer_idx} T{task.token_idx} "
                                f"H{task.head_idx if hasattr(task, 'head_idx') else '-'})"
                            )

                            try:
                                result = future.result()
                                # 成功した場合はカウンタをリセット
                                consecutive_memory_errors = 0

                                # 結果がNoneの場合は、エラーではなく警告として記録し、処理を続行
                                if result is None:
                                    logging.warning(
                                        "IGタスク結果がNone: %s (L%d T%d H%s)",
                                        task_key,
                                        task.layer_idx,
                                        task.token_idx,
                                        task.head_idx,
                                    )
                                    # Noneの結果も記録する（後でエラーハンドリング）
                                    results[task_key] = {
                                        "task": task,
                                        "result": None,
                                        "success": False,
                                        "error": "結果がNone",
                                        "error_type": "NoneResult",
                                    }
                                    completed_tasks += 1
                                else:
                                    results[task_key] = {
                                        "task": task,
                                        "result": result,
                                        "success": True,
                                    }
                                    completed_tasks += 1
                            except (
                                Exception
                            ) as exc:  # pragma: no cover - diagnostic logging path
                                import traceback

                                error_msg = str(exc)
                                error_type = type(exc).__name__
                                error_traceback = traceback.format_exc()

                                # メモリエラーの場合
                                if (
                                    "out of memory" in error_msg.lower()
                                    or "CUBLAS_STATUS_ALLOC_FAILED" in error_msg
                                ):
                                    consecutive_memory_errors += 1
                                    # メモリエラーを詳細に記録（エラーログの詳細化）
                                    logging.error(
                                        f"メモリエラー発生 (連続{consecutive_memory_errors}回目): {task_key} "
                                        f"(L{task.layer_idx} T{task.token_idx} H{task.head_idx if hasattr(task, 'head_idx') else '-'})\n"
                                        f"エラー種類: {error_type}\n"
                                        f"エラーメッセージ: {error_msg}\n"
                                        f"トレースバック:\n{error_traceback}"
                                    )

                                    # メモリクリーンアップを実行（連続エラーが多い場合、または定期的に）
                                    # 改善: エラーがなくても定期的にクリーンアップ（メモリリーク防止）
                                    should_cleanup = (
                                        consecutive_memory_errors % 5 == 0
                                        or completed_tasks % 100
                                        == 0  # 100タスクごとに予防的クリーンアップ
                                    )
                                    if should_cleanup:
                                        if torch.cuda.is_available():
                                            # すべてのGPUでメモリクリーンアップ
                                            for gpu_id in range(
                                                torch.cuda.device_count()
                                            ):
                                                try:
                                                    torch.cuda.set_device(gpu_id)
                                                    torch.cuda.empty_cache()
                                                    torch.cuda.synchronize()
                                                except Exception:
                                                    pass
                                            import gc

                                            gc.collect()

                                    # メモリエラーが発生したタスクを記録（後で再計算）
                                    results[task_key] = {
                                        "task": task,
                                        "result": None,
                                        "success": False,
                                        "error": error_msg[
                                            :200
                                        ],  # エラーメッセージを短縮
                                        "error_type": error_type,
                                        "retry": True,  # 再計算フラグ
                                    }
                                    completed_tasks += 1
                                    # 処理を続行（エラーを発生させない）
                                    continue

                                # メモリエラー以外のエラーも詳細に記録（エラーログの詳細化）
                                logging.error(
                                    f"IGタスク失敗: {task_key} "
                                    f"(L{task.layer_idx} T{task.token_idx} H{task.head_idx if hasattr(task, 'head_idx') else '-'})\n"
                                    f"エラー種類: {error_type}\n"
                                    f"エラーメッセージ: {error_msg}\n"
                                    f"トレースバック:\n{error_traceback}"
                                )
                                # エラーが発生したタスクも記録（後でエラーハンドリング）
                                results[task_key] = {
                                    "task": task,
                                    "result": None,
                                    "success": False,
                                    "error": error_msg,
                                    "error_type": error_type,
                                }
                                completed_tasks += 1
                            current_time = time.time()
                            should_update = completed_tasks > last_progress_update and (
                                completed_tasks % max(1, total_tasks // 20)
                                == 0  # 5%ごと
                                or current_time - last_progress_time
                                >= progress_update_interval
                            )
                            if progress_callback and should_update:
                                device_label = (
                                    "CPU"
                                    if device_id in (-1, None)
                                    else f"GPU{device_id}"
                                )
                                running_count = len(combined_futures)
                                progress_callback(
                                    completed_tasks,
                                    total_tasks,
                                    f"{device_label}: {completed_tasks:,}/{total_tasks:,} ({completed_tasks*100//total_tasks if total_tasks > 0 else 0}%)",
                                )
                                last_progress_update = completed_tasks
                                last_progress_time = current_time

                    # デバッグ: 処理完了時の統計を記録
                    logging.info("=" * 80)
                    logging.info(
                        f"compute_batch_ig: as_completedループ完了 "
                        f"(処理済みfutures: {processed_count}, "
                        f"スキップ済みfutures: {skipped_count}, "
                        f"完了タスク数: {completed_tasks}, "
                        f"results内のタスク数: {len(results)}, "
                        f"総タスク数: {total_tasks}, "
                        f"残りcombined_futures: {len(combined_futures)}, "
                        f"残りAttentionタスク: {len(remaining_attention_tasks)}, "
                        f"残りMLPタスク: {len(remaining_mlp_tasks)})"
                    )

                    # 残りのタスクがあれば、すべてスケジュールして処理を継続
                    if (
                        len(remaining_attention_tasks) > 0
                        or len(remaining_mlp_tasks) > 0
                    ):
                        logging.warning(
                            f"⚠️ 警告: 残りのタスクがあります "
                            f"(Attention: {len(remaining_attention_tasks)}, MLP: {len(remaining_mlp_tasks)}). "
                            f"残りのタスクをスケジュールします。"
                        )

                        # 残りのタスクをすべてスケジュール
                        while (
                            len(remaining_attention_tasks) > 0
                            or len(remaining_mlp_tasks) > 0
                        ):
                            if len(combined_futures) >= max_concurrent_futures:
                                # 同時実行数の上限に達した場合、完了を待つ
                                if len(combined_futures) > 0:
                                    # combined_futuresのコピーを作成してイテレーション
                                    futures_to_wait = dict(combined_futures)
                                    for future in as_completed(futures_to_wait.keys()):
                                        # 完了したタスクの処理（既存のロジックを再利用）
                                        task_info = None
                                        if future in combined_futures:
                                            task_info = combined_futures[future]
                                            del combined_futures[future]

                                        if task_info is None:
                                            continue

                                        if len(task_info) == 3 and task_info[2] is True:
                                            # バッチ処理
                                            if future in processed_futures:
                                                continue
                                            processed_futures.add(future)
                                            processed_count += 1
                                            batch_tasks, device_id, _ = task_info
                                            try:
                                                batch_results = future.result()
                                                if isinstance(batch_results, dict):
                                                    for task in batch_tasks:
                                                        task_key = generate_task_key(
                                                            task, text
                                                        )
                                                        if task_key in batch_results:
                                                            result = batch_results[
                                                                task_key
                                                            ]
                                                            results[task_key] = {
                                                                "task": task,
                                                                "result": result,
                                                                "success": result
                                                                is not None,
                                                            }
                                                            completed_tasks += 1
                                                        else:
                                                            results[task_key] = {
                                                                "task": task,
                                                                "result": None,
                                                                "success": False,
                                                                "error": "バッチ結果に存在しません",
                                                            }
                                                            completed_tasks += 1
                                            except Exception as exc:
                                                error_msg = str(exc)
                                                for task in batch_tasks:
                                                    task_key = generate_task_key(
                                                        task, text
                                                    )
                                                    if task_key not in results:
                                                        results[task_key] = {
                                                            "task": task,
                                                            "result": None,
                                                            "success": False,
                                                            "error": f"バッチ処理エラー: {error_msg}",
                                                        }
                                                        completed_tasks += 1
                                        else:
                                            # 通常処理
                                            processed_count += 1
                                            if len(task_info) == 3:
                                                task, device_id, _ = task_info
                                            elif len(task_info) == 2:
                                                task, device_id = task_info
                                            else:
                                                continue

                                            task_key = generate_task_key(task, text)
                                            try:
                                                result = future.result()
                                                consecutive_memory_errors = 0
                                                if result is None:
                                                    results[task_key] = {
                                                        "task": task,
                                                        "result": None,
                                                        "success": False,
                                                    }
                                                    completed_tasks += 1
                                                else:
                                                    results[task_key] = {
                                                        "task": task,
                                                        "result": result,
                                                        "success": True,
                                                    }
                                                    completed_tasks += 1
                                            except Exception as exc:
                                                error_msg = str(exc)
                                                results[task_key] = {
                                                    "task": task,
                                                    "result": None,
                                                    "success": False,
                                                    "error": error_msg,
                                                }
                                                completed_tasks += 1
                                else:
                                    break

                            # 新しいタスクをスケジュール
                            if remaining_attention_tasks:
                                task = remaining_attention_tasks.pop(0)
                                device_id = next(device_cycle)
                                model = self.device_pool.get_model(device_id)
                                inputs_for_device = device_inputs_map[device_id]
                                stream = self.device_pool.get_stream(device_id)

                                new_future = executor.submit(
                                    self.attention_executor.execute,
                                    task,
                                    text,
                                    inputs_for_device,
                                    fallback_model=self.unified_model,
                                    model=model,
                                    device_id=device_id,
                                    stream=stream,
                                    cached_hidden_states=cached_hidden_states,
                                )
                                combined_futures[new_future] = (task, device_id, False)
                            elif remaining_mlp_tasks:
                                task = remaining_mlp_tasks.pop(0)
                                device_id = next(device_cycle)
                                model = self.device_pool.get_model(device_id)
                                inputs_for_device = device_inputs_map[device_id]
                                stream = self.device_pool.get_stream(device_id)

                                new_future = executor.submit(
                                    self.mlp_executor.execute,
                                    task,
                                    text,
                                    inputs_for_device,
                                    fallback_model=self.unified_model,
                                    model=model,
                                    device_id=device_id,
                                    stream=stream,
                                )
                                combined_futures[new_future] = (task, device_id, False)
                            else:
                                break

                        # 残りのfuturesをすべて処理
                        if len(combined_futures) > 0:
                            logging.info(
                                f"残りの{len(combined_futures)}個のfutureを処理します"
                            )
                            # combined_futuresのコピーを作成してイテレーション
                            futures_to_wait = dict(combined_futures)
                            for future in as_completed(futures_to_wait.keys()):
                                task_info = None
                                if future in combined_futures:
                                    task_info = combined_futures[future]
                                    del combined_futures[future]

                                if task_info is None:
                                    continue

                                if len(task_info) == 3 and task_info[2] is True:
                                    # バッチ処理
                                    if future in processed_futures:
                                        continue
                                    processed_futures.add(future)
                                    processed_count += 1
                                    batch_tasks, device_id, _ = task_info
                                    try:
                                        batch_results = future.result()
                                        if isinstance(batch_results, dict):
                                            for task in batch_tasks:
                                                task_key = generate_task_key(task, text)
                                                if task_key in batch_results:
                                                    result = batch_results[task_key]
                                                    results[task_key] = {
                                                        "task": task,
                                                        "result": result,
                                                        "success": result is not None,
                                                    }
                                                    completed_tasks += 1
                                                else:
                                                    results[task_key] = {
                                                        "task": task,
                                                        "result": None,
                                                        "success": False,
                                                        "error": "バッチ結果に存在しません",
                                                    }
                                                    completed_tasks += 1
                                    except Exception as exc:
                                        error_msg = str(exc)
                                        for task in batch_tasks:
                                            task_key = generate_task_key(task, text)
                                            if task_key not in results:
                                                results[task_key] = {
                                                    "task": task,
                                                    "result": None,
                                                    "success": False,
                                                    "error": f"バッチ処理エラー: {error_msg}",
                                                }
                                                completed_tasks += 1
                                else:
                                    # 通常処理
                                    processed_count += 1
                                    if len(task_info) == 3:
                                        task, device_id, _ = task_info
                                    elif len(task_info) == 2:
                                        task, device_id = task_info
                                    else:
                                        continue

                                    task_key = generate_task_key(task, text)
                                    try:
                                        result = future.result()
                                        consecutive_memory_errors = 0
                                        if result is None:
                                            results[task_key] = {
                                                "task": task,
                                                "result": None,
                                                "success": False,
                                            }
                                            completed_tasks += 1
                                        else:
                                            results[task_key] = {
                                                "task": task,
                                                "result": result,
                                                "success": True,
                                            }
                                            completed_tasks += 1
                                    except Exception as exc:
                                        error_msg = str(exc)
                                        results[task_key] = {
                                            "task": task,
                                            "result": None,
                                            "success": False,
                                            "error": error_msg,
                                        }
                                        completed_tasks += 1

                    if len(combined_futures) > 0:
                        logging.error(
                            f"⚠️ 警告: {len(combined_futures)}個のfutureが未完了のまま残っています"
                        )
                    if completed_tasks != total_tasks:
                        logging.error(
                            f"⚠️ 警告: 完了タスク数({completed_tasks}) != 総タスク数({total_tasks})"
                        )
                    logging.info("=" * 80)
        finally:
            # 標準出力のリダイレクトは統一ログ設定が管理するため、ここでは何もしない
            pass

        # 失敗したタスクがないか確認（メモリエラーの場合は再計算を試みる）
        failed_tasks = [
            task_key
            for task_key, result_data in results.items()
            if not result_data.get("success", False)
        ]

        # メモリエラーで失敗したタスクを再計算（最大3回まで）
        memory_error_tasks = []
        other_error_tasks = []

        for task_key in failed_tasks:
            result_data = results[task_key]
            error_msg = result_data.get("error", "")
            if "out of memory" in str(
                error_msg
            ).lower() or "CUBLAS_STATUS_ALLOC_FAILED" in str(error_msg):
                memory_error_tasks.append((task_key, result_data))
            else:
                other_error_tasks.append((task_key, result_data))

        # メモリエラーで失敗したタスクを再計算
        if memory_error_tasks:
            logger.info(
                f"🔄 メモリエラーで失敗したタスク {len(memory_error_tasks)}個を再計算します..."
            )

            # メモリクリーンアップ
            if torch.cuda.is_available():
                for gpu_id in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.set_device(gpu_id)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                import gc

                gc.collect()

            # 失敗したタスクを再計算
            retry_results = {}
            max_retries = 3

            for retry_count in range(max_retries):
                retry_tasks_to_compute = []
                for task_key, result_data in memory_error_tasks:
                    task = result_data.get("task")
                    if task is None:
                        continue

                    # メモリが利用可能かチェック
                    device_id = (
                        preferred_device_id
                        if preferred_device_id is not None
                        else device_ids[0]
                    )
                    from utils.calculations.ig.optimized_runtime.executors import (
                        _wait_for_memory_available,
                    )

                    if _wait_for_memory_available(device_id, max_wait_seconds=30):
                        retry_tasks_to_compute.append((task_key, task))
                    else:
                        logger.warning(
                            f"GPU {device_id}のメモリが利用可能になりませんでした。"
                            f"タスク {task_key} の再計算をスキップします。"
                        )

                if not retry_tasks_to_compute:
                    break

                logger.info(
                    f"🔄 再計算試行 {retry_count + 1}/{max_retries}: "
                    f"{len(retry_tasks_to_compute)}個のタスクを再計算します..."
                )

                # 再計算を実行
                for task_key, task in retry_tasks_to_compute:
                    try:
                        # タスクを再実行
                        if task.task_type == "attention":
                            executor = self.attention_executor
                        else:
                            executor = self.mlp_executor

                        device_id = (
                            preferred_device_id
                            if preferred_device_id is not None
                            else device_ids[0]
                        )
                        model = self.device_pool.get_model(device_id)
                        inputs_for_device = device_inputs_map[device_id]
                        stream = self.device_pool.get_stream(device_id)

                        result = executor.execute(
                            task,
                            text,
                            inputs_for_device,
                            fallback_model=model,
                            model=model,
                            device_id=device_id,
                            stream=stream,
                            cached_hidden_states=(
                                cached_hidden_states
                                if task.task_type == "attention"
                                else None
                            ),
                        )

                        if result is not None:
                            retry_results[task_key] = {
                                "task": task,
                                "result": result,
                                "success": True,
                            }
                            logger.info(f"✅ 再計算成功: {task_key}")
                            # 成功したタスクをmemory_error_tasksから削除
                            memory_error_tasks = [
                                (tk, rd)
                                for tk, rd in memory_error_tasks
                                if tk != task_key
                            ]
                        else:
                            logger.warning(f"⚠️ 再計算結果がNone: {task_key}")
                    except Exception as exc:
                        error_msg = str(exc)
                        logger.warning(
                            f"⚠️ 再計算失敗: {task_key}, エラー: {error_msg[:200]}"
                        )
                        # 再計算に失敗したタスクはそのまま残す

                # 再計算が成功したタスクがあれば、結果を更新
                if retry_results:
                    results.update(retry_results)
                    logger.info(
                        f"✅ 再計算完了: {len(retry_results)}個のタスクが成功しました。"
                    )

            # 再計算後も失敗したタスクがある場合は警告
            if memory_error_tasks:
                logger.warning(
                    f"⚠️ {len(memory_error_tasks)}個のタスクが{max_retries}回の再計算後も失敗しました。"
                )

        # メモリエラー以外のエラーで失敗したタスクがある場合はエラーを発生
        if other_error_tasks:
            error_details = []
            for task_key, result_data in other_error_tasks[
                :10
            ]:  # 最初の10個のみ詳細を記録
                error_msg = result_data.get("error", "不明")
                error_type = result_data.get("error_type", "Unknown")
                # エラー情報をより詳細に記録
                if isinstance(error_msg, str) and len(error_msg) > 200:
                    error_msg_short = error_msg[:200] + "..."
                else:
                    error_msg_short = str(error_msg)
                error_details.append(
                    f"  {task_key}: error_type={error_type}, error={error_msg_short}"
                )
            if len(other_error_tasks) > 10:
                error_details.append(
                    f"  ... 他{len(other_error_tasks) - 10}個のタスクも失敗しています"
                )

            # エラーの種類を集計
            error_types = {}
            for task_key, result_data in other_error_tasks:
                error_type = result_data.get("error_type", "Unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1

            error_summary = ", ".join([f"{k}: {v}個" for k, v in error_types.items()])

            error_msg = (
                f"❌ {len(other_error_tasks)}個のIGタスクが失敗しました（メモリエラー以外）。計算を停止します。\n"
                f"エラー種類: {error_summary}\n"
                f"失敗したタスク:\n" + "\n".join(error_details)
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        # 再計算後も失敗したタスクがある場合は警告のみ（エラーにはしない）
        final_failed_tasks = [
            task_key
            for task_key, result_data in results.items()
            if not result_data.get("success", False)
        ]
        if final_failed_tasks:
            # 失敗タスクの統計情報を記録
            failed_stats = {
                "total": len(final_failed_tasks),
                "by_layer": {},
                "by_error_type": {},
            }

            for task_key in final_failed_tasks:
                result_data = results[task_key]
                task = result_data.get("task")
                error_type = result_data.get("error_type", "Unknown")

                if task:
                    layer_idx = task.layer_idx
                    failed_stats["by_layer"][layer_idx] = (
                        failed_stats["by_layer"].get(layer_idx, 0) + 1
                    )

                failed_stats["by_error_type"][error_type] = (
                    failed_stats["by_error_type"].get(error_type, 0) + 1
                )

            logger.warning(
                f"⚠️ {len(final_failed_tasks)}個のタスクが再計算後も失敗しました。"
                f"これらはメモリエラーが原因の可能性があります。"
            )
            logger.warning(
                f"📊 失敗タスクの統計: "
                f"レイヤー別={failed_stats['by_layer']}, "
                f"エラー種類別={failed_stats['by_error_type']}"
            )

        # すべてのタスクがresultsに含まれているか確認
        all_task_keys = {generate_task_key(task, text) for task in tasks}
        missing_in_results = all_task_keys - set(results.keys())
        if missing_in_results:
            logging.error("=" * 80)
            logging.error(
                f"❌ 理論通りに計算できませんでした: {len(missing_in_results)}個のタスクがresultsに含まれていません "
                f"(総タスク数: {len(tasks)}, results内のタスク数: {len(results)})"
            )
            # 見つからないタスクの詳細をログに記録
            missing_tasks_detail = []
            for task in tasks:
                task_key = generate_task_key(task, text)
                if task_key not in results:
                    task_detail = (
                        f"{task.task_type} L{task.layer_idx} "
                        f"T{task.token_idx} H{task.head_idx if hasattr(task, 'head_idx') else '-'}"
                    )
                    missing_tasks_detail.append((task_key, task_detail))
                    logging.error(f"  見つからないタスク: {task_key} ({task_detail})")

            # 最初の20個の見つからないタスクの詳細をエラーメッセージに含める
            error_details = []
            for task_key, task_detail in missing_tasks_detail[:20]:
                error_details.append(f"  {task_detail}: {task_key}")
            if len(missing_tasks_detail) > 20:
                error_details.append(
                    f"  ... 他{len(missing_tasks_detail) - 20}個のタスクも見つかりません"
                )

            error_msg = (
                f"❌ 理論通りに計算できませんでした: {len(missing_in_results)}個のIGタスクが結果に含まれていません。計算を停止します。\n"
                f"見つからないタスク:\n" + "\n".join(error_details)
            )
            logging.error(error_msg)
            logging.error("=" * 80)

            # 理論通りに計算できなかった場合はエラーを発生させて停止
            raise RuntimeError(error_msg)
        else:
            logging.info(
                f"✅ compute_batch_ig: すべてのタスクがresultsに含まれています "
                f"(総タスク数: {len(tasks)}, results内のタスク数: {len(results)})"
            )

        return results

    def _get_optimal_device_id(self, device_ids: List[int]) -> int:
        """
        GPUメモリ使用状況を考慮して最も空いているGPUを選択

        GPUアイドル時間を削減するため、メモリに余裕があるGPUを優先的に選択
        実行中のタスク数を追跡して、より正確にGPU使用状況を判断

        Returns:
            最適なGPU ID
        """
        if not torch.cuda.is_available() or len(device_ids) == 1:
            return device_ids[0] if device_ids else 0

        # 各GPUのメモリ使用状況を確認（高速）
        best_device = device_ids[0]
        best_score = -1.0

        # 実行中のタスク数を追跡（クラス変数として管理）
        if not hasattr(self, "_device_task_count"):
            self._device_task_count = {device_id: 0 for device_id in device_ids}

        for device_id in device_ids:
            try:
                # メモリ使用状況を取得（高速）
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
                free_gb = free_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
                memory_utilization = 1.0 - (free_gb / total_gb) if total_gb > 0 else 0.0

                # 実行中のタスク数を取得（簡易的なGPU使用率の指標）
                task_count = self._device_task_count.get(device_id, 0)
                # タスク数が多いほどGPU使用率が高いと仮定（簡易的な指標）
                # H100などの高性能GPUでは、より多くのタスクを同時実行可能
                # ワーカー数に応じて動的に調整
                # 2GPUの場合、各GPUでより多くのタスクを同時実行可能
                # GPU負荷分散を改善するため、タスクスコアの重みは調整済み
                # 同時実行可能タスク数はiter1の設定を維持（速度重視）
                optimal_workers = getattr(self, "_current_optimal_workers", 256)
                max_concurrent_tasks_per_gpu = max(
                    50, optimal_workers // max(1, len(device_ids)) if device_ids else 50
                )
                task_utilization = min(
                    task_count / float(max_concurrent_tasks_per_gpu), 1.0
                )

                # スコア計算: メモリに余裕があり、実行中タスクが少ないほど高スコア
                # GPU利用率を最大化するため、タスク数が少ないGPUを優先的に選択
                # GPU1の利用率が低いため、タスク数の重みをさらに上げる
                # メモリ使用率が低い（空きが多い）: 0.3の重み（0.4→0.3に変更）
                # 実行中タスクが少ない（アイドル）: 0.7の重み（0.6→0.7に変更、タスク数をさらに重視）
                memory_score = (1.0 - memory_utilization) * 0.3
                task_score = (1.0 - task_utilization) * 0.7
                total_score = memory_score + task_score

                if total_score > best_score:
                    best_score = total_score
                    best_device = device_id
            except Exception:
                # エラーが発生した場合は最初のGPUを使用
                continue

        # 選択されたGPUのタスク数を増やす
        self._device_task_count[best_device] = (
            self._device_task_count.get(best_device, 0) + 1
        )

        return best_device

    def _calculate_max_preload_tasks(
        self, device_ids: List[int], optimal_workers: int
    ) -> int:
        """
        GPUメモリ使用状況を確認して、プリロード可能なタスク数を計算

        メモリに余裕がある限り、タスクを事前にメモリにロードして待機させることで、
        GPU利用率が下がった時にすぐに次のタスクを投入できるようにする

        Returns:
            プリロード可能な最大タスク数
        """
        if not torch.cuda.is_available():
            return optimal_workers * 2

        # 各GPUのメモリ使用状況を確認
        total_free_gb = 0.0
        total_gb = 0.0

        for device_id in device_ids:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
                free_gb = free_bytes / (1024**3)
                total_gb_device = total_bytes / (1024**3)
                total_free_gb += free_gb
                total_gb += total_gb_device
            except Exception:
                continue

        # メモリに余裕がある場合、より多くのタスクをプリロード可能
        # H100（80GB）の場合、メモリに余裕がある限り、タスクを事前にロード
        if total_gb > 0:
            free_ratio = total_free_gb / total_gb

            # メモリに余裕がある場合、より多くのタスクをプリロード
            # H100（80GB）の場合、メモリに余裕がある限り、タスクを事前にロード
            # メモリ使用率が10%未満（90%以上空き）の場合、さらに積極的にプリロード
            if free_ratio >= 0.90:  # 90%以上空きがある場合（メモリ使用率10%未満）
                # メモリに大量の余裕がある限り、タスクを事前にロード（ワーカー数の20倍まで）
                max_preload = optimal_workers * 20
            elif free_ratio >= 0.85:  # 85%以上空きがある場合
                # メモリに余裕がある限り、タスクを事前にロード（ワーカー数の10倍まで）
                max_preload = optimal_workers * 10
            elif free_ratio >= 0.75:  # 75%以上空きがある場合
                max_preload = optimal_workers * 8
            elif free_ratio >= 0.6:  # 60%以上空きがある場合
                max_preload = optimal_workers * 6
            elif free_ratio >= 0.4:  # 40%以上空きがある場合
                max_preload = optimal_workers * 4
            else:
                max_preload = optimal_workers * 2

            # ログ出力を簡潔に（頻繁に呼ばれるため）
            # 詳細は定期的なパフォーマンスログで表示される
            if free_ratio >= 0.85 or free_ratio < 0.4:  # 極端な場合のみログ出力
                logging.info(
                    f"📊 GPUメモリ状況: {total_free_gb:.1f}GB / {total_gb:.1f}GB 空き "
                    f"({free_ratio*100:.1f}%), プリロード可能タスク数: {max_preload}"
                )
            return max_preload

        return optimal_workers * 2

    def _schedule_tasks(
        self,
        executor: ThreadPoolExecutor,
        tasks: List[IGTask],
        text: str,
        device_inputs_map: Dict[int, Dict[str, torch.Tensor]],
        device_ids: List[int],
        batch_size: int,
        optimal_workers: int,
        *,
        is_attention: bool,
        cached_hidden_states: Optional[Tuple] = None,  # 事前計算済みhidden states
        preferred_device_id: Optional[int] = None,  # 優先的に使用するGPU ID
    ):
        """
        タスクをスケジュールしてfuturesを返す。
        H100の場合は段階的スケジュールを行うため、残りのタスクも返す。

        Returns:
            (futures, remaining_tasks): futures辞書と残りのタスクリスト
        """
        futures = {}
        # 動的負荷分散とラウンドロビンのハイブリッド方式
        # タスクごとにメモリ使用状況を確認して最適なGPUを選択
        device_cycle = cycle(device_ids)
        executor_name = "Attention" if is_attention else "MLP"

        logging.info(
            "%s タスク %d個を並列実行開始 (バッチサイズ: %d, ワーカー数: %d)",
            executor_name,
            len(tasks),
            batch_size,
            optimal_workers,
        )

        # MLPの場合は、同じレイヤーのタスクをまとめてバッチ処理
        if not is_attention and hasattr(self.mlp_executor, "execute_batch"):
            # レイヤーごとにタスクをグループ化
            layer_groups = {}
            for task in tasks:
                layer_idx = task.layer_idx
                if layer_idx not in layer_groups:
                    layer_groups[layer_idx] = []
                layer_groups[layer_idx].append(task)

            # 各レイヤーグループを並列実行（異なるレイヤーを跨いで並列化）
            for layer_idx, layer_tasks in layer_groups.items():
                # H100の場合は、レイヤー内の全トークンを一度に処理
                effective_batch_size = (
                    len(layer_tasks)
                    if self.is_h100
                    else min(batch_size * 2, len(layer_tasks))
                )

                for start in range(0, len(layer_tasks), effective_batch_size):
                    batch_tasks = layer_tasks[start : start + effective_batch_size]
                    # preferred_device_idが指定されている場合は優先的に使用
                    # それ以外は動的負荷分散
                    device_id = (
                        preferred_device_id
                        if preferred_device_id is not None
                        and preferred_device_id in device_ids
                        else self._get_optimal_device_id(device_ids)
                    )
                    model = self.device_pool.get_model(device_id)
                    inputs_for_device = device_inputs_map[device_id]
                    stream = self.device_pool.get_stream(device_id)

                    # バッチ処理実行（異なるレイヤーを跨いで並列実行）
                    future = executor.submit(
                        self.mlp_executor.execute_batch,
                        batch_tasks,
                        text,
                        inputs_for_device,
                        fallback_model=self.unified_model,
                        model=model,
                        device_id=device_id,
                        stream=stream,
                    )
                    futures[future] = (batch_tasks, device_id, True)

            # MLPの場合は残りのタスクなし（全タスクをスケジュール）
            return futures, []
        else:
            # Attentionタスク: 異なるToken/Head/Layerを跨いで並列実行
            # タスクを段階的にスケジュール（全タスクを一度にスケジュールしない）
            # 一度にスケジュールするタスク数を制限（GPU利用率を最大化）
            max_concurrent_futures = optimal_workers * 2  # ワーカー数の2倍程度に制限

            if self.is_h100:
                # H100の場合でも、タスクを段階的にスケジュール
                # 一度にスケジュールするタスク数を制限してGPU利用率を最大化
                scheduled_count = 0
                for task in tasks:
                    # 一度にスケジュールするタスク数を制限
                    if scheduled_count >= max_concurrent_futures:
                        break

                    # preferred_device_idが指定されている場合は優先的に使用
                    # それ以外は動的負荷分散
                    device_id = (
                        preferred_device_id
                        if preferred_device_id is not None
                        and preferred_device_id in device_ids
                        else self._get_optimal_device_id(device_ids)
                    )
                    model = self.device_pool.get_model(device_id)
                    inputs_for_device = device_inputs_map[device_id]
                    stream = self.device_pool.get_stream(device_id)

                    executor_fn = (
                        self.attention_executor.execute
                        if is_attention
                        else self.mlp_executor.execute
                    )

                    # AttentionExecutorの場合はcached_hidden_statesを渡す
                    if is_attention and cached_hidden_states is not None:
                        future = executor.submit(
                            executor_fn,
                            task,
                            text,
                            inputs_for_device,
                            fallback_model=self.unified_model,
                            model=model,
                            device_id=device_id,
                            stream=stream,
                            cached_hidden_states=cached_hidden_states,
                        )
                    else:
                        future = executor.submit(
                            executor_fn,
                            task,
                            text,
                            inputs_for_device,
                            fallback_model=self.unified_model,
                            model=model,
                            device_id=device_id,
                            stream=stream,
                        )
                    futures[future] = (task, device_id, False)
                    scheduled_count += 1

                # 残りのタスクを返す（呼び出し側で段階的にスケジュール）
                remaining_tasks = tasks[scheduled_count:]
                return futures, remaining_tasks
            else:
                # 従来の方法（バッチサイズを大きく）
                effective_batch_size = batch_size * 2

                for start in range(0, len(tasks), effective_batch_size):
                    batch_tasks = tasks[start : start + effective_batch_size]

                    # 各タスクを並列実行（異なるToken/Head/Layerを跨いで）
                    for task in batch_tasks:
                        # preferred_device_idが指定されている場合は優先的に使用
                        # それ以外は動的負荷分散
                        device_id = (
                            preferred_device_id
                            if preferred_device_id is not None
                            and preferred_device_id in device_ids
                            else self._get_optimal_device_id(device_ids)
                        )
                        model = self.device_pool.get_model(device_id)
                        inputs_for_device = device_inputs_map[device_id]
                        stream = self.device_pool.get_stream(device_id)

                        executor_fn = (
                            self.attention_executor.execute
                            if is_attention
                            else self.mlp_executor.execute
                        )

                        # AttentionExecutorの場合はcached_hidden_statesを渡す
                        if is_attention and cached_hidden_states is not None:
                            future = executor.submit(
                                executor_fn,
                                task,
                                text,
                                inputs_for_device,
                                fallback_model=self.unified_model,
                                model=model,
                                device_id=device_id,
                                stream=stream,
                                cached_hidden_states=cached_hidden_states,
                            )
                        else:
                            future = executor.submit(
                                executor_fn,
                                task,
                                text,
                                inputs_for_device,
                                fallback_model=self.unified_model,
                                model=model,
                                device_id=device_id,
                                stream=stream,
                            )
                        futures[future] = (task, device_id, False)

                # 非H100の場合は残りのタスクなし
                return futures, []

    def get_computation_stats(self) -> Dict[str, Any]:
        cache_stats = self.cache.get_stats()
        return {
            "cache_stats": cache_stats,
            "active_computations": len(self.registry),
            "h100_optimized": self.is_h100,
            "max_batch_size": self.max_batch_size,
            "max_workers": self.max_workers,
        }


_ig_calculator: Optional[OptimizedIGCalculator] = None


def get_ig_calculator(
    unified_model,
    tokenizer,
    use_lightning_trainer: bool = False,
    total_samples: int = 0,
) -> OptimizedIGCalculator:
    global _ig_calculator
    if _ig_calculator is None:
        _ig_calculator = OptimizedIGCalculator(
            unified_model,
            tokenizer,
            use_lightning_trainer=use_lightning_trainer,
            total_samples=total_samples,
        )
    else:
        # 既存のインスタンスがある場合、total_samplesを更新
        _ig_calculator._total_samples = total_samples
    return _ig_calculator


__all__ = ["IGTask", "OptimizedIGCalculator", "get_ig_calculator"]
