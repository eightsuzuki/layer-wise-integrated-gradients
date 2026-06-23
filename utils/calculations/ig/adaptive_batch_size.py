"""
適応的バッチサイズ計算モジュール

実際のメモリ使用量を測定してバッチサイズを動的に決定する
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class AdaptiveBatchSizeCalculator:
    """
    GPU並列処理能力とメモリ使用量の実測に基づいた適応的バッチサイズ計算

    従来の固定値ベースではなく、実際のメモリ使用量を測定して
    バッチサイズを動的に決定する。
    """

    def __init__(self):
        self.gpu_name = None
        self.total_memory = 0.0
        self.sm_count = 0
        self.max_threads_per_sm = 64
        self.measured_memory_per_task = None

        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            self.total_memory = props.total_memory / 1024**3  # GB
            self.sm_count = props.multi_processor_count
            if hasattr(props, "max_threads_per_multi_processor"):
                self.max_threads_per_sm = props.max_threads_per_multi_processor
            elif hasattr(props, "max_threads_per_sm"):
                self.max_threads_per_sm = props.max_threads_per_sm

            # 大量サンプル実行時（1700以上）はログを抑制
            # logger.info(
            #     f"GPU: {self.gpu_name}, SM数: {self.sm_count}, メモリ: {self.total_memory:.1f}GB"
            # )

    def measure_memory_per_task(self, sample_task_fn, num_samples: int = 3) -> float:
        """
        サンプルタスクを実行して、タスクあたりのメモリ使用量を実測

        Args:
            sample_task_fn: サンプルタスクを実行する関数
            num_samples: 測定回数

        Returns:
            タスクあたりのメモリ使用量（MB）
        """
        if not torch.cuda.is_available():
            return 50.0  # デフォルト値

        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # ベースラインメモリ使用量を測定
            baseline_memory = torch.cuda.memory_allocated() / 1024**2  # MB

            # サンプルタスクを実行
            for _ in range(num_samples):
                sample_task_fn()

            # ピークメモリ使用量を測定
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

            # タスクあたりのメモリ使用量を計算
            memory_per_task = (peak_memory - baseline_memory) / num_samples

            self.measured_memory_per_task = memory_per_task
            logger.info(f"実測メモリ使用量: {memory_per_task:.1f}MB/タスク")

            return memory_per_task

        except Exception as e:
            logger.warning(f"メモリ測定エラー: {e}")
            return 50.0  # デフォルト値

    def calculate_optimal_batch_size(
        self,
        total_tasks: int,
        use_mixed_precision: bool = False,
        memory_safety_margin: float = 0.9,
    ) -> int:
        """
        GPU並列処理能力とメモリ使用量の実測に基づいてバッチサイズを計算

        Args:
            total_tasks: 総タスク数
            use_mixed_precision: 混合精度を使用するかどうか
            memory_safety_margin: メモリ安全マージン（0.9 = 90%まで使用）

        Returns:
            最適なバッチサイズ
        """
        if not torch.cuda.is_available():
            return min(8, total_tasks)

        # 利用可能メモリを取得（他のプロセスの使用も考慮）
        # torch.cuda.mem_get_info()は、システム全体での実際の空きメモリを返す
        free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(0)
        free_memory_mb = free_memory_bytes / (1024**2)  # B -> MB
        total_memory_mb = total_memory_bytes / (1024**2)  # B -> MB

        # 現在のプロセスのメモリ使用状況
        reserved_memory_mb = torch.cuda.memory_reserved(0) / (1024**2)  # MB
        allocated_memory_mb = torch.cuda.memory_allocated(0) / (1024**2)  # MB

        # 実際に使用可能なメモリ = システム全体の空きメモリに安全マージンを適用
        # 他のプロセスが使用しているメモリは既にfree_memory_mbに反映されている
        available_memory = free_memory_mb * memory_safety_margin

        # 最小値を保証（負の値にならないように）
        available_memory = max(available_memory, 100.0)  # 最低100MBは確保

        # タスクあたりのメモリ使用量を決定（実測値 or 推定値）
        if self.measured_memory_per_task is not None:
            memory_per_task = self.measured_memory_per_task
        else:
            # 推定値（混合精度を考慮）
            memory_per_task = 50 if use_mixed_precision else 80

        # メモリ制約に基づく最大バッチサイズ
        max_memory_batch = int(available_memory / memory_per_task)

        # GPU並列処理能力に基づく最大バッチサイズ
        max_parallel_batch = self.sm_count * self.max_threads_per_sm

        # 実際のバッチサイズを計算（制約の最小値）
        optimal_batch_size = min(max_memory_batch, max_parallel_batch, total_tasks)

        # 最小値の保証
        optimal_batch_size = max(optimal_batch_size, 4)

        # 詳細ログはDEBUGレベルに変更（ターミナル出力は抑制）
        logger.debug(f"バッチサイズ計算:")
        logger.debug(f"  システム全体総メモリ: {total_memory_mb:.1f}MB")
        logger.debug(f"  システム全体空きメモリ: {free_memory_mb:.1f}MB")
        logger.debug(f"  現在プロセス予約メモリ: {reserved_memory_mb:.1f}MB")
        logger.debug(f"  現在プロセス割り当てメモリ: {allocated_memory_mb:.1f}MB")
        logger.debug(
            f"  利用可能メモリ（安全マージン適用後）: {available_memory:.1f}MB"
        )
        logger.debug(f"  メモリあたり最大: {max_memory_batch}")
        logger.debug(f"  並列処理能力: {max_parallel_batch}")
        logger.debug(f"  総タスク数: {total_tasks}")
        logger.debug(f"  最適バッチサイズ: {optimal_batch_size}")

        # ターミナル出力は抑制（ログファイルには記録される）
        # print(f"🚀 適応的バッチサイズ計算:")
        # print(f"  GPU: {self.gpu_name} (SM数: {self.sm_count})")
        # print(f"  システム全体空きメモリ: {free_memory_mb:.1f}MB")
        # print(f"  利用可能メモリ（安全マージン適用後）: {available_memory:.1f}MB")
        # print(f"  メモリ/タスク: {memory_per_task:.1f}MB")
        # print(f"  メモリ制約最大: {max_memory_batch}")
        # print(f"  並列処理能力: {max_parallel_batch}")
        # print(f"  総タスク数: {total_tasks}")
        # print(f"  → 最適バッチサイズ: {optimal_batch_size}")

        # メモリ不足の警告
        if available_memory < 500.0:  # 500MB未満の場合
            logger.warning(
                f"⚠️ GPUメモリが不足しています。空きメモリ: {free_memory_mb:.1f}MB。"
                f"他のプロセスが使用している可能性があります。"
            )
            # ターミナル出力は抑制（ログファイルには記録される）
            # print(
            #     f"⚠️ 警告: GPUメモリ不足（空き: {free_memory_mb:.1f}MB）。バッチサイズを小さくしています。"
            # )

        return optimal_batch_size


# グローバルインスタンス
_calculator_instance: Optional[AdaptiveBatchSizeCalculator] = None


def get_adaptive_batch_calculator() -> AdaptiveBatchSizeCalculator:
    """適応的バッチサイズ計算器のシングルトンインスタンスを取得"""
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = AdaptiveBatchSizeCalculator()
    return _calculator_instance


def calculate_optimal_batch_size(
    total_tasks: int,
    use_mixed_precision: bool = False,
    memory_safety_margin: float = 0.9,
) -> int:
    """
    適応的バッチサイズを計算（簡易インターフェース）

    Args:
        total_tasks: 総タスク数
        use_mixed_precision: 混合精度を使用するかどうか
        memory_safety_margin: メモリ安全マージン（0.9 = 90%まで使用）

    Returns:
        最適なバッチサイズ
    """
    calculator = get_adaptive_batch_calculator()
    return calculator.calculate_optimal_batch_size(
        total_tasks, use_mixed_precision, memory_safety_margin
    )
