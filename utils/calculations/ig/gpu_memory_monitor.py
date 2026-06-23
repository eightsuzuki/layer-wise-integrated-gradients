"""
GPUメモリ監視と動的調整モジュール

処理中にGPUメモリをリアルタイムで監視し、バッチサイズやワーカー数を動的に調整する
"""

import logging
import time
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class GPUMemoryMonitor:
    """
    GPUメモリ使用状況を監視し、バッチサイズやワーカー数を動的に調整
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.memory_history: list = []
        self.last_check_time = time.time()
        self.check_interval = 1.0  # 1秒ごとにチェック
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device_id)
            self.total_memory_gb = props.total_memory / (1024**3)
        else:
            self.total_memory_gb = 0.0

    def get_memory_status(self) -> Dict[str, float]:
        """
        現在のGPUメモリ使用状況を取得
        
        Returns:
            メモリ情報の辞書（GB単位）
        """
        if not torch.cuda.is_available():
            return {
                "total_gb": 0.0,
                "free_gb": 0.0,
                "used_gb": 0.0,
                "reserved_gb": 0.0,
                "allocated_gb": 0.0,
                "utilization_percent": 0.0,
            }

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_id)
            reserved_bytes = torch.cuda.memory_reserved(self.device_id)
            allocated_bytes = torch.cuda.memory_allocated(self.device_id)

            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            used_gb = total_gb - free_gb
            reserved_gb = reserved_bytes / (1024**3)
            allocated_gb = allocated_bytes / (1024**3)
            utilization_percent = (used_gb / total_gb) * 100 if total_gb > 0 else 0.0

            return {
                "total_gb": total_gb,
                "free_gb": free_gb,
                "used_gb": used_gb,
                "reserved_gb": reserved_gb,
                "allocated_gb": allocated_gb,
                "utilization_percent": utilization_percent,
            }
        except Exception as e:
            logger.warning(f"GPUメモリ状態取得エラー: {e}")
            return {
                "total_gb": self.total_memory_gb,
                "free_gb": 0.0,
                "used_gb": 0.0,
                "reserved_gb": 0.0,
                "allocated_gb": 0.0,
                "utilization_percent": 0.0,
            }

    def check_memory_usage(self) -> Tuple[bool, Dict[str, float]]:
        """
        メモリ使用状況をチェック（一定間隔で実行）
        
        Returns:
            (チェックしたかどうか, メモリ情報)
        """
        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            return False, {}

        self.last_check_time = current_time
        memory_status = self.get_memory_status()
        self.memory_history.append(memory_status)

        # 履歴を保持（最新100件）
        if len(self.memory_history) > 100:
            self.memory_history.pop(0)

        return True, memory_status

    def calculate_optimal_batch_size(
        self,
        current_batch_size: int,
        min_batch_size: int = 4,
        max_batch_size: int = 512,
        target_memory_utilization: float = 0.90,  # デフォルトを90%に上げる
    ) -> int:
        """
        現在のメモリ使用状況に基づいて最適なバッチサイズを計算
        
        Args:
            current_batch_size: 現在のバッチサイズ
            min_batch_size: 最小バッチサイズ
            max_batch_size: 最大バッチサイズ
            target_memory_utilization: 目標メモリ使用率（0.85 = 85%）
        
        Returns:
            最適なバッチサイズ
        """
        memory_status = self.get_memory_status()
        if memory_status["total_gb"] == 0:
            return current_batch_size

        utilization = memory_status["utilization_percent"] / 100.0
        free_gb = memory_status["free_gb"]

        # メモリ使用率が高い場合はバッチサイズを減らす
        if utilization > target_memory_utilization + 0.1:  # 95%以上
            new_batch_size = int(current_batch_size * 0.7)  # 30%減らす
            logger.info(
                f"🔴 メモリ使用率が高いためバッチサイズを減らします: "
                f"{current_batch_size} → {new_batch_size} "
                f"(使用率: {utilization*100:.1f}%, 空き: {free_gb:.1f}GB)"
            )
            return max(new_batch_size, min_batch_size)

        # メモリに余裕がある場合はバッチサイズを増やす
        if utilization < target_memory_utilization - 0.2 and free_gb > 5.0:  # 70%以下かつ5GB以上空き
            new_batch_size = int(current_batch_size * 1.5)  # 50%増やす（より積極的に）
            logger.info(
                f"🟢 メモリに余裕があるためバッチサイズを増やします: "
                f"{current_batch_size} → {new_batch_size} "
                f"(使用率: {utilization*100:.1f}%, 空き: {free_gb:.1f}GB)"
            )
            return min(new_batch_size, max_batch_size)

        return current_batch_size

    def calculate_optimal_workers(
        self,
        current_workers: int,
        min_workers: int = 4,
        max_workers: int = 32,
        target_memory_utilization: float = 0.90,  # デフォルトを90%に上げる
    ) -> int:
        """
        現在のメモリ使用状況に基づいて最適なワーカー数を計算
        
        Args:
            current_workers: 現在のワーカー数
            min_workers: 最小ワーカー数
            max_workers: 最大ワーカー数
            target_memory_utilization: 目標メモリ使用率
        
        Returns:
            最適なワーカー数
        """
        memory_status = self.get_memory_status()
        if memory_status["total_gb"] == 0:
            return current_workers

        utilization = memory_status["utilization_percent"] / 100.0
        free_gb = memory_status["free_gb"]

        # メモリ使用率が高い場合はワーカー数を減らす
        if utilization > target_memory_utilization + 0.1:  # 95%以上
            new_workers = int(current_workers * 0.8)  # 20%減らす
            logger.info(
                f"🔴 メモリ使用率が高いためワーカー数を減らします: "
                f"{current_workers} → {new_workers} "
                f"(使用率: {utilization*100:.1f}%, 空き: {free_gb:.1f}GB)"
            )
            return max(new_workers, min_workers)

        # メモリに余裕がある場合はワーカー数を増やす
        if utilization < target_memory_utilization - 0.2 and free_gb > 10.0:  # 70%以下かつ10GB以上空き
            new_workers = int(current_workers * 1.3)  # 30%増やす（より積極的に）
            logger.info(
                f"🟢 メモリに余裕があるためワーカー数を増やします: "
                f"{current_workers} → {new_workers} "
                f"(使用率: {utilization*100:.1f}%, 空き: {free_gb:.1f}GB)"
            )
            return min(new_workers, max_workers)

        return current_workers

    def is_memory_critical(self, threshold: float = 0.95) -> bool:
        """
        メモリ使用率がクリティカルな状態かどうかをチェック
        
        Args:
            threshold: クリティカルとみなす使用率（デフォルト: 95%）
        
        Returns:
            クリティカルな場合はTrue
        """
        memory_status = self.get_memory_status()
        if memory_status["total_gb"] == 0:
            return False

        utilization = memory_status["utilization_percent"] / 100.0
        return utilization > threshold

    def get_memory_summary(self) -> str:
        """
        メモリ使用状況のサマリーを取得
        
        Returns:
            メモリ情報の文字列
        """
        memory_status = self.get_memory_status()
        return (
            f"GPU{self.device_id}: "
            f"{memory_status['used_gb']:.1f}GB / {memory_status['total_gb']:.1f}GB "
            f"({memory_status['utilization_percent']:.1f}%), "
            f"空き: {memory_status['free_gb']:.1f}GB"
        )


# グローバルインスタンス
_monitor_instance: Optional[GPUMemoryMonitor] = None


def get_gpu_memory_monitor(device_id: int = 0) -> GPUMemoryMonitor:
    """GPUメモリ監視器のシングルトンインスタンスを取得"""
    global _monitor_instance
    if _monitor_instance is None or _monitor_instance.device_id != device_id:
        _monitor_instance = GPUMemoryMonitor(device_id)
    return _monitor_instance

