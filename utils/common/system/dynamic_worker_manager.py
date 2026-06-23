"""
動的ワーカー数管理システム

H100デュアルGPU構成で最大48ワーカーまで活用し、
他のユーザーとの競合を避けるための動的リソース監視システム
"""

import torch
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

# psutilの代替実装
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

    # 基本的なシステム監視の代替実装
    class BasicSystemMonitor:
        @staticmethod
        def cpu_count():
            return os.cpu_count() or 4

        @staticmethod
        def cpu_percent(interval=1.0):
            # 簡易的なCPU使用率推定
            return 30.0  # デフォルト値

        @staticmethod
        def virtual_memory():
            class MemoryInfo:
                def __init__(self):
                    self.total = 16 * 1024**3  # 16GB想定
                    self.used = 8 * 1024**3  # 8GB使用想定
                    self.available = 8 * 1024**3  # 8GB空き想定

            return MemoryInfo()

    # psutilの代替
    psutil = BasicSystemMonitor()


@dataclass
class ResourceStatus:
    """リソース状態"""

    gpu_id: int
    gpu_name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    utilization_percent: float
    temperature_c: float
    power_usage_w: float
    is_available: bool = True
    estimated_available_workers: int = 0


@dataclass
class SystemResourceStatus:
    """システム全体のリソース状態"""

    cpu_count: int
    cpu_usage_percent: float
    memory_total_gb: float
    memory_used_gb: float
    memory_available_gb: float
    other_users_detected: bool = False
    recommended_max_workers: int = 16


class DynamicWorkerManager:
    """
    動的ワーカー数管理システム

    H100デュアルGPU構成で最大48ワーカーまで活用し、
    他のユーザーとの競合を避けるための動的リソース監視
    """

    def __init__(self):
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.resource_status: Dict[int, ResourceStatus] = {}
        self.system_status: Optional[SystemResourceStatus] = None
        self.lock = threading.Lock()
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None

        # H100デュアルGPU構成の理論的最大値
        self.max_theoretical_workers = 48  # 24 * 2 GPUs
        self.base_workers_per_gpu = 24  # H100単体の理論値

        # 安全マージン設定
        self.memory_safety_margin = 0.8  # 80%まで使用
        self.utilization_threshold = 0.7  # 70%以上で負荷軽減
        self.temperature_threshold = 80  # 80°C以上で負荷軽減

        self._initialize_resources()

    def _initialize_resources(self):
        """リソース初期化"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - using CPU mode")
            return

        for gpu_id in range(self.gpu_count):
            try:
                props = torch.cuda.get_device_properties(gpu_id)
                gpu_name = torch.cuda.get_device_name(gpu_id)
                total_memory_gb = props.total_memory / (1024**3)

                self.resource_status[gpu_id] = ResourceStatus(
                    gpu_id=gpu_id,
                    gpu_name=gpu_name,
                    total_memory_gb=total_memory_gb,
                    free_memory_gb=total_memory_gb,
                    used_memory_gb=0.0,
                    utilization_percent=0.0,
                    temperature_c=0.0,
                    power_usage_w=0.0,
                    is_available=True,
                    estimated_available_workers=self.base_workers_per_gpu,
                )

                logger.info(f"✅ GPU {gpu_id}: {gpu_name} ({total_memory_gb:.1f}GB)")

            except Exception as e:
                logger.error(f"❌ GPU {gpu_id} 初期化エラー: {e}")
                if gpu_id in self.resource_status:
                    self.resource_status[gpu_id].is_available = False

    def start_monitoring(self):
        """リソース監視開始"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_resources, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("🔍 動的リソース監視開始")

    def stop_monitoring(self):
        """リソース監視停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("⏹️ 動的リソース監視停止")

    def _monitor_resources(self):
        """リソース監視スレッド"""
        while self.monitoring_active:
            try:
                self._update_gpu_resources()
                self._update_system_resources()
                self._detect_other_users()
                time.sleep(2.0)  # 2秒間隔で更新
            except Exception as e:
                logger.error(f"リソース監視エラー: {e}")
                time.sleep(5.0)

    def _update_gpu_resources(self):
        """GPUリソース更新"""
        try:
            import pynvml

            pynvml.nvmlInit()

            for gpu_id, status in self.resource_status.items():
                if not status.is_available:
                    continue

                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

                    # メモリ情報
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    status.total_memory_gb = mem_info.total / (1024**3)
                    status.free_memory_gb = mem_info.free / (1024**3)
                    status.used_memory_gb = mem_info.used / (1024**3)

                    # 使用率
                    try:
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        status.utilization_percent = utilization.gpu
                    except:
                        status.utilization_percent = 0.0

                    # 温度
                    try:
                        status.temperature_c = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    except:
                        status.temperature_c = 0.0

                    # 電力使用量
                    try:
                        status.power_usage_w = (
                            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        )
                    except:
                        status.power_usage_w = 0.0

                    # 利用可能ワーカー数推定
                    status.estimated_available_workers = (
                        self._estimate_available_workers(status)
                    )

                except Exception as e:
                    logger.warning(f"GPU {gpu_id} 情報取得エラー: {e}")
                    status.is_available = False

        except ImportError:
            # pynvml利用不可の場合は基本的な監視を使用
            self._update_gpu_resources_basic()

    def _update_gpu_resources_basic(self):
        """基本的なGPUリソース更新（pynvmlなし）"""
        for gpu_id, status in self.resource_status.items():
            try:
                # PyTorchの基本情報のみ
                allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)

                status.used_memory_gb = allocated
                status.free_memory_gb = status.total_memory_gb - reserved

                # 基本的なワーカー数推定
                memory_usage_ratio = reserved / status.total_memory_gb
                status.estimated_available_workers = max(
                    1, int(self.base_workers_per_gpu * (1.0 - memory_usage_ratio))
                )

            except Exception as e:
                logger.warning(f"GPU {gpu_id} 基本情報取得エラー: {e}")
                status.is_available = False

    def _update_system_resources(self):
        """システムリソース更新"""
        try:
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()

            self.system_status = SystemResourceStatus(
                cpu_count=cpu_count,
                cpu_usage_percent=cpu_usage,
                memory_total_gb=memory.total / (1024**3),
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
            )

        except Exception as e:
            logger.warning(f"システムリソース取得エラー: {e}")

    def _detect_other_users(self):
        """他のユーザーの検出"""
        if not self.system_status:
            return

        # CPU使用率が高い場合、他のプロセスが動いている可能性
        if self.system_status.cpu_usage_percent > 50:
            self.system_status.other_users_detected = True
            logger.info(
                f"⚠️ 他のユーザーが検出されました (CPU使用率: {self.system_status.cpu_usage_percent:.1f}%)"
            )
        else:
            self.system_status.other_users_detected = False

    def _estimate_available_workers(self, status: ResourceStatus) -> int:
        """利用可能ワーカー数推定"""
        if not status.is_available:
            return 0

        # メモリ制約
        memory_usage_ratio = status.used_memory_gb / status.total_memory_gb
        memory_factor = max(0.1, 1.0 - memory_usage_ratio)

        # 使用率制約
        utilization_factor = max(0.1, 1.0 - (status.utilization_percent / 100.0))

        # 温度制約
        temp_factor = 1.0
        if status.temperature_c > self.temperature_threshold:
            temp_factor = max(
                0.3, 1.0 - (status.temperature_c - self.temperature_threshold) / 20.0
            )

        # 総合的な利用可能ワーカー数
        available_workers = int(
            self.base_workers_per_gpu * memory_factor * utilization_factor * temp_factor
        )

        return max(1, min(available_workers, self.base_workers_per_gpu))

    def get_optimal_worker_count(
        self, requested_workers: int = None
    ) -> Tuple[int, Dict[str, any]]:
        """
        最適なワーカー数を取得

        Args:
            requested_workers: 要求されたワーカー数（Noneの場合は自動決定）

        Returns:
            (optimal_workers, resource_info)
        """
        with self.lock:
            if not self.resource_status:
                return 4, {"reason": "No GPU available", "mode": "CPU"}

            # 各GPUの利用可能ワーカー数を計算
            total_available_workers = 0
            gpu_details = {}

            for gpu_id, status in self.resource_status.items():
                if status.is_available:
                    available_workers = status.estimated_available_workers
                    total_available_workers += available_workers

                    gpu_details[f"gpu_{gpu_id}"] = {
                        "name": status.gpu_name,
                        "available_workers": available_workers,
                        "memory_usage_gb": status.used_memory_gb,
                        "memory_free_gb": status.free_memory_gb,
                        "utilization_percent": status.utilization_percent,
                        "temperature_c": status.temperature_c,
                    }

            # 要求されたワーカー数が指定されている場合
            if requested_workers is not None:
                optimal_workers = min(requested_workers, total_available_workers)
                reason = f"Requested {requested_workers}, available {total_available_workers}"
            else:
                # 自動決定
                optimal_workers = self._auto_determine_workers(total_available_workers)
                reason = f"Auto-determined from {total_available_workers} available"

            # 他のユーザーが検出された場合の調整
            if self.system_status and self.system_status.other_users_detected:
                optimal_workers = max(4, int(optimal_workers * 0.7))  # 30%削減
                reason += " (reduced due to other users)"

            # 最小値の保証
            optimal_workers = max(4, optimal_workers)

            resource_info = {
                "optimal_workers": optimal_workers,
                "total_available_workers": total_available_workers,
                "reason": reason,
                "gpu_details": gpu_details,
                "system_status": (
                    self.system_status.__dict__ if self.system_status else None
                ),
                "other_users_detected": (
                    self.system_status.other_users_detected
                    if self.system_status
                    else False
                ),
            }

            logger.info(
                f"🎯 最適ワーカー数: {optimal_workers} (利用可能: {total_available_workers})"
            )
            return optimal_workers, resource_info

    def _auto_determine_workers(self, available_workers: int) -> int:
        """自動ワーカー数決定"""
        # H100デュアルGPU構成での最適化
        if self.gpu_count >= 2 and any(
            "H100" in status.gpu_name for status in self.resource_status.values()
        ):
            # デュアルH100の場合、理論最大値の80%を目標
            target_workers = int(self.max_theoretical_workers * 0.8)  # 38ワーカー
            return min(target_workers, available_workers)

        # 単一GPUまたはその他の構成
        return min(16, available_workers)

    def get_resource_summary(self) -> Dict[str, any]:
        """リソース状況サマリー取得"""
        with self.lock:
            summary = {
                "gpu_count": self.gpu_count,
                "monitoring_active": self.monitoring_active,
                "gpus": {},
            }

            for gpu_id, status in self.resource_status.items():
                summary["gpus"][gpu_id] = {
                    "name": status.gpu_name,
                    "available": status.is_available,
                    "memory_total_gb": status.total_memory_gb,
                    "memory_free_gb": status.free_memory_gb,
                    "utilization_percent": status.utilization_percent,
                    "temperature_c": status.temperature_c,
                    "available_workers": status.estimated_available_workers,
                }

            if self.system_status:
                summary["system"] = {
                    "cpu_count": self.system_status.cpu_count,
                    "cpu_usage_percent": self.system_status.cpu_usage_percent,
                    "memory_available_gb": self.system_status.memory_available_gb,
                    "other_users_detected": self.system_status.other_users_detected,
                }

            return summary


# グローバルインスタンス
dynamic_worker_manager = DynamicWorkerManager()
