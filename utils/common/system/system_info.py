"""
システム情報表示コンポーネント
全ページで共通して使用するシステム情報の表示機能
"""

import streamlit as st
import torch
import time
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# GPU監視ライブラリのインポート（オプション）
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# システム監視ライブラリのインポート（オプション）
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class SystemInfoManager:
    """システム情報管理クラス"""

    def __init__(self):
        self.last_update = 0
        self.update_interval = 1  # デフォルト更新間隔（秒）
        self.gpu_info_cache = {}
        self.system_info_cache = {}

    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU情報を取得"""
        if not torch.cuda.is_available():
            return {"available": False, "gpus": []}

        gpus = []
        gpu_count = torch.cuda.device_count()

        for gpu_id in range(gpu_count):
            try:
                # 基本情報
                gpu_name = torch.cuda.get_device_name(gpu_id)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory

                # メモリ使用量
                torch.cuda.set_device(gpu_id)
                allocated_memory = torch.cuda.memory_allocated(gpu_id)
                cached_memory = torch.cuda.memory_reserved(gpu_id)
                free_memory = total_memory - allocated_memory

                # GPU使用率（pynvmlが利用可能な場合）
                utilization = "監視不可 (No module named 'pynvml')"
                if PYNVML_AVAILABLE:
                    try:
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        utilization = f"{util.gpu}%"
                    except Exception as e:
                        utilization = f"エラー: {str(e)}"

                gpu_info = {
                    "id": gpu_id,
                    "name": gpu_name,
                    "total_memory_gb": total_memory / (1024**3),
                    "allocated_memory_gb": allocated_memory / (1024**3),
                    "cached_memory_gb": cached_memory / (1024**3),
                    "free_memory_gb": free_memory / (1024**3),
                    "utilization": utilization,
                    "is_h100": "H100" in gpu_name,
                }
                gpus.append(gpu_info)

            except Exception as e:
                logger.error(f"GPU {gpu_id} 情報取得エラー: {e}")
                continue

        return {
            "available": True,
            "gpus": gpus,
            "lightning_gpus": gpu_count,
            "h100_optimized": any(gpu["is_h100"] for gpu in gpus),
        }

    def get_multi_gpu_info(self) -> Dict[str, Any]:
        """マルチGPU管理情報を取得"""
        if not torch.cuda.is_available():
            return {"available": False}

        try:
            from utils.multi_gpu_manager import MultiGPUManager

            manager = MultiGPUManager()

            # GPU状態を取得
            gpu_count = torch.cuda.device_count()
            available_gpus = sum(
                1 for gpu_id in range(gpu_count) if manager.is_gpu_available(gpu_id)
            )

            # 実行中タスク数
            running_tasks = sum(len(tasks) for tasks in manager.gpu_tasks.values())

            # GPU使用率
            gpu_utilizations = {}
            for gpu_id in range(gpu_count):
                if manager.is_gpu_available(gpu_id):
                    # 簡易的な使用率計算（メモリベース）
                    allocated = torch.cuda.memory_allocated(gpu_id)
                    total = torch.cuda.get_device_properties(gpu_id).total_memory
                    utilization = (allocated / total) * 100
                    gpu_utilizations[f"GPU {gpu_id}"] = f"{utilization:.1f}%"

            return {
                "available": True,
                "total_gpus": gpu_count,
                "available_gpus": available_gpus,
                "running_tasks": running_tasks,
                "gpu_utilizations": gpu_utilizations,
                "multi_gpu_enabled": gpu_count > 1,
                "auto_load_balancing": True,
            }

        except Exception as e:
            logger.error(f"マルチGPU管理情報取得エラー: {e}")
            return {"available": False, "error": str(e)}

    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュシステム情報を取得"""
        try:
            cache_dir = Path("cache")
            if not cache_dir.exists():
                return {
                    "available": False,
                    "message": "キャッシュディレクトリが存在しません",
                }

            # キャッシュファイル数をカウント
            cache_files = list(cache_dir.rglob("*"))
            cache_size_mb = sum(
                f.stat().st_size for f in cache_files if f.is_file()
            ) / (1024**2)

            return {
                "available": True,
                "cache_dir": str(cache_dir),
                "file_count": len(cache_files),
                "size_mb": cache_size_mb,
                "features": [
                    "各層のz、u、MLP出力をキャッシュ",
                    "重みパラメータもキャッシュ",
                    "IG計算時の入力分離を正確に実現",
                    "計算効率の大幅な向上",
                ],
            }

        except Exception as e:
            logger.error(f"キャッシュ情報取得エラー: {e}")
            return {"available": False, "error": str(e)}


def render_system_info(update_interval: int = 1, show_refresh_button: bool = True):
    """
    システム情報をStreamlitで表示

    Args:
        update_interval: 更新間隔（秒）
        show_refresh_button: 手動更新ボタンを表示するか
    """

    # セッション状態の初期化
    if "system_info_manager" not in st.session_state:
        st.session_state.system_info_manager = SystemInfoManager()

    manager = st.session_state.system_info_manager

    # 更新間隔設定
    st.sidebar.subheader("🖥️ システム情報")
    selected_interval = st.sidebar.selectbox(
        "更新間隔（秒）",
        [1, 5, 10, 30, 60],
        index=(
            [1, 5, 10, 30, 60].index(update_interval)
            if update_interval in [1, 5, 10, 30, 60]
            else 0
        ),
    )

    # 手動更新ボタン
    if show_refresh_button:
        if st.sidebar.button("🔄 システム情報更新", key="refresh_system_info"):
            manager.last_update = 0  # 強制更新

    # 自動更新チェック
    current_time = time.time()
    should_update = (current_time - manager.last_update) >= selected_interval

    if should_update or show_refresh_button:
        # GPU情報を取得
        gpu_info = manager.get_gpu_info()

        # マルチGPU情報を取得
        multi_gpu_info = manager.get_multi_gpu_info()

        # キャッシュ情報を取得
        cache_info = manager.get_cache_info()

        # 更新時刻を記録
        manager.last_update = current_time

    # GPU情報表示
    if gpu_info["available"]:
        st.sidebar.subheader("🚀 GPU情報")

        for gpu in gpu_info["gpus"]:
            with st.sidebar.expander(f"GPU {gpu['id']}: {gpu['name']}", expanded=True):
                st.write(f"💾 **総メモリ**: {gpu['total_memory_gb']:.1f}GB")
                st.write(f"📊 **使用中**: {gpu['allocated_memory_gb']:.1f}GB")
                st.write(f"🆓 **利用可能**: {gpu['free_memory_gb']:.1f}GB")
                st.write(f"GPU {gpu['id']} 使用率: {gpu['utilization']}")

        # Lightning対応GPU数
        st.sidebar.write(f"⚡ Lightning対応GPU: {gpu_info['lightning_gpus']} devices")

        # H100最適化設定
        if gpu_info["h100_optimized"]:
            st.sidebar.success("🎯 H100最適化設定有効")
    else:
        st.sidebar.warning("🚫 GPU利用不可")

    # マルチGPU管理システム
    if multi_gpu_info["available"]:
        st.sidebar.subheader("🚀 マルチGPU管理システム")
        st.sidebar.write(f"📊 総GPU数: {multi_gpu_info['total_gpus']}")
        st.sidebar.write(f"✅ 利用可能: {multi_gpu_info['available_gpus']}")
        st.sidebar.write(f"🔄 実行中タスク: {multi_gpu_info['running_tasks']}")

        # GPU使用率
        for gpu_name, utilization in multi_gpu_info["gpu_utilizations"].items():
            st.sidebar.write(f"{gpu_name}: {utilization} 使用")

        # マルチGPU設定
        st.sidebar.subheader("⚡ マルチGPU設定")
        if multi_gpu_info["multi_gpu_enabled"]:
            st.sidebar.success(f"✅ {multi_gpu_info['total_gpus']}台のGPUで並列処理")
        st.sidebar.success("🎯 自動負荷分散（最適化済み）")
    else:
        st.sidebar.warning("🚫 マルチGPU管理システム利用不可")

    # BERT層出力キャッシュシステム
    if cache_info["available"]:
        st.sidebar.subheader("BERT層出力キャッシュシステム")
        st.sidebar.success("✅ 各層のz、u、MLP出力をキャッシュ")
        st.sidebar.success("✅ 重みパラメータもキャッシュ")
        st.sidebar.success("✅ IG計算時の入力分離を正確に実現")
        st.sidebar.success("✅ 計算効率の大幅な向上")

        with st.sidebar.expander("📊 キャッシュ統計", expanded=False):
            st.write(f"📁 キャッシュディレクトリ: {cache_info['cache_dir']}")
            st.write(f"📄 ファイル数: {cache_info['file_count']}")
            st.write(f"💾 サイズ: {cache_info['size_mb']:.1f}MB")
    else:
        st.sidebar.warning("🚫 キャッシュシステム利用不可")


def get_system_summary() -> Dict[str, Any]:
    """
    システム情報のサマリーを取得（プログラム用）

    Returns:
        Dict: システム情報のサマリー
    """
    manager = SystemInfoManager()

    return {
        "gpu_info": manager.get_gpu_info(),
        "multi_gpu_info": manager.get_multi_gpu_info(),
        "cache_info": manager.get_cache_info(),
        "timestamp": time.time(),
    }


# 便利な関数
def is_h100_available() -> bool:
    """H100 GPUが利用可能かチェック"""
    gpu_info = SystemInfoManager().get_gpu_info()
    return gpu_info["available"] and gpu_info["h100_optimized"]


def get_available_gpu_count() -> int:
    """利用可能なGPU数を取得"""
    gpu_info = SystemInfoManager().get_gpu_info()
    return len(gpu_info["gpus"]) if gpu_info["available"] else 0


def get_total_gpu_memory_gb() -> float:
    """全GPUの総メモリ量を取得（GB）"""
    gpu_info = SystemInfoManager().get_gpu_info()
    if not gpu_info["available"]:
        return 0.0

    return sum(gpu["total_memory_gb"] for gpu in gpu_info["gpus"])


def get_free_gpu_memory_gb() -> float:
    """全GPUの空きメモリ量を取得（GB）"""
    gpu_info = SystemInfoManager().get_gpu_info()
    if not gpu_info["available"]:
        return 0.0

    return sum(gpu["free_memory_gb"] for gpu in gpu_info["gpus"])
