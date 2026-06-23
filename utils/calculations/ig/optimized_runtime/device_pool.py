from __future__ import annotations

import copy
import logging
import os
from itertools import cycle
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class DevicePool:
    """
    Manage model replicas, CUDA streams, and input placement across devices.

    The handling mirrors the execution order summarised in
    `theory/1.transformerの記号体系の定義と計算の流れ.md`: inputs are
    distributed to the relevant device before the IG integrals are evaluated.
    """

    def __init__(self, unified_model: torch.nn.Module, *, workers_per_gpu: int, use_lightning_trainer: bool = False) -> None:
        self._source_model = unified_model
        self.use_lightning_trainer = use_lightning_trainer

        # Hardware profile -------------------------------------------------
        self.is_h100 = (
            torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0)
        )
        self.is_a100 = (
            torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0)
        )
        self.is_v100 = (
            torch.cuda.is_available() and "V100" in torch.cuda.get_device_name(0)
        )

        # 全GPUを使用
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # GPU認識状況をログに出力
        if torch.cuda.is_available():
            actual_gpu_count = torch.cuda.device_count()
            if actual_gpu_count > 0:
                logger.info(f"✅ GPU検出: {actual_gpu_count}個のGPUが利用可能")
                for gpu_id in range(actual_gpu_count):
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    logger.info(f"   GPU {gpu_id}: {gpu_name}")
            else:
                logger.warning("⚠️ torch.cuda.is_available()=True だが、device_count()=0。GPU設定を確認してください。")
        else:
            logger.warning("⚠️ GPUが利用できません（torch.cuda.is_available()=False）。Docker設定を確認してください。")
            logger.info("   DockerでGPUを使用する場合: docker-compose.ymlでruntime: nvidiaを設定")
            logger.info("   または: docker run --gpus all ... を使用")
        self.max_concurrent_tasks = self._estimate_concurrency()
        self.max_batch_size = self._estimate_batch_cap()
        # Phase 2.2: マルチストリーム処理の改善（最適化版）
        # GPUタイプごとにストリーム数を最適化（より積極的な並行処理）
        # 環境変数でストリーム数を制御可能（デフォルト: 自動最適化）
        env_streams = os.environ.get("PTB_CUDA_STREAMS_PER_DEVICE")
        if env_streams:
            try:
                self.streams_per_device = int(env_streams)
            except ValueError:
                # 環境変数が無効な場合は自動最適化を使用
                env_streams = None
        
        if not env_streams:
            if self.is_h100:
                # H100: GPU数に応じてストリーム数を最適化
                # H100 2枚: 各GPU 128-256ストリームで並行処理を最大化
                if self.gpu_count >= 2:
                    self.streams_per_device = max(128, min(256, workers_per_gpu * 2))
                else:
                    self.streams_per_device = max(128, min(256, workers_per_gpu * 2))
            elif self.is_a100:
                # A100: より多くのストリームで並行処理を最大化（64-128ストリーム）
                # GPU使用率を上げるため、ストリーム数を大幅に増加
                self.streams_per_device = max(64, min(128, workers_per_gpu * 4))
            elif self.is_v100:
                # V100: GPU数に応じてストリーム数を最適化
                # V100 4枚: 各GPU 16-32ストリームで並行処理を最大化
                if self.gpu_count >= 4:
                    self.streams_per_device = max(16, min(32, workers_per_gpu * 2))
                else:
                    self.streams_per_device = max(8, min(16, workers_per_gpu))
            else:
                self.streams_per_device = max(8, min(16, workers_per_gpu))  # デフォルト8-16ストリーム

        # Model replicas & streams -----------------------------------------
        self.cpu_device = torch.device("cpu")
        # 全GPUを使用（プライマリデバイスはGPU0）
        self.primary_device = (
            torch.device("cuda:0") if torch.cuda.is_available() else self.cpu_device
        )
        self.primary_device_id = 0 if torch.cuda.is_available() else -1
        self.device_models: Dict[int, torch.nn.Module] = {}
        self.device_streams: Dict[int, List[torch.cuda.Stream]] = {}
        self.device_stream_cycle: Dict[int, Any] = {}

        # Trainer使用時はモデル配置をスキップ（Trainerが自動配置）
        if not self.use_lightning_trainer:
            self._prepare_device_models()
        else:
            # Trainer使用時は、モデルは既にTrainerによってGPU0に配置されている
            # ただし、マルチGPU対応のため、各GPUにモデルを配置する必要がある
            if torch.cuda.is_available():
                # モデルが既に配置されているデバイスを確認
                model_device = next(unified_model.parameters()).device
                if model_device.type == "cuda":
                    device_id = model_device.index if model_device.index is not None else 0
                    # GPU0には元のモデルを使用
                    self.device_models[0] = unified_model
                    
                    # GPU1以降にはモデルのコピーを作成して配置
                    # Trainer使用時はstate_dict()を使ってコピー（deepcopyの問題を避ける）
                    for gpu_id in range(1, self.gpu_count):
                        device = torch.device(f"cuda:{gpu_id}")
                        # UnifiedBertModelの新しいインスタンスを作成してstate_dictをロード
                        from utils.unified_bert_model import UnifiedBertModel
                        model_copy = UnifiedBertModel(unified_model.model_name)
                        model_copy.load_state_dict(unified_model.state_dict())
                        model_copy.to(device)
                        model_copy.eval()
                        self.device_models[gpu_id] = model_copy
                    
                    logger.info(f"📦 PyTorch Lightning Trainerでモデルが自動配置済み（GPU0）。他のGPUにもコピーを配置: GPU 0-{self.gpu_count-1}")
                else:
                    self.device_models[-1] = unified_model
            else:
                self.device_models[-1] = unified_model
        
        self._prepare_device_streams()

    # ------------------------------------------------------------------ #
    # Hardware helpers
    # ------------------------------------------------------------------ #
    def _estimate_concurrency(self) -> int:
        if self.is_h100:
            # H100: GPU数に応じて並行度を最適化
            if self.gpu_count >= 2:
                return 132 * 64 * 2  # H100 2枚なら2倍
            return 132 * 64
        if self.is_a100:
            return 108 * 64
        if self.is_v100 and self.gpu_count >= 4:
            # V100 4枚なら並行度を増加
            return 80 * 32 * 4  # V100 4枚なら4倍
        return 1000

    def _estimate_batch_cap(self) -> int:
        if self.is_h100:
            return 2048  # H100なら2048に大幅増加
        if self.is_a100:
            return 2048  # A100も2048に増加（40GBメモリを活用）
        # V100: 4GPU環境では512に増加（従来の256から2倍）
        if self.gpu_count >= 4:
            return 512  # 4GPU以上なら512に増加
        return 256  # 1-3GPUなら256

    # ------------------------------------------------------------------ #
    # Model & stream placement
    # ------------------------------------------------------------------ #
    def _prepare_device_models(self) -> None:
        self.device_models.clear()
        try:
            self._source_model.eval()
        except Exception:
            pass

        if not torch.cuda.is_available():
            self._source_model.to(self.cpu_device)
            self.device_models[-1] = self._source_model
            return

        # 全GPUにモデルを配置
        base_device = torch.device("cuda:0")
        self._source_model.to(base_device)
        self._source_model.eval()
        
        # GPU0には元のモデルを使用
        self.device_models[0] = self._source_model
        
        # GPU1以降にはモデルのコピーを作成して配置
        for gpu_id in range(1, self.gpu_count):
            device = torch.device(f"cuda:{gpu_id}")
            model_copy = copy.deepcopy(self._source_model)
            model_copy.to(device)
            model_copy.eval()
            self.device_models[gpu_id] = model_copy
        
        logger.info(f"📦 {self.gpu_count}個のGPUにモデルを配置しました: GPU 0-{self.gpu_count-1}")

        if not self.device_models:
            self.device_models[-1] = self._source_model
        
        # Phase 1.3: メモリプールの事前確保（フラグメンテーション防止）
        if torch.cuda.is_available():
            self._preallocate_memory_pools()

    def _preallocate_memory_pools(self) -> None:
        """メモリプールを事前に確保してフラグメンテーションを防止"""
        for device_id in range(self.gpu_count):
            try:
                device = torch.device(f"cuda:{device_id}")
                # GPUメモリの10%を事前確保（フラグメンテーション防止）
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
                total_gb = total_bytes / (1024**3)
                preallocate_gb = min(total_gb * 0.1, 4.0)  # 最大4GBまで
                
                if preallocate_gb > 0.5:  # 0.5GB以上の場合のみ事前確保
                    dummy_tensor = torch.empty(
                        (int(preallocate_gb * 1024**3 // 4),),  # float32の要素数
                        dtype=torch.float32,
                        device=device
                    )
                    del dummy_tensor
                    torch.cuda.empty_cache()
                    logger.info(f"💾 GPU {device_id}: {preallocate_gb:.1f}GBのメモリプールを事前確保")
            except Exception as e:
                logger.warning(f"⚠️ GPU {device_id}のメモリプール事前確保に失敗: {e}")

    def _prepare_device_streams(self) -> None:
        if not torch.cuda.is_available():
            self.device_streams = {}
            self.device_stream_cycle = {}
            return

        for device_id in self.device_models.keys():
            if device_id == -1:
                continue
            device = torch.device(f"cuda:{device_id}")
            streams = [
                torch.cuda.Stream(device=device) for _ in range(self.streams_per_device)
            ]
            self.device_streams[device_id] = streams
            self.device_stream_cycle[device_id] = cycle(streams)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_model(self, device_id: Optional[int]) -> torch.nn.Module:
        if device_id is None:
            device_id = self.primary_device_id
        
        # 指定されたdevice_idのモデルが存在する場合はそれを返す
        if device_id in self.device_models:
            return self.device_models[device_id]
        
        # 存在しない場合は、device_idにモデルを配置してから返す
        if torch.cuda.is_available() and device_id >= 0:
            device = torch.device(f"cuda:{device_id}")
            # UnifiedBertModelの新しいインスタンスを作成してstate_dictをロード
            from utils.unified_bert_model import UnifiedBertModel
            model_copy = UnifiedBertModel(self._source_model.model_name)
            model_copy.load_state_dict(self._source_model.state_dict())
            model_copy.to(device)
            model_copy.eval()
            self.device_models[device_id] = model_copy
            logger.info(f"📦 オンデマンドでGPU{device_id}にモデルを配置しました")
            return model_copy
        
        # CPUまたはフォールバック
        return self._source_model

    def get_stream(self, device_id: Optional[int]) -> Optional[torch.cuda.Stream]:
        if not torch.cuda.is_available():
            return None
        if device_id is None:
            device_id = self.primary_device_id
        stream_cycle = self.device_stream_cycle.get(device_id)
        if stream_cycle is None:
            return None
        return next(stream_cycle)

    def prepare_inputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        device_inputs: Dict[int, Dict[str, torch.Tensor]] = {}
        if not self.device_models:
            device_inputs[-1] = {k: v.to(self.cpu_device) for k, v in inputs.items()}
            return device_inputs

        for device_id in self.device_models.keys():
            if device_id == -1:
                device_inputs[device_id] = {k: v.clone() for k, v in inputs.items()}
                continue

            device = torch.device(f"cuda:{device_id}")
            device_inputs[device_id] = {
                k: v.to(device, non_blocking=True) for k, v in inputs.items()
            }
        return device_inputs
