# parallel_ig_calculator.py
"""
マルチGPU対応並列IG計算システム
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .multi_gpu_manager import get_multi_gpu_manager, allocate_gpu_for_task, release_gpu_for_task
from .optimized_ig import OptimizedIGCalculator

logger = logging.getLogger(__name__)


class ParallelIGCalculator:
    """
    マルチGPU対応並列IG計算システム
    """
    
    def __init__(self):
        self.multi_gpu_manager = get_multi_gpu_manager()
        self.executor = ThreadPoolExecutor(max_workers=self.multi_gpu_manager.gpu_count * 2)
        
    def compute_parallel_mlp_ig(
        self,
        model,
        layer_idx: int,
        target_token_indices: List[int],
        target_head_indices: Optional[List[int]] = None,
        num_steps: int = 32,
        use_mixed_precision: bool = True
    ) -> List[Dict]:
        """
        並列MLP IG計算
        
        Args:
            model: BERTモデル
            layer_idx: 対象レイヤー
            target_token_indices: 対象トークンインデックスリスト
            target_head_indices: 対象ヘッドインデックスリスト
            num_steps: 積分分割数
            use_mixed_precision: 混合精度使用
        
        Returns:
            計算結果リスト
        """
        logger.info(f"🚀 並列MLP IG計算開始: {len(target_token_indices)} タスク")
        
        # タスク作成
        tasks = self._create_mlp_ig_tasks(
            layer_idx, target_token_indices, target_head_indices
        )
        
        # 並列実行
        results = self._execute_parallel_tasks(model, tasks, num_steps, use_mixed_precision)
        
        logger.info(f"✅ 並列MLP IG計算完了: {len(results)} 結果")
        return results
    
    def _create_mlp_ig_tasks(
        self,
        layer_idx: int,
        target_token_indices: List[int],
        target_head_indices: Optional[List[int]] = None
    ) -> List[Dict]:
        """MLP IG計算タスク作成"""
        tasks = []
        
        for i, token_idx in enumerate(target_token_indices):
            head_idx = target_head_indices[i] if target_head_indices else None
            
            task = {
                "task_id": f"mlp_ig_{layer_idx}_{token_idx}_{head_idx}_{uuid.uuid4().hex[:8]}",
                "layer_idx": layer_idx,
                "token_idx": token_idx,
                "head_idx": head_idx,
                "task_type": "mlp_ig_calculation",
                "estimated_memory": 2 * 1024 * 1024 * 1024,  # 2GB推定
                "priority": 1
            }
            tasks.append(task)
        
        return tasks
    
    def _execute_parallel_tasks(
        self,
        model,
        tasks: List[Dict],
        num_steps: int,
        use_mixed_precision: bool
    ) -> List[Dict]:
        """並列タスク実行"""
        # タスクをGPU間で分散
        gpu_tasks = self._distribute_tasks_to_gpus(tasks)
        
        # 並列実行
        futures = []
        results = []
        
        for gpu_id, gpu_task_list in gpu_tasks.items():
            if not gpu_task_list:
                continue
                
            future = self.executor.submit(
                self._execute_gpu_tasks,
                gpu_id, model, gpu_task_list, num_steps, use_mixed_precision
            )
            futures.append(future)
        
        # 結果収集
        for future in as_completed(futures):
            try:
                gpu_results = future.result()
                results.extend(gpu_results)
            except Exception as e:
                logger.error(f"並列タスク実行エラー: {e}")
        
        return results
    
    def _distribute_tasks_to_gpus(self, tasks: List[Dict]) -> Dict[int, List[Dict]]:
        """タスクをGPU間で分散"""
        gpu_tasks = {i: [] for i in range(self.multi_gpu_manager.gpu_count)}
        
        # 負荷分散アルゴリズム
        for i, task in enumerate(tasks):
            # 最適なGPUを選択
            try:
                optimal_gpu = self.multi_gpu_manager.get_optimal_gpu(
                    task["estimated_memory"],
                    task["task_type"]
                )
                gpu_tasks[optimal_gpu].append(task)
            except Exception as e:
                # フォールバック: ラウンドロビン
                gpu_id = i % self.multi_gpu_manager.gpu_count
                gpu_tasks[gpu_id].append(task)
        
        return gpu_tasks
    
    def _execute_gpu_tasks(
        self,
        gpu_id: int,
        model,
        tasks: List[Dict],
        num_steps: int,
        use_mixed_precision: bool
    ) -> List[Dict]:
        """GPU上でタスク実行"""
        try:
            # GPU設定
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            
            # モデルをGPUに移動
            model_gpu = model.to(device)
            
            # 混合精度設定
            if use_mixed_precision:
                model_gpu = model_gpu.half()
            
            results = []
            for task in tasks:
                try:
                    # GPU割り当て
                    allocated_gpu = allocate_gpu_for_task(
                        task["task_id"],
                        task["estimated_memory"],
                        task["task_type"]
                    )
                    
                    # タスク実行
                    result = self._execute_single_mlp_ig_task(
                        model_gpu, task, num_steps, device
                    )
                    result["gpu_id"] = gpu_id
                    result["task_id"] = task["task_id"]
                    results.append(result)
                    
                    # GPU解放
                    release_gpu_for_task(task["task_id"])
                    
                except Exception as e:
                    logger.error(f"タスク実行エラー (GPU {gpu_id}): {e}")
                    results.append({
                        "error": str(e),
                        "gpu_id": gpu_id,
                        "task_id": task["task_id"]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"GPU {gpu_id} タスク実行エラー: {e}")
            return [{"error": str(e), "gpu_id": gpu_id} for _ in tasks]
    
    def _execute_single_mlp_ig_task(
        self,
        model,
        task: Dict,
        num_steps: int,
        device
    ) -> Dict:
        """単一MLP IGタスク実行"""
        try:
            # IG計算実行
            ig_calculator = OptimizedIGCalculator(model, device)
            
            result = ig_calculator.compute_mlp_ig(
                layer_idx=task["layer_idx"],
                target_token_idx=task["token_idx"],
                target_head_idx=task["head_idx"],
                num_steps=num_steps
            )
            
            return {
                "success": True,
                "ig_values": result,
                "execution_time": time.time()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time()
            }
    
    def compute_parallel_attention_ig(
        self,
        model,
        layer_idx: int,
        target_token_indices: List[int],
        target_head_indices: Optional[List[int]] = None,
        num_steps: int = 32,
        use_mixed_precision: bool = True
    ) -> List[Dict]:
        """
        並列Attention IG計算
        
        Args:
            model: BERTモデル
            layer_idx: 対象レイヤー
            target_token_indices: 対象トークンインデックスリスト
            target_head_indices: 対象ヘッドインデックスリスト
            num_steps: 積分分割数
            use_mixed_precision: 混合精度使用
        
        Returns:
            計算結果リスト
        """
        logger.info(f"🚀 並列Attention IG計算開始: {len(target_token_indices)} タスク")
        
        # タスク作成
        tasks = self._create_attention_ig_tasks(
            layer_idx, target_token_indices, target_head_indices
        )
        
        # 並列実行
        results = self._execute_parallel_tasks(model, tasks, num_steps, use_mixed_precision)
        
        logger.info(f"✅ 並列Attention IG計算完了: {len(results)} 結果")
        return results
    
    def _create_attention_ig_tasks(
        self,
        layer_idx: int,
        target_token_indices: List[int],
        target_head_indices: Optional[List[int]] = None
    ) -> List[Dict]:
        """Attention IG計算タスク作成"""
        tasks = []
        
        for i, token_idx in enumerate(target_token_indices):
            head_idx = target_head_indices[i] if target_head_indices else None
            
            task = {
                "task_id": f"attn_ig_{layer_idx}_{token_idx}_{head_idx}_{uuid.uuid4().hex[:8]}",
                "layer_idx": layer_idx,
                "token_idx": token_idx,
                "head_idx": head_idx,
                "task_type": "attention_ig_calculation",
                "estimated_memory": 1 * 1024 * 1024 * 1024,  # 1GB推定
                "priority": 1
            }
            tasks.append(task)
        
        return tasks
    
    def get_parallel_execution_stats(self) -> Dict[str, Any]:
        """並列実行統計取得"""
        gpu_status = self.multi_gpu_manager.get_gpu_status()
        
        stats = {
            "total_gpus": gpu_status["gpu_count"],
            "available_gpus": gpu_status["available_gpus"],
            "running_tasks": gpu_status["running_tasks"],
            "gpu_utilization": {},
            "memory_usage": {},
            "temperature": {},
            "power_usage": {}
        }
        
        for gpu_id, details in gpu_status["gpu_details"].items():
            stats["gpu_utilization"][gpu_id] = details["utilization_percent"]
            stats["memory_usage"][gpu_id] = {
                "total_gb": details["total_memory_gb"],
                "used_gb": details["used_memory_gb"],
                "free_gb": details["free_memory_gb"]
            }
            stats["temperature"][gpu_id] = details["temperature_celsius"]
            stats["power_usage"][gpu_id] = details["power_usage_watts"]
        
        return stats


# グローバルインスタンス
parallel_ig_calculator = ParallelIGCalculator()


def get_parallel_ig_calculator() -> ParallelIGCalculator:
    """並列IG計算システム取得"""
    return parallel_ig_calculator


def compute_parallel_mlp_ig(
    model,
    layer_idx: int,
    target_token_indices: List[int],
    target_head_indices: Optional[List[int]] = None,
    num_steps: int = 32,
    use_mixed_precision: bool = True
) -> List[Dict]:
    """並列MLP IG計算（簡易版）"""
    return parallel_ig_calculator.compute_parallel_mlp_ig(
        model, layer_idx, target_token_indices, target_head_indices, num_steps, use_mixed_precision
    )


def compute_parallel_attention_ig(
    model,
    layer_idx: int,
    target_token_indices: List[int],
    target_head_indices: Optional[List[int]] = None,
    num_steps: int = 32,
    use_mixed_precision: bool = True
) -> List[Dict]:
    """並列Attention IG計算（簡易版）"""
    return parallel_ig_calculator.compute_parallel_attention_ig(
        model, layer_idx, target_token_indices, target_head_indices, num_steps, use_mixed_precision
    )


def get_parallel_execution_stats() -> Dict[str, Any]:
    """並列実行統計取得（簡易版）"""
    return parallel_ig_calculator.get_parallel_execution_stats()
