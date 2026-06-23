# resource_planner.py
"""
GPUリソースに基づいたIGバッチサイズの動的計画

固定値ではなく、実際に利用可能なGPUメモリ・SM数から
最適なバッチサイズを見積もる。
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def _get_threads_per_sm(props) -> int:
    # torchのプロパティによって名称が異なる場合がある
    threads = getattr(props, "max_threads_per_multi_processor", None)
    if threads is None:
        threads = getattr(props, "max_threads_per_sm", 2048)
    return max(int(threads), 1)


def plan_gpu_batch_size(
    total_tasks: int,
    *,
    base_batch_size: int = 8,
    use_mixed_precision: bool = False,
    memory_safety_margin: float = 0.9,
) -> int:
    """
    利用可能なGPUリソースを用いて最適なバッチサイズを推定する。

    Args:
        total_tasks: 実行すべきタスク数
        base_batch_size: 最小のベースバッチサイズ
        use_mixed_precision: 混合精度を使用する場合はTrue
        memory_safety_margin: どれくらいメモリに余裕を持つか（0-1）
    """
    total_tasks = max(int(total_tasks), 0)
    if total_tasks == 0:
        return 0

    per_task_memory_mb = 50 if use_mixed_precision else 80
    min_batch = max(base_batch_size, 4)

    if not torch.cuda.is_available():
        return min(total_tasks, min_batch)

    original_device = torch.cuda.current_device()
    device_count = torch.cuda.device_count()

    aggregate_concurrency = 0
    memory_task_limits: list[int] = []

    try:
        for device_idx in range(device_count):
            torch.cuda.set_device(device_idx)
            props = torch.cuda.get_device_properties(device_idx)

            sm_count = props.multi_processor_count
            threads_per_sm = _get_threads_per_sm(props)

            # タスクをできるだけ多く載せたいので、warp単位よりさらに粗い粒度で見積もる
            concurrent_tasks_per_device = sm_count * max(threads_per_sm // 8, 64)
            aggregate_concurrency += concurrent_tasks_per_device

            total_memory_gb = props.total_memory / 1024**3
            reserved_gb = torch.cuda.memory_reserved(device_idx) / 1024**3
            allocated_gb = torch.cuda.memory_allocated(device_idx) / 1024**3
            available_gb = max(total_memory_gb - max(reserved_gb, allocated_gb), 0.0)
            available_gb *= memory_safety_margin

            mem_limit = int((available_gb * 1024) / per_task_memory_mb)
            if mem_limit > 0:
                memory_task_limits.append(mem_limit)

            logger.debug(
                "device=%d name=%s SM=%d threads/SM=%d available=%.2fGB concurrent=%d mem_limit=%d",
                device_idx,
                props.name,
                sm_count,
                threads_per_sm,
                available_gb,
                concurrent_tasks_per_device,
                mem_limit,
            )
    finally:
        torch.cuda.set_device(original_device)

    if not memory_task_limits:
        memory_task_limit = total_tasks
    else:
        memory_task_limit = max(1, min(memory_task_limits))

    aggregate_concurrency = max(aggregate_concurrency, min_batch)

    suggested = min(total_tasks, aggregate_concurrency, memory_task_limit)

    # MultiGPU の場合 aggregate_concurrency が極端に大きくなる可能性があるので安全に上限を設ける
    max_cap = max(min(total_tasks, aggregate_concurrency), 2048)
    suggested = min(suggested, max_cap)

    if suggested < min_batch:
        suggested = min(total_tasks, min_batch)

    logger.debug(
        "plan_gpu_batch_size -> total=%d, concurrency=%d, mem_limit=%d, suggested=%d",
        total_tasks,
        aggregate_concurrency,
        memory_task_limit,
        suggested,
    )

    return suggested
