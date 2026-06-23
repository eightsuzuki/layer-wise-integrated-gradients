from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IGTask:
    """
    Atomic IG calculation unit passed to the optimized runtime.

    Attributes:
        task_id: External identifier supplied by the scheduler.
        task_type: Either `"attention"` or `"mlp"`.
        layer_idx: Layer index targeted by the task.
        token_idx: Token index used for the computation.
        head_idx: Optional head index (None for MLP final tasks).
        num_steps: Number of integration steps.
        priority: Scheduling hint (higher -> earlier execution).
    """

    task_id: str
    task_type: str
    layer_idx: int
    token_idx: int
    head_idx: Optional[int]
    num_steps: int
    priority: int = 0


def generate_task_key(task: IGTask, text: str) -> str:
    """
    Compose the cache key for a task following the theoretical definition:

    R^{(l,h)}_{i -> j} is uniquely determined by the layer, token, head and
    integration steps for a specific input text.
    """

    head_idx = task.head_idx if task.head_idx is not None else 0
    return (
        f"{task.task_type}_{task.layer_idx}_{task.token_idx}_{head_idx}_"
        f"{task.num_steps}_{hash(text)}"
    )
