# Optimized IG runtime package
"""
Shared runtime components for the optimized Integrated Gradients calculator.

The modules in this package decomposed the original monolithic implementation
into smaller, testable units that mirror the theoretical formulation documented
under `theory/`.
"""

from .tasks import IGTask, generate_task_key
from .device_pool import DevicePool
from .registry import ActiveComputationRegistry
from .executors import AttentionExecutor, MLPExecutor

__all__ = [
    "IGTask",
    "generate_task_key",
    "DevicePool",
    "ActiveComputationRegistry",
    "AttentionExecutor",
    "MLPExecutor",
]
