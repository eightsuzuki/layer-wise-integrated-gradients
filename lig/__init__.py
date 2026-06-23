"""
Layer-wise Integrated Gradients (LIG) — public API.

Quick start::

    from lig import explain
    result = explain("The cat sat on the mat.", model="bert-base-uncased")
    # result is a JSON-serializable dict
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["explain", "describe_boundaries", "LIGConfig", "__version__"]
__version__ = "0.1.0"

if TYPE_CHECKING:
    from lig.api import describe_boundaries as describe_boundaries
    from lig.api import explain as explain
    from lig.config import LIGConfig as LIGConfig


def __getattr__(name: str):
    """Lazy imports so optional tooling does not require torch at import time."""
    if name == "explain":
        from lig.api import explain

        return explain
    if name == "describe_boundaries":
        from lig.api import describe_boundaries

        return describe_boundaries
    if name == "LIGConfig":
        from lig.config import LIGConfig

        return LIGConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
