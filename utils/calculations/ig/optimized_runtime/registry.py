from __future__ import annotations

import threading
from typing import Set


class ActiveComputationRegistry:
    """
    Thread-safe registry tracking in-flight IG computations.

    The registry prevents duplicated work when multiple workers request the same
    (layer, token, head) combination for an identical input sentence. The
    original implementation stored this inline inside the calculator; it is now
    extracted for clarity.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: Set[str] = set()

    def is_active(self, key: str) -> bool:
        with self._lock:
            return key in self._active

    def start(self, key: str) -> None:
        with self._lock:
            self._active.add(key)

    def finish(self, key: str) -> None:
        with self._lock:
            self._active.discard(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._active)
