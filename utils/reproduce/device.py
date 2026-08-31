"""Pick a CUDA device without evicting someone else's job.

These scripts are meant to be run on shared lab machines. A bare
``torch.device("cuda")`` always lands on card 0, so two people starting a run
at the same time collide there while the other cards sit idle. ``resolve``
defaults to the card with the most free memory and refuses to start on one that
is already nearly full, which is the behaviour you want when the machine is not
yours alone.
"""

from __future__ import annotations

import sys

import torch

# BERT-base IG with the default settings peaks near 3GB; keep some slack so we
# do not squeeze in beside a job that is about to grow.
MIN_FREE_BYTES = 4 * 1024**3


def resolve(requested: str = "auto", *, min_free_bytes: int = MIN_FREE_BYTES) -> torch.device:
    """Resolve a --device value to a concrete torch.device.

    ``auto``      pick the CUDA card with the most free memory, else CPU
    ``cuda:N``    use that card, but say so loudly if it is already busy
    ``cpu``       force CPU
    """
    if requested == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if requested != "auto":
            print(f"CUDA not available; ignoring --device {requested}", file=sys.stderr)
        return torch.device("cpu")

    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda":
            index = device.index if device.index is not None else 0
            free, total = torch.cuda.mem_get_info(index)
            if free < min_free_bytes:
                print(
                    f"warning: cuda:{index} has only {free / 1024**3:.1f}GB of "
                    f"{total / 1024**3:.1f}GB free. Someone else may be using it; "
                    f"consider --device auto.",
                    file=sys.stderr,
                )
        return device

    best_index, best_free = None, 0
    for index in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(index)
        if free > best_free:
            best_index, best_free = index, free

    if best_index is None or best_free < min_free_bytes:
        print(
            f"No CUDA device with {min_free_bytes / 1024**3:.0f}GB free "
            f"(most free: {best_free / 1024**3:.1f}GB); falling back to CPU. "
            f"Pass --device cuda:N to override.",
            file=sys.stderr,
        )
        return torch.device("cpu")

    print(
        f"Using cuda:{best_index} ({best_free / 1024**3:.1f}GB free). "
        f"Pass --device cuda:N to pin a different card.",
        file=sys.stderr,
    )
    return torch.device(f"cuda:{best_index}")
