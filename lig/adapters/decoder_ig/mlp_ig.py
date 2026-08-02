"""Shared MLP u→z IG helpers for decoder adapters."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients


def run_mlp_input_ig(
    *,
    u: torch.Tensor,
    baseline_u: torch.Tensor,
    forward_scalar: Callable[[torch.Tensor], torch.Tensor],
    num_steps: int,
    num_heads: int,
    head_dim: int,
    target_mode: str = "l2_delta",
) -> Tuple[np.ndarray, float, np.ndarray, dict]:
    """
    Integrated Gradients on MLP input u with a scalar forward target.

    Returns (contributions, total, per_head, verification).
    """
    ig = IntegratedGradients(forward_scalar)
    attributions = ig.attribute(
        inputs=u.float(),
        baselines=baseline_u.float(),
        n_steps=num_steps,
        return_convergence_delta=False,
    )
    attr = attributions.squeeze(0)
    per_head = []
    for h_idx in range(num_heads):
        start = h_idx * head_dim
        end = start + head_dim
        per_head.append(float(attr[start:end].sum().detach().cpu().item()))
    per_head_np = np.asarray(per_head, dtype=np.float64)
    total = float(attr.sum().detach().cpu().item())

    with torch.no_grad():
        actual = float(forward_scalar(u.float()).detach().cpu().item())
        base = float(forward_scalar(baseline_u.float()).detach().cpu().item())
    theoretical_diff = actual - base
    relative_error = (
        abs(total - theoretical_diff) / abs(theoretical_diff)
        if abs(theoretical_diff) > 1e-8
        else abs(total - theoretical_diff)
    )
    verification = {
        "theoretical_diff": theoretical_diff,
        "ig_sum": total,
        "relative_error": float(relative_error),
        "is_valid": bool(relative_error < 0.2),
        "target_mode": target_mode,
        "baseline_u_norm": float(torch.norm(baseline_u).detach().cpu().item()),
        "target_u_norm": float(torch.norm(u).detach().cpu().item()),
        "input_delta_norm": float(torch.norm(u - baseline_u).detach().cpu().item()),
    }
    return (
        np.asarray(attr.detach().cpu().numpy(), dtype=np.float64),
        total,
        per_head_np,
        verification,
    )


def make_probe_direction_forward(
    mlp_output_fn: Callable[[torch.Tensor], torch.Tensor],
    probe_w: np.ndarray,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Scalar target f(u) = w · z^(l+1)(u) for probe-direction IG."""
    w = None

    def forward_fn(u_interp: torch.Tensor) -> torch.Tensor:
        nonlocal w
        if w is None:
            w = torch.as_tensor(probe_w, dtype=torch.float32, device=u_interp.device)
        z_next = mlp_output_fn(u_interp)
        return (z_next * w).sum(dim=-1)

    return forward_fn
