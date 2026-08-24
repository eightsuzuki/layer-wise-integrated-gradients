"""Shared MLP u→z IG helpers for decoder adapters."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients

from lig.adapters.decoder_ig.base import check_completeness


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
        "is_valid": check_completeness(relative_error, where="MLP u->z IG"),
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


def run_mlp_head_space_ig(
    *,
    head_concat: torch.Tensor,
    baseline_head_concat: torch.Tensor,
    to_mlp_input: Callable[[torch.Tensor], torch.Tensor],
    mlp_output_fn: Callable[[torch.Tensor], torch.Tensor],
    num_steps: int,
    num_heads: int,
    head_dim: int,
    probe_w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, dict]:
    """Per-head MLP attribution taken in *head space*.

    ``run_mlp_input_ig`` attributes to the MLP input ``u = z + W_o concat_h a_h``
    and then slices ``u``'s residual coordinates into ``head_dim`` blocks.  Those
    blocks are not heads: ``W_o`` has already mixed them.  The paper defines
    ``u^(l,h)`` as the head output *before* ``W_o`` (Eq. mlp-o), so the
    attribution has to be taken with respect to ``concat_h a_h`` and only then
    summed within each head's ``head_dim`` slice.

    ``z`` is held at its actual value, matching Eq. (mlp-o) where the
    attribution index runs over ``h`` alone.

    Returns (per_head, total, verification).
    """
    if probe_w is not None:
        forward_scalar = make_probe_direction_forward(
            lambda x: mlp_output_fn(to_mlp_input(x)), probe_w
        )
        target_mode = "probe_direction"
    else:
        with torch.no_grad():
            base_out = mlp_output_fn(to_mlp_input(baseline_head_concat)).detach()

        def forward_scalar(x: torch.Tensor) -> torch.Tensor:
            out = mlp_output_fn(to_mlp_input(x))
            return torch.norm(out - base_out, dim=-1)

        target_mode = "l2_delta"

    ig = IntegratedGradients(forward_scalar)
    attributions = ig.attribute(
        inputs=head_concat.float(),
        baselines=baseline_head_concat.float(),
        n_steps=num_steps,
        return_convergence_delta=False,
    )
    attr = attributions.squeeze(0)
    per_head = attr.view(num_heads, head_dim).sum(-1)
    per_head_np = per_head.detach().cpu().numpy().astype(np.float64)
    total = float(attr.sum().detach().cpu().item())

    with torch.no_grad():
        actual = float(forward_scalar(head_concat.float()).detach().cpu().item())
        base = float(forward_scalar(baseline_head_concat.float()).detach().cpu().item())
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
        "is_valid": check_completeness(relative_error, where="MLP head-space IG"),
        "target_mode": target_mode,
        "per_head_boundary": "pre_proj",
    }
    return per_head_np, total, verification
