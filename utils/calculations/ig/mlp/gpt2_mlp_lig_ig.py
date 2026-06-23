"""MLP IG (u -> z) for GPT-2 with L2 scalarization (LIG)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from transformers import GPT2Model

from utils.calculations.ig.gpt2.block_forward import create_causal_mask


def _forward_gpt2_mlp_z(
    gpt2: GPT2Model,
    layer_idx: int,
    z_layer: torch.Tensor,
    target_token_idx: int,
    u_j: torch.Tensor,
    include_residual_connection: bool,
) -> torch.Tensor:
    hidden = z_layer.expand(1, -1, -1).clone()
    hidden[0, target_token_idx, :] = u_j
    mlp_in = gpt2.h[layer_idx].ln_2(hidden)
    mlp_out = gpt2.h[layer_idx].mlp(mlp_in)
    if include_residual_connection:
        z_next = hidden + mlp_out
    else:
        z_next = mlp_out
    return z_next[0, target_token_idx, :]


class _GPT2MLPL2IGWrapper(nn.Module):
    def __init__(
        self,
        gpt2: GPT2Model,
        layer_idx: int,
        target_token_idx: int,
        baseline_z: torch.Tensor,
        include_residual_connection: bool = True,
    ) -> None:
        super().__init__()
        self.gpt2 = gpt2
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self.baseline_z = baseline_z
        self.include_residual_connection = include_residual_connection
        self._z_context: torch.Tensor | None = None

    def set_context(self, z_layer: torch.Tensor) -> None:
        self._z_context = z_layer

    def forward(self, u_j: torch.Tensor) -> torch.Tensor:
        if self._z_context is None:
            raise RuntimeError("call set_context(z_layer) first")
        if u_j.dim() == 1:
            u_j = u_j.unsqueeze(0)
        z = torch.stack(
            [
                _forward_gpt2_mlp_z(
                    self.gpt2,
                    self.layer_idx,
                    self._z_context,
                    self.target_token_idx,
                    u_j[b],
                    self.include_residual_connection,
                )
                for b in range(u_j.shape[0])
            ],
            dim=0,
        )
        if self.baseline_z.device != z.device:
            self.baseline_z = self.baseline_z.to(z.device)
        return torch.norm(z - self.baseline_z.unsqueeze(0), p=2, dim=-1)


def compute_gpt2_mlp_lig_single_token(
    gpt2: GPT2Model,
    *,
    layer_idx: int,
    z_layer: torch.Tensor,
    target_mlp_input: torch.Tensor,
    baseline_mlp_input: torch.Tensor,
    target_token_idx: int,
    num_steps: int = 32,
    include_residual_connection: bool = True,
) -> Dict:
    """GPT-2 MLP boundary: post-attn u at j -> z_j^(l+1), L2-scalarized IG."""
    if target_mlp_input.dim() == 1:
        target_mlp_input = target_mlp_input.unsqueeze(0)
    if baseline_mlp_input.dim() == 1:
        baseline_mlp_input = baseline_mlp_input.unsqueeze(0)

    with torch.no_grad():
        baseline_z = _forward_gpt2_mlp_z(
            gpt2,
            layer_idx,
            z_layer,
            target_token_idx,
            baseline_mlp_input.squeeze(0),
            include_residual_connection,
        ).detach()
        actual_z = _forward_gpt2_mlp_z(
            gpt2,
            layer_idx,
            z_layer,
            target_token_idx,
            target_mlp_input.squeeze(0),
            include_residual_connection,
        ).detach()

    wrapper = _GPT2MLPL2IGWrapper(
        gpt2,
        layer_idx,
        target_token_idx,
        baseline_z,
        include_residual_connection=include_residual_connection,
    )
    wrapper.set_context(z_layer)
    attr = IntegratedGradients(wrapper).attribute(
        inputs=target_mlp_input,
        baselines=baseline_mlp_input,
        n_steps=num_steps,
        method="riemann_trapezoid",
    )
    contributions = attr.squeeze(0).detach().cpu().tolist()
    l2_target = torch.norm(actual_z - baseline_z, p=2).item()
    l2_reconstructed = float(sum(contributions))
    completeness_error = l2_reconstructed - l2_target

    return {
        "contributions": contributions,
        "l2_total": l2_target,
        "completeness_error": completeness_error,
        "mean_abs_completeness_error": abs(completeness_error),
        "max_abs_completeness_error": abs(completeness_error),
    }
