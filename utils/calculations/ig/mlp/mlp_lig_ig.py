"""MLP Integrated Gradients (u -> z) with L2 scalarization (LIG)."""

from __future__ import annotations

from typing import Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from .mlp_ig import _apply_layernorm_with_optional_residual


def _get_encoder_layer(model_mlp: L.LightningModule, layer_idx: int):
    if hasattr(model_mlp, "bert"):
        return model_mlp.bert.encoder.layer[layer_idx]
    return model_mlp.encoder.layer[layer_idx]


def _forward_mlp_z(
    model_mlp: L.LightningModule,
    layer_idx: int,
    u: torch.Tensor,
    include_residual_connection: bool,
) -> torch.Tensor:
    """One-token MLP input u [hidden] -> z [hidden] after LayerNorm."""
    u_exp = u.reshape(1, 1, -1)
    encoder_layer = _get_encoder_layer(model_mlp, layer_idx)
    inter = encoder_layer.intermediate.dense(u_exp)
    inter = F.gelu(inter)
    mlp_out = encoder_layer.output.dense(inter)
    ln_out = _apply_layernorm_with_optional_residual(
        encoder_layer=encoder_layer,
        mlp_input=u_exp,
        mlp_output=mlp_out,
        include_residual_connection=include_residual_connection,
    )
    return ln_out[0, 0, :]


class _MLPL2IGWrapper(nn.Module):
    """u [batch, hidden] -> ||z(u) - z(baseline)||_2."""

    def __init__(
        self,
        model_mlp: L.LightningModule,
        layer_idx: int,
        baseline_z: torch.Tensor,
        include_residual_connection: bool = True,
    ) -> None:
        super().__init__()
        self.model_mlp = model_mlp
        self.layer_idx = layer_idx
        self.baseline_z = baseline_z
        self.include_residual_connection = include_residual_connection

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.dim() == 1:
            u = u.unsqueeze(0)
        z = torch.stack(
            [
                _forward_mlp_z(
                    self.model_mlp,
                    self.layer_idx,
                    u[b],
                    self.include_residual_connection,
                )
                for b in range(u.shape[0])
            ],
            dim=0,
        )
        if self.baseline_z.device != z.device:
            self.baseline_z = self.baseline_z.to(z.device)
        return torch.norm(z - self.baseline_z.unsqueeze(0), p=2, dim=-1)


def compute_mlp_lig_single_token(
    model_mlp: L.LightningModule,
    layer_idx: int,
    target_mlp_input: torch.Tensor,
    baseline_mlp_input: torch.Tensor,
    num_steps: int = 32,
    include_residual_connection: bool = True,
) -> Dict:
    """
    MLP (u -> z) IG with L2 scalarization for one target token.

    Returns per-dimension attributions on u (length hidden) whose sum equals
    ||z(u) - z(baseline)||_2.
    """
    device = next(model_mlp.parameters()).device
    dtype = next(model_mlp.parameters()).dtype
    target_mlp_input = target_mlp_input.to(device=device, dtype=dtype).reshape(1, -1)
    baseline_mlp_input = baseline_mlp_input.to(device=device, dtype=dtype).reshape(1, -1)

    with torch.no_grad():
        baseline_z = _forward_mlp_z(
            model_mlp, layer_idx, baseline_mlp_input.squeeze(0), include_residual_connection
        ).detach()
        actual_z = _forward_mlp_z(
            model_mlp, layer_idx, target_mlp_input.squeeze(0), include_residual_connection
        ).detach()

    need_fp32 = dtype in (torch.bfloat16, torch.float16)
    t_in = target_mlp_input.float() if need_fp32 else target_mlp_input
    b_in = baseline_mlp_input.float() if need_fp32 else baseline_mlp_input

    wrapper = _MLPL2IGWrapper(
        model_mlp=model_mlp,
        layer_idx=layer_idx,
        baseline_z=baseline_z.float() if need_fp32 else baseline_z,
        include_residual_connection=include_residual_connection,
    )
    attr = IntegratedGradients(wrapper).attribute(
        inputs=t_in,
        baselines=b_in,
        n_steps=num_steps,
        method="riemann_trapezoid",
    )
    if attr is None:
        raise ValueError("MLP LIG attribution failed")
    contributions = attr.squeeze(0).detach().float().cpu().tolist()
    l2_target = torch.norm(actual_z.float() - baseline_z.float(), p=2).item()
    l2_reconstructed = float(sum(contributions))
    completeness_error = l2_reconstructed - l2_target

    return {
        "layer_idx": layer_idx,
        "contributions": contributions,
        "l2_total": l2_target,
        "l2_reconstructed": l2_reconstructed,
        "completeness_error": completeness_error,
        "mean_abs_completeness_error": abs(completeness_error),
    }
