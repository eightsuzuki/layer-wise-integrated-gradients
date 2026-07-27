"""Gemma 3 ``u -> z`` IG with ``u`` at the concatenated-head boundary.

For Gemma 3, ``u`` is the pre-output-projection attention result with shape
``[num_attention_heads * head_dim]``. The downstream map evaluated here is::

    u (concatenated heads)
      -> attention output projection
      -> post-attention RMSNorm
      -> + fixed block-input residual z
      -> pre-FFN RMSNorm -> MLP -> post-FFN RMSNorm
      -> + residual
      -> z_next

This keeps the ``u`` endpoint identical to the one used by attention IG and
allows its feature attributions to be grouped by the original attention heads.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from utils.calculations.ig.gemma3.block_forward import get_gemma3_text_model, mlp_branch


def _forward_gemma3_mlp_z(
    text_model: nn.Module,
    layer_idx: int,
    z_layer: torch.Tensor,
    target_token_idx: int,
    u_pre_j: torch.Tensor,
    include_residual_connection: bool,
) -> torch.Tensor:
    """
    Map concatenated attention heads ``u_pre_j`` to ``z_next_j``.

    ``z_layer`` is the fixed block-input residual stream. Only the pre-output-
    projection attention representation is interpolated by IG.
    """
    block = text_model.layers[layer_idx]
    projected = block.self_attn.o_proj(u_pre_j)
    attention_branch = block.post_attention_layernorm(projected)
    residual_z_j = z_layer[0, target_token_idx, :]
    post_attention_j = residual_z_j + attention_branch

    mlp_out = mlp_branch(block, post_attention_j)
    if include_residual_connection:
        z_next = post_attention_j + mlp_out
    else:
        z_next = mlp_out
    return z_next


class _Gemma3MLPL2IGWrapper(nn.Module):
    def __init__(
        self,
        text_model: nn.Module,
        layer_idx: int,
        target_token_idx: int,
        baseline_z: torch.Tensor,
        include_residual_connection: bool = True,
    ) -> None:
        super().__init__()
        self.text_model = text_model
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self.baseline_z = baseline_z
        self.include_residual_connection = include_residual_connection
        self._z_context: torch.Tensor | None = None

    def set_context(self, z_layer: torch.Tensor) -> None:
        self._z_context = z_layer

    def forward(self, u_pre_j: torch.Tensor) -> torch.Tensor:
        if self._z_context is None:
            raise RuntimeError("call set_context(z_layer) first")
        if u_pre_j.dim() == 1:
            u_pre_j = u_pre_j.unsqueeze(0)
        z = torch.stack(
            [
                _forward_gemma3_mlp_z(
                    self.text_model,
                    self.layer_idx,
                    self._z_context,
                    self.target_token_idx,
                    u_pre_j[b],
                    self.include_residual_connection,
                )
                for b in range(u_pre_j.shape[0])
            ],
            dim=0,
        )
        if self.baseline_z.device != z.device:
            self.baseline_z = self.baseline_z.to(z.device)
        return torch.norm(z - self.baseline_z.unsqueeze(0), p=2, dim=-1)


def compute_gemma3_mlp_lig_single_token(
    text_model: nn.Module,
    *,
    layer_idx: int,
    z_layer: torch.Tensor,
    target_mlp_input: torch.Tensor,
    baseline_mlp_input: torch.Tensor,
    target_token_idx: int,
    num_steps: int = 32,
    include_residual_connection: bool = True,
) -> Dict:
    """
    Gemma3 boundary: concatenated heads ``u`` -> ``z_j^(l+1)``.

    ``target_mlp_input`` and ``baseline_mlp_input`` have width
    ``num_attention_heads * head_dim``. The output projection, post-attention
    RMSNorm, fixed ``z_layer`` residual, and feed-forward branch are all inside
    the differentiated mapping.
    """
    text_model = get_gemma3_text_model(text_model)
    if target_mlp_input.dim() == 1:
        target_mlp_input = target_mlp_input.unsqueeze(0)
    if baseline_mlp_input.dim() == 1:
        baseline_mlp_input = baseline_mlp_input.unsqueeze(0)
    cfg = text_model.config
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    expected_width = int(cfg.num_attention_heads) * head_dim
    if target_mlp_input.shape[-1] != expected_width:
        raise ValueError(
            "Gemma3 u must be the concatenated-head tensor before the attention "
            f"output projection: expected width {expected_width}, got "
            f"{target_mlp_input.shape[-1]}."
        )
    if baseline_mlp_input.shape != target_mlp_input.shape:
        raise ValueError(
            "Gemma3 u baseline must match the concatenated-head input shape: "
            f"{tuple(baseline_mlp_input.shape)} != {tuple(target_mlp_input.shape)}."
        )

    with torch.no_grad():
        baseline_z = _forward_gemma3_mlp_z(
            text_model,
            layer_idx,
            z_layer,
            target_token_idx,
            baseline_mlp_input.squeeze(0),
            include_residual_connection,
        ).detach()
        actual_z = _forward_gemma3_mlp_z(
            text_model,
            layer_idx,
            z_layer,
            target_token_idx,
            target_mlp_input.squeeze(0),
            include_residual_connection,
        ).detach()

    wrapper = _Gemma3MLPL2IGWrapper(
        text_model,
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
        "input_width": expected_width,
        "l2_total": l2_target,
        "completeness_error": completeness_error,
        "mean_abs_completeness_error": abs(completeness_error),
        "max_abs_completeness_error": abs(completeness_error),
    }
