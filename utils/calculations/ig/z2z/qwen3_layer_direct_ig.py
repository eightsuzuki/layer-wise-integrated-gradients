"""Layer-whole IG (z -> z') for one dense Qwen3 decoder block."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from utils.calculations.ig.qwen3.block_forward import (
    create_qwen3_causal_mask,
    forward_qwen3_block,
    get_qwen3_model,
    qwen3_position_embeddings,
)
from utils.calculations.ig.shared.release_scope import reject_otb_baseline


class Qwen3LayerDirectIGWrapper(nn.Module):
    """Interpolate block input z and scalarize the target-token z' delta."""

    def __init__(
        self,
        qwen3: nn.Module,
        layer_idx: int,
        target_token_idx: int,
        position_ids: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.qwen3 = get_qwen3_model(qwen3)
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self.position_ids = position_ids
        self.attention_mask_2d = attention_mask_2d
        self._baseline_output_j: Optional[torch.Tensor] = None

    def _forward_block(self, z: torch.Tensor) -> torch.Tensor:
        mask = create_qwen3_causal_mask(
            z.shape[1], z.device, z.dtype, self.attention_mask_2d
        )
        position_embeddings = qwen3_position_embeddings(
            self.qwen3, z, self.position_ids
        )
        return forward_qwen3_block(
            self.qwen3.layers[self.layer_idx], z, position_embeddings, mask
        )

    def set_baseline_output(self, baseline_z: torch.Tensor) -> None:
        with torch.no_grad():
            out = self._forward_block(baseline_z)
        self._baseline_output_j = out[:, self.target_token_idx, :].clone()

    def forward(self, z_interp: torch.Tensor) -> torch.Tensor:
        if self._baseline_output_j is None:
            raise RuntimeError("call set_baseline_output() before attribute()")
        out = self._forward_block(z_interp)
        target = out[:, self.target_token_idx, :]
        return torch.norm(target - self._baseline_output_j, p=2, dim=-1)


def _baseline_z(
    method: Literal["zero", "self_input_token"],
    z_layer: torch.Tensor,
    target_token_idx: int,
) -> torch.Tensor:
    reject_otb_baseline(method)  # type: ignore[arg-type]
    if method == "zero":
        return torch.zeros_like(z_layer)
    if method == "self_input_token":
        z_j = z_layer[:, target_token_idx, :].clone()
        return z_j.unsqueeze(1).expand_as(z_layer).clone()
    raise ValueError(f"Unknown baseline_method: {method}")


def compute_qwen3_layer_direct_ig_single_target(
    qwen3: nn.Module,
    z_layer: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_steps: int = 32,
    baseline_method: Literal["zero", "self_input_token"] = "zero",
) -> np.ndarray:
    baseline_z = _baseline_z(baseline_method, z_layer, target_token_idx)
    wrapper = Qwen3LayerDirectIGWrapper(
        qwen3,
        layer_idx,
        target_token_idx,
        position_ids,
        attention_mask,
    )
    wrapper.eval()
    wrapper.set_baseline_output(baseline_z)
    ig = IntegratedGradients(wrapper)
    attr = ig.attribute(
        inputs=z_layer.float(),
        baselines=baseline_z.float(),
        n_steps=num_steps,
        method="riemann_trapezoid",
    )
    return attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()


def compute_qwen3_layer_direct_ig_all_targets(
    qwen3: nn.Module,
    z_layer: torch.Tensor,
    layer_idx: int,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_steps: int = 32,
    baseline_method: Literal["zero", "self_input_token"] = "zero",
) -> np.ndarray:
    seq_len = z_layer.shape[1]
    out = np.zeros((seq_len, seq_len), dtype=np.float32)
    for target_idx in range(seq_len):
        out[:, target_idx] = compute_qwen3_layer_direct_ig_single_target(
            qwen3=qwen3,
            z_layer=z_layer,
            layer_idx=layer_idx,
            target_token_idx=target_idx,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_steps=num_steps,
            baseline_method=baseline_method,
        )
    return out
