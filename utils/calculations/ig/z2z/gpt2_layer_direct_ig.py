"""Layer-whole IG (z->z) for one GPT-2 block."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from transformers import GPT2Model

from utils.calculations.ig.gpt2.block_forward import create_causal_mask, forward_gpt2_block
from utils.calculations.ig.shared.release_scope import reject_otb_baseline


class GPT2LayerDirectIGWrapper(nn.Module):
    """Interpolate z^(l) at block input; scalarize ||z_j^(l+1)(a) - z_j^(l+1)(0)||_2."""

    def __init__(
        self,
        gpt2: GPT2Model,
        layer_idx: int,
        target_token_idx: int,
    ) -> None:
        super().__init__()
        self.gpt2 = gpt2
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self._baseline_output_j: Optional[torch.Tensor] = None
        self._attention_mask: Optional[torch.Tensor] = None

    def set_baseline_output(self, baseline_z: torch.Tensor) -> None:
        assert baseline_z.shape[0] == 1
        seq_len = baseline_z.shape[1]
        mask = create_causal_mask(seq_len, baseline_z.device, baseline_z.dtype)
        self._attention_mask = mask
        block = self.gpt2.h[self.layer_idx]
        with torch.no_grad():
            out = forward_gpt2_block(block, baseline_z, mask)
        self._baseline_output_j = out[:, self.target_token_idx, :].clone()

    def forward(self, z_interp: torch.Tensor) -> torch.Tensor:
        if self._baseline_output_j is None or self._attention_mask is None:
            raise RuntimeError("call set_baseline_output() before attribute()")
        block = self.gpt2.h[self.layer_idx]
        out = forward_gpt2_block(block, z_interp, self._attention_mask)
        target = out[:, self.target_token_idx, :]
        diff = target - self._baseline_output_j
        return torch.norm(diff, p=2, dim=-1)


def _baseline_z(
    method: Literal["zero", "self_input_token"],
    z_layer: torch.Tensor,
    gpt2: GPT2Model,
    layer_idx: int,
    target_token_idx: int,
) -> torch.Tensor:
    reject_otb_baseline(method)  # type: ignore[arg-type]
    _, seq_len, hidden = z_layer.shape
    device, dtype = z_layer.device, z_layer.dtype
    if method == "zero":
        return torch.zeros(1, seq_len, hidden, device=device, dtype=dtype)
    if method == "self_input_token":
        z_j = z_layer[0, target_token_idx, :].clone()
        return z_j.unsqueeze(0).unsqueeze(0).expand(1, seq_len, hidden)
    raise ValueError(f"Unknown baseline_method: {method}")


def compute_gpt2_layer_direct_ig_single_target(
    gpt2: GPT2Model,
    z_layer: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
    num_steps: int = 32,
    baseline_method: Literal["zero", "self_input_token"] = "zero",
) -> np.ndarray:
    baseline_z = _baseline_z(
        baseline_method, z_layer, gpt2, layer_idx, target_token_idx
    )
    wrapper = GPT2LayerDirectIGWrapper(gpt2, layer_idx, target_token_idx)
    wrapper.eval()
    wrapper.set_baseline_output(baseline_z)
    ig = IntegratedGradients(wrapper)
    attr = ig.attribute(
        inputs=z_layer,
        baselines=baseline_z,
        n_steps=num_steps,
        method="riemann_trapezoid",
    )
    return attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()


def compute_gpt2_layer_direct_ig_all_targets(
    gpt2: GPT2Model,
    z_layer: torch.Tensor,
    layer_idx: int,
    num_steps: int = 32,
    baseline_method: Literal["zero", "self_input_token"] = "zero",
) -> np.ndarray:
    seq_len = z_layer.shape[1]
    out = np.zeros((seq_len, seq_len), dtype=np.float32)
    for j in range(seq_len):
        out[:, j] = compute_gpt2_layer_direct_ig_single_target(
            gpt2=gpt2,
            z_layer=z_layer,
            layer_idx=layer_idx,
            target_token_idx=j,
            num_steps=num_steps,
            baseline_method=baseline_method,
        )
    return out
