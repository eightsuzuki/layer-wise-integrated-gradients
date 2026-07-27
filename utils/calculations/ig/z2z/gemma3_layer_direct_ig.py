"""Layer-whole IG (z -> z) for one Gemma 3 pre+post-LN block."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from utils.calculations.ig.gemma3.block_forward import (
    forward_gemma3_block,
    get_gemma3_text_model,
    layer_attention_mask,
    rope_position_embeddings,
)
from utils.calculations.ig.shared.release_scope import reject_otb_baseline


class Gemma3LayerDirectIGWrapper(nn.Module):
    """Interpolate ``z^(l)``; scalarize ``||z_j^(l+1)(a) - z_j^(l+1)(0)||_2``."""

    def __init__(
        self,
        text_model: nn.Module,
        layer_idx: int,
        target_token_idx: int,
    ) -> None:
        super().__init__()
        self.text_model = get_gemma3_text_model(text_model)
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self._baseline_output_j: Optional[torch.Tensor] = None
        self._attention_mask: Optional[torch.Tensor] = None
        self._global_pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._local_pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def set_baseline_output(self, baseline_z: torch.Tensor) -> None:
        assert baseline_z.shape[0] == 1
        seq_len = baseline_z.shape[1]
        mask = layer_attention_mask(
            self.text_model, self.layer_idx, seq_len, baseline_z.device, baseline_z.dtype
        )
        self._attention_mask = mask
        global_pe, local_pe = rope_position_embeddings(self.text_model, baseline_z)
        self._global_pe = global_pe
        self._local_pe = local_pe
        block = self.text_model.layers[self.layer_idx]
        with torch.no_grad():
            out = forward_gemma3_block(block, baseline_z, mask, global_pe, local_pe)
        self._baseline_output_j = out[:, self.target_token_idx, :].clone()

    def forward(self, z_interp: torch.Tensor) -> torch.Tensor:
        if (
            self._baseline_output_j is None
            or self._attention_mask is None
            or self._global_pe is None
            or self._local_pe is None
        ):
            raise RuntimeError("call set_baseline_output() before attribute()")
        # Recompute RoPE for interpolated activations (same position_ids).
        global_pe, local_pe = rope_position_embeddings(self.text_model, z_interp)
        block = self.text_model.layers[self.layer_idx]
        out = forward_gemma3_block(
            block, z_interp, self._attention_mask, global_pe, local_pe
        )
        target = out[:, self.target_token_idx, :]
        diff = target - self._baseline_output_j
        return torch.norm(diff, p=2, dim=-1)


def _baseline_z(
    method: Literal["zero", "self_input_token"],
    z_layer: torch.Tensor,
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


def compute_gemma3_layer_direct_ig_single_target(
    text_model: nn.Module,
    z_layer: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
    num_steps: int = 32,
    baseline_method: Literal["zero", "self_input_token"] = "zero",
) -> np.ndarray:
    baseline_z = _baseline_z(baseline_method, z_layer, target_token_idx)
    wrapper = Gemma3LayerDirectIGWrapper(text_model, layer_idx, target_token_idx)
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


def compute_gemma3_layer_direct_ig_all_targets(
    text_model: nn.Module,
    z_layer: torch.Tensor,
    layer_idx: int,
    num_steps: int = 32,
    baseline_method: Literal["zero", "self_input_token"] = "zero",
) -> np.ndarray:
    seq_len = z_layer.shape[1]
    out = np.zeros((seq_len, seq_len), dtype=np.float32)
    for j in range(seq_len):
        out[:, j] = compute_gemma3_layer_direct_ig_single_target(
            text_model=text_model,
            z_layer=z_layer,
            layer_idx=layer_idx,
            target_token_idx=j,
            num_steps=num_steps,
            baseline_method=baseline_method,
        )
    return out
