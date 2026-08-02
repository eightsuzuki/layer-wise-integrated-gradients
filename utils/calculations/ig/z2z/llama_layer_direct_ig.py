"""Layer-whole IG (z->z) for one Llama-family decoder block."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from utils.calculations.ig.llama.block_forward import forward_llama_block, make_position_embeddings
from utils.calculations.ig.shared.release_scope import reject_otb_baseline


class LlamaLayerDirectIGWrapper(nn.Module):
    """Interpolate z^(l) at block input; scalarize ||z_j^(l+1)(a) - z_j^(l+1)(0)||_2."""

    def __init__(
        self,
        llama: nn.Module,
        layer_idx: int,
        target_token_idx: int,
        position_ids: torch.Tensor,
    ) -> None:
        super().__init__()
        self.llama = llama
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self.position_ids = position_ids
        self._baseline_output_j: Optional[torch.Tensor] = None

    def set_baseline_output(self, baseline_z: torch.Tensor) -> None:
        assert baseline_z.shape[0] == 1
        layer = self.llama.layers[self.layer_idx]
        pos_emb = make_position_embeddings(self.llama, baseline_z, self.position_ids)
        with torch.no_grad():
            out = forward_llama_block(layer, baseline_z, pos_emb)
        self._baseline_output_j = out[:, self.target_token_idx, :].clone()

    def forward(self, z_interp: torch.Tensor) -> torch.Tensor:
        if self._baseline_output_j is None:
            raise RuntimeError("call set_baseline_output() before attribute()")
        layer = self.llama.layers[self.layer_idx]
        pos_emb = make_position_embeddings(self.llama, z_interp, self.position_ids)
        out = forward_llama_block(layer, z_interp, pos_emb)
        target = out[:, self.target_token_idx, :]
        return torch.norm(target - self._baseline_output_j, p=2, dim=-1)


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


def compute_llama_layer_direct_ig_single_target(
    llama: nn.Module,
    z_layer: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
    position_ids: torch.Tensor,
    num_steps: int = 32,
    baseline_method: Literal["zero", "self_input_token"] = "zero",
) -> np.ndarray:
    baseline_z = _baseline_z(baseline_method, z_layer, target_token_idx)
    wrapper = LlamaLayerDirectIGWrapper(llama, layer_idx, target_token_idx, position_ids)
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


def compute_llama_layer_direct_ig_all_targets(
    llama: nn.Module,
    z_layer: torch.Tensor,
    layer_idx: int,
    position_ids: torch.Tensor,
    num_steps: int = 32,
    baseline_method: Literal["zero", "self_input_token"] = "zero",
) -> np.ndarray:
    seq_len = z_layer.shape[1]
    out = np.zeros((seq_len, seq_len), dtype=np.float32)
    for j in range(seq_len):
        out[:, j] = compute_llama_layer_direct_ig_single_target(
            llama=llama,
            z_layer=z_layer,
            layer_idx=layer_idx,
            target_token_idx=j,
            position_ids=position_ids,
            num_steps=num_steps,
            baseline_method=baseline_method,
        )
    return out
