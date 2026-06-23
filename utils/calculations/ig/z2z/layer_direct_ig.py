"""
Layer (一気通貫) Integrated Gradients

理論:
  層入力 z^{(l)} を補間し、層全体の forward で z^{(l+1)} を求め、
  スカラー L_j^{(l)}(a) = ||z_j^{(l+1)}(a) - z_j^{(l+1)}(0)||_2 に対する IG
    IG_{i,j}^{Layer} = (z_i^{(l)} - z_i^{(l)}(0)) · ∫_0^1 ∂L_j^{(l)}(a) / ∂z_i^{(l)} da
  を計算する。

実装上の注意:
  Captum の IntegratedGradients は補間点を baseline + α*(input - baseline) の
  全 α を dim=0 で連結した1バッチで forward に渡す。そのため forward 内で
  「最初の呼び出し＝baseline」とみなして _baseline_output_j を保存すると、
  実際には「全ステップの出力」が保存され diff=0 となり結果が全て 0 になる。
  正しくは ig.attribute() の前に set_baseline_output(baseline_z) で
  z_j^{(l+1)}(0) を [1, hidden] で1回だけ計算・保存すること。
"""

from typing import Literal, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from utils.calculations.ig.shared.encoder_access import (
    forward_encoder_layer,
    get_encoder_layers,
)
from utils.calculations.ig.shared.release_scope import reject_otb_baseline
from utils.calculations.shared.device_utils import ensure_model_on_device


class LayerDirectIGWrapper(nn.Module):
    """
    層入力 z^{(l)}(a) を受け取り、1 層の forward で z^{(l+1)}(a) を計算し、
    ターゲット j について L_j(a) = ||z_j^{(l+1)}(a) - z_j^{(l+1)}(0)||_2 を返す。
    ベースライン出力 z_j^{(l+1)}(0) は set_baseline_output() で事前に設定すること。
    """

    def __init__(
        self,
        bert_model: L.LightningModule,
        layer_idx: int,
        target_token_idx: int,
        attention_mask: torch.Tensor,
    ):
        super().__init__()
        self.bert_model = bert_model
        self.layer_idx = layer_idx
        self.target_token_idx = target_token_idx
        self.attention_mask = attention_mask
        self._baseline_output_j: Optional[torch.Tensor] = None  # [1, hidden]

    def set_target_token_idx(self, target_token_idx: int) -> None:
        self.target_token_idx = target_token_idx

    def set_baseline_output_from_tensor(self, baseline_output_j: torch.Tensor) -> None:
        """z_j^{(l+1)}(0) [1, hidden] を直接設定（zero baseline の一括 forward 用）。"""
        self._baseline_output_j = baseline_output_j.clone()

    def set_baseline_output(self, baseline_z: torch.Tensor) -> None:
        """
        ベースライン入力 z^{(l)}(0) [1, seq_len, hidden] で1回 forward し、
        z_j^{(l+1)}(0) を保存する。ig.attribute() の前に必ず1回呼ぶこと。
        """
        assert baseline_z.shape[0] == 1
        layer = get_encoder_layers(self.bert_model)[self.layer_idx]
        with torch.no_grad():
            layer_output = forward_encoder_layer(
                self.bert_model,
                layer,
                baseline_z,
                self.attention_mask,
                layer_idx=self.layer_idx,
            )
        self._baseline_output_j = layer_output[:, self.target_token_idx, :].clone()

    def forward(self, z_interp: torch.Tensor) -> torch.Tensor:
        if self._baseline_output_j is None:
            raise RuntimeError(
                "LayerDirectIGWrapper: set_baseline_output(baseline_z) を attribute() の前に呼んでください。"
            )
        layer = get_encoder_layers(self.bert_model)[self.layer_idx]
        layer_output = forward_encoder_layer(
            self.bert_model,
            layer,
            z_interp,
            self.attention_mask,
            layer_idx=self.layer_idx,
        )
        target_out = layer_output[:, self.target_token_idx, :]
        diff = target_out - self._baseline_output_j
        return torch.norm(diff, p=2, dim=-1)


def _compute_baseline_z(
    baseline_method: Literal["zero", "self_input_token"],
    z_layer: torch.Tensor,
    bert_model: L.LightningModule,
    layer_idx: int,
    target_token_idx: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    reject_otb_baseline(baseline_method)  # type: ignore[arg-type]
    _, seq_len, hidden = z_layer.shape
    device = z_layer.device
    dtype = z_layer.dtype

    if baseline_method == "zero":
        return torch.zeros(1, seq_len, hidden, device=device, dtype=dtype)

    if baseline_method == "self_input_token":
        z_j = z_layer[0, target_token_idx, :].clone()
        return z_j.unsqueeze(0).unsqueeze(0).expand(1, seq_len, hidden)

    raise ValueError(f"Unknown baseline_method: {baseline_method}")


def _compute_zero_baseline_layer_output(
    bert_model: L.LightningModule,
    z_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """zero baseline で層を 1 回 forward し [1, seq_len, hidden] を返す。"""
    _, seq_len, hidden = z_layer.shape
    device = z_layer.device
    dtype = z_layer.dtype
    baseline_z = torch.zeros(1, seq_len, hidden, device=device, dtype=dtype)
    layer = get_encoder_layers(bert_model)[layer_idx]
    with torch.no_grad():
        return forward_encoder_layer(
            bert_model,
            layer,
            baseline_z,
            attention_mask,
            layer_idx=layer_idx,
        )


def compute_layer_direct_ig_single_target(
    bert_model: L.LightningModule,
    z_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
    num_steps: int = 32,
    baseline_method: Literal["zero", "self_input_token"] = "zero",
    *,
    wrapper: Optional[LayerDirectIGWrapper] = None,
    ig: Optional[IntegratedGradients] = None,
    zero_baseline_layer_output: Optional[torch.Tensor] = None,
) -> np.ndarray:
    device = ensure_model_on_device(bert_model)
    model_dtype = next(bert_model.parameters()).dtype
    z_layer = z_layer.to(device=device, dtype=model_dtype)
    z_layer = z_layer.detach().requires_grad_(True)
    attention_mask = attention_mask.to(device=device)

    baseline_z = _compute_baseline_z(
        baseline_method=baseline_method,
        z_layer=z_layer,
        bert_model=bert_model,
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
        attention_mask=attention_mask,
    )
    baseline_z = baseline_z.to(device=device, dtype=model_dtype)

    if wrapper is None:
        wrapper = LayerDirectIGWrapper(
            bert_model=bert_model,
            layer_idx=layer_idx,
            target_token_idx=target_token_idx,
            attention_mask=attention_mask,
        )
        wrapper.to(device)
        wrapper.train(bert_model.training)
    else:
        wrapper.set_target_token_idx(target_token_idx)

    need_fp32 = model_dtype in (torch.bfloat16, torch.float16)
    z_in = z_layer.to(torch.float32) if need_fp32 else z_layer
    baseline_in = baseline_z.to(torch.float32) if need_fp32 else baseline_z

    if baseline_method == "zero" and zero_baseline_layer_output is not None:
        out_j = zero_baseline_layer_output[:, target_token_idx, :]
        if need_fp32:
            out_j = out_j.to(torch.float32)
        wrapper.set_baseline_output_from_tensor(out_j)
    else:
        wrapper.set_baseline_output(baseline_in if need_fp32 else baseline_z)

    if ig is None:
        ig = IntegratedGradients(wrapper)

    attr = ig.attribute(
        inputs=z_in,
        baselines=baseline_in,
        n_steps=num_steps,
        method="riemann_trapezoid",
    )
    ig_per_token = attr[0].sum(dim=-1).detach()
    if need_fp32:
        ig_per_token = ig_per_token.to(model_dtype)
    return ig_per_token.cpu().numpy()


def compute_layer_direct_ig_all_targets(
    bert_model: L.LightningModule,
    z_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    num_steps: int = 32,
    baseline_method: Literal["zero", "self_input_token"] = "zero",
) -> np.ndarray:
    device = ensure_model_on_device(bert_model)
    model_dtype = next(bert_model.parameters()).dtype
    z_layer = z_layer.to(device=device, dtype=model_dtype)
    attention_mask = attention_mask.to(device=device)
    seq_len = z_layer.shape[1]

    wrapper = LayerDirectIGWrapper(
        bert_model=bert_model,
        layer_idx=layer_idx,
        target_token_idx=0,
        attention_mask=attention_mask,
    )
    wrapper.to(device)
    wrapper.train(bert_model.training)
    ig = IntegratedGradients(wrapper)

    zero_baseline_layer_output: Optional[torch.Tensor] = None
    if baseline_method == "zero":
        zero_baseline_layer_output = _compute_zero_baseline_layer_output(
            bert_model, z_layer, attention_mask, layer_idx
        )

    out = np.zeros((seq_len, seq_len), dtype=np.float64)
    for j in range(seq_len):
        out[:, j] = compute_layer_direct_ig_single_target(
            bert_model=bert_model,
            z_layer=z_layer,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            target_token_idx=j,
            num_steps=num_steps,
            baseline_method=baseline_method,
            wrapper=wrapper,
            ig=ig,
            zero_baseline_layer_output=zero_baseline_layer_output,
        )
    return out
