"""GPT-2 decoder adapter with ATT/MLP Integrated Gradients baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from captum.attr import IntegratedGradients
from transformers import GPT2Model, GPT2TokenizerFast

from lig.adapters.decoder_ig.base import DecoderIGAdapter


# LIG public baseline names (see lig/config.py)
ATT_ZERO = "zero"
ATT_SELF_INPUT_TOKEN = "self_input_token"
ATT_ITB_MAP_RATIO = "itb_map_ratio"
ATT_ITB_ZERO_RATIO = "itb_zero_ratio"
ATT_BASELINES = (
    ATT_ZERO,
    ATT_SELF_INPUT_TOKEN,
    ATT_ITB_MAP_RATIO,
    ATT_ITB_ZERO_RATIO,
)

MLP_ZERO = "zero"
MLP_ITB = "itb"
MLP_ATT_ITB_A0 = "att_itb_a0"
MLP_BASELINES = (MLP_ZERO, MLP_ITB, MLP_ATT_ITB_A0)


@dataclass
class AttentionIGResult:
    """Single target-token attention attribution result."""

    values: np.ndarray
    baseline: str
    raw_values: Optional[np.ndarray]
    zero_values: Optional[np.ndarray]
    attention_weights: Optional[np.ndarray]
    verification: Dict[str, Any]


@dataclass
class MLPIGResult:
    """Single target-token MLP attribution result."""

    total: float
    per_head: np.ndarray
    contributions: np.ndarray
    baseline: str
    verification: Dict[str, Any]


def normalize_attention_baseline(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "baseline_zero": ATT_ZERO,
        "zero": ATT_ZERO,
        "itb": ATT_SELF_INPUT_TOKEN,
        "itb_raw": ATT_SELF_INPUT_TOKEN,
        "self_input_token": ATT_SELF_INPUT_TOKEN,
        "self_input_token_direct_zero": ATT_SELF_INPUT_TOKEN,
        "direct_zero": ATT_SELF_INPUT_TOKEN,
        "itb_attentionmap": ATT_ITB_MAP_RATIO,
        "itb_attention_map": ATT_ITB_MAP_RATIO,
        "attention_map": ATT_ITB_MAP_RATIO,
        "att_map_ratio": ATT_ITB_MAP_RATIO,
        "itb_map_ratio": ATT_ITB_MAP_RATIO,
        "self_contrib_att_map_ratio": ATT_ITB_MAP_RATIO,
        "itb_igzero": ATT_ITB_ZERO_RATIO,
        "itb_ig_zero": ATT_ITB_ZERO_RATIO,
        "itb_zero_ratio": ATT_ITB_ZERO_RATIO,
        "zero_base_ratio": ATT_ITB_ZERO_RATIO,
        "self_contrib_zero_base_ratio": ATT_ITB_ZERO_RATIO,
    }
    if key not in aliases:
        raise ValueError(f"Unknown GPT-2 ATT baseline: {name!r}")
    return aliases[key]


def normalize_mlp_baseline(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace("=", "")
    aliases = {
        "baseline_zero": MLP_ZERO,
        "zero": MLP_ZERO,
        "itb": MLP_ITB,
        "self_input_token": MLP_ITB,
        "mlp_itb": MLP_ITB,
        "attitba0": MLP_ATT_ITB_A0,
        "attitba_0": MLP_ATT_ITB_A0,
        "att_itb_a0": MLP_ATT_ITB_A0,
        "att_itb_attitba0": MLP_ATT_ITB_A0,
        "baseline_att_itb_attitba0": MLP_ATT_ITB_A0,
    }
    if key not in aliases:
        raise ValueError(f"Unknown GPT-2 MLP baseline: {name!r}")
    return aliases[key]


def _safe_divide(num: np.ndarray, den: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros_like(np.asarray(num, dtype=np.float64), dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    np.divide(num, den, out=out, where=np.abs(den) > eps)
    return out


def _correct_itb_self_term_1d(
    *,
    raw: np.ndarray,
    target_token_idx: int,
    estimator: str,
    eps: float = 1e-8,
    zero_values: Optional[np.ndarray] = None,
    attention_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fill the ITB raw self term and rescale to preserve the raw column sum."""
    base = np.asarray(raw, dtype=np.float64).copy()
    n = base.shape[0]
    j = int(target_token_idx)
    if j < 0 or j >= n:
        raise IndexError(f"target_token_idx={j} out of range for length {n}")

    col_sum_base = float(base.sum())
    diag_base = float(base[j])
    sum_other_base = col_sum_base - diag_base

    if estimator == ATT_ITB_ZERO_RATIO:
        if zero_values is None:
            raise ValueError("zero_values is required for itb_igzero")
        z = np.asarray(zero_values, dtype=np.float64)
        if z.shape[0] < n:
            raise ValueError("zero_values is shorter than raw ITB values")
        diag_zero = float(z[j])
        sum_other_zero = float(z[:n].sum() - diag_zero)
        est_self = float(diag_zero * _safe_divide(np.array(sum_other_base), np.array(sum_other_zero), eps))
    elif estimator == ATT_ITB_MAP_RATIO:
        if attention_weights is None:
            raise ValueError("attention_weights is required for itb_attention_map")
        a = np.asarray(attention_weights, dtype=np.float64)
        if a.ndim != 1 or a.shape[0] < n:
            raise ValueError("attention_weights must be a source-token vector")
        alpha_diag = float(a[j])
        alpha_ratio = float(_safe_divide(np.array(alpha_diag), np.array(1.0 - alpha_diag), eps))
        est_self = sum_other_base * alpha_ratio
    else:
        raise ValueError(f"Unknown ITB self-term estimator: {estimator}")

    provisional = base.copy()
    provisional[j] = est_self
    provisional_sum = float(provisional.sum())
    if abs(provisional_sum) > eps:
        provisional *= col_sum_base / provisional_sum
    return provisional


class GPT2Adapter(DecoderIGAdapter):
    """Adapter exposing GPT-2 internals for decoder layer IG.

    Definitions used here:
    - ``z``: residual stream entering a GPT-2 block.
    - ATT output: attention sublayer delta before residual addition.
    - MLP input: residual stream after attention residual addition.
    - MLP output: residual stream after MLP residual addition.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        model: Optional[GPT2Model] = None,
        tokenizer: Optional[GPT2TokenizerFast] = None,
        device: Optional[torch.device | str] = None,
    ):
        self.model_name = model_name
        self.model = model or GPT2Model.from_pretrained(model_name)
        self.tokenizer = tokenizer or GPT2TokenizerFast.from_pretrained(
            model_name, add_prefix_space=True
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        self.cache_data: Dict[str, Any] = {}

    @property
    def num_layers(self) -> int:
        return int(self.model.config.n_layer)

    @property
    def num_heads(self) -> int:
        return int(self.model.config.n_head)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.n_embd)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def encode(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def tokens_from_inputs(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu().tolist())

    def cache(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        input_ids = inputs["input_ids"].to(self.device)
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        hidden = self.model.wte(input_ids) + self.model.wpe(position_ids)

        z_cache: Dict[int, torch.Tensor] = {}
        attn_delta_cache: Dict[int, torch.Tensor] = {}
        mlp_input_cache: Dict[int, torch.Tensor] = {}
        mlp_delta_cache: Dict[int, torch.Tensor] = {}
        z_next_cache: Dict[int, torch.Tensor] = {}
        attention_weights_cache: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            for layer_idx, block in enumerate(self.model.h):
                z_cache[layer_idx] = hidden.detach()

                attn_outputs = block.attn(
                    block.ln_1(hidden),
                    attention_mask=None,
                    output_attentions=True,
                )
                attn_delta = attn_outputs[0]
                maybe_weights = attn_outputs[-1] if len(attn_outputs) > 1 else None
                if isinstance(maybe_weights, torch.Tensor) and maybe_weights.dim() == 4:
                    attention_weights_cache[layer_idx] = maybe_weights.detach()

                hidden_after_attn = hidden + attn_delta
                mlp_delta = block.mlp(block.ln_2(hidden_after_attn))
                hidden = hidden_after_attn + mlp_delta

                attn_delta_cache[layer_idx] = attn_delta.detach()
                mlp_input_cache[layer_idx] = hidden_after_attn.detach()
                mlp_delta_cache[layer_idx] = mlp_delta.detach()
                z_next_cache[layer_idx] = hidden.detach()

        self.cache_data = {
            "input_ids": input_ids.detach(),
            "z": z_cache,
            "attn_delta": attn_delta_cache,
            "mlp_input": mlp_input_cache,
            "mlp_delta": mlp_delta_cache,
            "z_next": z_next_cache,
            "attention_weights": attention_weights_cache,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
        return self.cache_data

    def ensure_cached(self) -> None:
        if not self.cache_data:
            raise RuntimeError("GPT2Adapter cache is empty. Call cache(inputs) first.")

    def get_z(self, layer_idx: int) -> torch.Tensor:
        self.ensure_cached()
        return self.cache_data["z"][layer_idx]

    def get_mlp_input(self, layer_idx: int) -> torch.Tensor:
        self.ensure_cached()
        return self.cache_data["mlp_input"][layer_idx]

    def _manual_attention_weights(self, layer_idx: int, z: torch.Tensor) -> torch.Tensor:
        """Compute GPT-2 causal attention probabilities from Q/K states."""
        block = self.model.h[layer_idx]
        attn = block.attn
        with torch.no_grad():
            z_norm = block.ln_1(z)
            query, key, value = attn.c_attn(z_norm).split(self.hidden_size, dim=2)
            shape = (*query.shape[:-1], -1, self.head_dim)
            query = query.view(shape).transpose(1, 2)
            key = key.view(shape).transpose(1, 2)
            value = value.view(shape).transpose(1, 2)

            weights = torch.matmul(query, key.transpose(-1, -2))
            if getattr(attn, "scale_attn_weights", True):
                scale = torch.full(
                    [],
                    value.size(-1) ** 0.5,
                    dtype=weights.dtype,
                    device=weights.device,
                )
                weights = weights / scale
            if getattr(attn, "scale_attn_by_inverse_layer_idx", False):
                weights = weights / float(getattr(attn, "layer_idx", layer_idx) + 1)

            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = attn.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.full(
                [],
                torch.finfo(weights.dtype).min,
                dtype=weights.dtype,
                device=weights.device,
            )
            weights = torch.where(causal_mask, weights, mask_value)
            weights = torch.nn.functional.softmax(weights, dim=-1)
            weights = weights.type(value.dtype)
        return weights

    def get_attention_weights(
        self,
        layer_idx: int,
        z: Optional[torch.Tensor] = None,
        head_idx: Optional[int] = None,
        target_token_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if z is None:
            self.ensure_cached()
            weights = self.cache_data["attention_weights"].get(layer_idx)
            if weights is None:
                z = self.get_z(layer_idx)
            else:
                selected = weights
                if head_idx is not None:
                    selected = selected[:, head_idx, :, :]
                if target_token_idx is not None:
                    selected = selected[..., target_token_idx, :]
                return selected

        weights = self._manual_attention_weights(layer_idx, z)
        if head_idx is not None:
            weights = weights[:, head_idx, :, :]
        if target_token_idx is not None:
            weights = weights[..., target_token_idx, :]
        return weights

    def attention_output(
        self,
        layer_idx: int,
        z: torch.Tensor,
        target_token_idx: int,
        head_idx: Optional[int] = None,
    ) -> torch.Tensor:
        block = self.model.h[layer_idx]
        z_norm = block.ln_1(z)
        if head_idx is None:
            outputs = block.attn(z_norm, attention_mask=None, output_attentions=False)
            return outputs[0][:, target_token_idx, :]

        try:
            qkv = block.attn.c_attn(z_norm)
            query, key, value = qkv.split(self.hidden_size, dim=2)
            query = block.attn._split_heads(query, self.num_heads, self.head_dim)
            key = block.attn._split_heads(key, self.num_heads, self.head_dim)
            value = block.attn._split_heads(value, self.num_heads, self.head_dim)
            attn_outputs = block.attn._attn(
                query,
                key,
                value,
                attention_mask=None,
                head_mask=None,
            )
            attn_output = attn_outputs[0] if isinstance(attn_outputs, tuple) else attn_outputs
            return attn_output[:, head_idx, target_token_idx, :]
        except Exception:
            outputs = block.attn(z_norm, attention_mask=None, output_attentions=False)
            attn_delta = outputs[0]
            start = head_idx * self.head_dim
            end = start + self.head_dim
            return attn_delta[:, target_token_idx, start:end]

    def mlp_output(self, layer_idx: int, mlp_input: torch.Tensor) -> torch.Tensor:
        block = self.model.h[layer_idx]
        squeeze_token_dim = False
        if mlp_input.dim() == 2:
            mlp_input = mlp_input.unsqueeze(1)
            squeeze_token_dim = True
        mlp_delta = block.mlp(block.ln_2(mlp_input))
        out = mlp_input + mlp_delta
        if squeeze_token_dim:
            out = out[:, 0, :]
        return out

    def make_attention_baseline_z(
        self,
        z: torch.Tensor,
        target_token_idx: int,
        baseline: str,
    ) -> torch.Tensor:
        baseline = normalize_attention_baseline(baseline)
        if baseline == ATT_ZERO:
            return torch.zeros_like(z)
        if baseline in (ATT_SELF_INPUT_TOKEN, ATT_ITB_MAP_RATIO, ATT_ITB_ZERO_RATIO):
            z_j = z[:, target_token_idx, :].clone()
            return z_j.unsqueeze(1).expand_as(z).clone()
        raise ValueError(f"Unsupported ATT baseline: {baseline}")

    def make_mlp_baseline_u(
        self,
        layer_idx: int,
        target_token_idx: int,
        baseline: str,
    ) -> torch.Tensor:
        baseline = normalize_mlp_baseline(baseline)
        u = self.get_mlp_input(layer_idx)[:, target_token_idx, :]
        if baseline == MLP_ZERO:
            return torch.zeros_like(u)
        if baseline == MLP_ITB:
            return u.clone()
        if baseline == MLP_ATT_ITB_A0:
            z = self.get_z(layer_idx)
            baseline_z = self.make_attention_baseline_z(z, target_token_idx, ATT_SELF_INPUT_TOKEN)
            block = self.model.h[layer_idx]
            with torch.no_grad():
                attn_delta = block.attn(
                    block.ln_1(baseline_z),
                    attention_mask=None,
                    output_attentions=False,
                )[0]
                baseline_u_full = baseline_z + attn_delta
            return baseline_u_full[:, target_token_idx, :].detach()
        raise ValueError(f"Unsupported MLP baseline: {baseline}")

    def compute_attention_ig(
        self,
        layer_idx: int,
        target_token_idx: int,
        head_idx: Optional[int] = None,
        baseline: str = ATT_ZERO,
        num_steps: int = 32,
        zero_values: Optional[np.ndarray] = None,
    ) -> AttentionIGResult:
        baseline = normalize_attention_baseline(baseline)
        z = self.get_z(layer_idx).detach()
        baseline_z = self.make_attention_baseline_z(z, target_token_idx, baseline)

        baseline_out = self.attention_output(
            layer_idx, baseline_z, target_token_idx, head_idx
        ).detach()

        def forward_fn(z_interp: torch.Tensor) -> torch.Tensor:
            out = self.attention_output(layer_idx, z_interp, target_token_idx, head_idx)
            return torch.norm(out - baseline_out, dim=-1)

        ig = IntegratedGradients(forward_fn)
        attributions = ig.attribute(
            inputs=z.float(),
            baselines=baseline_z.float(),
            n_steps=num_steps,
            return_convergence_delta=False,
        )
        values = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        # Future tokens are masked by GPT-2 causality; zero tiny numerical noise explicitly.
        if target_token_idx + 1 < values.shape[0]:
            values[target_token_idx + 1 :] = 0.0

        with torch.no_grad():
            actual = float(forward_fn(z.float()).detach().cpu().item())
            base = float(forward_fn(baseline_z.float()).detach().cpu().item())
        ig_sum = float(np.sum(values))
        theoretical_diff = actual - base
        relative_error = (
            abs(ig_sum - theoretical_diff) / abs(theoretical_diff)
            if abs(theoretical_diff) > 1e-8
            else abs(ig_sum - theoretical_diff)
        )
        verification = {
            "theoretical_diff": theoretical_diff,
            "ig_sum": ig_sum,
            "relative_error": float(relative_error),
            "is_valid": bool(relative_error < 0.2),
        }

        raw_values: Optional[np.ndarray] = None
        attention_weights_np: Optional[np.ndarray] = None
        zero_np: Optional[np.ndarray] = zero_values
        if baseline in (ATT_ITB_MAP_RATIO, ATT_ITB_ZERO_RATIO):
            raw_values = values.copy()
            if baseline == ATT_ITB_ZERO_RATIO and zero_np is None:
                zero_np = self.compute_attention_ig(
                    layer_idx=layer_idx,
                    target_token_idx=target_token_idx,
                    head_idx=head_idx,
                    baseline=ATT_ZERO,
                    num_steps=num_steps,
                ).values
            if baseline == ATT_ITB_MAP_RATIO:
                weights = self.get_attention_weights(
                    layer_idx,
                    z=z,
                    head_idx=head_idx,
                    target_token_idx=target_token_idx,
                )
                if head_idx is None and weights.dim() == 3:
                    attention_weights_np = weights.mean(dim=1).squeeze(0).detach().cpu().numpy()
                else:
                    attention_weights_np = weights.squeeze(0).detach().cpu().numpy()
            values = _correct_itb_self_term_1d(
                raw=raw_values,
                target_token_idx=target_token_idx,
                estimator=baseline,
                zero_values=zero_np,
                attention_weights=attention_weights_np,
            )
            verification = {
                **verification,
                "postprocess": baseline,
                "raw_itb_sum": float(np.sum(raw_values)),
                "corrected_sum": float(np.sum(values)),
                "self_value_before": float(raw_values[target_token_idx]),
                "self_value_after": float(values[target_token_idx]),
            }

        return AttentionIGResult(
            values=np.asarray(values, dtype=np.float64),
            baseline=baseline,
            raw_values=raw_values,
            zero_values=zero_np,
            attention_weights=attention_weights_np,
            verification=verification,
        )

    def compute_attention_matrix(
        self,
        layer_idx: int,
        head_idx: int,
        baseline: str = ATT_ZERO,
        num_steps: int = 32,
        target_token_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        self.ensure_cached()
        seq_len = int(self.cache_data["seq_len"])
        targets = list(target_token_indices) if target_token_indices is not None else list(range(seq_len))
        matrix = np.zeros((seq_len, seq_len), dtype=np.float64)
        zero_cache: Dict[int, np.ndarray] = {}
        canonical = normalize_attention_baseline(baseline)
        for target_idx in targets:
            zero_values = None
            if canonical == ATT_ITB_ZERO_RATIO:
                zero_values = zero_cache.get(target_idx)
                if zero_values is None:
                    zero_values = self.compute_attention_ig(
                        layer_idx, target_idx, head_idx, ATT_ZERO, num_steps
                    ).values
                    zero_cache[target_idx] = zero_values
            result = self.compute_attention_ig(
                layer_idx, target_idx, head_idx, canonical, num_steps, zero_values=zero_values
            )
            matrix[:, target_idx] = result.values
        return matrix

    def compute_mlp_ig(
        self,
        layer_idx: int,
        target_token_idx: int,
        baseline: str = MLP_ZERO,
        num_steps: int = 32,
    ) -> MLPIGResult:
        baseline = normalize_mlp_baseline(baseline)
        u = self.get_mlp_input(layer_idx)[:, target_token_idx, :].detach()
        baseline_u = self.make_mlp_baseline_u(layer_idx, target_token_idx, baseline)
        baseline_out = self.mlp_output(layer_idx, baseline_u).detach()

        def forward_fn(u_interp: torch.Tensor) -> torch.Tensor:
            out = self.mlp_output(layer_idx, u_interp)
            return torch.norm(out - baseline_out, dim=-1)

        ig = IntegratedGradients(forward_fn)
        attributions = ig.attribute(
            inputs=u.float(),
            baselines=baseline_u.float(),
            n_steps=num_steps,
            return_convergence_delta=False,
        )
        attr = attributions.squeeze(0)
        per_head = []
        for head_idx in range(self.num_heads):
            start = head_idx * self.head_dim
            end = start + self.head_dim
            per_head.append(float(attr[start:end].sum().detach().cpu().item()))
        per_head_np = np.asarray(per_head, dtype=np.float64)
        total = float(attr.sum().detach().cpu().item())

        with torch.no_grad():
            actual = float(forward_fn(u.float()).detach().cpu().item())
            base = float(forward_fn(baseline_u.float()).detach().cpu().item())
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
            "baseline_u_norm": float(torch.norm(baseline_u).detach().cpu().item()),
            "target_u_norm": float(torch.norm(u).detach().cpu().item()),
            "input_delta_norm": float(torch.norm(u - baseline_u).detach().cpu().item()),
        }
        if baseline == MLP_ITB:
            verification["note"] = (
                "MLP ITB uses the target token's own MLP input as baseline; "
                "for token-wise MLPs this makes input_delta_norm zero."
            )

        return MLPIGResult(
            total=total,
            per_head=per_head_np,
            contributions=np.asarray(attr.detach().cpu().numpy(), dtype=np.float64),
            baseline=baseline,
            verification=verification,
        )

    def compute_mlp_table(
        self,
        layer_idx: int,
        baseline: str = MLP_ZERO,
        num_steps: int = 32,
        target_token_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        self.ensure_cached()
        seq_len = int(self.cache_data["seq_len"])
        targets = list(target_token_indices) if target_token_indices is not None else list(range(seq_len))
        table = np.zeros((seq_len, self.num_heads), dtype=np.float64)
        for target_idx in targets:
            table[target_idx, :] = self.compute_mlp_ig(
                layer_idx, target_idx, baseline, num_steps
            ).per_head
        return table
