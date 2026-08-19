"""Dense Qwen3 decoder adapter with post-attention-residual ``u`` boundaries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients
from transformers import AutoModel, AutoTokenizer

from lig.adapters.decoder_ig.base import DecoderIGAdapter
from lig.adapters.decoder_ig.gpt2 import (
    ATT_ITB_MAP_RATIO,
    ATT_ITB_ZERO_RATIO,
    ATT_SELF_INPUT_TOKEN,
    ATT_ZERO,
    MLP_ATT_ITB_A0,
    MLP_ATT_ITB_MAP_A0,
    MLP_ATT_ITB_ZR_A0,
    MLP_ITB,
    MLP_ZERO,
    AttentionIGResult,
    MLPIGResult,
    _correct_itb_self_term_1d,
    normalize_attention_baseline,
    normalize_mlp_baseline,
)
from lig.adapters.decoder_ig.mlp_ig import make_probe_direction_forward, run_mlp_input_ig
from utils.calculations.ig.qwen3.block_forward import (
    attention_pre_oproj_output,
    create_qwen3_causal_mask,
    get_qwen3_model,
    hidden_after_attn_residual,
    project_qwen3_head,
    qwen3_position_embeddings,
)


class Qwen3IGAdapter(DecoderIGAdapter):
    """LIG adapter for dense Qwen3 models, including Qwen3-1.7B.

    Boundaries:
    - ``z``: residual stream entering a decoder layer.
    - ``u``: post-attention residual, ``z + o_proj(concat(heads))``.
    - ``z_next``: post-MLP residual, i.e. the decoder-layer output.

    For a selected head, attention IG targets ``delta_u_h``: that head's
    pre-output-projection value transformed by its slice of ``o_proj`` into the
    residual space.  It never slices ``u`` into fake head-sized chunks.
    """

    attention_description = (
        "ATT IG (z -> delta_u_h), Qwen3 causal GQA; each head is projected through "
        "its o_proj weight slice into residual space; u is the full post-attention residual"
    )
    mlp_description = (
        "MLP IG (u -> z), Qwen3 post-attention residual u through RMSNorm, "
        "SwiGLU MLP, and residual addition"
    )
    head_output_space = "residual_delta"

    def __init__(
        self,
        model_name: str,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device | str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model_name = model_name
        load_kwargs: Dict[str, Any] = {"attn_implementation": "eager"}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        self.model = model or AutoModel.from_pretrained(model_name, **load_kwargs)
        if getattr(self.model.config, "model_type", None) != "qwen3":
            raise ValueError(
                "Qwen3IGAdapter supports dense model_type='qwen3' only; "
                f"got {getattr(self.model.config, 'model_type', None)!r}."
            )
        layer_types = getattr(self.model.config, "layer_types", None) or ()
        if getattr(self.model.config, "use_sliding_window", False) or any(
            "sliding" in str(layer_type) for layer_type in layer_types
        ):
            raise NotImplementedError(
                "Qwen3IGAdapter currently supports full causal attention only."
            )
        # Direct attention calls must use eager mode so attention weights are
        # available and our explicit additive causal mask is honored.
        self.model.config._attn_implementation = "eager"
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        self.cache_data: Dict[str, Any] = {}

    @property
    def decoder(self) -> torch.nn.Module:
        return get_qwen3_model(self.model)

    @property
    def num_layers(self) -> int:
        return int(self.decoder.config.num_hidden_layers)

    @property
    def num_heads(self) -> int:
        return int(self.decoder.config.num_attention_heads)

    @property
    def num_key_value_heads(self) -> int:
        return int(self.decoder.config.num_key_value_heads)

    @property
    def hidden_size(self) -> int:
        return int(self.decoder.config.hidden_size)

    @property
    def head_dim(self) -> int:
        return int(self.decoder.config.head_dim)

    @property
    def head_output_width(self) -> int:
        return self.hidden_size

    def encode(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def tokens_from_inputs(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        ids = inputs["input_ids"][0].detach().cpu().tolist()
        return self.tokenizer.convert_ids_to_tokens(ids)

    def cache(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        input_ids = inputs["input_ids"].to(self.device)
        batch_size, seq_len = input_ids.shape
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self.device)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        u_cache: Dict[int, torch.Tensor] = {}
        handles = []
        for layer_idx, layer in enumerate(self.decoder.layers):
            def capture_u(
                _module: torch.nn.Module,
                args: tuple[torch.Tensor, ...],
                idx: int = layer_idx,
            ) -> None:
                u_cache[idx] = args[0].detach()

            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(capture_u))

        try:
            with torch.no_grad():
                outputs = self.decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) < self.num_layers:
            raise RuntimeError("Qwen3 model did not return decoder-layer hidden states")
        missing_u = sorted(set(range(self.num_layers)) - set(u_cache))
        if missing_u:
            raise RuntimeError(f"Failed to capture Qwen3 post-attention residuals: {missing_u}")

        self.cache_data = {
            "input_ids": input_ids.detach(),
            "attention_mask": attention_mask.detach(),
            "position_ids": position_ids.detach(),
            "z": {idx: hidden_states[idx].detach() for idx in range(self.num_layers)},
            "mlp_input": u_cache,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
        return self.cache_data

    def ensure_cached(self) -> None:
        if not self.cache_data:
            raise RuntimeError("Qwen3IGAdapter cache is empty. Call cache(inputs) first.")

    def _position_embeddings(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.ensure_cached()
        return qwen3_position_embeddings(self.decoder, z, self.cache_data["position_ids"])

    def _attention_mask(self, z: torch.Tensor) -> torch.Tensor:
        self.ensure_cached()
        return create_qwen3_causal_mask(
            z.shape[1],
            z.device,
            z.dtype,
            self.cache_data["attention_mask"],
        )

    def get_z(self, layer_idx: int) -> torch.Tensor:
        self.ensure_cached()
        return self.cache_data["z"][layer_idx]

    def get_mlp_input(self, layer_idx: int) -> torch.Tensor:
        self.ensure_cached()
        return self.cache_data["mlp_input"][layer_idx]

    def attention_components(
        self, layer_idx: int, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        layer = self.decoder.layers[layer_idx]
        return attention_pre_oproj_output(
            layer,
            z,
            self._position_embeddings(z),
            self._attention_mask(z),
        )

    def attention_output(
        self,
        layer_idx: int,
        z: torch.Tensor,
        target_token_idx: int,
        head_idx: Optional[int] = None,
    ) -> torch.Tensor:
        layer = self.decoder.layers[layer_idx]
        if head_idx is None:
            u = hidden_after_attn_residual(
                layer,
                z,
                self._position_embeddings(z),
                self._attention_mask(z),
            )
            return u[:, target_token_idx, :]
        heads, _ = self.attention_components(layer_idx, z)
        delta_u_h = project_qwen3_head(layer, heads, head_idx)
        return delta_u_h[:, target_token_idx, :]

    def get_attention_weights(
        self,
        layer_idx: int,
        z: Optional[torch.Tensor] = None,
        head_idx: Optional[int] = None,
        target_token_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if z is None:
            z = self.get_z(layer_idx)
        _, weights = self.attention_components(layer_idx, z)
        if weights is None:
            raise RuntimeError("Qwen3 eager attention returned no attention weights")
        if head_idx is not None:
            weights = weights[:, head_idx, :, :]
        if target_token_idx is not None:
            weights = weights[..., target_token_idx, :]
        return weights

    def mlp_output(self, layer_idx: int, mlp_input: torch.Tensor) -> torch.Tensor:
        layer = self.decoder.layers[layer_idx]
        squeeze_token_dim = False
        if mlp_input.dim() == 2:
            mlp_input = mlp_input.unsqueeze(1)
            squeeze_token_dim = True
        out = mlp_input + layer.mlp(layer.post_attention_layernorm(mlp_input))
        if squeeze_token_dim:
            out = out[:, 0, :]
        return out

    def make_attention_baseline_z(
        self, z: torch.Tensor, target_token_idx: int, baseline: str
    ) -> torch.Tensor:
        baseline = normalize_attention_baseline(baseline)
        if baseline == ATT_ZERO:
            return torch.zeros_like(z)
        if baseline in (ATT_SELF_INPUT_TOKEN, ATT_ITB_MAP_RATIO, ATT_ITB_ZERO_RATIO):
            z_j = z[:, target_token_idx, :].clone()
            return z_j.unsqueeze(1).expand_as(z).clone()
        raise ValueError(f"Unsupported Qwen3 ATT baseline: {baseline}")

    def make_mlp_baseline_u(
        self, layer_idx: int, target_token_idx: int, baseline: str
    ) -> torch.Tensor:
        baseline = normalize_mlp_baseline(baseline)
        u = self.get_mlp_input(layer_idx)[:, target_token_idx, :]
        if baseline == MLP_ZERO:
            return torch.zeros_like(u)
        if baseline == MLP_ITB:
            return u.clone()
        att_a0_baselines = {
            MLP_ATT_ITB_A0: ATT_SELF_INPUT_TOKEN,
            MLP_ATT_ITB_ZR_A0: ATT_ITB_ZERO_RATIO,
            MLP_ATT_ITB_MAP_A0: ATT_ITB_MAP_RATIO,
        }
        if baseline in att_a0_baselines:
            z = self.get_z(layer_idx)
            baseline_z = self.make_attention_baseline_z(
                z, target_token_idx, att_a0_baselines[baseline]
            )
            with torch.no_grad():
                return self.attention_output(
                    layer_idx, baseline_z, target_token_idx, head_idx=None
                ).detach()
        raise ValueError(f"Unsupported Qwen3 MLP baseline: {baseline}")

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
            return torch.norm(out - baseline_out, p=2, dim=-1)

        ig = IntegratedGradients(forward_fn)
        attributions = ig.attribute(
            inputs=z.float(),
            baselines=baseline_z.float(),
            n_steps=num_steps,
            return_convergence_delta=False,
        )
        values = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

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
        verification: Dict[str, Any] = {
            "theoretical_diff": theoretical_diff,
            "ig_sum": ig_sum,
            "relative_error": float(relative_error),
            "is_valid": bool(relative_error < 0.2),
            "output_space": (
                self.head_output_space
                if head_idx is not None
                else "post_attention_residual"
            ),
            "output_width": self.hidden_size,
        }

        raw_values: Optional[np.ndarray] = None
        attention_weights_np: Optional[np.ndarray] = None
        zero_np = zero_values
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
            verification.update(
                {
                    "postprocess": baseline,
                    "raw_itb_sum": float(np.sum(raw_values)),
                    "corrected_sum": float(np.sum(values)),
                    "self_value_before": float(raw_values[target_token_idx]),
                    "self_value_after": float(values[target_token_idx]),
                }
            )

        return AttentionIGResult(
            values=np.asarray(values, dtype=np.float64),
            baseline=baseline,
            raw_values=raw_values,
            zero_values=zero_np,
            attention_weights=attention_weights_np,
            verification=verification,
        )

    def compute_mlp_ig(
        self,
        layer_idx: int,
        target_token_idx: int,
        baseline: str = MLP_ZERO,
        num_steps: int = 32,
        probe_w: Optional[np.ndarray] = None,
    ) -> MLPIGResult:
        baseline = normalize_mlp_baseline(baseline)
        u = self.get_mlp_input(layer_idx)[:, target_token_idx, :].detach()
        baseline_u = self.make_mlp_baseline_u(layer_idx, target_token_idx, baseline)

        if probe_w is not None:
            forward_fn = make_probe_direction_forward(
                lambda u_interp: self.mlp_output(layer_idx, u_interp), probe_w
            )
            target_mode = "probe_direction"
        else:
            baseline_out = self.mlp_output(layer_idx, baseline_u).detach()

            def forward_fn(u_interp: torch.Tensor) -> torch.Tensor:
                out = self.mlp_output(layer_idx, u_interp)
                return torch.norm(out - baseline_out, p=2, dim=-1)

            target_mode = "l2_delta"

        # u is already in residual space after o_proj, so its feature axes are
        # not attention-head axes.  Return no misleading per-head aggregation.
        contributions, total, per_head, verification = run_mlp_input_ig(
            u=u,
            baseline_u=baseline_u,
            forward_scalar=forward_fn,
            num_steps=num_steps,
            num_heads=0,
            head_dim=0,
            target_mode=target_mode,
        )
        verification["per_head_available"] = False
        return MLPIGResult(
            total=total,
            per_head=per_head,
            contributions=contributions,
            baseline=baseline,
            verification=verification,
        )
