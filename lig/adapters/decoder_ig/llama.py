"""Llama-family decoder adapter (Llama 2/3, Mistral, Qwen2, Gemma) with ATT/MLP IG."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients
from transformers import AutoModel, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

from lig.adapters.decoder_ig.base import DecoderIGAdapter
from lig.adapters.decoder_ig.mlp_ig import make_probe_direction_forward, run_mlp_input_ig
from lig.adapters.decoder_ig.gpt2 import (
    ATT_BASELINES,
    ATT_ITB_MAP_RATIO,
    ATT_ITB_ZERO_RATIO,
    ATT_SELF_INPUT_TOKEN,
    ATT_ZERO,
    MLP_ATT_ITB_A0,
    MLP_ATT_ITB_MAP_A0,
    MLP_ATT_ITB_ZR_A0,
    MLP_BASELINES,
    MLP_ITB,
    MLP_ZERO,
    AttentionIGResult,
    MLPIGResult,
    _correct_itb_self_term_1d,
    normalize_attention_baseline,
    normalize_mlp_baseline,
)
from utils.calculations.ig.llama.block_forward import (
    hidden_after_attn_residual,
    make_causal_mask,
    make_position_embeddings,
)


class LlamaIGAdapter(DecoderIGAdapter):
    """
    Decoder IG for Llama-style blocks (pre-LN, RoPE, SwiGLU MLP).

    Boundaries match GPT-2 semantics:
    - z: residual before block
    - u: post-attention residual (MLP input)
    - z_next: after MLP residual
    """

    def __init__(
        self,
        model_name: str,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device | str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name
        load_kwargs: Dict[str, Any] = {"attn_implementation": "eager"}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        self.model = model or AutoModel.from_pretrained(model_name, **load_kwargs)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        self.cache_data: Dict[str, Any] = {}

    @property
    def decoder(self) -> torch.nn.Module:
        """Inner decoder (``model`` for AutoModel, ``model.model`` for CausalLM)."""
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
        return self.model

    def _rope_module(self) -> torch.nn.Module:
        if hasattr(self.model, "rotary_emb"):
            return self.model
        return self.decoder

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def num_heads(self) -> int:
        return int(self.model.config.num_attention_heads)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

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
        return self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu().tolist())

    def cache(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        input_ids = inputs["input_ids"].to(self.device)
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        hidden = self.decoder.embed_tokens(input_ids)
        pos_emb = make_position_embeddings(self._rope_module(), hidden, position_ids)

        z_cache: Dict[int, torch.Tensor] = {}
        mlp_input_cache: Dict[int, torch.Tensor] = {}
        z_next_cache: Dict[int, torch.Tensor] = {}
        attention_weights_cache: Dict[int, torch.Tensor] = {}

        with torch.no_grad():
            for layer_idx, layer in enumerate(self.decoder.layers):
                z_cache[layer_idx] = hidden.detach()
                residual = hidden
                normed = layer.input_layernorm(hidden)
                attn_out, attn_weights = layer.self_attn(
                    normed,
                    position_embeddings=pos_emb,
                    attention_mask=self._causal_mask(normed),
                )
                hidden_after_attn = residual + attn_out
                mlp_input_cache[layer_idx] = hidden_after_attn.detach()
                if isinstance(attn_weights, torch.Tensor):
                    attention_weights_cache[layer_idx] = attn_weights.detach()
                residual = hidden_after_attn
                mlp_out = layer.mlp(layer.post_attention_layernorm(hidden_after_attn))
                hidden = residual + mlp_out
                z_next_cache[layer_idx] = hidden.detach()
                pos_emb = make_position_embeddings(self._rope_module(), hidden, position_ids)

        self.cache_data = {
            "input_ids": input_ids.detach(),
            "position_ids": position_ids.detach(),
            "z": z_cache,
            "mlp_input": mlp_input_cache,
            "z_next": z_next_cache,
            "attention_weights": attention_weights_cache,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
        return self.cache_data

    def ensure_cached(self) -> None:
        if not self.cache_data:
            raise RuntimeError("LlamaIGAdapter cache is empty. Call cache(inputs) first.")

    def _position_embeddings(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return make_position_embeddings(
            self._rope_module(), hidden_states, self.cache_data["position_ids"]
        )

    def _causal_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Additive causal mask for hand-rolled block forwards.

        ``LlamaAttention`` takes the mask from its caller; ``LlamaModel`` builds it,
        we do not, so every ``self_attn`` call below has to pass it explicitly or
        the block runs as bidirectional attention.
        """
        return make_causal_mask(hidden_states)

    def get_z(self, layer_idx: int) -> torch.Tensor:
        self.ensure_cached()
        return self.cache_data["z"][layer_idx]

    def get_mlp_input(self, layer_idx: int) -> torch.Tensor:
        self.ensure_cached()
        return self.cache_data["mlp_input"][layer_idx]

    def attention_output(
        self,
        layer_idx: int,
        z: torch.Tensor,
        target_token_idx: int,
        head_idx: Optional[int] = None,
    ) -> torch.Tensor:
        layer = self.decoder.layers[layer_idx]
        normed = layer.input_layernorm(z)
        pos_emb = self._position_embeddings(z)
        if head_idx is None:
            attn_out, _ = layer.self_attn(
                normed,
                position_embeddings=pos_emb,
                attention_mask=self._causal_mask(normed),
            )
            return attn_out[:, target_token_idx, :]

        # Keep the head boundary before o_proj.  Slicing o_proj's output would
        # only split mixed residual coordinates, not recover per-head outputs.
        attn = layer.self_attn
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query = attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        key = attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        value = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *pos_emb)
        head_outputs, _ = eager_attention_forward(
            attn,
            query,
            key,
            value,
            attention_mask=self._causal_mask(normed),
            dropout=0.0,
            scaling=attn.scaling,
        )
        return head_outputs[:, target_token_idx, head_idx, :]

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
            if weights is not None:
                selected = weights
                if head_idx is not None:
                    selected = selected[:, head_idx, :, :]
                if target_token_idx is not None:
                    selected = selected[..., target_token_idx, :]
                return selected
            z = self.get_z(layer_idx)

        layer = self.decoder.layers[layer_idx]
        normed = layer.input_layernorm(z)
        pos_emb = self._position_embeddings(z)
        _, weights = layer.self_attn(
            normed,
            position_embeddings=pos_emb,
            attention_mask=self._causal_mask(normed),
        )
        if weights is None:
            raise RuntimeError(f"Layer {layer_idx} returned no attention weights")
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
        residual = mlp_input
        mlp_delta = layer.mlp(layer.post_attention_layernorm(mlp_input))
        out = residual + mlp_delta
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
            layer = self.decoder.layers[layer_idx]
            pos_emb = self._position_embeddings(baseline_z)
            with torch.no_grad():
                u_full = hidden_after_attn_residual(layer, baseline_z, pos_emb)
            return u_full[:, target_token_idx, :].detach()
        raise ValueError(f"Unsupported MLP baseline: {baseline}")

    def compute_attention_ig(
        self,
        layer_idx: int,
        target_token_idx: int,
        head_idx: Optional[int] = None,
        baseline: str = ATT_ZERO,
        num_steps: int = 32,
        zero_values: Optional[np.ndarray] = None,
        probe_w: Optional[np.ndarray] = None,
    ) -> AttentionIGResult:
        baseline = normalize_attention_baseline(baseline)
        z = self.get_z(layer_idx).detach()
        baseline_z = self.make_attention_baseline_z(z, target_token_idx, baseline)
        baseline_out = self.attention_output(
            layer_idx, baseline_z, target_token_idx, head_idx
        ).detach()

        if probe_w is None:
            # Paper default: L2 scalarization A_j(a) = ||u_j(a) - u_j(0)||_2.
            def forward_fn(z_interp: torch.Tensor) -> torch.Tensor:
                out = self.attention_output(layer_idx, z_interp, target_token_idx, head_idx)
                return torch.norm(out - baseline_out, dim=-1)

            target_mode = "l2_delta"
        else:
            # Signed linear read-out, matching compute_mlp_ig(probe_w=...).  Needed
            # to share one estimand with logit-difference methods (EAP / EAP-IG).
            def forward_fn(z_interp: torch.Tensor) -> torch.Tensor:
                out = self.attention_output(layer_idx, z_interp, target_token_idx, head_idx)
                w = torch.as_tensor(probe_w, dtype=out.dtype, device=out.device)
                return (out - baseline_out) @ w

            target_mode = "probe_direction"

        ig = IntegratedGradients(forward_fn)
        attributions, convergence_delta = ig.attribute(
            inputs=z.float(),
            baselines=baseline_z.float(),
            n_steps=num_steps,
            return_convergence_delta=True,
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
        verification = {
            "target_mode": target_mode,
            "theoretical_diff": theoretical_diff,
            "ig_sum": ig_sum,
            "relative_error": float(relative_error),
            "captum_convergence_delta": float(convergence_delta.abs().max().detach().cpu()),
            "is_valid": bool(relative_error < 0.02),
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
                    probe_w=probe_w,
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
                lambda u_interp: self.mlp_output(layer_idx, u_interp),
                probe_w,
            )
            target_mode = "probe_direction"
        else:
            baseline_out = self.mlp_output(layer_idx, baseline_u).detach()

            def forward_fn(u_interp: torch.Tensor) -> torch.Tensor:
                out = self.mlp_output(layer_idx, u_interp)
                return torch.norm(out - baseline_out, dim=-1)

            target_mode = "l2_delta"

        contributions, total, per_head_np, verification = run_mlp_input_ig(
            u=u,
            baseline_u=baseline_u,
            forward_scalar=forward_fn,
            num_steps=num_steps,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            target_mode=target_mode,
        )

        return MLPIGResult(
            total=total,
            per_head=per_head_np,
            contributions=contributions,
            baseline=baseline,
            verification=verification,
        )
