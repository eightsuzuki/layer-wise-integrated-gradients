"""
Unified LIG API — one call, JSON output.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from lig.config import LIGConfig, validate_release_baselines

from utils.calculations.ig.attention.attention_ig import (
    compute_attention_ig_global_analysis_multi_layer,
)
from utils.calculations.ig.attention.attention_map_alpha import (
    get_encoder_self_attention_alpha,
)
from utils.calculations.ig.mlp.att_itb_mlp_baseline import get_mlp_baseline_att_itb_eq_zero
from utils.calculations.ig.mlp.mlp_lig_ig import compute_mlp_lig_single_token
from utils.calculations.ig.shared.itb_self_contrib import (
    apply_itb_column_map_ratio,
    apply_itb_column_zero_ratio,
)
from utils.calculations.ig.z2z.layer_direct_ig import compute_layer_direct_ig_all_targets
from utils.calculations.ig.z2z.layer_itb_zero_ratio import apply_layer_z2z_itb_zero_base_ratio


def explain(
    text: str,
    *,
    model: str = "bert-base-uncased",
    num_steps: int = 32,
    granularity: Union[str, List[str]] = "all",
    baseline_att: str = "self_input_token",
    baseline_mlp: str = "zero",
    baseline_layer: str = "self_input_token",
    layers: Optional[List[int]] = None,
    target_tokens: Optional[List[int]] = None,
    target_head: Optional[int] = None,
    device: Optional[str] = None,
    include_residual_connection: bool = True,
    config: Optional[LIGConfig] = None,
) -> Dict[str, Any]:
    """
    Compute Layer-wise Integrated Gradients for one sentence.

    Returns a JSON-serializable dict with z→u (ATT), u→z (MLP), and z→z (layer) attributions.

    Example::

        from lig import explain
        result = explain("Hello world", num_steps=16, granularity="all")
    """
    cfg = config or LIGConfig(
        model=model,
        num_steps=num_steps,
        granularity=granularity,  # type: ignore[arg-type]
        baseline_att=baseline_att,  # type: ignore[arg-type]
        baseline_mlp=baseline_mlp,  # type: ignore[arg-type]
        baseline_layer=baseline_layer,  # type: ignore[arg-type]
        layers=layers,
        target_tokens=target_tokens,
        target_head=target_head,
        device=device,
        include_residual_connection=include_residual_connection,
    )
    validate_release_baselines(cfg)
    return _run_explain(text, cfg)


def explain_to_json(text: str, **kwargs: Any) -> str:
    """Same as :func:`explain` but returns a JSON string."""
    return json.dumps(explain(text, **kwargs), indent=2, ensure_ascii=False)


def _run_explain(text: str, cfg: LIGConfig) -> Dict[str, Any]:
    from transformers import AutoConfig

    from lig.adapters.decoder import DECODER_FAMILY_TYPES
    from lig.adapters.encoder import EncoderAdapter

    model_type = getattr(AutoConfig.from_pretrained(cfg.model), "model_type", "unknown")
    modes = cfg.resolved_granularity()

    if model_type == "gpt2":
        return _run_explain_gpt2(text, cfg)
    if model_type in DECODER_FAMILY_TYPES:
        raise NotImplementedError(
            f"Decoder model '{cfg.model}' ({model_type}) is not implemented yet. "
            f"GPT-2 supports granularity att/mlp/layer via lig.explain(). "
            f"Requested: {modes}. See docs/DECODER_DESIGN.md."
        )

    adapter = EncoderAdapter.from_pretrained(cfg.model, device=cfg.device)
    tokenized = adapter.tokenize(text)
    tokens = adapter.tokens_as_strings(text)
    seq_len = tokenized["input_ids"].shape[1]
    hidden_states = adapter.forward_hidden_states(tokenized)
    ig_inputs = adapter.inputs_for_ig(tokenized)

    layer_indices = cfg.layers if cfg.layers is not None else list(range(adapter.num_layers))
    token_indices = (
        cfg.target_tokens if cfg.target_tokens is not None else list(range(seq_len))
    )
    modes = cfg.resolved_granularity()
    supported = adapter.supported_granularity
    unsupported = set(modes) - supported
    if unsupported:
        raise NotImplementedError(
            f"Model '{cfg.model}' ({adapter.model_type}) supports granularity "
            f"{sorted(supported)} only. Requested unsupported mode(s): {sorted(unsupported)}. "
            + (
                "MoE (Switch) and Mamba use block-level z→z only; "
                "DistilBERT/ModernBERT lack a BERT-style ATT/MLP split."
                if unsupported <= {"att", "mlp"}
                else ""
            )
        )

    result: Dict[str, Any] = {
        "text": text,
        "tokens": tokens,
        "model": cfg.model,
        "model_type": adapter.model_type,
        "config": {
            "num_steps": cfg.num_steps,
            "granularity": modes,
            "baseline_att": cfg.baseline_att,
            "baseline_mlp": cfg.baseline_mlp,
            "baseline_layer": cfg.baseline_layer,
            "layers": layer_indices,
            "target_tokens": token_indices,
            "target_head": cfg.target_head,
            "include_residual_connection": cfg.include_residual_connection,
        },
        "boundaries": {
            "z": adapter.boundaries.z_node,
            "u": adapter.boundaries.u_node,
            "z_next": adapter.boundaries.z_next_node,
            **adapter.boundaries.as_dict(),
        },
        "layers": {},
    }

    for layer_idx in layer_indices:
        layer_out: Dict[str, Any] = {"layer_idx": layer_idx}
        z_layer = adapter.z_at_layer(hidden_states, layer_idx)
        attention_mask = adapter.attention_mask_for_layer(tokenized, z_layer)

        if "layer" in modes:
            matrix = _compute_encoder_layer_z2z(
                adapter=adapter,
                z_layer=z_layer,
                attention_mask=attention_mask,
                layer_idx=layer_idx,
                cfg=cfg,
            )
            layer_out["z2z"] = {
                "baseline": cfg.baseline_layer,
                "description": "Layer-whole IG (z -> z), L2-scalarized",
                "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "matrix": _tolist(matrix),
                "token_to_token": _matrix_with_labels(matrix, tokens, token_indices),
            }

        targets_out: Dict[str, Any] = {}
        for t_idx in token_indices:
            target_entry: Dict[str, Any] = {"target_token_idx": t_idx, "target_token": tokens[t_idx]}

            if "att" in modes:
                target_entry["z2u"] = _encoder_z2u(
                    adapter=adapter,
                    ig_inputs=ig_inputs,
                    layer_idx=layer_idx,
                    target_token_idx=t_idx,
                    cfg=cfg,
                )

            if "mlp" in modes:
                u_j = adapter.u_from_z(layer_idx, z_layer, attention_mask, t_idx)
                baseline_u = _mlp_baseline(
                    adapter=adapter,
                    cfg=cfg,
                    layer_idx=layer_idx,
                    z_layer=z_layer,
                    attention_mask=attention_mask,
                    target_token_idx=t_idx,
                    u_j=u_j,
                )
                mlp = compute_mlp_lig_single_token(
                    model_mlp=adapter.model,
                    layer_idx=layer_idx,
                    target_mlp_input=u_j.unsqueeze(0),
                    baseline_mlp_input=baseline_u.unsqueeze(0),
                    num_steps=cfg.num_steps,
                    include_residual_connection=cfg.include_residual_connection,
                )
                target_entry["u2z"] = {
                    "baseline": cfg.baseline_mlp,
                    "description": "MLP IG (u -> z), L2-scalarized",
                    "contributions": mlp["contributions"],
                    "l2_total": mlp["l2_total"],
                    "completeness": {
                        "error": mlp["completeness_error"],
                        "mean_abs_error": mlp["mean_abs_completeness_error"],
                    },
                }

            targets_out[str(t_idx)] = target_entry

        if "att" in modes or "mlp" in modes:
            layer_out["targets"] = targets_out

        result["layers"][str(layer_idx)] = layer_out

    return result


def _run_explain_gpt2(text: str, cfg: LIGConfig) -> Dict[str, Any]:
    """GPT-2 decoder: z→u (ATT), u→z (MLP), z→z (layer) via :class:`GPT2Adapter`."""
    from lig.adapters.decoder_ig import GPT2Adapter
    from lig.boundaries import detect_boundaries

    modes = cfg.resolved_granularity()
    if cfg.baseline_mlp not in ("zero", "att_itb_a0"):
        raise NotImplementedError(
            "GPT-2 u→z supports baseline_mlp in {'zero', 'att_itb_a0'} only. "
            f"Got: {cfg.baseline_mlp!r}"
        )

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    adapter = GPT2Adapter(model_name=cfg.model, device=device)
    inputs = adapter.encode(text, max_length=512)
    adapter.cache(inputs)
    tokens = adapter.tokens_from_inputs(inputs)
    seq_len = len(tokens)

    boundaries = detect_boundaries(adapter.model)
    layer_indices = cfg.layers if cfg.layers is not None else list(range(adapter.num_layers))
    token_indices = cfg.target_tokens if cfg.target_tokens is not None else list(range(seq_len))
    token_indices = [i for i in token_indices if 0 <= i < seq_len]
    if not token_indices:
        raise ValueError(f"No valid target_tokens for seq_len={seq_len}")
    head_indices = (
        [cfg.target_head] if cfg.target_head is not None else list(range(adapter.num_heads))
    )

    result: Dict[str, Any] = {
        "text": text,
        "tokens": tokens,
        "model": cfg.model,
        "model_type": "gpt2",
        "architecture": "decoder",
        "config": {
            "num_steps": cfg.num_steps,
            "granularity": modes,
            "baseline_att": cfg.baseline_att,
            "baseline_mlp": cfg.baseline_mlp,
            "baseline_layer": cfg.baseline_layer,
            "layers": layer_indices,
            "target_tokens": token_indices,
            "target_head": cfg.target_head,
            "include_residual_connection": cfg.include_residual_connection,
        },
        "boundaries": {
            "z": boundaries.z_node,
            "u": boundaries.u_node,
            "z_next": boundaries.z_next_node,
            **boundaries.as_dict(),
        },
        "layers": {},
    }

    for layer_idx in layer_indices:
        z_layer = adapter.get_z(layer_idx)
        layer_out: Dict[str, Any] = {"layer_idx": layer_idx}

        if "layer" in modes:
            matrix = _compute_gpt2_layer_z2z(
                gpt2=adapter.model,
                z_layer=z_layer,
                layer_idx=layer_idx,
                cfg=cfg,
            )
            layer_out["z2z"] = {
                "baseline": cfg.baseline_layer,
                "description": "Layer-whole IG (z -> z), GPT-2 block, L2-scalarized",
                "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "matrix": _tolist(matrix),
                "token_to_token": _matrix_with_labels(matrix, tokens, token_indices),
            }

        targets_out: Dict[str, Any] = {}
        for t_idx in token_indices:
            target_entry: Dict[str, Any] = {
                "target_token_idx": t_idx,
                "target_token": tokens[t_idx],
            }

            if "att" in modes:
                target_entry["z2u"] = {
                    "baseline": cfg.baseline_att,
                    "description": "ATT IG (z -> u), GPT-2 causal; per-head token scores",
                    "heads": {},
                }
                for h_idx in head_indices:
                    att_result = adapter.compute_attention_ig(
                        layer_idx=layer_idx,
                        target_token_idx=t_idx,
                        head_idx=h_idx,
                        baseline=cfg.baseline_att,
                        num_steps=cfg.num_steps,
                    )
                    scores = att_result.values.tolist()
                    target_entry["z2u"]["heads"][str(h_idx)] = {
                        "contributions": scores,
                        "token_l2_norm": float(np.linalg.norm(scores)),
                    }

            if "mlp" in modes:
                mlp_result = adapter.compute_mlp_ig(
                    layer_idx=layer_idx,
                    target_token_idx=t_idx,
                    baseline=cfg.baseline_mlp,
                    num_steps=cfg.num_steps,
                )
                contributions = mlp_result.contributions.tolist()
                l2_total = float(mlp_result.total)
                ig_sum = float(mlp_result.verification.get("ig_sum", l2_total))
                theoretical_diff = float(mlp_result.verification.get("theoretical_diff", l2_total))
                completeness_error = ig_sum - theoretical_diff
                target_entry["u2z"] = {
                    "baseline": cfg.baseline_mlp,
                    "description": "MLP IG (u -> z), GPT-2 post-attn u to z^(l+1), L2-scalarized",
                    "contributions": contributions,
                    "l2_total": l2_total,
                    "completeness": {
                        "mean_abs_error": abs(completeness_error),
                        "max_abs_error": abs(completeness_error),
                    },
                }

            targets_out[str(t_idx)] = target_entry

        if "att" in modes or "mlp" in modes:
            layer_out["targets"] = targets_out
        result["layers"][str(layer_idx)] = layer_out

    return result


def _run_explain_gpt2_att(text: str, cfg: LIGConfig) -> Dict[str, Any]:
    """Backward-compatible alias: att-only path."""
    narrow = LIGConfig(
        model=cfg.model,
        num_steps=cfg.num_steps,
        granularity="att",
        baseline_att=cfg.baseline_att,
        layers=cfg.layers,
        target_tokens=cfg.target_tokens,
        target_head=cfg.target_head,
        device=cfg.device,
    )
    return _run_explain_gpt2(text, narrow)


def _encoder_z2u(
    *,
    adapter: Any,
    ig_inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    target_token_idx: int,
    cfg: LIGConfig,
) -> Dict[str, Any]:
    """ATT z→u with L2-scalarized IG (LIG)."""
    num_heads = int(adapter.model.config.num_attention_heads)
    head_indices = [cfg.target_head] if cfg.target_head is not None else list(range(num_heads))
    heads_out: Dict[str, Any] = {}
    for h_idx in head_indices:
        scores = _encoder_att_head_contributions(
            adapter=adapter,
            ig_inputs=ig_inputs,
            layer_idx=layer_idx,
            target_token_idx=target_token_idx,
            head_idx=h_idx,
            baseline_att=cfg.baseline_att,
            num_steps=cfg.num_steps,
        )
        heads_out[str(h_idx)] = {
            "contributions": scores,
            "token_l2_norm": float(np.linalg.norm(scores)),
        }
    desc = "ATT IG (z -> u), L2-scalarized"
    if cfg.baseline_att == "itb_zero_ratio":
        desc += " (ITB-zeroRatio)"
    elif cfg.baseline_att == "itb_map_ratio":
        desc += " (ITB-mapRatio)"
    return {
        "baseline": cfg.baseline_att,
        "description": desc,
        "heads": heads_out,
    }


def _att_ig_column_scores(
    *,
    model: Any,
    inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    target_token_idx: int,
    head_idx: int,
    baseline_method: str,
    num_steps: int,
) -> List[float]:
    att = compute_attention_ig_global_analysis_multi_layer(
        bert_model=model,
        inputs=inputs,
        layer_indices=[layer_idx],
        target_token_idx=target_token_idx,
        target_head_idx=head_idx,
        num_steps=num_steps,
        baseline_method=baseline_method,
        input_type="z",
    )
    return att[layer_idx]["ig_values"]


def _encoder_att_head_contributions(
    *,
    adapter: Any,
    ig_inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    target_token_idx: int,
    head_idx: int,
    baseline_att: str,
    num_steps: int,
) -> List[float]:
    if baseline_att in ("itb_zero_ratio", "itb_map_ratio"):
        itb = np.asarray(
            _att_ig_column_scores(
                model=adapter.model,
                inputs=ig_inputs,
                layer_idx=layer_idx,
                target_token_idx=target_token_idx,
                head_idx=head_idx,
                baseline_method="self_input_token",
                num_steps=num_steps,
            ),
            dtype=np.float64,
        )
        if baseline_att == "itb_zero_ratio":
            zero = np.asarray(
                _att_ig_column_scores(
                    model=adapter.model,
                    inputs=ig_inputs,
                    layer_idx=layer_idx,
                    target_token_idx=target_token_idx,
                    head_idx=head_idx,
                    baseline_method="zero",
                    num_steps=num_steps,
                ),
                dtype=np.float64,
            )
            out = apply_itb_column_zero_ratio(itb, zero, target_token_idx)
        else:
            alpha = get_encoder_self_attention_alpha(
                adapter.model,
                ig_inputs,
                layer_idx=layer_idx,
                head_idx=head_idx,
                token_idx=target_token_idx,
            )
            out = apply_itb_column_map_ratio(itb, alpha, target_token_idx)
        return out.tolist()

    return _att_ig_column_scores(
        model=adapter.model,
        inputs=ig_inputs,
        layer_idx=layer_idx,
        target_token_idx=target_token_idx,
        head_idx=head_idx,
        baseline_method=baseline_att,
        num_steps=num_steps,
    )


def _compute_encoder_layer_z2z(
    *,
    adapter: Any,
    z_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    cfg: LIGConfig,
) -> np.ndarray:
    if cfg.baseline_layer == "itb_zero_ratio":
        z_itb = compute_layer_direct_ig_all_targets(
            bert_model=adapter.model,
            z_layer=z_layer,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            num_steps=cfg.num_steps,
            baseline_method="self_input_token",
        )
        z_zero = compute_layer_direct_ig_all_targets(
            bert_model=adapter.model,
            z_layer=z_layer,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            num_steps=cfg.num_steps,
            baseline_method="zero",
        )
        return apply_layer_z2z_itb_zero_base_ratio(z_itb[np.newaxis, ...], z_zero[np.newaxis, ...])[0]
    return compute_layer_direct_ig_all_targets(
        bert_model=adapter.model,
        z_layer=z_layer,
        attention_mask=attention_mask,
        layer_idx=layer_idx,
        num_steps=cfg.num_steps,
        baseline_method=cfg.baseline_layer,  # type: ignore[arg-type]
    )


def _compute_gpt2_layer_z2z(
    *,
    gpt2: Any,
    z_layer: torch.Tensor,
    layer_idx: int,
    cfg: LIGConfig,
) -> np.ndarray:
    from utils.calculations.ig.z2z.gpt2_layer_direct_ig import (
        compute_gpt2_layer_direct_ig_all_targets,
    )

    if cfg.baseline_layer == "itb_zero_ratio":
        z_itb = compute_gpt2_layer_direct_ig_all_targets(
            gpt2=gpt2,
            z_layer=z_layer,
            layer_idx=layer_idx,
            num_steps=cfg.num_steps,
            baseline_method="self_input_token",
        )
        z_zero = compute_gpt2_layer_direct_ig_all_targets(
            gpt2=gpt2,
            z_layer=z_layer,
            layer_idx=layer_idx,
            num_steps=cfg.num_steps,
            baseline_method="zero",
        )
        return apply_layer_z2z_itb_zero_base_ratio(z_itb[np.newaxis, ...], z_zero[np.newaxis, ...])[0]
    return compute_gpt2_layer_direct_ig_all_targets(
        gpt2=gpt2,
        z_layer=z_layer,
        layer_idx=layer_idx,
        num_steps=cfg.num_steps,
        baseline_method=cfg.baseline_layer,  # type: ignore[arg-type]
    )


def _mlp_baseline(
    *,
    adapter: Any,
    cfg: LIGConfig,
    layer_idx: int,
    z_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_idx: int,
    u_j: torch.Tensor,
) -> torch.Tensor:
    if cfg.baseline_mlp == "zero":
        return torch.zeros_like(u_j)
    if cfg.baseline_mlp == "att_itb_a0":
        return get_mlp_baseline_att_itb_eq_zero(
            bert_model=adapter.model,
            z_layer=z_layer,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            target_token_idx=target_token_idx,
        )
    raise ValueError(f"Unknown MLP baseline: {cfg.baseline_mlp}")


def _tolist(arr: np.ndarray) -> List[List[float]]:
    return arr.tolist()


def _matrix_with_labels(
    matrix: np.ndarray,
    tokens: List[str],
    token_indices: List[int],
) -> List[Dict[str, Any]]:
    rows = []
    for j in token_indices:
        row = {
            "target_token_idx": j,
            "target_token": tokens[j],
            "contributions": [
                {
                    "source_token_idx": i,
                    "source_token": tokens[i],
                    "value": float(matrix[i, j]),
                }
                for i in token_indices
            ],
        }
        rows.append(row)
    return rows


def describe_boundaries(
    model: str,
    *,
    device: Optional[str] = None,
    load_weights: bool = False,
) -> Dict[str, Any]:
    """
    Summarize auto-detected z / u / z_next boundaries for a Hugging Face model id.

    With ``load_weights=False`` (default), uses ``AutoConfig`` only — no checkpoint download.
    With ``load_weights=True``, loads the model and runs module introspection via
    :func:`lig.boundaries.detect_boundaries`.
    """
    from transformers import AutoConfig

    from lig.adapters.decoder import DECODER_FAMILY_TYPES
    from lig.adapters.encoder import ENCODER_TYPES
    from lig.adapters.factory import load_adapter
    from lig.boundaries import describe_boundaries_from_config, detect_boundaries

    if not load_weights:
        out = describe_boundaries_from_config(model)
        out["model"] = model
        return out

    config = AutoConfig.from_pretrained(model)
    model_type = getattr(config, "model_type", "unknown")
    if model_type in ENCODER_TYPES:
        adapter = load_adapter(model, device=device)
        info = adapter.boundaries.as_dict()
        info["model"] = model
        return info
    if model_type in DECODER_FAMILY_TYPES:
        adapter = load_adapter(model, device=device, allow_decoder_stub=True)
        info = detect_boundaries(adapter.model).as_dict()
        info["model"] = model
        info["ig_ready"] = model_type == "gpt2"
        return info

    raise ValueError(
        f"Unsupported model_type '{model_type}' for '{model}'. "
        f"Encoders: {sorted(ENCODER_TYPES)}. Decoders: {sorted(DECODER_FAMILY_TYPES)}."
    )
