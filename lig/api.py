"""
Unified LIG API — one call, JSON output.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from lig.config import LIGConfig, validate_release_baselines

from utils.calculations.ig.shared.itb_self_contrib import (
    apply_itb_column_map_ratio,
    apply_itb_column_zero_ratio,
)
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

    from lig.adapters.decoder_ig import DECODER_IG_MODEL_TYPES

    if model_type == "gemma3":
        return _run_explain_gemma3(text, cfg)
    if model_type in DECODER_IG_MODEL_TYPES:
        return _run_explain_decoder(text, cfg, model_type)
    if model_type in DECODER_FAMILY_TYPES:
        raise NotImplementedError(
            f"Decoder model '{cfg.model}' ({model_type}) is not implemented yet. "
            f"Supported decoder IG: {sorted(DECODER_IG_MODEL_TYPES | {'gemma3'})}. "
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
                # Encoder-only implementation. Keep this import out of decoder
                # startup because it depends on Lightning/TorchMetrics.
                from utils.calculations.ig.mlp.mlp_lig_ig import (
                    compute_mlp_lig_single_token,
                )

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


def _run_explain_decoder(text: str, cfg: LIGConfig, model_type: str) -> Dict[str, Any]:
    """Decoder (GPT-2 / Llama family): z→u, u→z, z→z via :class:`DecoderIGAdapter`."""
    from lig.adapters.decoder_ig import load_decoder_ig_adapter
    from lig.boundaries import detect_boundaries

    modes = cfg.resolved_granularity()
    if cfg.baseline_mlp not in ("zero", "att_itb_a0"):
        raise NotImplementedError(
            "Decoder u→z supports baseline_mlp in {'zero', 'att_itb_a0'} only. "
            f"Got: {cfg.baseline_mlp!r}"
        )

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    adapter = load_decoder_ig_adapter(cfg.model, device=str(device))
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
        "model_type": model_type,
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
            matrix = _compute_decoder_layer_z2z(
                adapter=adapter,
                z_layer=z_layer,
                layer_idx=layer_idx,
                cfg=cfg,
            )
            layer_out["z2z"] = {
                "baseline": cfg.baseline_layer,
                "description": "Layer-whole IG (z -> z), decoder block, L2-scalarized",
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
                    "description": "ATT IG (z -> u), causal decoder; per-head token scores",
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
                    "description": "MLP IG (u -> z), post-attn u to z^(l+1), L2-scalarized",
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


def _run_explain_gpt2(text: str, cfg: LIGConfig) -> Dict[str, Any]:
    """Backward-compatible alias for GPT-2 decoder explain."""
    return _run_explain_decoder(text, cfg, "gpt2")


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


@contextmanager
def _torchvision_disabled():
    """Hide torchvision while Transformers imports the multimodal Gemma3 class.

    LIG uses only the language model, so this optional vision dependency must
    not block text analysis when it is absent or binary-incompatible with the
    installed PyTorch. The flag is restored on exit, so image pipelines running
    later in the same process are unaffected.
    """
    import transformers.utils.import_utils as import_utils

    original = import_utils._torchvision_available
    import_utils._torchvision_available = False
    try:
        yield
    finally:
        import_utils._torchvision_available = original


def _gemma3_baseline_embeddings(
    input_embeddings: torch.Tensor,
    baseline_method: str,
    target_token_idx: int,
) -> torch.Tensor:
    if baseline_method in ("zero", "itb_zero_ratio"):
        return torch.zeros_like(input_embeddings)
    if baseline_method in ("self_input_token", "itb_map_ratio"):
        z_j = input_embeddings[0, target_token_idx, :].clone()
        return z_j.unsqueeze(0).unsqueeze(0).expand_as(input_embeddings)
    raise ValueError(
        "Gemma3 baseline_att must be one of "
        "{'zero', 'self_input_token', 'itb_zero_ratio', 'itb_map_ratio'}. "
        f"Got: {baseline_method!r}"
    )


def _run_explain_gemma3(text: str, cfg: LIGConfig) -> Dict[str, Any]:
    """Gemma3 decoder: z→u (ATT), u→z (MLP), z→z (layer) with pre+post-LN blocks."""
    import warnings

    modes = cfg.resolved_granularity()
    if cfg.baseline_mlp not in ("zero", "att_itb_a0"):
        raise NotImplementedError(
            "Gemma3 u→z supports baseline_mlp in {'zero', 'att_itb_a0'} only. "
            f"Got: {cfg.baseline_mlp!r}"
        )

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Gemma3 is multimodal, so Transformers probes torchvision while importing
    # its model class; LIG only needs the language model.
    with _torchvision_disabled():
        from transformers import AutoModel, AutoTokenizer

        from lig.boundaries import detect_boundaries
        from utils.calculations.ig.gemma3.block_forward import (
            attn_pre_oproj_output,
            embed_tokens,
            get_gemma3_text_model,
            layer_attention_mask,
            rope_position_embeddings,
        )
        from utils.calculations.ig.mlp.gemma3_mlp_lig_ig import (
            compute_gemma3_mlp_lig_single_token,
        )

        loaded = AutoModel.from_pretrained(
            cfg.model,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        ).to(device)
    loaded.eval()
    text_model = get_gemma3_text_model(loaded)
    text_model.eval()
    boundaries = detect_boundaries(text_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to(device)
    seq_len = int(input_ids.shape[1])
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    with torch.no_grad():
        input_embeddings = embed_tokens(text_model, input_ids)
        hs_out = text_model(
            inputs_embeds=input_embeddings,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = hs_out.hidden_states
    assert hidden_states is not None

    n_layers = int(text_model.config.num_hidden_layers)
    n_heads = int(text_model.config.num_attention_heads)
    layer_indices = cfg.layers if cfg.layers is not None else list(range(n_layers))
    token_indices = cfg.target_tokens if cfg.target_tokens is not None else list(range(seq_len))
    token_indices = [i for i in token_indices if 0 <= i < seq_len]
    if not token_indices:
        raise ValueError(f"No valid target_tokens for seq_len={seq_len}")
    head_indices = (
        [cfg.target_head] if cfg.target_head is not None else list(range(n_heads))
    )

    if "att" in modes and (cfg.layers is None or cfg.target_tokens is None or cfg.target_head is None):
        warnings.warn(
            "Gemma3 ATT IG over all layers/tokens/heads is expensive "
            f"(n_layer={n_layers}, seq_len={seq_len}, n_head={n_heads}). "
            "Prefer layers=, target_tokens=, and target_head=.",
            stacklevel=2,
        )

    result: Dict[str, Any] = {
        "text": text,
        "tokens": tokens,
        "model": cfg.model,
        "model_type": "gemma3",
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
        z_layer = hidden_states[layer_idx]
        layer_out: Dict[str, Any] = {"layer_idx": layer_idx}
        mask = layer_attention_mask(text_model, layer_idx, seq_len, device, z_layer.dtype)
        global_pe, local_pe = rope_position_embeddings(text_model, z_layer)
        gemma_att_itb_u_cache: Dict[str, torch.Tensor] = {}
        gemma_u_pre_full: torch.Tensor | None = None
        if "mlp" in modes:
            block = text_model.layers[layer_idx]
            pe = local_pe if block.self_attn.is_sliding else global_pe
            with torch.no_grad():
                gemma_u_pre_full = attn_pre_oproj_output(
                    block,
                    block.input_layernorm(z_layer),
                    mask,
                    pe,
                )

        if "layer" in modes:
            matrix = _compute_gemma3_layer_z2z(
                text_model=text_model,
                z_layer=z_layer,
                layer_idx=layer_idx,
                cfg=cfg,
            )
            layer_out["z2z"] = {
                "baseline": cfg.baseline_layer,
                "description": "Layer-whole IG (z -> z), Gemma3 pre+post-LN block, L2-scalarized",
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
                    "description": (
                        "ATT IG (z -> u), Gemma3 causal/sliding; per-head scores from "
                        "pre-o_proj attention output (head_dim space; not residual dim). "
                        "IG input is the token embedding propagated through blocks 0..l-1, "
                        "not z^(l) as in the GPT-2/Llama adapters"
                    ),
                    "heads": {},
                }
                for h_idx in head_indices:
                    scores = _gemma3_att_head_contributions(
                        text_model=text_model,
                        input_embeddings=input_embeddings,
                        layer_idx=layer_idx,
                        target_token_idx=t_idx,
                        head_idx=h_idx,
                        baseline_att=cfg.baseline_att,
                        num_steps=cfg.num_steps,
                    )
                    target_entry["z2u"]["heads"][str(h_idx)] = {
                        "contributions": scores,
                        "token_l2_norm": float(np.linalg.norm(scores)),
                    }

            if "mlp" in modes:
                assert gemma_u_pre_full is not None
                u_pre_j = gemma_u_pre_full[0, t_idx, :].clone()
                baseline_u_pre = _gemma3_mlp_baseline_u_pre(
                    text_model=text_model,
                    baseline_mlp=cfg.baseline_mlp,
                    input_embeddings=input_embeddings,
                    layer_idx=layer_idx,
                    target_token_idx=t_idx,
                    u_pre_j=u_pre_j,
                    cache=gemma_att_itb_u_cache,
                )
                mlp = compute_gemma3_mlp_lig_single_token(
                    text_model,
                    layer_idx=layer_idx,
                    z_layer=z_layer,
                    target_mlp_input=u_pre_j.unsqueeze(0),
                    baseline_mlp_input=baseline_u_pre.unsqueeze(0),
                    target_token_idx=t_idx,
                    num_steps=cfg.num_steps,
                    include_residual_connection=cfg.include_residual_connection,
                )
                target_entry["u2z"] = {
                    "baseline": cfg.baseline_mlp,
                    "description": (
                        "Downstream IG (u -> z), Gemma3 concatenated heads before the "
                        "attention output projection to z^(l+1), including output projection, "
                        "post-attention RMSNorm, fixed z residual, and FFN, L2-scalarized"
                    ),
                    "contributions": mlp["contributions"],
                    "input_width": mlp["input_width"],
                    "head_shape": [
                        int(text_model.config.num_attention_heads),
                        int(text_model.config.head_dim),
                    ],
                    "l2_total": mlp["l2_total"],
                    "completeness": {
                        "mean_abs_error": mlp["mean_abs_completeness_error"],
                        "max_abs_error": mlp["max_abs_completeness_error"],
                    },
                }

            targets_out[str(t_idx)] = target_entry

        if "att" in modes or "mlp" in modes:
            layer_out["targets"] = targets_out
        result["layers"][str(layer_idx)] = layer_out

    return result


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
    # Encoder-only implementation; importing it eagerly also imports Lightning.
    from utils.calculations.ig.attention.attention_ig import (
        compute_attention_ig_global_analysis_multi_layer,
    )

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
            from utils.calculations.ig.attention.attention_map_alpha import (
                get_encoder_self_attention_alpha,
            )

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
    from utils.calculations.ig.z2z.layer_direct_ig import (
        compute_layer_direct_ig_all_targets,
    )

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


def _compute_decoder_layer_z2z(
    *,
    adapter: Any,
    z_layer: torch.Tensor,
    layer_idx: int,
    cfg: LIGConfig,
) -> np.ndarray:
    from lig.adapters.decoder_ig.factory import LLAMA_DECODER_TYPES

    model_type = getattr(adapter.model.config, "model_type", "unknown")
    if model_type == "gpt2":
        return _compute_gpt2_layer_z2z(
            gpt2=adapter.model,
            z_layer=z_layer,
            layer_idx=layer_idx,
            cfg=cfg,
        )
    if model_type in LLAMA_DECODER_TYPES:
        from utils.calculations.ig.z2z.llama_layer_direct_ig import (
            compute_llama_layer_direct_ig_all_targets,
        )

        position_ids = adapter.cache_data["position_ids"]
        if cfg.baseline_layer == "itb_zero_ratio":
            z_itb = compute_llama_layer_direct_ig_all_targets(
                llama=adapter.model,
                z_layer=z_layer,
                layer_idx=layer_idx,
                position_ids=position_ids,
                num_steps=cfg.num_steps,
                baseline_method="self_input_token",
            )
            z_zero = compute_llama_layer_direct_ig_all_targets(
                llama=adapter.model,
                z_layer=z_layer,
                layer_idx=layer_idx,
                position_ids=position_ids,
                num_steps=cfg.num_steps,
                baseline_method="zero",
            )
            return apply_layer_z2z_itb_zero_base_ratio(
                z_itb[np.newaxis, ...], z_zero[np.newaxis, ...]
            )[0]
        return compute_llama_layer_direct_ig_all_targets(
            llama=adapter.model,
            z_layer=z_layer,
            layer_idx=layer_idx,
            position_ids=position_ids,
            num_steps=cfg.num_steps,
            baseline_method=cfg.baseline_layer,  # type: ignore[arg-type]
        )
    raise NotImplementedError(f"Layer z2z not implemented for decoder {model_type}")


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


def _compute_gemma3_layer_z2z(
    *,
    text_model: Any,
    z_layer: torch.Tensor,
    layer_idx: int,
    cfg: LIGConfig,
) -> np.ndarray:
    from utils.calculations.ig.z2z.gemma3_layer_direct_ig import (
        compute_gemma3_layer_direct_ig_all_targets,
    )

    if cfg.baseline_layer == "itb_zero_ratio":
        z_itb = compute_gemma3_layer_direct_ig_all_targets(
            text_model=text_model,
            z_layer=z_layer,
            layer_idx=layer_idx,
            num_steps=cfg.num_steps,
            baseline_method="self_input_token",
        )
        z_zero = compute_gemma3_layer_direct_ig_all_targets(
            text_model=text_model,
            z_layer=z_layer,
            layer_idx=layer_idx,
            num_steps=cfg.num_steps,
            baseline_method="zero",
        )
        return apply_layer_z2z_itb_zero_base_ratio(z_itb[np.newaxis, ...], z_zero[np.newaxis, ...])[0]
    return compute_gemma3_layer_direct_ig_all_targets(
        text_model=text_model,
        z_layer=z_layer,
        layer_idx=layer_idx,
        num_steps=cfg.num_steps,
        baseline_method=cfg.baseline_layer,  # type: ignore[arg-type]
    )


def _gemma3_att_head_contributions(
    *,
    text_model: Any,
    input_embeddings: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
    head_idx: int,
    baseline_att: str,
    num_steps: int,
) -> List[float]:
    from captum.attr import IntegratedGradients

    from utils.calculations.ig.attention.attention_map_alpha import get_gemma3_self_attention_alpha
    from utils.calculations.ig.attention.gemma3_attention_models import (
        create_gemma3_attention_model,
    )

    def _scores_for_baseline(baseline_method: str) -> np.ndarray:
        baseline_emb = _gemma3_baseline_embeddings(
            input_embeddings, baseline_method, target_token_idx
        )
        ig_model = create_gemma3_attention_model(
            text_model=text_model,
            layer_idx=layer_idx,
            target_token_idx=target_token_idx,
            target_head_idx=head_idx,
            use_last_token=False,
            debug=False,
        )
        ig_model.eval()
        attr = IntegratedGradients(ig_model).attribute(
            inputs=input_embeddings,
            baselines=baseline_emb,
            n_steps=num_steps,
            method="riemann_trapezoid",
        )
        return attr.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    if baseline_att in ("itb_zero_ratio", "itb_map_ratio"):
        itb = np.asarray(_scores_for_baseline("self_input_token"), dtype=np.float64)
        if baseline_att == "itb_zero_ratio":
            zero = np.asarray(_scores_for_baseline("zero"), dtype=np.float64)
            out = apply_itb_column_zero_ratio(itb, zero, target_token_idx)
        else:
            alpha = get_gemma3_self_attention_alpha(
                text_model,
                inputs_embeds=input_embeddings,
                layer_idx=layer_idx,
                head_idx=head_idx,
                token_idx=target_token_idx,
            )
            out = apply_itb_column_map_ratio(itb, alpha, target_token_idx)
        return out.tolist()

    return _scores_for_baseline(baseline_att).tolist()


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
        from utils.calculations.ig.mlp.att_itb_mlp_baseline import (
            get_mlp_baseline_att_itb_eq_zero,
        )

        return get_mlp_baseline_att_itb_eq_zero(
            bert_model=adapter.model,
            z_layer=z_layer,
            attention_mask=attention_mask,
            layer_idx=layer_idx,
            target_token_idx=target_token_idx,
        )
    raise ValueError(f"Unknown MLP baseline: {cfg.baseline_mlp}")


def _gemma3_mlp_baseline_u_pre(
    *,
    text_model: Any,
    baseline_mlp: str,
    input_embeddings: torch.Tensor,
    layer_idx: int,
    target_token_idx: int,
    u_pre_j: torch.Tensor,
    cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    from utils.calculations.ig.gemma3.block_forward import (
        attn_pre_oproj_output,
        layer_attention_mask,
        rope_position_embeddings,
        run_blocks_up_to,
    )

    if baseline_mlp == "zero":
        return torch.zeros_like(u_pre_j)
    if baseline_mlp != "att_itb_a0":
        raise ValueError(f"Unknown Gemma3 MLP baseline: {baseline_mlp}")

    cache_key = f"{layer_idx}:{target_token_idx}"
    if cache_key in cache:
        return cache[cache_key]

    with torch.no_grad():
        baseline_emb = _gemma3_baseline_embeddings(
            input_embeddings=input_embeddings,
            baseline_method="self_input_token",
            target_token_idx=target_token_idx,
        )
        z_layer_itb = run_blocks_up_to(
            text_model=text_model,
            hidden_states=baseline_emb,
            layer_idx=layer_idx,
        )
        seq_len = z_layer_itb.shape[1]
        mask = layer_attention_mask(
            text_model, layer_idx, seq_len, z_layer_itb.device, z_layer_itb.dtype
        )
        global_pe, local_pe = rope_position_embeddings(text_model, z_layer_itb)
        block = text_model.layers[layer_idx]
        pe = local_pe if block.self_attn.is_sliding else global_pe
        u_pre_full_itb = attn_pre_oproj_output(
            block,
            block.input_layernorm(z_layer_itb),
            mask,
            pe,
        )
        baseline_u_pre = u_pre_full_itb[0, target_token_idx, :].clone()
    cache[cache_key] = baseline_u_pre
    return baseline_u_pre


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
        probe = adapter.model
        if model_type == "gemma3":
            from utils.calculations.ig.gemma3.block_forward import get_gemma3_text_model

            try:
                probe = get_gemma3_text_model(adapter.model)
            except AttributeError:
                probe = adapter.model
        info = detect_boundaries(probe).as_dict()
        info["model"] = model
        info["ig_ready"] = model_type in (
            "gpt2",
            "llama",
            "mistral",
            "qwen2",
            "gemma",
            "gemma3",
        )
        return info

    raise ValueError(
        f"Unsupported model_type '{model_type}' for '{model}'. "
        f"Encoders: {sorted(ENCODER_TYPES)}. Decoders: {sorted(DECODER_FAMILY_TYPES)}."
    )
