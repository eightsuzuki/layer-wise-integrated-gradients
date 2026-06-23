"""
Auto-detect Transformer block boundaries for LIG (z, u, z_next).

Inspects Hugging Face module layout instead of hard-coding every ``model_type``.
Known edge cases (Mamba, Switch, DistilBERT, …) can still override detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from lig.encoder_access import forward_encoder_layer, get_encoder_layers, get_model_type


class BlockLayout(str, Enum):
    """How a single Transformer block wires ATT and MLP around the residual stream."""

    POST_LN_ENCODER = "post_ln_encoder"
    """Post-LN encoder (BERT-family): ``layer.attention`` then ``layer.intermediate`` / ``layer.output``."""

    PRE_LN_DECODER = "pre_ln_decoder"
    """Pre-LN decoder (GPT-2): ``ln_1 → attn → +residual`` gives *u*, then ``ln_2 → mlp → +residual``."""

    BLOCK_ONLY = "block_only"
    """Whole-block forward only; no stable BERT-style ATT/MLP split for z→u / u→z."""


FULL_GRANULARITY: FrozenSet[str] = frozenset({"att", "mlp", "layer"})
LAYER_GRANULARITY: FrozenSet[str] = frozenset({"layer"})

# model_type overrides when module introspection is ambiguous or IG is not wired yet
_LAYER_ONLY_MODEL_TYPES = frozenset(
    {
        "distilbert",
        "modernbert",
        "switch_transformers",
        "mamba",
        "mpnet",
    }
)

_POST_LN_ENCODER_TYPES = frozenset(
    {
        "bert",
        "roberta",
        "xlm-roberta",
        "electra",
        "deberta",
        "deberta-v2",
        "camembert",
        "albert",
    }
)

_PRE_LN_DECODER_TYPES = frozenset({"gpt2", "gpt_neox"})


@dataclass(frozen=True)
class IgHookPoints:
    """Named modules at ATT/MLP boundaries (for introspection or optional forward hooks)."""

    att_input: str
    att_core: Optional[str] = None
    att_output: Optional[str] = None
    mlp_input: Optional[str] = None
    mlp_core: Optional[str] = None
    mlp_output: Optional[str] = None


@dataclass(frozen=True)
class BlockBoundaries:
    """Detected z / u / z_next semantics and supported LIG granularity for one model."""

    layout: BlockLayout
    architecture: str
    model_type: str
    num_layers: int
    supported_granularity: FrozenSet[str]
    z_node: str
    u_node: str
    z_next_node: str
    hook_points: IgHookPoints
    detection: str = "introspection"
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layout": self.layout.value,
            "architecture": self.architecture,
            "model_type": self.model_type,
            "num_layers": self.num_layers,
            "supported_granularity": sorted(self.supported_granularity),
            "nodes": {
                "z": self.z_node,
                "u": self.u_node,
                "z_next": self.z_next_node,
            },
            "hook_points": {
                "att_input": self.hook_points.att_input,
                "att_core": self.hook_points.att_core,
                "att_output": self.hook_points.att_output,
                "mlp_input": self.hook_points.mlp_input,
                "mlp_core": self.hook_points.mlp_core,
                "mlp_output": self.hook_points.mlp_output,
            },
            "detection": self.detection,
            "notes": list(self.notes),
        }


def _layer_has(name: str, layer: nn.Module) -> bool:
    return hasattr(layer, name)


def _inspect_layer_layout(layer: nn.Module) -> BlockLayout:
    """Infer block wiring from the first layer module."""
    if _layer_has("attn", layer) and _layer_has("mlp", layer) and _layer_has("ln_1", layer):
        return BlockLayout.PRE_LN_DECODER
    if _layer_has("attention", layer) and _layer_has("intermediate", layer):
        return BlockLayout.POST_LN_ENCODER
    return BlockLayout.BLOCK_ONLY


def _architecture_for(model_type: str, layout: BlockLayout) -> str:
    if layout == BlockLayout.PRE_LN_DECODER or model_type in _PRE_LN_DECODER_TYPES:
        return "decoder"
    return "encoder"


def _layout_metadata(layout: BlockLayout) -> Tuple[str, str, str, IgHookPoints, FrozenSet[str]]:
    if layout == BlockLayout.PRE_LN_DECODER:
        return (
            "Residual stream at block input z^(l) (Pre-LN)",
            "Post-attention residual z + attn(ln_1(z)) — MLP input u",
            "Block output after MLP residual z^(l+1)",
            IgHookPoints(
                att_input="block input (residual stream)",
                att_core="attn (on ln_1(z))",
                att_output="z + attn(ln_1(z))",
                mlp_input="post-attention residual",
                mlp_core="mlp (on ln_2(u))",
                mlp_output="block output",
            ),
            FULL_GRANULARITY,
        )
    if layout == BlockLayout.POST_LN_ENCODER:
        return (
            "ATT input hidden_states[layer_idx] (residual stream before attention)",
            "ATT output / MLP input (layer.attention output, post-residual)",
            "MLP output hidden_states[layer_idx+1] (next-layer z)",
            IgHookPoints(
                att_input="hidden_states[layer_idx]",
                att_core="attention.self",
                att_output="attention",
                mlp_input="attention output",
                mlp_core="intermediate.dense → output.dense",
                mlp_output="output.LayerNorm",
            ),
            FULL_GRANULARITY,
        )
    return (
        "Block input hidden_states[layer_idx]",
        "N/A — no ATT/MLP split exposed for z→u / u→z",
        "Block output hidden_states[layer_idx+1]",
        IgHookPoints(
            att_input="hidden_states[layer_idx]",
            att_core=None,
            att_output=None,
            mlp_input=None,
            mlp_core=None,
            mlp_output="block output",
        ),
        LAYER_GRANULARITY,
    )


def detect_boundaries(model: nn.Module) -> BlockBoundaries:
    """
    Detect z / u / z_next boundaries and IG hook points from ``model`` architecture.

    Uses ``config.model_type`` overrides for known block-only families, then falls
    back to inspecting the first encoder/decoder block module.
    """
    model_type = get_model_type(model)
    layers = get_encoder_layers(model)
    num_layers = len(layers)
    sample_layer = layers[0]

    if model_type in _LAYER_ONLY_MODEL_TYPES:
        layout = BlockLayout.BLOCK_ONLY
        detection = f"model_type override ({model_type})"
    elif model_type in _POST_LN_ENCODER_TYPES:
        layout = BlockLayout.POST_LN_ENCODER
        detection = f"model_type ({model_type})"
    elif model_type in _PRE_LN_DECODER_TYPES:
        layout = BlockLayout.PRE_LN_DECODER
        detection = f"model_type ({model_type})"
    else:
        layout = _inspect_layer_layout(sample_layer)
        detection = "module introspection"

    z_node, u_node, z_next_node, hook_points, granularity = _layout_metadata(layout)
    architecture = _architecture_for(model_type, layout)
    notes: List[str] = []
    if layout == BlockLayout.BLOCK_ONLY:
        notes.append("Use granularity='layer' (z→z block IG only).")
    if layout == BlockLayout.PRE_LN_DECODER:
        notes.append("Causal mask; u is post-attention residual at the target token.")

    return BlockBoundaries(
        layout=layout,
        architecture=architecture,
        model_type=model_type,
        num_layers=num_layers,
        supported_granularity=granularity,
        z_node=z_node,
        u_node=u_node,
        z_next_node=z_next_node,
        hook_points=hook_points,
        detection=detection,
        notes=tuple(notes),
    )


def z_at_layer(hidden_states: Tuple[torch.Tensor, ...], layer_idx: int) -> torch.Tensor:
    """z^(l): residual stream input to block ``layer_idx``."""
    return hidden_states[layer_idx]


def u_from_z(
    *,
    model: nn.Module,
    boundaries: BlockBoundaries,
    layer_idx: int,
    z_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_idx: int,
) -> torch.Tensor:
    """u_j: ATT output = MLP input at ``target_token_idx``."""
    if boundaries.layout == BlockLayout.BLOCK_ONLY:
        raise NotImplementedError(
            f"{boundaries.model_type} has no ATT/MLP boundary; use granularity='layer'."
        )

    layer = get_encoder_layers(model)[layer_idx]

    if boundaries.layout == BlockLayout.POST_LN_ENCODER:
        with torch.no_grad():
            attn_out = layer.attention(z_layer, attention_mask)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        return attn_out[0, target_token_idx, :].clone()

    if boundaries.layout == BlockLayout.PRE_LN_DECODER:
        from utils.calculations.ig.gpt2.block_forward import hidden_after_attn_residual

        with torch.no_grad():
            u_full = hidden_after_attn_residual(layer, z_layer, attention_mask)
        return u_full[0, target_token_idx, :].clone()

    raise NotImplementedError(f"Unsupported layout: {boundaries.layout}")


def forward_block(
    *,
    model: nn.Module,
    boundaries: BlockBoundaries,
    layer_idx: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """One block forward: z^(l) → z^(l+1)."""
    layer = get_encoder_layers(model)[layer_idx]

    if boundaries.layout == BlockLayout.PRE_LN_DECODER:
        from utils.calculations.ig.gpt2.block_forward import forward_gpt2_block

        return forward_gpt2_block(layer, hidden_states, attention_mask)

    return forward_encoder_layer(
        model,
        layer,
        hidden_states,
        attention_mask,
        layer_idx=layer_idx,
    )


def resolve_hook_modules(
    layer: nn.Module, boundaries: BlockBoundaries
) -> Mapping[str, Optional[nn.Module]]:
    """
    Resolve IG boundary modules on one block (for debugging / optional hooks).

    Values may be ``None`` when the layout does not expose that boundary.
    """
    out: Dict[str, Optional[nn.Module]] = {
        "att_core": None,
        "att_output": None,
        "mlp_core": None,
        "mlp_output": None,
    }
    if boundaries.layout == BlockLayout.POST_LN_ENCODER:
        att = getattr(layer, "attention", None)
        out["att_core"] = getattr(att, "self", None) if att is not None else None
        out["att_output"] = att
        out["mlp_core"] = getattr(layer, "intermediate", None)
        out["mlp_output"] = getattr(layer, "output", None)
    elif boundaries.layout == BlockLayout.PRE_LN_DECODER:
        out["att_core"] = getattr(layer, "attn", None)
        out["att_output"] = layer
        out["mlp_core"] = getattr(layer, "mlp", None)
        out["mlp_output"] = layer
    return out


def describe_boundaries_from_config(model_name: str) -> Dict[str, Any]:
    """
    Lightweight boundary summary from ``AutoConfig`` only (no weight download).

    Falls back to typical layout per ``model_type`` when weights are unavailable.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    model_type = str(getattr(config, "model_type", "unknown"))
    num_layers = int(
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None)
        or getattr(config, "num_layers", 0)
    )

    if model_type in _LAYER_ONLY_MODEL_TYPES:
        layout = BlockLayout.BLOCK_ONLY
        detection = f"config model_type ({model_type})"
    elif model_type in _POST_LN_ENCODER_TYPES:
        layout = BlockLayout.POST_LN_ENCODER
        detection = f"config model_type ({model_type})"
    elif model_type in _PRE_LN_DECODER_TYPES:
        layout = BlockLayout.PRE_LN_DECODER
        detection = f"config model_type ({model_type})"
    else:
        layout = BlockLayout.BLOCK_ONLY
        detection = "config fallback (unknown layout — load model for introspection)"

    z_node, u_node, z_next_node, hook_points, granularity = _layout_metadata(layout)
    architecture = _architecture_for(model_type, layout)
    notes: List[str] = []
    if layout == BlockLayout.BLOCK_ONLY and model_type not in _LAYER_ONLY_MODEL_TYPES:
        notes.append("Load the model and call detect_boundaries(model) to refine layout.")
    elif layout == BlockLayout.BLOCK_ONLY:
        notes.append("Block-level z→z only.")

    return BlockBoundaries(
        layout=layout,
        architecture=architecture,
        model_type=model_type,
        num_layers=num_layers,
        supported_granularity=granularity,
        z_node=z_node,
        u_node=u_node,
        z_next_node=z_next_node,
        hook_points=hook_points,
        detection=detection,
        notes=tuple(notes),
    ).as_dict()
