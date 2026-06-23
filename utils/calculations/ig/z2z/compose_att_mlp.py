"""
Compose within-layer z→z from cached ATT and MLP IG matrices (product mode).

IG^prod_{i,j} = Σ_h IG_ATT[i, j, h] * IG_MLP[j, h]
"""

from __future__ import annotations

from typing import Any, List

import numpy as np


def _prepare_att_mlp_arrays(attns: Any, mlp: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if attns is None or mlp is None:
        return None, None
    attns_array = np.array(attns, dtype=np.float64)
    mlp_array = np.array(mlp, dtype=np.float64)
    if attns_array.ndim != 4 or mlp_array.ndim != 3:
        return None, None
    return attns_array, mlp_array


def _align_attn_mlp_layer(
    attn_layer: np.ndarray,
    mlp_layer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    att_output_tokens = attn_layer.shape[2]
    mlp_tokens = mlp_layer.shape[0]
    if att_output_tokens != mlp_tokens:
        n = min(att_output_tokens, mlp_tokens)
        attn_layer = attn_layer[:, :n, :n]
        mlp_layer = mlp_layer[:n, :]
    return attn_layer, mlp_layer


def _compute_layer_z2z_prod(
    attn_layer: np.ndarray, mlp_layer: np.ndarray, num_heads: int
) -> np.ndarray:
    att_input_tokens, att_output_tokens = attn_layer.shape[1], attn_layer.shape[2]
    layer_z2z = np.zeros((att_input_tokens, att_output_tokens), dtype=np.float32)
    for h in range(num_heads):
        layer_z2z += attn_layer[h] * mlp_layer[:, h][np.newaxis, :]
    return layer_z2z


def compute_z2z_from_att_mlp(attns: Any, mlp: Any) -> List[List[List[float]]]:
    """ATT and MLP IG tensors → per-layer z→z matrices (product composition)."""
    prepared = _prepare_att_mlp_arrays(attns, mlp)
    if prepared[0] is None:
        return []
    attns_array, mlp_array = prepared
    num_layers = attns_array.shape[0]
    num_heads = attns_array.shape[1]
    z2z_results: List[List[List[float]]] = []
    for layer_idx in range(num_layers):
        attn_layer, mlp_layer = _align_attn_mlp_layer(
            attns_array[layer_idx], mlp_array[layer_idx]
        )
        layer_z2z = _compute_layer_z2z_prod(attn_layer, mlp_layer, num_heads)
        z2z_results.append(layer_z2z.tolist())
    return z2z_results
