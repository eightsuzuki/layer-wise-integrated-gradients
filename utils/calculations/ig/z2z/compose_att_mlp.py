"""
Compose within-layer z→z from cached ATT and MLP IG matrices.

Two modes:

``affine`` (the paper, Eq. layer-decomp-hat) normalizes each boundary column by
its own IG total -- the completeness denominator g(1) - g(0) -- before chaining,
so what is multiplied are unit-sum allocation coefficients:

    IG~^ATT[:, j, h] = IG^ATT[:, j, h] / sum_i IG^ATT[i, j, h]
    IG~^MLP[:, j]    = IG^MLP[j, :]    / sum_h IG^MLP[j, h]
    IG^affine[i, j]  = sum_h IG~^ATT[i, j, h] * IG~^MLP[h, j]

``prod`` chains the raw (unnormalized) scores instead:

    IG^prod[i, j] = sum_h IG^ATT[i, j, h] * IG^MLP[j, h]

The two are not proportional: prod is dominated by columns with a large IG
total, whereas the paper's criterion compares allocations. Reproducing the
published tables requires ``affine``; ``prod`` is kept for older artifacts.
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


_NORM_EPS = 1e-12


def normalize_ig_column_to_affine(vec: np.ndarray, eps: float = _NORM_EPS) -> np.ndarray:
    """IG column -> unit-sum allocation (paper Eq. ig-att-norm).

    Divides by the signed total, which by IG completeness equals g(1) - g(0).
    A column whose total vanishes carries no allocation to distribute; it is
    returned as zeros, and callers must exclude such columns from an average
    rather than scoring them as a distance to the zero vector.
    """
    v = np.asarray(vec, dtype=np.float64)
    s = float(v.sum())
    if abs(s) <= eps:
        return np.zeros_like(v)
    return v / s


def _compute_layer_z2z_affine(
    attn_layer: np.ndarray, mlp_layer: np.ndarray, num_heads: int
) -> np.ndarray:
    att_input_tokens, att_output_tokens = attn_layer.shape[1], attn_layer.shape[2]
    layer_z2z = np.zeros((att_input_tokens, att_output_tokens), dtype=np.float64)
    for j in range(att_output_tokens):
        w_mlp = normalize_ig_column_to_affine(mlp_layer[j, :])
        for h in range(num_heads):
            w_att = normalize_ig_column_to_affine(attn_layer[h, :, j])
            layer_z2z[:, j] += w_att * w_mlp[h]
    return layer_z2z.astype(np.float32)


_LAYER_FN = {"affine": _compute_layer_z2z_affine, "prod": _compute_layer_z2z_prod}


def compute_z2z_from_att_mlp(
    attns: Any, mlp: Any, mode: str = "affine"
) -> List[List[List[float]]]:
    """ATT and MLP IG tensors -> per-layer z->z matrices.

    ``mode`` is ``affine`` (the paper) or ``prod``; see the module docstring.
    """
    if mode not in _LAYER_FN:
        raise ValueError(f"unknown composition mode {mode!r} (affine|prod)")
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
        layer_z2z = _LAYER_FN[mode](attn_layer, mlp_layer, num_heads)
        z2z_results.append(layer_z2z.tolist())
    return z2z_results
