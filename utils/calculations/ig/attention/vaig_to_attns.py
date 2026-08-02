"""
Collapse VAIG ``vaig_vectors`` to scalar ATT ``attns`` layout for z2z composition.

Each VAIG cell is ``[seq_len, output_dim]`` per (layer, head, target_j).
Summing over ``output_dim`` yields a scalar IG per input token i, matching
ATT cache layout ``attns[layer][head][i][j]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def collapse_vaig_vectors_to_scalar(vaig_vectors: Any) -> np.ndarray:
    """
    Sum VAIG over output dimensions.

    Args:
        vaig_vectors: ``[seq_len, output_dim]`` or already-collapsed ``[seq_len]``.

    Returns:
        1-D float64 array of length ``seq_len``.
    """
    arr = np.asarray(vaig_vectors, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.sum(axis=1)
    raise ValueError(
        f"vaig_vectors must be 1-D or 2-D, got shape {arr.shape}"
    )


def _layer_head_target_vaig_to_matrix(
    layer_data: Any,
    *,
    num_heads: int | None = None,
    seq_len: int | None = None,
) -> np.ndarray | None:
    """
    Convert one layer's VAIG payload to ``[H, T, T]``.

    Supported ``layer_data`` shapes:
    - dict ``head -> target_j -> {vaig_vectors}`` or ``head -> [targets...]``
    - list ``[head][target_j]`` with dict or raw vectors
    """
    if isinstance(layer_data, dict):
        head_keys = sorted(
            (int(k) if str(k).isdigit() else k for k in layer_data.keys()),
            key=lambda x: int(x) if isinstance(x, int) or str(x).isdigit() else str(x),
        )
        if not head_keys:
            return None
        if num_heads is None:
            num_heads = max(int(k) for k in head_keys) + 1
        heads_out: list[np.ndarray | None] = [None] * num_heads
        for hk in head_keys:
            h_idx = int(hk)
            mat = _head_targets_to_matrix(layer_data[str(hk)] if str(hk) in layer_data else layer_data[hk], seq_len=seq_len)
            if mat is not None:
                heads_out[h_idx] = mat
                if seq_len is None:
                    seq_len = mat.shape[0]
        if all(h is None for h in heads_out):
            return None
        assert seq_len is not None
        filled = [
            h if h is not None else np.zeros((seq_len, seq_len), dtype=np.float64)
            for h in heads_out
        ]
        return np.stack(filled, axis=0)

    if isinstance(layer_data, list):
        if not layer_data:
            return None
        heads = [_head_targets_to_matrix(hd, seq_len=seq_len) for hd in layer_data]
        if all(h is None for h in heads):
            return None
        seq_len = seq_len or next(h.shape[0] for h in heads if h is not None)
        filled = [
            h if h is not None else np.zeros((seq_len, seq_len), dtype=np.float64)
            for h in heads
        ]
        return np.stack(filled, axis=0)

    return None


def _head_targets_to_matrix(
    head_data: Any,
    *,
    seq_len: int | None = None,
) -> np.ndarray | None:
    """Build ``[T, T]`` for one head (rows=i, cols=j)."""
    if isinstance(head_data, dict):
        target_keys = sorted(
            (int(k) if str(k).isdigit() else k for k in head_data.keys()),
            key=lambda x: int(x) if isinstance(x, int) or str(x).isdigit() else str(x),
        )
        if not target_keys:
            return None
        cols: dict[int, np.ndarray] = {}
        for tk in target_keys:
            j = int(tk)
            cell = head_data[str(tk)] if str(tk) in head_data else head_data[tk]
            vec = _extract_vaig_vectors(cell)
            if vec is None:
                continue
            cols[j] = vec
            if seq_len is None:
                seq_len = len(vec)
        if not cols or seq_len is None:
            return None
        mat = np.zeros((seq_len, seq_len), dtype=np.float64)
        for j, vec in cols.items():
            n = min(seq_len, len(vec))
            mat[:n, j] = vec[:n]
        return mat

    if isinstance(head_data, list):
        if not head_data:
            return None
        cols = []
        for cell in head_data:
            vec = _extract_vaig_vectors(cell)
            cols.append(vec)
        if not any(c is not None for c in cols):
            return None
        seq_len = seq_len or max(len(c) for c in cols if c is not None)
        mat = np.zeros((seq_len, seq_len), dtype=np.float64)
        for j, vec in enumerate(cols):
            if vec is None:
                continue
            n = min(seq_len, len(vec))
            mat[:n, j] = vec[:n]
        return mat

    return None


def _extract_vaig_vectors(cell: Any) -> np.ndarray | None:
    if cell is None:
        return None
    if isinstance(cell, dict):
        if "vaig_vectors" in cell:
            return collapse_vaig_vectors_to_scalar(cell["vaig_vectors"])
        if "attns" in cell:
            arr = np.asarray(cell["attns"], dtype=np.float64)
            return arr if arr.ndim == 1 else arr.sum(axis=-1)
    if isinstance(cell, (list, tuple, np.ndarray)):
        return collapse_vaig_vectors_to_scalar(cell)
    return None


def vaig_nested_to_attns(vaig_root: Any) -> list[list[list[list[float]]]]:
    """
    Convert nested VAIG cache to ATT list form ``[L][H][T][T]``.
    """
    if isinstance(vaig_root, dict):
        layer_keys = sorted(
            (int(k) if str(k).isdigit() else k for k in vaig_root.keys()),
            key=lambda x: int(x) if isinstance(x, int) or str(x).isdigit() else str(x),
        )
        layers: list[np.ndarray] = []
        seq_len: int | None = None
        for lk in layer_keys:
            ld = vaig_root[str(lk)] if str(lk) in vaig_root else vaig_root[lk]
            mat = _layer_head_target_vaig_to_matrix(ld, seq_len=seq_len)
            if mat is None:
                continue
            if seq_len is None:
                seq_len = mat.shape[1]
            layers.append(mat)
        if not layers:
            raise ValueError("No VAIG layers could be parsed from dict")
        return [layer.tolist() for layer in layers]

    if isinstance(vaig_root, list):
        layers = []
        seq_len: int | None = None
        for ld in vaig_root:
            mat = _layer_head_target_vaig_to_matrix(ld, seq_len=seq_len)
            if mat is None:
                continue
            if seq_len is None:
                seq_len = mat.shape[1]
            layers.append(mat)
        if not layers:
            raise ValueError("No VAIG layers could be parsed from list")
        return [layer.tolist() for layer in layers]

    raise ValueError(f"Unsupported vaig_root type: {type(vaig_root)}")


def vaig_cache_to_att_data(vaig_data: dict[str, Any]) -> dict[str, Any]:
    """
    Build an ATT-like sample dict with ``attns`` from a VAIG cache JSON payload.

    If ``attns`` is already present, returns a shallow copy unchanged.
    Otherwise reads ``vaig`` / ``vaig_attns`` and collapses ``vaig_vectors``.
    """
    if "attns" in vaig_data and vaig_data["attns"]:
        out = dict(vaig_data)
        return out

    if "vaig_attns" in vaig_data and vaig_data["vaig_attns"]:
        out = dict(vaig_data)
        out["attns"] = vaig_data["vaig_attns"]
        return out

    vaig_root = vaig_data.get("vaig")
    if vaig_root is None:
        raise KeyError("VAIG cache must contain 'attns', 'vaig_attns', or 'vaig'")

    attns = vaig_nested_to_attns(vaig_root)
    out = {k: v for k, v in vaig_data.items() if k != "vaig"}
    out["attns"] = attns
    return out
