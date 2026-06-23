"""
Layer z2z（一気通貫）の ITB 寄与に対し、ATT ITB-zeroRatio と同型の自己項推定と再スケーリングを列ごとに適用する。

各層 l・各出力トークン j について、寄与ベクトル c_i = z2z[l, i, j] を ATT の 1 列とみなし、
zero baseline の z2z と同じ j における比率で自己項を推定し、列和（ITB raw の総和）をアンカーとして一様スケールする。
"""

from __future__ import annotations

import numpy as np


def apply_layer_z2z_column_zero_base_ratio(
    itb_col: np.ndarray,
    zero_col: np.ndarray,
    j: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    単一列（固定 j）について ITB-zeroRatio 変換を適用する。

    Args:
        itb_col: 長さ T の ITB（raw）寄与 z2z[:, j] に相当
        zero_col: 同じ形状の zero baseline 寄与
        j: 出力トークンインデックス（自己項の位置）
        eps: 数値安定用

    Returns:
        長さ T の変換後ベクトル
    """
    itb = np.asarray(itb_col, dtype=np.float64).ravel()
    zer = np.asarray(zero_col, dtype=np.float64).ravel()
    n = min(itb.shape[0], zer.shape[0])
    itb = itb[:n].copy()
    zer = zer[:n].copy()
    if j < 0 or j >= n:
        raise IndexError(f"target index j={j} out of range for length {n}")

    sum_other_itb = float(itb.sum() - itb[j])
    sum_other_zero = float(zer.sum() - zer[j])
    if abs(sum_other_zero) > eps:
        est_self = float(zer[j]) * sum_other_itb / sum_other_zero
    else:
        est_self = 0.0

    provisional = itb.copy()
    provisional[j] = est_self
    anchor = float(itb.sum())
    provisional_sum = float(provisional.sum())
    if abs(provisional_sum) <= eps:
        gamma = 1.0
    else:
        gamma = anchor / provisional_sum
    return provisional * gamma


def apply_layer_z2z_itb_zero_base_ratio(
    z_itb: np.ndarray,
    z_zero: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    z2z テンソル全体 [L, T, T] に ITB-zeroRatio を適用する（各 (l, j) で列方向に処理）。
    """
    a = np.asarray(z_itb, dtype=np.float64)
    b = np.asarray(z_zero, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: z_itb {a.shape} vs z_zero {b.shape}")
    if a.ndim != 3:
        raise ValueError(f"expected z2z [L,T,T], got ndim={a.ndim}")
    l_dim, t0, t1 = a.shape
    if t0 != t1:
        raise ValueError(f"expected square token dims, got {t0} x {t1}")
    out = np.zeros_like(a)
    for l in range(l_dim):
        for j in range(t0):
            out[l, :, j] = apply_layer_z2z_column_zero_base_ratio(
                a[l, :, j], b[l, :, j], j, eps=eps
            )
    return out
