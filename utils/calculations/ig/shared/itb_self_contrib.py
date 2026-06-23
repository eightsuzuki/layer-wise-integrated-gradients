"""
ITB 自己項の zeroRatio / mapRatio 補完（LIG 公開実装 §4 相当）。

ATT (z→u) 列ベクトルと Layer (z→z) 列ベクトルに同型の変換を適用する。
"""

from __future__ import annotations

import numpy as np

from utils.calculations.ig.z2z.layer_itb_zero_ratio import (
    apply_layer_z2z_column_zero_base_ratio,
)


def apply_itb_column_zero_ratio(
    itb_col: np.ndarray,
    zero_col: np.ndarray,
    j: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """ITB-zeroRatio: zero baseline 自己項比率で ITB 自己項を推定し列和を保持。"""
    return apply_layer_z2z_column_zero_base_ratio(itb_col, zero_col, j, eps=eps)


def apply_itb_column_map_ratio(
    itb_col: np.ndarray,
    alpha_jj: float,
    j: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    ITB-mapRatio: Attention 重み α_{j,j} から自己項を推定し列和を保持。

    est_self ≈ (Σ_{k≠j} IG_{k,j}^{ITB}) · α_{j,j} / (1 - α_{j,j})
    """
    itb = np.asarray(itb_col, dtype=np.float64).ravel()
    n = itb.shape[0]
    if j < 0 or j >= n:
        raise IndexError(f"target index j={j} out of range for length {n}")

    sum_other_itb = float(itb.sum() - itb[j])
    denom = 1.0 - float(alpha_jj)
    if abs(denom) > eps:
        alpha_ratio = float(alpha_jj) / denom
    else:
        alpha_ratio = 0.0
    est_self = sum_other_itb * alpha_ratio

    provisional = itb.copy()
    provisional[j] = est_self
    anchor = float(itb.sum())
    provisional_sum = float(provisional.sum())
    if abs(provisional_sum) <= eps:
        gamma = 1.0
    else:
        gamma = anchor / provisional_sum
    return provisional * gamma
