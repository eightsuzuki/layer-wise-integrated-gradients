"""ITB zeroRatio / mapRatio 列変換のユニットテスト。"""

import numpy as np

from utils.calculations.ig.shared.itb_self_contrib import (
    apply_itb_column_map_ratio,
    apply_itb_column_zero_ratio,
)


def test_zero_ratio_preserves_column_sum():
    itb = np.array([0.4, 0.0, 0.6], dtype=np.float64)
    zero = np.array([0.1, 0.5, 0.2], dtype=np.float64)
    j = 1
    out = apply_itb_column_zero_ratio(itb, zero, j)
    assert np.isclose(out.sum(), itb.sum())
    assert out[j] != 0.0 or np.isclose(itb.sum(), 0.0)


def test_map_ratio_preserves_column_sum():
    itb = np.array([0.3, 0.0, 0.7], dtype=np.float64)
    j = 1
    out = apply_itb_column_map_ratio(itb, alpha_jj=0.25, j=j)
    assert np.isclose(out.sum(), itb.sum())
    assert out.shape == itb.shape
