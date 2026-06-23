"""layer_itb_zero_ratio: 列和アンカーと自己項推定のスモークテスト。"""

import numpy as np

from utils.calculations.ig.z2z.layer_itb_zero_ratio import (
    apply_layer_z2z_column_zero_base_ratio,
    apply_layer_z2z_itb_zero_base_ratio,
)


def test_column_preserves_sum():
    itb = np.array([0.4, 0.0, 0.6], dtype=np.float64)
    zero = np.array([0.1, 0.5, 0.2], dtype=np.float64)
    j = 1
    out = apply_layer_z2z_column_zero_base_ratio(itb, zero, j)
    assert np.isclose(out.sum(), itb.sum())
    assert out.shape == itb.shape


def test_full_tensor_runs():
    L, T = 2, 4
    z_itb = np.random.RandomState(0).rand(L, T, T).astype(np.float64)
    for j in range(T):
        z_itb[:, j, j] = 0.0
    z_zero = np.random.RandomState(1).rand(L, T, T).astype(np.float64)
    out = apply_layer_z2z_itb_zero_base_ratio(z_itb, z_zero)
    assert out.shape == z_itb.shape
    for l in range(L):
        for j in range(T):
            assert np.isclose(out[l, :, j].sum(), z_itb[l, :, j].sum())
