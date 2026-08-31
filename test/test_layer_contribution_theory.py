#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer間貢献度計算の理論検証テスト

理論: 5.キャッシュからLayer間貢献度を計算する理論.md
理論式: IG_Layer[i, i'] = Σ_h IG_ATT[i, i', h] * IG_MLP[h, i']
"""

import sys
from pathlib import Path
import numpy as np
import unittest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.calculations.ig.z2z.compose_att_mlp import compute_z2z_from_att_mlp


class TestLayerContributionTheory(unittest.TestCase):
    """Layer間貢献度計算の理論検証"""
    
    def setUp(self):
        """テストデータの準備"""
        # テスト用の小さなデータセット
        self.num_layers = 2
        self.num_heads = 3
        self.num_tokens = 4
        
        # ATTのIGデータ: [num_layers, num_heads, num_tokens, num_tokens]
        # 各要素は IG_ATT[i_input, i_output, h] を表す
        self.attns = np.random.randn(
            self.num_layers, 
            self.num_heads, 
            self.num_tokens, 
            self.num_tokens
        ).tolist()
        
        # MLPのIGデータ: [num_layers, num_tokens, num_heads]
        # 各要素は IG_MLP[i_output, h] を表す
        self.mlp = np.random.randn(
            self.num_layers,
            self.num_tokens,
            self.num_heads
        ).tolist()
    
    def test_z2z_computation_formula(self):
        """理論式の検証: IG_Layer[i, i'] = Σ_h IG_ATT[i, i', h] * IG_MLP[h, i']"""
        z2z_results = compute_z2z_from_att_mlp(self.attns, self.mlp, mode="prod")
        
        attns_array = np.array(self.attns)
        mlp_array = np.array(self.mlp)
        
        # 各層について検証
        for layer_idx in range(self.num_layers):
            z2z_layer = np.array(z2z_results[layer_idx])
            attn_layer = attns_array[layer_idx]  # [num_heads, num_tokens, num_tokens]
            mlp_layer = mlp_array[layer_idx]  # [num_tokens, num_heads]
            
            # 理論式に基づいて手動計算
            expected_z2z = np.zeros((self.num_tokens, self.num_tokens))
            
            for h in range(self.num_heads):
                attn_h = attn_layer[h]  # [num_tokens, num_tokens]
                mlp_h = mlp_layer[:, h]  # [num_tokens]
                
                # 理論: IG_ATT[i_input, i_output, h] * IG_MLP[i_output, h]
                head_contribution = attn_h * mlp_h[np.newaxis, :]
                expected_z2z += head_contribution
            
            # 計算結果が理論式と一致するか確認
            np.testing.assert_allclose(
                z2z_layer,
                expected_z2z,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Layer {layer_idx} の計算結果が理論式と一致しません"
            )
    
    def test_z2z_affine_matches_paper_formula(self):
        """affine（既定, 論文 Eq. layer-decomp-hat）: 列を総和で正規化してから積を取る。

        prod との違いは規約の違いではなく別の推定量である。既定を取り違えると
        論文の表は再現しないので、両方を固定しておく。
        """
        from utils.calculations.ig.z2z.compose_att_mlp import (
            normalize_ig_column_to_affine as nrm,
        )

        z2z_results = compute_z2z_from_att_mlp(self.attns, self.mlp)  # 既定 = affine
        attns_array = np.array(self.attns)
        mlp_array = np.array(self.mlp)

        for layer_idx in range(self.num_layers):
            z2z_layer = np.array(z2z_results[layer_idx])
            attn_layer = attns_array[layer_idx]
            mlp_layer = mlp_array[layer_idx]

            expected = np.zeros((self.num_tokens, self.num_tokens))
            for j in range(self.num_tokens):
                w_mlp = nrm(mlp_layer[j, :])
                for h in range(self.num_heads):
                    expected[:, j] += nrm(attn_layer[h, :, j]) * w_mlp[h]

            np.testing.assert_allclose(
                z2z_layer, expected, rtol=1e-5, atol=1e-5,
                err_msg=f"Layer {layer_idx}: affine 合成が論文の式と一致しません",
            )

    def test_z2z_affine_columns_sum_to_one(self):
        """affine の列は総和 1（Eq. layer-decomp-hat の直後で述べている性質）。"""
        z2z_results = compute_z2z_from_att_mlp(self.attns, self.mlp)
        for layer_idx in range(self.num_layers):
            col_sums = np.array(z2z_results[layer_idx]).sum(axis=0)
            np.testing.assert_allclose(
                col_sums, np.ones_like(col_sums), rtol=1e-4, atol=1e-4,
                err_msg=f"Layer {layer_idx}: affine 合成の列和が 1 ではありません",
            )

    def test_z2z_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            compute_z2z_from_att_mlp(self.attns, self.mlp, mode="bogus")

    def test_z2z_shape(self):
        """出力形状の検証"""
        z2z_results = compute_z2z_from_att_mlp(self.attns, self.mlp)
        
        # 層数の確認
        self.assertEqual(len(z2z_results), self.num_layers)
        
        # 各層の形状確認
        for layer_idx in range(self.num_layers):
            z2z_layer = np.array(z2z_results[layer_idx])
            self.assertEqual(
                z2z_layer.shape,
                (self.num_tokens, self.num_tokens),
                f"Layer {layer_idx} の形状が正しくありません"
            )
    
    def test_z2z_commutativity(self):
        """可換性の検証: ヘッドの順序を変えても結果が同じ"""
        z2z_results1 = compute_z2z_from_att_mlp(self.attns, self.mlp)
        
        # ヘッドの順序を変更
        attns_reordered = [layer[:] for layer in self.attns]
        mlp_reordered = [layer[:] for layer in self.mlp]
        
        # ヘッド0とヘッド1を入れ替え
        for layer_idx in range(self.num_layers):
            attns_reordered[layer_idx][0], attns_reordered[layer_idx][1] = (
                attns_reordered[layer_idx][1],
                attns_reordered[layer_idx][0],
            )
            for tok_idx in range(len(mlp_reordered[layer_idx])):
                row = mlp_reordered[layer_idx][tok_idx]
                row[0], row[1] = row[1], row[0]
        
        z2z_results2 = compute_z2z_from_att_mlp(attns_reordered, mlp_reordered)
        
        # 結果が同じであることを確認（合計なので順序に依存しない）
        for layer_idx in range(self.num_layers):
            np.testing.assert_allclose(
                np.array(z2z_results1[layer_idx]),
                np.array(z2z_results2[layer_idx]),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Layer {layer_idx} でヘッドの順序を変えた結果が異なります"
            )
    
    def test_z2z_zero_mlp(self):
        """MLPがゼロの場合、結果もゼロになることを確認"""
        mlp_zero = [[[0.0] * self.num_heads for _ in range(self.num_tokens)] 
                    for _ in range(self.num_layers)]
        
        z2z_results = compute_z2z_from_att_mlp(self.attns, mlp_zero)
        
        for layer_idx in range(self.num_layers):
            z2z_layer = np.array(z2z_results[layer_idx])
            np.testing.assert_allclose(
                z2z_layer,
                np.zeros((self.num_tokens, self.num_tokens)),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Layer {layer_idx} でMLPがゼロの場合、結果もゼロになるべきです"
            )
    
    def test_z2z_zero_att(self):
        """ATTがゼロの場合、結果もゼロになることを確認"""
        attns_zero = [[[[0.0] * self.num_tokens for _ in range(self.num_tokens)]
                      for _ in range(self.num_heads)] 
                     for _ in range(self.num_layers)]
        
        z2z_results = compute_z2z_from_att_mlp(attns_zero, self.mlp)
        
        for layer_idx in range(self.num_layers):
            z2z_layer = np.array(z2z_results[layer_idx])
            np.testing.assert_allclose(
                z2z_layer,
                np.zeros((self.num_tokens, self.num_tokens)),
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Layer {layer_idx} でATTがゼロの場合、結果もゼロになるべきです"
            )


if __name__ == "__main__":
    unittest.main()








