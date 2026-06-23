# LIG 計算の概要（実装・計算量・性能）

Layer-wise Integrated Gradients (LIG) リポジトリにおける z2z 可視化用データの計算経路をまとめる。

## 1. Layer-direct z2z（一気通貫 IG）

層入力 z^(l) をベースラインから実入力まで補間し、1 層全体を forward して z^(l+1) を得る。ターゲット j について
L_j = ||z_j^(l+1)(a) - z_j^(l+1)(0)||_2 に対する IG を各入力トークン i で求める。

- 実装: `utils/calculations/ig/z2z/layer_direct_ig.py`
- バッチ: `scripts/reproduce/run_layer_direct_ig.py`
- 出力: `z2z[layer][target_j][source_i]`

Captum は全補間点を 1 バッチで forward するため、`ig.attribute()` 前に `set_baseline_output()` で z_j^(l+1)(0) を事前計算する必要がある。

### ベースライン

| 名前 | baseline_method | 備考 |
|------|-----------------|------|
| Zero | zero | ゼロベクトル |
| ITB | self_input_token | ターゲット j の入力を全位置に展開 |
| ITB-zeroRatio | 後処理 | ITB+Zero から `layer_itb_zero_ratio.py` で導出 |

## 2. z2u / u2z

| 経路 | 実装 |
|------|------|
| z2u | `utils/calculations/ig/attention/attention_ig.py` |
| u2z | `utils/calculations/ig/mlp/mlp_lig_ig.py`, `mlp_ig.py` |

## 3. 積合成 z2z

IG^prod_{i,j} = sum_h IG_ATT[i,j,h] * IG_MLP[j,h]

- `utils/calculations/ig/z2z/compose_att_mlp.py`
- `scripts/reproduce/compose_z2z.py`（8 組 → `z2z/composed/`）

## 4. 計算量

| 処理 | オーダー |
|------|----------|
| モデル load | O(1) per batch run |
| full BERT forward | O(1) |
| layer-direct / 層 | O(T * (B + S*F)) |
| 全体 | O(L * T * (B + S*F)) |
| 積合成 | O(L * H * T^2) numpy |

T=seq_len, L=12, S=32, B=baseline 層 forward 回数。

zero baseline 最適化後: 層あたり B=1（従来 T 回）。

## 5. モデル・データ移動

- モデルはサンプルループ外で 1 回 load（再 load なし）
- 主ボトルネックは Captum IG の層 forward 回数

## 6. キャッシュ

`PTB_CACHE_ROOT/samples/dev/{att,mlp,z2z}/.../sample_XXXXX.json`

## 7. プロファイル結果

- sample: dev #410, baseline: `zero`
- device: `cuda` (CUDA_VISIBLE_DEVICES=3)
- seq_len: 41, layers profiled: 12

| metric | seconds |
|--------|---------|
| model_load | 1.781 |
| full_forward | 0.159 |
| layer_ig_total | 1.731 |
| layer_ig_mean | 0.144 |
| est_layer_forwards | 16236 |

Per-layer IG time (s): 0.255, 0.134, 0.134, 0.134, 0.134, 0.134, 0.134, 0.134, 0.134, 0.135, 0.135, 0.135

_Profile JSON: `scripts/verify/reports/logs/profile_sample_00410_zero_20260621_191728.json`_

## 8. 関連スクリプト

- `scripts/verify/profile_layer_direct_ig.py`
- `scripts/verify/validate_sample_00410_cache.py`
- `scripts/reproduce/compare_layer_vs_composed.py`
