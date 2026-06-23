# 計算モジュール構造

このディレクトリは、BERT分析の計算ロジックを整理した構造です。

## 📁 ディレクトリ構造

```
utils/calculations/
├── ig/                          # Integrated Gradients (IG) 計算
│   ├── attention/               # Attention IG 計算
│   │   ├── attention_ig.py      # Attention IG メイン実装
│   │   └── attention_models.py  # Attention モデル定義
│   ├── mlp/                     # MLP IG 計算
│   │   ├── mlp_ig.py           # MLP IG メイン実装
│   │   └── mlp_models.py       # MLP モデル定義
│   ├── global_analysis.py       # 全体分析（IG統合）
│   ├── optimized_ig.py         # 最適化IG計算器
│   └── parallel_ig_calculator.py # 並列IG計算器
├── lrp/                         # Layer-wise Relevance Propagation (LRP) 計算
│   ├── lrp_analysis.py         # LRP メイン実装
│   └── important_path_analysis.py # 重要経路分析
└── shared/                      # 共通ユーティリティ
    ├── device_utils.py          # デバイス管理
    └── ig_calculations.py       # IG計算共通関数
```

## 🔬 IG (Integrated Gradients) 計算

### z2u (z→u) - Attention IG
- **実装**: `ig/attention/attention_ig.py`
- **入力**: `z^(l)` (層lの隠れ状態)
- **出力**: `u^(l,h)` (層lのヘッドhのAttention出力)
- **ベースライン**: Zero, Self-Input-Token, Self-Output-Token
- **スカラー化**: L2ノルム差 `||u_{i'}(a) - u_{i'}(0)||_2`
- **説明**: z_i^(l) → u_{i'}^(l,h) の貢献度計算

### v2u (v→u) - Direct Computation
- **実装**: `ig/attention/direct_computation.py`
- **入力**: `v_j` (Valueベクトル)
- **出力**: `u_i` (Attention出力)
- **ベースライン**: Zero, Self-Input-Token, Self-Output-Token
- **計算方法**: 線形性を利用した直接計算（IGの数値積分は不要）
- **説明**: v_j → u_i の貢献度計算（理論文書5.3節に基づく）

### u2z (u→z) - MLP IG
- **実装**: `ig/mlp/mlp_ig.py`
- **入力**: `u^(l,h)` (層lのヘッドhのAttention出力)
- **出力**: `z^(l+1)` (層l+1の隠れ状態)
- **ベースライン**: Zero, Self-Input-Token
- **スカラー化**: L2ノルム差 `||z_{i'}(a) - z_{i'}(0)||_2`
- **説明**: u_i^(l,h) → z_i^(l+1) の貢献度計算

### z2z (z→z) - Layer間貢献度
- **実装**: `ig/z2z/global_z2z_analysis.py`
- **入力**: `z^(l)` (層lの隠れ状態)
- **出力**: `z^(l+1)` (層l+1の隠れ状態)
- **計算方法**: ATTとMLPの貢献度を合成（IG_{i,i'}^{Layer} = Σ_h IG_{i,i'}^{ATT} * IG_{h,i'}^{MLP}）
- **説明**: z_i^(l) → z_{i'}^(l+1) の貢献度計算（Layer間貢献度）

## 🔄 LRP (Layer-wise Relevance Propagation) 計算

### 逆伝播計算
- **最終層初期化**: `A(z_i^(L)) = 1`
- **MLP部分**: `A(u_i^(l,h)) = A(z_i^(l+1)) * R^MLP,l_{i,(h)}`
- **Attention部分**: `A(z_i^(l)) = Σ_{i'} A(u_{i'}^(l,h)) * R^Attn,l_{h,(i',i)}`
- **最終貢献度**: `A(z_i^(0))` = 入力トークンiの最終貢献度

## 🚀 使用方法

### IG計算
```python
from utils.calculations.ig.global_analysis import compute_global_ig_analysis
from utils.calculations.ig.attention.attention_ig import compute_attention_ig_with_verification
from utils.calculations.ig.mlp.mlp_ig import compute_mlp_ig_theoretical_with_cache
```

### LRP計算
```python
from utils.calculations.lrp.lrp_analysis import compute_lrp_analysis
from utils.calculations.lrp.important_path_analysis import compute_lrp_backpropagation_paths
```

### 共通関数
```python
from utils.calculations.shared.ig_calculations import calculate_attention_relevance
from utils.calculations.shared.device_utils import ensure_model_on_device
```

## 📊 理論的整合性

- **IG**: 理論文書「2.transformerのLRPについて.md」に完全準拠
- **LRP**: 層ごとの関連性伝播を数学的に厳密に実装
- **ベースライン**: ゼロベクトル（理論通り）
- **スカラー化**: L2ノルム（理論通り）
- **数値積分**: 32分割近似（理論推奨値）

## 🔧 最適化

- **H100対応**: 混合精度計算、マルチGPU並列処理
- **キャッシュシステム**: 階層構造 `./cache/{cache_type}/{text_hash}/`
- **並列処理**: ThreadPoolExecutor による効率的なバッチ処理
- **重複排除**: 同一計算の重複実行を防止
