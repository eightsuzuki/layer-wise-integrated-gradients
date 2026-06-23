# IG計算モジュール

## 概要

このモジュールは、BERTモデルのAttentionとMLPのIntegrated Gradients (IG) 計算を統一して管理します。
理論文書に基づく正しいIG計算を提供し、page17とpage18で共通利用できるようにします。

**注意**: 実際の実装は `utils/calculations/ig/` ディレクトリにあります。

## 主な機能

### 1. 理論文書準拠のIG計算関数

#### Attention IG計算
```python
from utils.ig_calculations import compute_attention_ig_theoretical

# 理論文書準拠のAttention IG計算
ig_values = compute_attention_ig_theoretical(
    model_attn, inputs, layer_idx, target_token_idx, num_steps=32
)
```

#### MLP IG計算
```python
from utils.ig_calculations import compute_mlp_ig_theoretical

# 理論文書準拠のMLP IG計算
ig_matrix = compute_mlp_ig_theoretical(
    model_mlp, inputs, layer_idx, target_token_idx, num_steps=32
)
```

### 2. 実用的なIG計算関数（最終層用）

#### 最終層MLP IG計算
```python
from utils.ig_calculations import compute_final_layer_mlp_ig

# 最終層MLPのIG計算（実用的アプローチ）
ig_values = compute_final_layer_mlp_ig(
    model_mlp, inputs, target_token_idx, num_steps=32
)
```

#### 最終層Attention IG計算
```python
from utils.ig_calculations import compute_final_layer_attention_ig

# 最終層AttentionのIG計算
ig_values = compute_final_layer_attention_ig(
    model_attn, inputs, target_token_idx, num_steps=32
)
```

### 3. デバッグ用IG計算関数

#### Attention IGデバッグ
```python
from utils.ig_calculations import compute_attention_ig_debug

# Attention IGデバッグ計算
attn_ig = compute_attention_ig_debug(
    model_attn, tokenizer, text, layer_idx, target_token_idx, 
    target_head_idx=None, num_steps=32
)
```

#### MLP IG計算（統一インターフェース）
```python
from utils.ig_calculations import compute_mlp_beta_contributions

# MLP IG計算（統一インターフェース）
mlp_ig = compute_mlp_beta_contributions(
    model_mlp, tokenizer, text, layer_idx, target_token_idx, num_steps=32
)
```

### 4. 統一インターフェース関数

#### MLP入力β貢献度計算（page18用）
```python
from utils.ig_calculations import compute_mlp_beta_contributions

# MLP入力β貢献度計算の統一インターフェース
beta_contributions = compute_mlp_beta_contributions(
    model_mlp, tokenizer, text, layer_idx, target_token_idx, num_steps=32
)
```

#### Attention貢献度計算（page17用）
```python
from utils.ig_calculations import compute_attention_contributions

# Attention貢献度計算の統一インターフェース
attn_contributions = compute_attention_contributions(
    model_attn, tokenizer, text, layer_idx, target_token_idx, 
    target_head_idx=None, num_steps=32
)
```

## 使用例

### page18での使用例
```python
# page18_mlp_input_analysis.py
from utils.ig_calculations import compute_mlp_beta_contributions

# 最終層のインデックスを取得
final_layer_idx = model_lightning.config.num_hidden_layers - 1

# 統一インターフェース関数を使用して最終層のMLP IGを計算
beta_contributions = compute_mlp_beta_contributions(
    model_mlp, tokenizer, text, final_layer_idx, target_token_idx, num_steps
)
```

### page17での使用例
```python
# page17_mlp_lrp.py
from utils.ig_calculations import compute_attention_contributions
from lrp_calculation_core import compute_mlp_ig_theoretical

# Attention IG計算
attn_ig = compute_attention_contributions(
    model_attn, tokenizer, text, layer_idx, target_token_idx, head_idx, num_steps
)

# MLP IG計算（直接理論関数を使用）
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
mlp_ig = compute_mlp_ig_theoretical(
    model_mlp, inputs, layer_idx, target_token_idx, num_steps
)
```

## 理論文書準拠

### Attention IG計算
理論式: `IG_{i,i'}^{Attn} = (α_i - α_i^base) · ∫_0^1 ∂A_{i'}(a)/∂α_i da`

- 入力: `{α_i}_i`
- ベースライン: `{α_i^base}_i = 0`
- 出力: `β_{i'}(a) = F_{i'}({α_i}_i:a)`
- IGで評価する関数: `A_{i'}(a) = ||F_{i'}(a) - F_{i'}(0)||_2`

### MLP IG計算
理論式: `IG_{h,h'}^{MLP} = (β^{(l,h)} - β^{(l,h), base}) · ∫_0^1 ∂M_{h'}^{(l)}(a)/∂β^{(l,h)} da`

- 入力: `{β^{(l,h)}_h`
- ベースライン: `{β^{(l,h), base}_h = 0`
- 出力: `α^{(l+1,h')}(a) = G^{(l+1,h')}({β^{(l,h)}_h:a)`
- IGで評価する関数: `M_{h'}^{(l)}(a) = ||G^{(l+1,h')}(a) - G^{(l+1,h')}(0)||_2`

## エラーハンドリング

各関数は以下のエラーハンドリング機能を提供します：

1. **理論文書準拠計算の失敗時**: 簡略化された計算にフォールバック
2. **フォールバック計算の失敗時**: デフォルト値（均等分配）を返す
3. **例外発生時**: 適切なエラーメッセージとデフォルト値を返す

## キャッシュ機能

このモジュールは既存のキャッシュシステムと統合されており、以下の機能を提供します：

- 計算結果の自動キャッシュ
- キャッシュヒット時の高速取得
- キャッシュミス時の新規計算

## 依存関係

- `torch`: PyTorch
- `numpy`: 数値計算
- `streamlit`: UI表示
- `captum`: Integrated Gradients計算（オプション）

## 注意事項

1. **計算コスト**: IG計算は計算コストが高いため、短い文・小さい分割数でまず試してください
2. **メモリ使用量**: 大きなモデルや長い文ではメモリ使用量に注意してください
3. **GPU利用**: GPUが利用可能な場合は自動的に使用されます
4. **理論準拠**: 理論文書に基づく正しい計算を提供しますが、近似計算のため完全な精度は保証されません 