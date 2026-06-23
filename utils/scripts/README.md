# ユーティリティスクリプト

このディレクトリには、データ準備、整理、修正、進捗確認などのユーティリティスクリプトが含まれています。

## 主要スクリプト

### layer_consistency 補助（scripts から移管）

- `utils/scripts/layer_consistency/run_uas_layer_ig_vs_composed.py`
- `utils/scripts/layer_consistency/summarize_uas_layer_ig_vs_composed.py`
- `utils/scripts/layer_consistency/phase_c_vs_uas_report.py`
- `utils/scripts/layer_consistency/generate_layer_ig_itb_zero_ratio_cache.py`

備考:
- `scripts/` 側には後方互換のラッパーを残しているため、旧コマンドでも動作します。

### データ準備スクリプト

#### 1. MLP IG計算

**`run_ptb_mlp_ig.py`**

PTBサンプルに対してMLP IG計算を実行し、MLPデータを生成するスクリプト。
既存のPTB JSONファイルにMLP IG計算結果を追加します。

**実装**: `utils/calculations/ig/mlp/mlp_ig.py` の `compute_mlp_ig_theoretical_with_cache` を使用

```bash
python utils/scripts/run_ptb_mlp_ig.py \
  --split dev \
  --num-samples 1700 \
  --start-sample 0 \
  --end-sample 500 \
  --ig-num-steps 32 \
  --baseline-method zero \
  --no-mlp-residual-connection \
  --log-file logs/mlp_ig.log
```

**出力**:
- MLPデータ: `cache/ptb_ig_analysis/samples/{split}/mlp/steps32_..._u_to_z_baseline_{baseline_method}_mlp_residual_{on|off}/`

#### 2. Layer間貢献度（z2z）計算

**`compute_z2z_from_cache.py`**

PTBキャッシュからLayer間貢献度（z->z）を計算するスクリプト。
既存のPTB JSONファイルからATTとMLPのIG計算結果を読み込み、
理論「5.キャッシュからLayer間貢献度を計算する理論.md」に基づいて
Layer間の貢献度 z_i^(l) -> z_{i'}^(l+1) を計算し、JSONファイルに追加します。

**実装**: `utils/calculations/ig/z2z/global_z2z_analysis.py` を使用

```bash
python utils/scripts/compute_z2z_from_cache.py \
  --split dev \
  --num-samples 1700 \
  --start-sample 0 \
  --end-sample 100 \
  --baseline-method zero \
  --mlp-residual-mode on \
  --num-workers 4
```

**入力**:
- ATTデータ: `cache/ptb_ig_analysis/samples/{split}/att/steps32_..._z_to_u_baseline_{baseline_method}/`
- MLPデータ: `cache/ptb_ig_analysis/samples/{split}/mlp/steps32_..._u_to_z_baseline_{baseline_method}_mlp_residual_{on|off}/`

**出力**:
- z2zデータ: `cache/ptb_ig_analysis/samples/{split}/z2z/steps32_..._z_to_z_baseline_{baseline_method}[_mlp_residual_off]/`

### データ整理スクリプト

#### 3. ATT/MLPデータの再整理

**`reorganize_attn_mlp_data.py`**

ATT/MLPデータ分離と再構成スクリプト。
既存の`steps32_bert-base-uncased_maxlen128_z_to_u_baseline_zero`ディレクトリから
ATTデータとMLPデータを分離し、`attn/`と`mlp/`ディレクトリに再構成します。

```bash
python utils/scripts/reorganize_attn_mlp_data.py \
  --base-dir cache/ptb_ig_analysis/samples/dev/z_to_u \
  --source-dir-name steps32_bert-base-uncased_maxlen128_z_to_u_baseline_zero \
  --execute
```

**実行モード**:
- `--dry-run`: ドライラン（実際の操作は行わない）
- `--execute`: 実際に実行（バックアップ自動作成）
- `--backup-only`: バックアップのみ作成

### データ修正スクリプト

#### 4. v_to_u_directデータの修正

**`fix_v_to_u_direct_both.py`**

`v_to_u_direct`のboth方向データを正しく修正するスクリプト。
`dep->head`と`head<-dep`の結果から`both`方向のデータを再生成します。

```bash
python utils/scripts/fix_v_to_u_direct_both.py
```

### 進捗確認スクリプト

#### 5. Herculesサーバーの進捗確認

**`check_hercules_progress.sh`**

Herculesサーバーの進捗確認スクリプト。

```bash
bash utils/scripts/check_hercules_progress.sh
```

## IG計算コードの場所

実際のIG計算の実装は `utils/calculations/ig/` ディレクトリにあります：

- **z2u (z→u)**: `utils/calculations/ig/attention/attention_ig.py`
  - z_i^(l) → u_{i'}^(l,h) の貢献度計算（Attention IG）
  - ベースライン: Zero, Self-Input-Token, Self-Output-Token

- **v2u (v→u)**: `utils/calculations/ig/attention/direct_computation.py`
  - v_j → u_i の貢献度計算（直接計算、線形性を利用）
  - ベースライン: Zero, Self-Input-Token, Self-Output-Token

- **u2z (u→z)**: `utils/calculations/ig/mlp/mlp_ig.py`
  - u_i^(l,h) → z_i^(l+1) の貢献度計算（MLP IG）
  - ベースライン: Zero, Self-Input-Token

- **z2z (z→z)**: `utils/calculations/ig/z2z/global_z2z_analysis.py`
  - z_i^(l) → z_{i'}^(l+1) の貢献度計算（Layer間貢献度）
  - ATTとMLPの貢献度を合成して計算

詳細は [`utils/calculations/README.md`](../calculations/README.md) を参照してください。
