# UV 環境での実行（Phase A 再計算など）

Docker を使わず、ホストで UV により仮想環境を用意して Phase A（Layer IG 再計算）などを実行する手順。

## 前提

- Python 3.10 以上
- [uv](https://github.com/astral-sh/uv) がインストールされていること  
  - インストール: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- （GPU を使う場合）NVIDIA ドライバと CUDA 対応の PyTorch が利用可能な環境

## 初回セットアップ

プロジェクトルートで:

```bash
bash scripts/ops/setup_uv_env.sh
```

- `.venv` が作成される
- PyTorch（CUDA 12.1 用 wheel）が `https://download.pytorch.org/whl/cu121` からインストールされる
- `pyproject.toml` の依存がインストールされる

## Phase A 再計算の実行

```bash
bash scripts/layer_consistency/run_phase_a_uv.sh
```

- A1（zero）→ A2（ITB）→ A3（OTB）の順で実行される
- ログは `logs/phase_a_recompute.log` および `logs/layer_direct_*.log`
- GPU が利用可能な環境なら自動で使用される

## Streamlit UI の起動

```bash
./detect_and_run.sh
# または: make start
```

- 既定ポートは **8503**（以前 `docker-compose` のホスト側ポートと揃えています）。変更する場合は `STREAMLIT_SERVER_PORT=8501 ./detect_and_run.sh` など。
- Docker で Streamlit を動かす場合は `BERT_USE_DOCKER=1 ./detect_and_run.sh` または `make start-docker`。

## その他の実行方法

- 仮想環境を有効化してから任意のスクリプトを実行:
  ```bash
  source .venv/bin/activate
  python utils/scripts/run_ptb_layer_direct_ig.py --split dev --start-sample 0 --end-sample 9 --baseline-method zero --ig-num-steps 32
  ```
- 進捗確認: `bash scripts/layer_consistency/check_status.sh` または `bash scripts/watch_phase_a_recompute.sh`

## 補足

- `.venv` は `.gitignore` に含まれており、リポジトリにはコミットしない
- 段階移行のため旧コマンド（`scripts/setup_uv_env.sh` など）も互換ラッパー経由で実行可能
- PyTorch を CPU のみにしたい場合は、`setup_uv_env.sh` の `--index-url` を `https://download.pytorch.org/whl/cpu` に変更する
- **Permission denied が出る場合**: キャッシュが Docker (root) で作られていると、UV で書き込めない。プロジェクトルートで `sudo chown -R $USER:$USER cache/` を実行してから再実行する。
