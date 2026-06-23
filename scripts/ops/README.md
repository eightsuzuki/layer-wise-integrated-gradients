# scripts/ops

公開リポジトリ専用の運用スクリプトです（`manifest.txt` には含めず、ここで直接管理）。

- `setup_uv_env.sh` — uv で `.venv` を作成し PyTorch + 本パッケージをインストール

```bash
bash scripts/ops/setup_uv_env.sh
bash scripts/ops/setup_uv_env.sh --cpu
```
