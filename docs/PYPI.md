# PyPI 公開手順

## 事前準備

1. [PyPI](https://pypi.org) でアカウント作成（初回のみ）
2. **API token** を発行（Account settings → API tokens → scope: プロジェクト全体または `layer-wise-integrated-gradients`）
3. ローカル: `pip install -e ".[dev]"`（`build` / `twine` / `pytest` を含む）

## ビルド

```bash
cd layer-wise-integrated-gradients
bash scripts/publish_pypi.sh --dry-run  # ビルドのみ（下記スクリプトは --test 前に手動で build 確認推奨）

rm -rf dist/ build/ *.egg-info layer_wise_integrated_gradients.egg-info
python -m build
twine check dist/*
```

## GitHub Actions（推奨）

1. GitHub リポジトリ → **Settings → Secrets → Actions** に `PYPI_API_TOKEN` を登録
2. **Actions → Publish to PyPI → Run workflow** を実行  
   または GitHub **Release** を publish（`v0.1.0` など）

ワークフロー: [.github/workflows/publish-pypi.yml](../.github/workflows/publish-pypi.yml)

## ローカルからアップロード

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...   # API token
bash scripts/publish_pypi.sh
```

## TestPyPI（任意）

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...   # TestPyPI token
bash scripts/publish_pypi.sh --test
```

検証:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  layer-wise-integrated-gradients==0.1.0
lig explain "test" --steps 2 --granularity layer --layers 0 --device cpu
```

## 公開後のインストール

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install layer-wise-integrated-gradients
```

PyPI: https://pypi.org/project/layer-wise-integrated-gradients/

## バージョン更新

1. `pyproject.toml` と `lig/__init__.py` の `version` / `__version__` を上げる
2. `git tag v0.1.1 && git push origin v0.1.1`
3. Release 作成または workflow 再実行

同じバージョンは再アップロードできません。

## 注意

- **PyTorch** は依存に含めていません。ユーザーは先に `torch` をインストールしてください。
- 開発・テスト用: `pip install -e ".[dev]"`
