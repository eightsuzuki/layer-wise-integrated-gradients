#!/usr/bin/env bash
# Create .venv and install PyTorch + this package.
# Usage: bash scripts/ops/setup_uv_env.sh [--cpu]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

TORCH_INDEX="https://download.pytorch.org/whl/cu121"
if [[ "${1:-}" == "--cpu" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

echo "[setup] Creating virtualenv in ${ROOT}/.venv"
uv venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] Installing PyTorch from ${TORCH_INDEX}"
uv pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"

echo "[setup] Installing layer-wise-integrated-gradients (editable, dev extras)"
uv pip install -e ".[dev]"

echo "[setup] Done. Run: source .venv/bin/activate"
