#!/usr/bin/env bash
# Build and upload layer-wise-integrated-gradients to PyPI.
#
# Usage:
#   export TWINE_USERNAME=__token__
#   export TWINE_PASSWORD=pypi-...   # PyPI API token
#   bash scripts/publish_pypi.sh
#
# Optional: upload to TestPyPI first
#   bash scripts/publish_pypi.sh --test
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

TEST=0
[[ "${1:-}" == "--test" ]] && TEST=1

python3 -m pip install --quiet build twine

rm -rf dist build *.egg-info layer_wise_integrated_gradients.egg-info
python3 -m build
twine check dist/*

if [[ "${TEST}" -eq 1 ]]; then
  twine upload --repository testpypi dist/*
else
  twine upload dist/*
fi

echo "Done. Verify: pip install layer-wise-integrated-gradients"
