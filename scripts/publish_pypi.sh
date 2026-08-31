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
DRY_RUN=0
case "${1:-}" in
  --test)    TEST=1 ;;
  --dry-run) DRY_RUN=1 ;;
  "")        ;;
  *)         echo "unknown option: $1 (use --test, --dry-run, or no argument)" >&2; exit 2 ;;
esac

python3 -m pip install --quiet build twine

rm -rf dist build *.egg-info layer_wise_integrated_gradients.egg-info
python3 -m build
twine check dist/*

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "--dry-run: built and checked dist/, not uploading."
  ls -l dist/
elif [[ "${TEST}" -eq 1 ]]; then
  twine upload --repository testpypi dist/*
else
  twine upload dist/*
fi

echo "Done. Verify: pip install layer-wise-integrated-gradients"
