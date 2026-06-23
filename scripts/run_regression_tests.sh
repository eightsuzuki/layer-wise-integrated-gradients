#!/usr/bin/env bash
# Quick regression suite for layer-wise-integrated-gradients
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif [[ -d "$ROOT/../.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/../.venv/bin/activate"
fi

echo "=== pytest (unit + slow IG smoke) ==="
pytest test/ -v --tb=short

echo "=== lig CLI smoke ==="
lig explain "Regression test sentence." \
  --model bert-base-uncased \
  --steps 2 \
  --granularity layer \
  --layers 0 \
  --target-tokens 1,2 \
  --device cpu \
  -o /tmp/lig_regression_smoke.json

python -c "import json; d=json.load(open('/tmp/lig_regression_smoke.json')); assert 'layers' in d and '0' in d['layers']"

echo "=== release scope guard ==="
python3 scripts/check_no_otb_in_release.py

echo "=== demo HTML build ==="
python3 scripts/build_demo_html.py
test -f docs/index.html
test -f docs/githubpage/z2z_embed.html
test -f docs/githubpage/z2z_token_contribution.html
test -f docs/githubpage/z2z_token_contribution_sample_00016.html
test -f docs/githubpage/z2z_token_contribution_sample_00410.html
grep -q 'demo-source-select' docs/githubpage/z2z_token_contribution.html
grep -q 'applyDemoSelection' docs/githubpage/z2z_token_contribution.html
grep -q 'z2z-layout-normal' docs/githubpage/z2z_token_contribution.html
test ! -f docs/githubpage/z2z_token_contribution_sample_00410__layer_ig_zero.html
test -f examples/paper_demo/json/sample_00016__layer_ig_zero.json
test -f examples/paper_demo/json/sample_00410__layer_ig_itb.json

echo "=== OK ==="
