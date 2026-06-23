#!/usr/bin/env bash
# Prepare ATT + MLP IG caches for Experiment A (delegates to parent monorepo when available).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MONOREPO_ROOT="${MONOREPO_ROOT:-$(cd "${RELEASE_ROOT}/.." && pwd)}"

SPLIT="dev"
START=0
END=1699
IG_STEPS=32

while [[ $# -gt 0 ]]; do
  case $1 in
    --split) SPLIT="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

Z2U="${MONOREPO_ROOT}/uas_syntax_parsing/01_prepare_data/run_z2u_ig.sh"
MLP="${MONOREPO_ROOT}/uas_syntax_parsing/01_prepare_data/run_mlp_ig.sh"

if [[ ! -f "${Z2U}" || ! -f "${MLP}" ]]; then
  cat >&2 <<EOF
ERROR: PTB batch runners not found under MONOREPO_ROOT=${MONOREPO_ROOT}

Experiment A needs ATT/MLP caches. Either:
  1) Clone the development monorepo beside this repo and set MONOREPO_ROOT, then re-run; or
  2) Point PTB_CACHE_ROOT at an existing cache tree (see docs/REPRODUCTION.md).

Required scripts:
  uas_syntax_parsing/01_prepare_data/run_z2u_ig.sh
  uas_syntax_parsing/01_prepare_data/run_mlp_ig.sh
EOF
  exit 1
fi

export PTB_DEPPARSE_DIR="${PTB_DEPPARSE_DIR:-${MONOREPO_ROOT}/data/depparse}"
if [[ ! -f "${PTB_DEPPARSE_DIR}/${SPLIT}.txt" ]]; then
  echo "ERROR: PTB not found at ${PTB_DEPPARSE_DIR}/${SPLIT}.txt (LDC99T42)" >&2
  exit 1
fi

cd "${MONOREPO_ROOT}"

bash "${Z2U}" --split "${SPLIT}" --start-sample "${START}" --end-sample "${END}" \
  --ig-num-steps "${IG_STEPS}" --baseline-method zero --self-contribution-estimator direct_zero

bash "${Z2U}" --split "${SPLIT}" --start-sample "${START}" --end-sample "${END}" \
  --ig-num-steps "${IG_STEPS}" --baseline-method self_input_token --self-contribution-estimator direct_zero

bash "${Z2U}" --split "${SPLIT}" --start-sample "${START}" --end-sample "${END}" \
  --ig-num-steps "${IG_STEPS}" --baseline-method self_input_token --self-contribution-estimator att_map_ratio

bash "${Z2U}" --split "${SPLIT}" --start-sample "${START}" --end-sample "${END}" \
  --ig-num-steps "${IG_STEPS}" --baseline-method self_input_token --self-contribution-estimator zero_base_ratio

bash "${MLP}" --split "${SPLIT}" --start-sample "${START}" --end-sample "${END}" \
  --ig-num-steps "${IG_STEPS}" --baseline-method zero

bash "${MLP}" --split "${SPLIT}" --start-sample "${START}" --end-sample "${END}" \
  --ig-num-steps "${IG_STEPS}" --baseline-method self_input_token --no-mlp-residual-connection

echo "Done. Set PTB_CACHE_ROOT=${MONOREPO_ROOT}/cache/ptb_ig_analysis for reproduce scripts in layer-wise-integrated-gradients."
