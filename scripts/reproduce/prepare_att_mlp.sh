#!/usr/bin/env bash
# Build the ATT (z->u) and MLP (u->z) IG caches Experiment A composes from.
#
# The MLP side runs entirely from this repository. The ATT side still needs the
# development monorepo's batch runner: it drives a multi-GPU scheduler that has
# not been factored out yet. If you do not have the monorepo, use --skip-att and
# point PTB_CACHE_ROOT at ATT caches you obtained separately; this script tells
# you exactly which directories are missing rather than failing later inside the
# composition step.
#
# PTB (LDC99T42) is required and is not distributed here -- see docs/REPRODUCTION.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MONOREPO_ROOT="${MONOREPO_ROOT:-$(cd "${RELEASE_ROOT}/.." && pwd)}"

SPLIT="dev"
START=0
END=1699
ATT_STEPS=256   # the ATT boundary needs this many to satisfy completeness
MLP_STEPS=32    # the head-space MLP boundary is already tight at 32
DEVICE="auto"
SKIP_ATT=0
SKIP_MLP=0

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --split NAME        PTB split (default: dev)
  --start N           first sentence index (default: 0)
  --end N             last sentence index, inclusive (default: 1699)
  --att-steps N       IG steps for z->u (default: 256)
  --mlp-steps N       IG steps for u->z (default: 32)
  --device SPEC       auto | cuda:N | cpu (default: auto, picks the freest card)
  --skip-att          do not build ATT caches (use ones already on disk)
  --skip-mlp          do not build MLP caches
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --split) SPLIT="$2"; shift 2 ;;
    --start) START="$2"; shift 2 ;;
    --end) END="$2"; shift 2 ;;
    --att-steps) ATT_STEPS="$2"; shift 2 ;;
    --mlp-steps) MLP_STEPS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --skip-att) SKIP_ATT=1; shift ;;
    --skip-mlp) SKIP_MLP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

PTB_DEPPARSE_DIR="${PTB_DEPPARSE_DIR:-${MONOREPO_ROOT}/data/depparse}"
if [[ ! -f "${PTB_DEPPARSE_DIR}/${SPLIT}.txt" ]]; then
  echo "ERROR: PTB not found at ${PTB_DEPPARSE_DIR}/${SPLIT}.txt" >&2
  echo "       Obtain Treebank-3 (LDC99T42) and set PTB_DEPPARSE_DIR." >&2
  exit 1
fi
export PTB_DEPPARSE_DIR
export PTB_CACHE_ROOT="${PTB_CACHE_ROOT:-${RELEASE_ROOT}/cache/ptb_ig_analysis}"
echo "PTB      : ${PTB_DEPPARSE_DIR}"
echo "cache    : ${PTB_CACHE_ROOT}"
echo "device   : ${DEVICE}"

# ---------------------------------------------------------------- MLP (u -> z)
if [[ "${SKIP_MLP}" -eq 0 ]]; then
  echo "== MLP (u -> z), head space, ${MLP_STEPS} steps"
  python "${SCRIPT_DIR}/run_mlp_head_space_ig.py" \
    --split "${SPLIT}" --start "${START}" --end "${END}" \
    --num-steps "${MLP_STEPS}" --device "${DEVICE}"
fi

# ---------------------------------------------------------------- ATT (z -> u)
Z2U="${MONOREPO_ROOT}/uas_syntax_parsing/01_prepare_data/run_z2u_ig.sh"
if [[ "${SKIP_ATT}" -eq 0 ]]; then
  if [[ ! -f "${Z2U}" ]]; then
    cat >&2 <<EOF
ERROR: the ATT batch runner was not found at
  ${Z2U}

The z->u caches are produced by the development monorepo's multi-GPU runner,
which is not part of this release. Either:
  1) clone the monorepo next to this repo and set MONOREPO_ROOT, then re-run; or
  2) re-run with --skip-att and point PTB_CACHE_ROOT at existing ATT caches.

Composition needs these directories under
  \${PTB_CACHE_ROOT}/samples/${SPLIT}/att/ :
  steps${ATT_STEPS}_bert-base-uncased_maxlen128_z_to_u_baseline_zero
  steps${ATT_STEPS}_bert-base-uncased_maxlen128_z_to_u_baseline_self_input_token_direct_zero
  steps${ATT_STEPS}_bert-base-uncased_maxlen128_z_to_u_baseline_self_input_token_self_contrib_zero_base_ratio
  steps${ATT_STEPS}_bert-base-uncased_maxlen128_z_to_u_baseline_self_input_token_self_contrib_att_map_ratio
EOF
    exit 1
  fi
  echo "== ATT (z -> u), ${ATT_STEPS} steps (delegating to the monorepo runner)"
  cd "${MONOREPO_ROOT}"
  # self_input_token also produces the two ratio-completed variants.
  for BASELINE in zero self_input_token; do
    bash "${Z2U}" --split "${SPLIT}" --start-sample "${START}" --end-sample "${END}" \
      --ig-num-steps "${ATT_STEPS}" --baseline-method "${BASELINE}" \
      --self-contribution-estimator direct_zero
  done
  for EST in zero_base_ratio att_map_ratio; do
    bash "${Z2U}" --split "${SPLIT}" --start-sample "${START}" --end-sample "${END}" \
      --ig-num-steps "${ATT_STEPS}" --baseline-method self_input_token \
      --self-contribution-estimator "${EST}"
  done
fi

cat <<EOF

Done. Next:
  python scripts/reproduce/run_layer_direct_ig.py --split ${SPLIT} \\
      --start-sample ${START} --end-sample ${END} --baseline-method zero --device ${DEVICE}
  python scripts/reproduce/compose_z2z.py --split ${SPLIT} --start ${START} --end ${END}
  python scripts/reproduce/compare_layer_vs_composed.py --split ${SPLIT} --start ${START} --end ${END} \\
      --csv-out results/summary_layer_vs_composed.csv
EOF
