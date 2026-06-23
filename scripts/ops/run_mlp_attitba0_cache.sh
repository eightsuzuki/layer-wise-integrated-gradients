#!/usr/bin/env bash
# ATTITBa=0 MLP キャッシュ生成の共通ランナー。
# 例:
#   bash scripts/ops/run_mlp_attitba0_cache.sh --gpus 4 --mode parallel
#   bash scripts/ops/run_mlp_attitba0_cache.sh --gpus 2 --mode sequential
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PROJECT_ROOT}/.venv/bin/python"
[ -x "$PY" ] || PY=python

GPUS=4
MODE="parallel" # parallel | sequential
START=0
END=1699

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"; shift 2 ;;
    --mode)
      MODE="$2"; shift 2 ;;
    --start)
      START="$2"; shift 2 ;;
    --end)
      END="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

if [[ "$GPUS" != "2" && "$GPUS" != "4" ]]; then
  echo "--gpus は 2 または 4 を指定してください" >&2
  exit 2
fi
if [[ "$MODE" != "parallel" && "$MODE" != "sequential" ]]; then
  echo "--mode は parallel または sequential を指定してください" >&2
  exit 2
fi

run_one() {
  local gpu_id="$1" s="$2" e="$3"
  echo "[GPU ${gpu_id}] ATTITBa=0 MLP samples ${s}-${e}"
  env CUDA_VISIBLE_DEVICES="$gpu_id" "$PY" scripts/attribution/layer_consistency/run_mlp_attitb_zero_cache.py --start "$s" --end "$e"
}

if [[ "$GPUS" == "2" ]]; then
  RANGES=("0:${START}:849" "1:850:${END}")
else
  RANGES=("0:${START}:424" "1:425:849" "2:850:1274" "3:1275:${END}")
fi

echo "=== ATTITBa=0 MLP キャッシュ (${GPUS} GPU, mode=${MODE}, samples ${START}-${END}) ==="
if [[ "$MODE" == "parallel" ]]; then
  for spec in "${RANGES[@]}"; do
    IFS=":" read -r gpu s e <<< "$spec"
    run_one "$gpu" "$s" "$e" &
  done
  wait
else
  for spec in "${RANGES[@]}"; do
    IFS=":" read -r gpu s e <<< "$spec"
    run_one "$gpu" "$s" "$e"
  done
fi

echo "Done."

