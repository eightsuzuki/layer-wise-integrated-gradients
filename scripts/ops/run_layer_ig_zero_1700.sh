#!/usr/bin/env bash
# Layer-direct z2z zero baseline: regenerate PTB dev caches.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU="${CUDA_VISIBLE_DEVICES:-3}"
START="${1:-0}"
END="${2:-1699}"
LOG_FILE="${3:-$PROJECT_ROOT/logs/layer_ig_zero_1700_gpu${GPU}_${START}_${END}.log}"
mkdir -p "$PROJECT_ROOT/logs"

export CUDA_VISIBLE_DEVICES="$GPU"
export PTB_CACHE_ROOT="${PTB_CACHE_ROOT:-/home/data/eight/bert_token_embedding_visualization/cache/ptb_ig_analysis}"
export PTB_DEPPARSE_DIR="${PTB_DEPPARSE_DIR:-/home/data/eight/bert_token_embedding_visualization/data/depparse}"
MONOREPO_PTB="${PTB_DEPPARSE_DIR}"

echo "[$(date -Iseconds)] layer-direct zero baseline: GPU=$GPU samples $START-$END"
echo "PTB_CACHE_ROOT=$PTB_CACHE_ROOT"
echo "log=$LOG_FILE"

uv run python scripts/reproduce/run_layer_direct_ig.py \
  --split dev \
  --start-sample "$START" \
  --end-sample "$END" \
  --num-samples 1700 \
  --baseline-method zero \
  --ig-num-steps 32 \
  --ptb-data-dir "$MONOREPO_PTB" \
  --no-cache \
  --log-file "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -Iseconds)] done GPU=$GPU $START-$END"
