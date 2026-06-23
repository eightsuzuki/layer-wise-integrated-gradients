#!/usr/bin/env bash
# LIG performance + cache audit for sample_00410 (tmux-friendly).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MONOREPO="$(cd "$ROOT/.." && pwd)"
export PTB_DEPPARSE_DIR="${PTB_DEPPARSE_DIR:-$MONOREPO/data/depparse}"
export PTB_CACHE_ROOT="${PTB_CACHE_ROOT:-$MONOREPO/cache/ptb_ig_analysis}"

GPU="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"
LOG_DIR="$ROOT/scripts/verify/reports/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/lig_audit_${STAMP}.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$MAIN_LOG"; }

log "=== LIG audit start (GPU=$GPU) ==="
log "ROOT=$ROOT"
log "MAIN_LOG=$MAIN_LOG"
log "PTB_DEPPARSE_DIR=$PTB_DEPPARSE_DIR"
log "PTB_CACHE_ROOT=$PTB_CACHE_ROOT"

log "uv sync (dev extra)..."
uv sync --extra dev 2>&1 | tee -a "$MAIN_LOG"

log "pytest (fast, no slow)..."
uv run pytest test/test_layer_direct_ig.py test/test_layer_contribution_theory.py test/test_layer_itb_zero_ratio.py -m 'not slow' -q 2>&1 | tee -a "$MAIN_LOG"

PROFILE_JSON="$LOG_DIR/profile_sample_00410_zero_${STAMP}.json"
log "profile_layer_direct_ig (zero baseline)..."
uv run python scripts/verify/profile_layer_direct_ig.py   --sample-idx 410   --baseline zero   --json-out "$PROFILE_JSON" 2>&1 | tee -a "$MAIN_LOG"

log "append profile results to docs/LIG_COMPUTATION.md..."
uv run python scripts/verify/append_profile_results.py "$PROFILE_JSON" --gpu "$GPU" 2>&1 | tee -a "$MAIN_LOG"

log "validate cache (composed + derived ITB-zeroRatio, no recompute yet)..."
uv run python scripts/verify/validate_sample_00410_cache.py   --json-out "$LOG_DIR/validate_quick_${STAMP}.json" 2>&1 | tee -a "$MAIN_LOG"

log "validate cache (--recompute-layer-ig, may take 10+ min)..."
uv run python scripts/verify/validate_sample_00410_cache.py   --recompute-layer-ig   --json-out "$LOG_DIR/validate_recompute_${STAMP}.json" 2>&1 | tee -a "$MAIN_LOG"

log "=== LIG audit done ==="
log "Report: $ROOT/scripts/verify/reports/sample_00410_cache_validation.md"
