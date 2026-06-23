#!/usr/bin/env bash
# 1700 samples layer-direct zero baseline in tmux (GPU 1 + 3 parallel).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION_NAME="layer_ig_zero1700"
CACHE_ROOT="/home/data/eight/bert_token_embedding_visualization/cache/ptb_ig_analysis"
NEW_DIR="steps32_bert-base-uncased_maxlen128_z_to_z_layer_ig_baseline_zero"
mkdir -p "$PROJECT_ROOT/logs"

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"

tmux new-session -d -s "$SESSION_NAME" -n gpu3 -c "$PROJECT_ROOT" \
  "CUDA_VISIBLE_DEVICES=3 bash scripts/ops/run_layer_ig_zero_1700.sh 0 849; echo DONE gpu3; exec bash"

tmux new-window -t "$SESSION_NAME" -n gpu1 -c "$PROJECT_ROOT" \
  "CUDA_VISIBLE_DEVICES=1 bash scripts/ops/run_layer_ig_zero_1700.sh 850 1699; echo DONE gpu1; exec bash"

tmux new-window -t "$SESSION_NAME" -n monitor -c "$PROJECT_ROOT" \
  "bash scripts/ops/monitor_layer_ig_zero_1700.sh"

echo "tmux session: $SESSION_NAME"
echo "  attach: tmux attach -t $SESSION_NAME"
echo "  GPU3: samples 0-849"
echo "  GPU1: samples 850-1699"
