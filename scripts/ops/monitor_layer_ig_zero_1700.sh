#!/usr/bin/env bash
CACHE_ROOT="/home/data/eight/bert_token_embedding_visualization/cache/ptb_ig_analysis"
NEW_DIR="$CACHE_ROOT/samples/dev/z2z/layer_ig/steps32_bert-base-uncased_maxlen128_z_to_z_layer_ig_baseline_zero"
while true; do
  echo "=== $(date -Iseconds) ==="
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk 'NR==2 || NR==4'
  count=0
  if [ -d "$NEW_DIR" ]; then count=$(ls "$NEW_DIR" | wc -l); fi
  echo "new cache: ${count}/1700"
  tail -2 logs/layer_ig_zero_1700_gpu3_0_849.log 2>/dev/null | sed 's/^/  gpu3: /' || true
  tail -2 logs/layer_ig_zero_1700_gpu1_850_1699.log 2>/dev/null | sed 's/^/  gpu1: /' || true
  echo
  sleep 120
done
