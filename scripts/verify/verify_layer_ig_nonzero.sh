#!/usr/bin/env bash
# Phase A 再計算後、Layer IG キャッシュが非零かどうかを 1 サンプルで確認する。
# 使用例: bash scripts/verify_layer_ig_nonzero.sh

set -e
ROOT="${1:-.}"
[ -d "$ROOT/cache" ] || ROOT="."
BASE="$ROOT/cache/ptb_ig_analysis/samples/dev/z2z/layer_ig"

check_nonzero() {
  local dir="$1"
  local name="$2"
  local sample="${3:-sample_00000.json}"
  local f="$dir/$sample"
  if [ ! -f "$f" ]; then
    echo "SKIP $name: $f が存在しません（再計算未完了）"
    return 0
  fi
  local out
  out=$(python3 -c "
import json, numpy as np
d = json.load(open('$f'))
a = np.array(d.get('z2z', []))
s, m = float(a.sum()), float(np.abs(a).max())
print(s, m)
" 2>/dev/null) || { echo "ERR $name: 読み込み失敗"; return 1; }
  local sum max_abs
  read -r sum max_abs <<< "$out"
  if [ -z "$sum" ] || [ -z "$max_abs" ]; then
    echo "ERR $name: 出力解析失敗"
    return 1
  fi
  if [ "${sum}" = "0.0" ] && [ "${max_abs}" = "0.0" ]; then
    echo "FAIL $name: 全て 0 (sum=$sum max_abs=$max_abs)"
    return 1
  fi
  echo "OK   $name: sum=$sum max_abs=$max_abs"
  return 0
}

failed=0
checked=0
for subdir in steps32_bert-base-uncased_maxlen128_z_to_z_layer_ig_baseline_zero \
              steps32_bert-base-uncased_maxlen128_z_to_z_layer_ig_baseline_self_input_token; do
  [ -d "$BASE/$subdir" ] || continue
  checked=$((checked+1))
  name="${subdir##*baseline_}"; name="Layer IG ${name}"
  check_nonzero "$BASE/$subdir" "$name" || failed=$((failed+1))
done

if [ $checked -eq 0 ]; then
  echo "--- Layer IG キャッシュがありません（再計算未実行）。torch 環境で scripts/run_phase_a_layer_ig_recompute.sh を実行してください。"
  exit 0
fi
if [ $failed -gt 0 ]; then
  echo "--- 1 件以上が全て 0 です。再計算は torch 環境で scripts/run_phase_a_layer_ig_recompute.sh を実行してください。"
  exit 1
fi
echo "--- 確認済み: いずれも非零"
