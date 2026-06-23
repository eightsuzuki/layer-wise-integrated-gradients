#!/usr/bin/env bash
# §3.7.4 ATTITBa=0 の MLP キャッシュ 1700 サンプルを tmux でデタッチ実行する。
# 進捗: tmux attach -t attitb_mlp1700
# ログ: tail -f logs/attitb_mlp_1700.log

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION_NAME="attitb_mlp1700"
LOG_FILE="$PROJECT_ROOT/logs/attitb_mlp_1700.log"
mkdir -p "$PROJECT_ROOT/logs"

tmux has-session -t "$SESSION_NAME" 2>/dev/null && tmux kill-session -t "$SESSION_NAME"

tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" \
  "bash scripts/ops/run_mlp_attitba0_cache.sh --gpus 4 --mode parallel --start 0 --end 1699 2>&1 | tee \"$LOG_FILE\"; echo ''; echo '=== 完了。exit でウィンドウを閉じる ==='; exec bash"

echo "tmux セッション '$SESSION_NAME' で ATTITBa=0（MLP）1700 サンプルを開始しました。"
echo "  進捗: tmux attach -t $SESSION_NAME"
echo "  ログ: tail -f $LOG_FILE"
