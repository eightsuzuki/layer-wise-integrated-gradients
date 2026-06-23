#!/usr/bin/env bash
# Poll lig-audit tmux session / latest log every N seconds.
INTERVAL="${1:-120}"
LOG_DIR="$(cd "$(dirname "$0")/../.." && pwd)/scripts/verify/reports/logs"

while true; do
  echo "===== $(date -Iseconds) ====="
  if tmux has-session -t lig-audit 2>/dev/null; then
    echo "[tmux] lig-audit: running"
    for win in main profile-gpu1 monitor; do
      if tmux list-windows -t lig-audit -F '#{window_name}' 2>/dev/null | grep -qx "$win"; then
        echo "[tmux] lig-audit:$win (last 10 lines):"
        tmux capture-pane -t "lig-audit:$win" -p | tail -10
      fi
    done
  else
    echo "[tmux] lig-audit: not running"
  fi
  LATEST="$(ls -t "$LOG_DIR"/lig_audit_*.log 2>/dev/null | head -1)"
  if [[ -n "$LATEST" ]]; then
    echo "[log] $LATEST (last 8 lines):"
    tail -8 "$LATEST"
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -4 || true
  echo ""
  sleep "$INTERVAL"
done
