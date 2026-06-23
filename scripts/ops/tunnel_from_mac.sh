#!/usr/bin/env bash
# ★ Mac の Terminal.app で実行（Cursor のターミナルではない）
# 使い方: bash tunnel_from_mac.sh
# ブラウザ: http://localhost:18504/hello.html

set -euo pipefail
HOST="${STREAMLIT_REMOTE_HOST:-eight@hercules.murata.eb.waseda.ac.jp}"

echo "トンネルを張ります（このウィンドウは閉じないでください）"
echo "  → ブラウザで http://localhost:18504/hello.html"
echo "  → Streamlit: http://localhost:18503/"
echo ""

exec ssh -N \
  -o ServerAliveInterval=30 \
  -L 18504:127.0.0.1:8504 \
  -L 18503:127.0.0.1:8503 \
  "${HOST}"
