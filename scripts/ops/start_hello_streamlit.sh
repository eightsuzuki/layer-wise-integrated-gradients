#!/usr/bin/env bash
# 接続確認用: 静的 HTML (8504) + Streamlit (8503)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT"

pkill -f "streamlit run hello_streamlit.py" 2>/dev/null || true
pkill -f "http.server 8504" 2>/dev/null || true
sleep 1

echo "静的ページ: 127.0.0.1:8504"
nohup python3 -m http.server 8504 --bind 0.0.0.0 --directory "$ROOT" >> logs/hello_http.log 2>&1 &

echo "Streamlit: 127.0.0.1:8503"
nohup uv run streamlit run hello_streamlit.py >> logs/hello_streamlit.log 2>&1 &

for _ in $(seq 1 20); do
  curl -sf http://127.0.0.1:8504/hello.html >/dev/null && curl -sf http://127.0.0.1:8503/_stcore/health >/dev/null && break
  sleep 1
done

{
  echo "=== 接続確認 URL（リモート側） ==="
  echo "静的 HTML:  http://127.0.0.1:8504/hello.html"
  echo "Streamlit:  http://127.0.0.1:8503"
  echo ""
  echo "=== Cursor で開く手順 ==="
  echo "1. 下部 Ports タブを開く"
  echo "2. 63346 など古い行はすべて削除（×）"
  echo "3. Forward a Port → 8504 を追加（まずこちらを試す）"
  echo "4. 8504 の行の地球アイコンをクリック（表示された URL をそのまま使う）"
  echo "5. ダメなら 8503 も同様に Forward"
  echo ""
  echo "※ localhost:63346 は壊れた古い転送です。使わないでください。"
} | tee logs/streamlit_access_urls.txt
