#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
QUERY=${1:-"Tóm tắt Điều 10"}
python main.py ask --query "$QUERY" \
  --index-dir /Users/coinhat/Documents/AI/lawrag/faiss_index \
  --provider local --local-model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k 5 --groq-model llama-3.3-70b-versatile
