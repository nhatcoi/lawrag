#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python main.py all \
  --pdf-path /Users/coinhat/Documents/AI/lawrag/luat_lao_dong.pdf \
  --split-dir /Users/coinhat/Documents/AI/lawrag/output_dieu_luat \
  --index-dir /Users/coinhat/Documents/AI/lawrag/faiss_index \
  --provider local --local-model sentence-transformers/all-MiniLM-L6-v2
