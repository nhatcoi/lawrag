#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
: "${INDEX_DIR:=/Users/coinhat/Documents/AI/lawrag/faiss_index}"
: "${PUBLIC_DIR:=/Users/coinhat/Documents/AI/lawrag/public}"
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
