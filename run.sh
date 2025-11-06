#!/bin/bash

VENV_DIR="venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# Kiểm tra và tạo virtual environment
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Tạo virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
}

# Kích hoạt virtual environment
activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        source "${VENV_DIR}/bin/activate"
    fi
}

# Build
build() {
    echo "Cài đặt dependencies..."
    setup_venv
    "$PIP" install -r requirements.txt
    echo "Đã cài đặt xong!"
}

# Split PDF
split() {
    setup_venv
    activate_venv
    echo "Tách PDF..."
    "$PYTHON" rag.py split --pdf luat_lao_dong.pdf --output output_dieu_luat
}

# Embed
embed() {
    setup_venv
    activate_venv
    echo "Tạo embeddings..."
    "$PYTHON" rag.py embed --split-dir output_dieu_luat --index-dir faiss_index
}

# Ask
ask() {
    if [ -z "$1" ]; then
        echo "Usage: ./run.sh ask \"Câu hỏi của bạn\""
        exit 1
    fi
    setup_venv
    activate_venv
    "$PYTHON" rag.py ask --query "$1" --index-dir faiss_index --top-k 5
}

# All (split + embed)
all() {
    setup_venv
    activate_venv
    echo "Chạy tổng thể (split + embed)..."
    "$PYTHON" rag.py all --pdf luat_lao_dong.pdf --split-dir output_dieu_luat --index-dir faiss_index
}

# Main
case "$1" in
    build)
        build
        ;;
    split)
        split
        ;;
    embed)
        embed
        ;;
    ask)
        ask "$2"
        ;;
    all)
        all
        ;;
    *)
        echo "Usage: $0 {build|split|embed|ask|all}"
        echo "  build  - Cài đặt dependencies"
        echo "  split  - Tách PDF thành các điều luật"
        echo "  embed  - Tạo embeddings và lưu index"
        echo "  ask    - Hỏi câu hỏi (cần câu hỏi trong dấu ngoặc kép)"
        echo "  all    - Chạy split + embed"
        exit 1
        ;;
esac

