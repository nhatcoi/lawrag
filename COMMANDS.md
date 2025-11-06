# Các lệnh RAG System

## 1. Build (Cài đặt dependencies)

Script sẽ tự động tạo virtual environment và cài đặt:

```bash
./run.sh build
```

Hoặc thủ công:
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 2. Chạy từng bước

### Bước 1: Tách PDF thành các điều luật
```bash
./run.sh split
```

### Bước 2: Tạo embeddings và lưu index
```bash
./run.sh embed
```

### Bước 3: Hỏi câu hỏi
```bash
./run.sh ask "Nội dung Điều 1 Bộ luật Lao động quy định gì?"
```

## 3. Chạy tổng thể (Split + Embed)

```bash
./run.sh all
```

## 4. Ví dụ đầy đủ

```bash
# Cài đặt dependencies (tự động tạo venv)
./run.sh build

# Build index (split + embed)
./run.sh all

# Hỏi câu hỏi
./run.sh ask "Điều 1 quy định gì?"
```

## 5. Lệnh thủ công (nếu không dùng script)

```bash
# Kích hoạt virtual environment
source venv/bin/activate

# Chạy các lệnh
python3 rag.py split --pdf luat_lao_dong.pdf --output output_dieu_luat
python3 rag.py embed --split-dir output_dieu_luat --index-dir faiss_index
python3 rag.py ask --query "Câu hỏi" --index-dir faiss_index
```

