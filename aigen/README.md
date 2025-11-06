Law RAG


# 1) Tạo môi trường và cài thư viện
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Xem hướng dẫn CLI
python main.py -h

# 3) Tách PDF và build FAISS index (local, không cần API key)
python main.py all --pdf-path /Users/coinhat/Documents/AI/lawrag/luat_lao_dong.pdf \
  --split-dir /Users/coinhat/Documents/AI/lawrag/output_dieu_luat \
  --index-dir /Users/coinhat/Documents/AI/lawrag/faiss_index \
  --provider local --local-model sentence-transformers/all-MiniLM-L6-v2

# 4) Hỏi đáp (cần GROQ_API_KEY cho sinh câu trả lời)
export GROQ_API_KEY="...your_key..."
python main.py ask --query "Xin tóm tắt nội dung chính của Điều 10" \
  --index-dir /Users/coinhat/Documents/AI/lawrag/faiss_index \
  --provider local --local-model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k 5 --groq-model llama-3.3-70b-versatile

## Chạy server API

```bash
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
# Tuỳ chọn: chỉ định thư mục index/static nếu khác mặc định
# export INDEX_DIR="/Users/coinhat/Documents/AI/lawrag/faiss_index"
# export PUBLIC_DIR="/Users/coinhat/Documents/AI/lawrag/public"
# Tuỳ chọn: cần để sinh câu trả lời bằng Groq
# export GROQ_API_KEY="...your_key..."

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

- Mở tài liệu API: http://127.0.0.1:8000/docs
- Mở web app tĩnh (nếu có `public/`): http://127.0.0.1:8000/app

## Gọi API mẫu

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tóm tắt Điều 10",
    "top_k": 2
  }'
```

Ghi chú: cần `GROQ_API_KEY` trong môi trường để sinh câu trả lời (sinh từ Groq). Nếu không có, vẫn có thể nhận nguồn trích dẫn nhưng phần sinh câu trả lời có thể lỗi.


## Tích hợp Lobechat (OpenAI-compatible)

1) Khởi chạy Lobechat (Docker):
```bash
docker run -d --name lobechat -p 3210:3210 lobehub/lobechat:latest
```

2) Chạy API của dự án (đã có /v1/chat/completions):
```bash
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
# export INDEX_DIR="/Users/coinhat/Documents/AI/lawrag/faiss_index"
# export PUBLIC_DIR="/Users/coinhat/Documents/AI/lawrag/public"
# export GROQ_API_KEY="...your_key..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

3) Cấu hình trong Lobechat:
- API Base: http://127.0.0.1:8000
- API Key: có thể đặt chuỗi bất kỳ (không dùng)
- Model: rag-law (hoặc tên bất kỳ)
- Bật Custom API Base / Self-hosted

4) Kiểm tra nhanh bằng curl (OpenAI schema):
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rag-law","messages":[{"role":"user","content":"Tóm tắt Điều 10"}]}'
```

## Scripts tiện lợi

Đã tạo các script ngắn gọn trong `scripts/`:

```bash
# 1. Cài đặt dependencies
./scripts/setup.sh

# 2. Build FAISS index (tách PDF + embed)
./scripts/build_index.sh

# 3. Chạy server API
./scripts/run_server.sh

# 4. Hỏi đáp nhanh
./scripts/ask.sh "Tóm tắt Điều 10"
```

Lưu ý: Server sẽ tự động reload khi code thay đổi.
