# Legal RAG System

Hệ thống RAG (Retrieval-Augmented Generation) cho pháp luật Việt Nam - Clean & Simple.

## Cấu trúc dự án

```
├── main.py              # CLI chính
├── splitter.py          # Tách PDF thành điều luật
├── vectorstore.py       # Quản lý vector store
├── retriever.py         # Truy xuất thông tin
├── generator.py         # Tạo câu trả lời với GROQ
├── rag.py              # RAG system chính
├── api.py              # FastAPI server
└── requirements.txt     # Dependencies
```

## Cài đặt

```bash
# Clone và setup
git clone <repo>
cd bai6
python3 -m venv venv
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt

# Cấu hình API keys
echo "GROQ_API_KEY=your_groq_api_key" > .env
```

## Sử dụng

### 1. CLI Commands

```bash
# Build knowledge base
python main.py build --pdf-path luat_lao_dong.pdf

# Đặt câu hỏi
python main.py ask --question "Nội dung Điều 33 quy định gì?"

# Build và test
python main.py all --pdf-path luat_lao_dong.pdf

# Chạy API server
python main.py server
```

### 2. API Server

```bash
# Khởi động server
python main.py server

# Truy cập
# Web UI: http://localhost:8000/app
# API Docs: http://localhost:8000/docs
```

### 3. Web Interface

Mở trình duyệt và truy cập: `http://localhost:8000/app`

## Quy trình hoạt động

1. **Splitter**: Tách PDF thành các điều luật riêng biệt
2. **VectorStore**: Tạo embeddings và lưu vào FAISS index
3. **Retriever**: Tìm kiếm thông tin liên quan
4. **Generator**: Tạo câu trả lời với GROQ LLM
5. **API**: Cung cấp REST API cho web interface

## Tính năng

- ✅ Tách PDF thành điều luật tự động
- ✅ Vector search với FAISS
- ✅ Heuristic cho điều luật cụ thể
- ✅ GROQ LLM integration
- ✅ FastAPI với web interface
- ✅ Clean code, dễ hiểu

## API Endpoints

- `GET /` - Root endpoint
- `POST /ask` - Đặt câu hỏi
- `GET /health` - Health check
- `GET /docs` - API documentation