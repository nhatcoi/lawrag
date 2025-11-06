# RAG System - Legal Document Assistant

## Cấu trúc dự án

```
be/
├── app/
│   ├── main.py                # Khởi tạo FastAPI, router
│   ├── api/                   # Endpoints
│   ├── core/                   # Xử lý RAG core
│   ├── models/                # Pydantic models
│   ├── db.py                  # SQLAlchemy (optional)
│   ├── config.py              # .env config
│   └── utils/                 # hash file, text clean, logging
├── storage/
│   ├── files/                 # file upload lưu local (.pdf, .docx,…)
│   │   ├── <document_id>_v1.pdf
│   │   └── <document_id>_v2.pdf
│   ├── index/                 # vector store FAISS
│   │   └── faiss.index
│   └── data/                  # JSON data & SQLite DB
├── .env
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # hoặc venv\Scripts\activate trên Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## Cấu hình

Tạo file `.env`:
```
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...  # (optional, nếu dùng OpenAI embeddings)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GROQ_MODEL=llama-3.3-70b-versatile
```

## Sử dụng

### 1. Build Vector Store

```bash
cd app
python main.py build
```

### 2. Chạy API Server

```bash
cd app
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Hoặc:
```bash
cd app
../venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Truy cập

- **API Docs**: http://localhost:8000/docs
- **API**: http://localhost:8000/api

## Modules

- **app/api/**: FastAPI endpoints (chat, upload, history, sources)
- **app/core/**: RAG pipeline (embedding, retrieval, generation, rag)
- **app/models/**: Pydantic models cho request/response
- **app/utils/**: Utilities (hash file, text cleaning, filename sanitization)
- **app/db.py**: Database models (optional, hiện tại dùng JSON)

