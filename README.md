# RAG System - Hệ thống truy xuất pháp luật

Hệ thống RAG đơn giản để hỏi đáp về pháp luật Việt Nam từ file PDF.

## Công nghệ

- **Embedding**: `sentence-transformers/all-MiniLM-L12-v2`
- **LLM**: Groq `llama-3.3-70b-versatile`
- **Vector Store**: FAISS

## Cài đặt

```bash
./run.sh build
```

## Sử dụng

```bash
# Build index từ PDF
./run.sh all

# Hỏi câu hỏi
./run.sh ask "Điều 1 quy định gì?"
```

## Cấu hình

Tạo file `.env` với:
```
GROQ_API_KEY=your_api_key_here
```

Xem thêm: [COMMANDS.md](COMMANDS.md)
