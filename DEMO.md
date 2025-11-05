# Cách chạy RAG System

## 1. Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Cài đặt dependencies (nếu chưa có)
pip install -r requirements.txt
```

## 2. Chạy toàn bộ pipeline (Split + Embed)
```bash
python main.py all --pdf-path luat_lao_dong.pdf --split-dir output_dieu_luat --index-dir vector_store/faiss_index --provider local --local-model sentence-transformers/all-MiniLM-L12-v2 --batch-size 32
```

## 3. Hỏi đáp
```bash
python main.py ask --query "Nội dung Điều 1 Bộ luật Lao động quy định gì?" --index-dir vector_store/faiss_index --provider local --local-model sentence-transformers/all-MiniLM-L12-v2 --top-k 20
```

## 4. Các lệnh riêng lẻ
```bash
# Chỉ split PDF
python main.py split

# Chỉ tạo embeddings
python main.py embed --provider local --local-model sentence-transformers/all-MiniLM-L12-v2

# Hỏi câu hỏi khác
python main.py ask --query "Điều 2 quy định gì?" --index-dir vector_store/faiss_index --provider local --local-model sentence-transformers/all-MiniLM-L12-v2
```

## 5. Kết quả mẫu
**Câu hỏi:** "Nội dung Điều 1 Bộ luật Lao động quy định gì?"

**Trả lời:** Điều 1 Bộ luật Lao động quy định về phạm vi điều chỉnh, bao gồm tiêu chuẩn lao động, quyền, nghĩa vụ, trách nhiệm của người lao động, người sử dụng lao động và quản lý nhà nước về lao động.

## 6. Lưu ý
- Đảm bảo có file `.env` với `GROQ_API_KEY` để tạo câu trả lời
- Dùng `--provider local` để tránh cost OpenAI API
- `--top-k 20` để lấy nhiều ngữ cảnh hơn cho câu trả lời chính xác
