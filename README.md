### Hệ thống RAG Bộ luật Lao động — Báo cáo và Hướng dẫn sử dụng

Nền tảng này triển khai một hệ thống RAG (Retrieval-Augmented Generation) cho Bộ luật Lao động Việt Nam: trích xuất văn bản từ PDF, tách theo Điều luật, tạo vector embeddings, lập chỉ mục bằng FAISS, truy hồi đoạn liên quan và sinh câu trả lời bằng ChatGroq.

---

### 1) Kiến trúc tổng quan
- `src/splitter.py` (Document Processing)
  - Đọc PDF bằng `pdfplumber` (đảm bảo Unicode/tiếng Việt ổn định).
  - Tách nội dung theo tiêu đề Điều khớp regex: `^\s*Điều\s+(\d+)\b`.
  - Gom các dòng từ tiêu đề này tới ngay trước tiêu đề kế tiếp → một Điều → ghi `.txt`.
  - Hàm chính: `extract_text_from_pdf`, `split_articles`, `write_articles`.
- `src/embedder.py` (Embedding & Storage Processing)
  - Local embeddings: `sentence-transformers` (mặc định `all-MiniLM-L6-v2`; tiếng Việt nên dùng `paraphrase-multilingual-MiniLM-L12-v2`).
  - OpenAI embeddings: `text-embedding-3-small` (cần `OPENAI_API_KEY`).
  - Chuẩn hóa L2 các vector và dùng `faiss.IndexFlatIP` (inner product ≈ cosine khi đã normalize).
  - Lưu `faiss_index/index.faiss` (vector + cấu trúc index) và `faiss_index/metadata.json` (ánh xạ id → path).
- `src/retriever.py` (Retrieval)
  - Embed truy vấn (local/OpenAI), normalize, `index.search(q, top_k)` lấy id + score.
  - Đọc nguồn từ `metadata.json` và nội dung `.txt` tương ứng.
  - Heuristic: nếu query chứa “Điều <số>”, đọc trực tiếp `điều_<số>.txt` prepend vào ngữ cảnh (đảm bảo hit chính xác theo số điều).
- `src/generator.py` (Generate)
  - Gọi ChatGroq (`llama-3.3-70b-versatile`) với system prompt + ngữ cảnh cắt gọn (`max_chars=20000`) + câu hỏi.
  - Nếu thiếu thông tin trong ngữ cảnh → trả “Không đủ dữ liệu”.
- `main.py`: CLI duy nhất (subcommands `split`, `embed`, `all`, `ask`).
- `src/api.py`: FastAPI — `POST /ask` và phục vụ UI tĩnh ở `/app` (đã bật CORS rộng).
- `public/`: UI chat cơ bản (HTML/CSS/JS), có loader, cấu hình API Base.

Luồng xử lý tổng quát:
1) PDF → Điều (`.txt`)
2) `.txt` → embeddings → FAISS + metadata
3) Query → embed → search top-k → tập ngữ cảnh
4) Groq LLM → câu trả lời dựa trên ngữ cảnh

---

### 2) Cài đặt
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tạo `.env` (ví dụ):
```
OPENAI_API_KEY=sk-...   # nếu dùng OpenAI cho embeddings
GROQ_API_KEY=gsk_...    # bắt buộc để sinh câu trả lời
```

---

### 2) Document Processing (chi tiết)
- Input: `luat_lao_dong.pdf`
- Quy trình:
  1. Đọc từng trang bằng `pdfplumber` với `x_tolerance/y_tolerance` để giữ định dạng dòng.
  2. Tách dòng (`splitlines()`), duyệt tuần tự.
  3. Khi gặp tiêu đề khớp `^\s*Điều\s+(\d+)\b` → mở Điều mới; đẩy Điều cũ vào danh sách.
  4. Bỏ qua phần trước Điều đầu tiên.
  5. Ghi mỗi Điều thành một file `.txt` tên dạng `điều_<số>.txt` (đã sanitize tên file).
- Output: `output_dieu_luat/điều_*.txt`
- Ưu điểm: Tách theo Điều giúp retrieval chính xác hơn, dễ mapping.

### 3) Embedding & Storage Processing (chi tiết)
- Embedding (2 lựa chọn):
  - Local: `SentenceTransformer(model_name)` → `encode(texts)` → `float32` vectors.
  - OpenAI: `client.embeddings.create(model, input=batch)`.
- Chuẩn hóa vector: `faiss.normalize_L2(vectors)` (cần để IP ≈ cosine).
- FAISS index: `IndexFlatIP(dim)` → `index.add(vectors)`.
- Metadata: JSON mảng các object `{ id, path }` theo đúng thứ tự vector đã add.
- Lưu trữ: `faiss_index/index.faiss` + `faiss_index/metadata.json`.
- Khuyến nghị tiếng Việt: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Điều chỉnh kích thước batch bằng `EMBED_BATCH_SIZE` nếu cần.

Lý do ưu tiên `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` thay vì OpenAI embeddings:
- Miễn phí/không phụ thuộc quota, chạy cục bộ (không phát sinh chi phí API).
- Hỗ trợ đa ngôn ngữ, phù hợp tiếng Việt hơn so với các model thuần tiếng Anh (`all-MiniLM-L6-v2`).
- Độ trễ ổn định và kiểm soát hoàn toàn về dữ liệu (không rời khỏi máy).
- Có thể chuyển sang OpenAI đơn giản bằng `--provider openai` khi cần chất lượng cao hơn và chấp nhận chi phí.

### 4) Retrieval (chi tiết)
- Embedded query (local/OpenAI) → normalize → `index.search(q, top_k)`.
- Ánh xạ id → `{path}` qua `metadata.json`, đọc nội dung `.txt` → tạo `contexts`.
- Heuristic “Điều <số>”: match regex, chèn thẳng nội dung `điều_<số>.txt` lên đầu `contexts` để đảm bảo recall tuyệt đối khi hỏi theo số điều.
- Chọn `top_k`:
  - Mặc định 10–20; nếu thiếu dữ liệu, tăng 20–50.
  - Khi có rerank (nếu tích hợp), có thể lấy 40–100 và giảm còn 5–10 sau rerank.

### 5) Generate (chi tiết)
- LLM: ChatGroq `llama-3.3-70b-versatile`.
- Prompting:
  - System: yêu cầu chỉ dùng thông tin trong ngữ cảnh pháp luật, thiếu thì nói không đủ dữ liệu.
  - User: gồm Ngữ cảnh (cắt tối đa ~20000 ký tự) + Câu hỏi.
- Tham số: `temperature=0.2`, `max_tokens=800` (tùy chỉnh).
- Điều kiện “Không đủ dữ liệu”: ngữ cảnh không có nội dung phù hợp (do retrieval/embedding chưa hit đúng hoặc query quá mơ hồ).

### 6) Sử dụng CLI (main.py)
- Tách PDF:
```bash
python main.py split \
  --pdf-path /Users/coinhat/Documents/PROJECT/AI/RAG/bai6/luat_lao_dong.pdf \
  --output-dir /Users/coinhat/Documents/PROJECT/AI/RAG/bai6/output_dieu_luat
```

- Tạo embeddings + FAISS (local — khuyến nghị):
```bash
python main.py embed \
  --split-dir /Users/coinhat/Documents/PROJECT/AI/RAG/bai6/output_dieu_luat \
  --index-dir /Users/coinhat/Documents/PROJECT/AI/RAG/bai6/faiss_index \
  --provider local \
  --local-model sentence-transformers/all-MiniLM-L6-v2
```

- Tạo embeddings + FAISS (OpenAI):
```bash
export OPENAI_API_KEY="sk-..."
python main.py embed \
  --split-dir /Users/coinhat/Documents/PROJECT/AI/RAG/bai6/output_dieu_luat \
  --index-dir /Users/coinhat/Documents/PROJECT/AI/RAG/bai6/faiss_index \
  --provider openai \
  --model text-embedding-3-small
```

- Chạy full quy trình (split + embed):
```bash
python main.py all \
  --pdf-path /Users/coinhat/Documents/PROJECT/AI/RAG/bai6/luat_lao_dong.pdf \
  --split-dir /Users/coinhat/Documents/PROJECT/AI/RAG/bai6/output_dieu_luat \
  --index-dir /Users/coinhat/Documents/PROJECT/AI/RAG/bai6/faiss_index \
  --provider local --local-model sentence-transformers/all-MiniLM-L6-v2
```

- Hỏi (RAG):
```bash
python main.py ask \
  --query "Nội dung Điều 1 Bộ luật Lao động quy định gì?" \
  --index-dir /Users/coinhat/Documents/PROJECT/AI/RAG/bai6/faiss_index \
  --provider local --local-model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k 10 --groq-model llama-3.3-70b-versatile
```

Gợi ý: tiếng Việt hoạt động tốt hơn với mô hình embed đa ngôn ngữ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

---

### 7) Sử dụng API và UI (ngắn gọn)
- Khởi chạy API:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

- Gọi API (mẫu):
```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Nội dung Điều 1 Bộ luật Lao động quy định gì?",
    "index_dir": "/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/faiss_index",
    "provider": "local",
    "local_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 10,
    "groq_model": "llama-3.3-70b-versatile"
  }'
```

- Mở UI (đã bật CORS, static mount):
  - Same-origin: `http://localhost:8000/app/index.html`
  - Hoặc Live Server: `http://127.0.0.1:5500/public/index.html?api=http://localhost:8000`

---

### 8) API (ngắn gọn)
- Endpoint: `POST /ask`
- Body:
```json
{
  "query": "...",
  "index_dir": ".../faiss_index",
  "provider": "local"|"openai",
  "local_model": "sentence-transformers/all-MiniLM-L6-v2",
  "top_k": 10,
  "groq_model": "llama-3.3-70b-versatile"
}
```
- Response: `{ answer: string, sources: [{rank, score, id, path}, ...] }`

---

### 6) Biến môi trường
- `.env`:
  - `OPENAI_API_KEY`: Khóa cho embeddings OpenAI (tùy chọn).
  - `GROQ_API_KEY`: Khóa cho ChatGroq (bắt buộc để sinh câu trả lời).
  - `EMBED_MODEL`, `EMBED_BATCH_SIZE`: có thể override mặc định.

---

### 7) Khắc phục sự cố
- “Không đủ dữ liệu”: tăng `--top-k`, dùng model embed đa ngôn ngữ, đặt câu hỏi rõ hơn.
- CORS khi dùng Live Server: thêm `?api=http://localhost:8000` hoặc mở qua `http://localhost:8000/app/index.html`.
- Tokenizers cảnh báo: `export TOKENIZERS_PARALLELISM=false`.

---

### 8) Thư mục đầu ra
- `output_dieu_luat/`: các file `.txt` cho từng Điều.
- `faiss_index/`:
  - `index.faiss`: FAISS index (vector + cấu trúc tìm kiếm).
  - `metadata.json`: ánh xạ id → path nguồn để hiển thị kết quả.


