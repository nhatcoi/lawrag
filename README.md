# RAG System - Hệ thống truy xuất pháp luật

Hệ thống RAG đơn giản để hỏi đáp về pháp luật Việt Nam từ file PDF (ví dụ: `luat_lao_dong.pdf`).

## Cấu hình nhanh

```bash
./run.sh build
```

Tạo file `.env` với:
```
GROQ_API_KEY=your_api_key_here
```

Xem thêm: `COMMANDS.md`.

---

## ✅ 1. Tổng quan về RAG (Retrieval-Augmented Generation)

**RAG** là kỹ thuật kết hợp hai thành phần:

| Thành phần | Vai trò |
| --------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Retrieval (Tìm kiếm)** | Truy xuất thông tin liên quan trong knowledge base (FAISS, Pinecone, Chroma, Elastic…). |
| **Generation (Sinh câu trả lời)** | Dùng mô hình LLM (GPT, LLaMA, Groq…) để tạo câu trả lời dựa trên truy vấn + tài liệu liên quan. |

Mục tiêu của RAG là giúp mô hình trả lời dựa trên **dữ liệu bên ngoài** (PDF, luật, tài liệu riêng), không phải chỉ dựa vào kiến thức nội bộ của LLM.

---

## ✅ 2. Các bước chính trong Flow RAG

### Bước 1: Data Preparation (Chuẩn bị dữ liệu)

- Đọc và tách văn bản từ PDF, Word, web…
- Trong CLI của dự án này: `split` → Tách PDF Luật Lao động thành các file `.txt` theo từng điều.
- Công cụ dùng: `pdfplumber` (đã có trong `requirements.txt`).

Chạy:

```bash
./run.sh split
```

---

### Bước 2: Chunking (Chia văn bản thành đoạn nhỏ)

Vì một tài liệu dài không thể nhét vào prompt của LLM → ta chia thành các “chunk” nhỏ. Ở dự án này, đơn vị chunk là từng **Điều** (Điều 1, Điều 2, …).

| Lý do cần chunking | Kỹ thuật thường dùng |
| ---------------------- | ------------------------------------------------------------------ |
| Giới hạn token của LLM | Chia theo độ dài ký tự (500, 1000), hoặc theo đoạn |
| Đảm bảo ngữ nghĩa | Tách theo `\n\n`, `.` hoặc theo tiêu đề như Điều 3, Điều 5 |

---

### Bước 3: Embedding (Chuyển chunk thành vector)

Embedding = Biến câu chữ thành vector số (mảng float) để so sánh mức độ giống nhau. Dự án này dùng `sentence-transformers/all-MiniLM-L12-v2`.

Ví dụ các lựa chọn (tham khảo):

```python
# --provider openai → model = "text-embedding-3-small"
# --provider local  → model = "sentence-transformers/all-MiniLM-L6-v2"
```

| Provider | Ưu điểm | Nhược điểm |
| ----------------------- | ---------------------- | ----------------------- |
| **OpenAI** | Chính xác cao | Tính phí / cần internet |
| **Local (HuggingFace)** | Miễn phí, chạy offline | Độ chính xác thấp hơn |

---

### Bước 4: Indexing (Lưu vector vào kho tìm kiếm)

Vector + metadata (nguồn, tên điều luật, số trang…) được lưu vào hệ thống tìm kiếm như **FAISS**. Trong CLI của dự án → lệnh `embed` tạo thư mục `faiss_index`.

```bash
./run.sh embed
```

---

### Bước 5: Retrieval (Truy xuất khi người dùng hỏi)

Khi người dùng hỏi:

1. Convert query → embedding vector.
2. So sánh với vector trong FAISS → tìm K đoạn liên quan nhất (`top-k=5`).
3. Ghép “context” với câu hỏi và gửi sang LLM.

Lệnh CLI tương ứng:

```bash
./run.sh ask "Thời giờ làm việc tối đa là bao nhiêu?"
```

---

### Bước 6: Generation (Sinh câu trả lời cuối cùng)

LLM nhận input dạng:

```
Context:
- Điều 105: Thời giờ làm việc bình thường...
- Điều 106: Làm thêm giờ, giới hạn thời gian...

Question: "Thời gian làm việc tối đa là bao nhiêu theo Luật Lao động?"

Answer:
```

Mô hình tạo ra câu trả lời chính xác dựa trên dữ liệu luật. Ở dự án này dùng Groq `llama-3.3-70b-versatile` (cần `GROQ_API_KEY`).

---

## ✅ 3. Tóm tắt Kiến thức cốt lõi cần nắm

| Khái niệm | Hiểu đơn giản |
| --------------------------- | ------------------------------------------------ |
| **Embedding** | Mã hóa văn bản thành vector để máy so sánh được. |
| **Vector Database (FAISS)** | Kho lưu vector + metadata để tìm kiếm nhanh. |
| **Chunking** | Chia tài liệu lớn thành đoạn nhỏ hợp lý. |
| **Top-K Retrieval** | Tìm K đoạn có vector giống câu hỏi nhất. |
| **Prompt Augmentation** | Ghép câu hỏi + context để gửi LLM. |
| **LLM Inference** | Mô hình ngôn ngữ sinh câu trả lời cuối cùng. |

---

## ✅ 4. Flow RAG trực quan

```
PDF Luật Lao động 
    ↓ (split) 
Tách thành từng điều luật .txt 
    ↓ (chunk + embed) 
Vector embeddings (FAISS index)
    ↓
Người dùng hỏi → Embedding câu hỏi 
    ↓ 
So khớp cosine similarity → Top-K tài liệu 
    ↓ 
LLM (GPT/Llama/Groq) sinh câu trả lời kèm trích dẫn
```

---

## Nhanh: Lệnh thường dùng

```bash
# Cài dependencies
./run.sh build

# Tách PDF và tạo FAISS index (split + embed)
./run.sh all

# Hỏi câu hỏi
./run.sh ask "Điều 1 quy định gì?"
```

## Công nghệ chính

- **Embedding**: `sentence-transformers/all-MiniLM-L12-v2`
- **LLM**: Groq `llama-3.3-70b-versatile`
- **Vector Store**: FAISS
