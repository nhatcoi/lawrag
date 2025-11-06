# LỜI NÓI ĐẦU

## 1. Nội dung bài tập lớn

Bài tập lớn này tập trung vào việc xây dựng và triển khai hệ thống RAG (Retrieval-Augmented Generation) qua ba bài tập chính:

- **RAG Core**: Xây dựng RAG từ đầu bằng Python thuần, không sử dụng framework, để hiểu sâu các thành phần cốt lõi (chunking, embedding, retrieval, generation)
- **LangChain → App**: Áp dụng RAG core đã xây dựng để phát triển ứng dụng chat hoàn chỉnh với LangChain, FastAPI và React
- **RAGFlow**: Sử dụng framework mã nguồn mở RAGFlow để xây dựng pipeline RAG với giao diện kéo thả, so sánh với cách tiếp cận tự code

## 2. Mô tả tổng quan chương trình, các kiến thức sử dụng

Chương trình được thiết kế theo hướng tiếp cận từ cơ bản đến nâng cao:

- **Kiến thức nền tảng**: RAG architecture, embeddings, vector search, LLM integration
- **Công nghệ sử dụng**: Python, FastAPI, React, TypeScript, LangChain, FAISS, Groq API, RAGFlow
- **Phương pháp**: Xây dựng từ đầu → Tích hợp framework → Sử dụng công cụ có sẵn

## 3. Mô tả chi tiết các kiến thức cơ bản sử dụng trong bài tập lớn

**RAG Core (Python thuần):**
- Text processing: PDF parsing, text chunking theo pattern (Điều luật)
- Embeddings: SentenceTransformers, vector representation
- Vector search: Cosine similarity, FAISS indexing
- Generation: Prompt template, rule-based answer formatting

**LangChain → App:**
- Document loaders và text splitters (LangChain)
- Vector stores (FAISS với LangChain wrapper)
- RetrievalQA chains với custom prompts
- FastAPI backend với RESTful API
- React frontend với chat UI

**RAGFlow:**
- Docker deployment
- Pipeline configuration (Ingest → Embed → Retrieve → Answer)
- Heuristic rules cho luật pháp Việt Nam

## 4. Mô tả kết quả

**RAG Core**: Đã xây dựng thành công hệ thống RAG hoàn chỉnh từ đầu, hiểu rõ từng thành phần, khả năng tùy chỉnh cao nhưng phát triển lâu hơn.

**LangChain → App**: Đã phát triển ứng dụng chat hoàn chỉnh với giao diện React, API backend, tích hợp LangChain giúp code ngắn gọn và dễ maintain.

**RAGFlow**: Đã triển khai pipeline RAG với RAGFlow, nhanh chóng nhưng ít tùy chỉnh hơn so với tự code.

## 5. Kết luận

Ba bài tập lớn đã thể hiện rõ sự khác biệt giữa các cách tiếp cận:
- **RAG Core**: Phù hợp cho học thuật, hiểu sâu, tùy chỉnh cao
- **LangChain → App**: Phù hợp cho phát triển sản phẩm thực tế, cân bằng giữa tùy chỉnh và tốc độ phát triển
- **RAGFlow**: Phù hợp cho người dùng không chuyên, triển khai nhanh với ít code

Mỗi cách tiếp cận có ưu nhược điểm riêng, tùy vào mục tiêu và yêu cầu cụ thể mà lựa chọn phương pháp phù hợp.

---

## CHƯƠNG 1: GIỚI THIỆU TỔNG QUAN

### 1.1. Lý do chọn đề tài
### 1.2. Vấn đề thực tế & nhu cầu khai thác tri thức từ văn bản
### 1.3. Mục tiêu dự án
### 1.4. Phạm vi thực hiện
### 1.5. Phương pháp tiếp cận (RAG – Retrieval Augmented Generation)
### 1.6. Cấu trúc báo cáo

## CHƯƠNG 2: KIẾN THỨC NỀN TẢNG & TỔNG QUAN RAG

### 2.1. Khái niệm RAG (Retrieval-Augmented Generation)
### 2.2. Các thành phần cốt lõi: Chunking – Embedding – Indexing – Retrieval – Generation
### 2.3. So sánh RAG với Fine-tuning, Prompting truyền thống
### 2.4. Ưu điểm, hạn chế & bài toán phù hợp
### 2.5. Các mô hình RAG hiện đại (OpenAI, Meta Llama, Gemini, haystack…)
### 2.6. Kiến trúc tổng thể hệ thống RAG

## CHƯƠNG 3: XÂY DỰNG RAG CORE – PYTHON THUẦN

### 3.1. Mục tiêu chương: Hiểu sâu bên trong RAG
### 3.2. Tiền xử lý & Chunking tài liệu (PDF/Text → đoạn nhỏ)
### 3.3. Embedding thủ công (SentenceTransformers)
### 3.4. Vector Search: cosine similarity hoặc kNN tuyến tính
### 3.5. Sinh câu trả lời (Simple Prompt Template Rule-based)
### 3.6. Demo CLI/Mini API
### 3.7. Kết luận: ưu điểm & giới hạn của cách làm thủ công

## CHƯƠNG 4: ÁP DỤNG RAG CORE VỚI LANGCHAIN

### 4.1. Mục tiêu chương

Chương 3 đã xây dựng RAG core từ đầu bằng Python thuần để hiểu rõ các thành phần cốt lõi. Chương 4 này áp dụng phần core đó để phát triển ứng dụng RAG hoàn chỉnh với LangChain, bao gồm:

- **Backend**: FastAPI + LangChain pipeline (thay thế các hàm thủ công bằng LangChain components)
- **Frontend**: React Chat UI đơn giản
- **Vector Store**: FAISS với LangChain integration
- **LLM**: Groq API (ChatGroq) thông qua LangChain

### 4.2. Kiến trúc hệ thống

```
React Chat UI → FastAPI → LangChain RAG Pipeline → FAISS → ChatGroq LLM
```

**Luồng xử lý:**
1. User query → FastAPI endpoint
2. LangChain RetrievalQA chain:
   - Load FAISS vector store (từ core đã xây dựng)
   - Similarity search với metadata filter
   - Custom prompt template
   - ChatGroq LLM generation
3. Trả về answer + source documents

### 4.3. Tích hợp LangChain vào RAG Core

#### 4.3.1. Thay thế các thành phần thủ công

**a) Document Loading & Splitting**

Thay thế manual text splitting bằng LangChain:

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
loader = TextLoader(file_path)
documents = loader.load()

# Split với LangChain
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = splitter.split_documents(documents)
```

**b) Embedding & Vector Store**

Thay thế manual embedding bằng LangChain FAISS:

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Embedding với HuggingFace
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Tạo FAISS vector store
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local(index_path)
```

**c) Retrieval & Generation**

Thay thế manual retrieval + generation bằng LangChain RetrievalQA:

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatGroq

# Load vector store
vectorstore = FAISS.load_local(index_path, embeddings)

# Tạo RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGroq(model="llama-3.1-70b-versatile", temperature=0.7),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Thực thi
result = qa_chain({"query": user_query})
answer = result["result"]
sources = result["source_documents"]
```

#### 4.3.2. Custom Prompt Template

Thay thế rule-based prompt bằng LangChain prompt template:

```python
from langchain.prompts import PromptTemplate

prompt_template = """Bạn là một luật sư AI chuyên nghiệp.

Sử dụng thông tin sau để trả lời câu hỏi:
{context}

Câu hỏi: {question}

Trả lời:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    ...,
    chain_type_kwargs={"prompt": prompt}
)
```

#### 4.3.3. Metadata Filtering

Áp dụng metadata filter từ core vào LangChain retriever:

```python
# Từ core: law_type metadata đã được thêm vào documents
# Áp dụng filter trong LangChain
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10, "filter": {"law_type": "doanh-nghiep"}}
)
```

### 4.4. Backend API với FastAPI

**Cấu trúc đơn giản:**

```
be/app/
├── main.py              # FastAPI app
├── api/
│   └── chat.py         # POST /api/chat
└── core/
    ├── rag.py          # ask() - LangChain RAG pipeline
    └── embedding.py    # build_vector_store() - LangChain FAISS
```

**API Endpoint:**

```python
@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    """Main chat endpoint với LangChain"""
    result = ask(
        query=req.query,
        index_dir=INDEX_DIR,
        law_type_filter=req.law_type_filter
    )
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"]
    )
```

### 4.5. Frontend đơn giản

React Chat UI với các tính năng cơ bản:
- Input field để nhập câu hỏi
- Hiển thị messages (user + assistant)
- Hiển thị sources/citations
- Law type selector (nếu có)

### 4.6. So sánh Core vs LangChain

| Thành phần | Core (Python thuần) | LangChain |
|------------|---------------------|-----------|
| Document Loading | Manual file reading | `TextLoader` |
| Text Splitting | Manual regex split | `RecursiveCharacterTextSplitter` |
| Embedding | SentenceTransformers trực tiếp | `HuggingFaceEmbeddings` wrapper |
| Vector Store | FAISS thủ công | `FAISS.from_documents()` |
| Retrieval | Manual similarity search | `vectorstore.as_retriever()` |
| Generation | OpenAI API trực tiếp | `RetrievalQA` chain |
| Prompt | F-string template | `PromptTemplate` |

**Ưu điểm của LangChain:**
- Code ngắn gọn, dễ maintain
- Tích hợp sẵn nhiều components
- Dễ dàng thay đổi LLM/embedding provider
- Hỗ trợ memory, streaming, callbacks

**Nhược điểm:**
- Abstraction layer che giấu implementation details
- Phụ thuộc vào thư viện lớn

### 4.7. Kết luận chương

Chương 4 đã áp dụng thành công RAG core từ chương 3 vào ứng dụng thực tế với LangChain:
- **Thay thế**: Các hàm thủ công bằng LangChain components
- **Giữ nguyên**: Logic core, cấu trúc dữ liệu, metadata filtering
- **Nâng cấp**: Thêm FastAPI backend, React frontend đơn giản
- **Kết quả**: Ứng dụng RAG hoàn chỉnh, dễ maintain và mở rộng

## CHƯƠNG 5: ỨNG DỤNG RAGFLOW

### 5.1. Lý do sử dụng công cụ mã nguồn mở
### 5.2. Giới thiệu RAGFlow (UI kéo thả pipeline)
### 5.3. Cài đặt bằng Docker
### 5.4. Xây dựng flow Ingest → Embed → Retrieve → Answer
### 5.5. Tích hợp heuristic riêng (như "Điều 101 – Bộ luật…")
### 5.6. So sánh với LangChain tự code

## CHƯƠNG 6: TỔNG HỢP & SO SÁNH BA CÁCH TIẾP CẬN

| Tiêu chí | Core Python | LangChain + App | RAGFlow |
|----------|-------------|-----------------|---------|
| Độ tùy chỉnh | Cao nhất | Vừa | Trung bình |
| Thời gian phát triển | Lâu | Nhanh hơn | Nhanh nhất |
| Khả năng mở rộng | Khó | Tốt | Tốt |
| Giao diện người dùng | Không | Có (React) | Có sẵn UI |
| Người dùng phù hợp | Học thuật | Dev, startup | Non-dev, enterprise |

## CHƯƠNG 7: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 7.1. Những gì đã đạt được
### 7.2. So sánh mục tiêu ban đầu – kết quả thực tế
### 7.3. Khó khăn & bài học kinh nghiệm
### 7.4. Hướng phát triển tiếp
- Agent RAG đa bước
- Kết hợp đồ thị tri thức (Knowledge Graph + RAG)
- Tích hợp giọng nói (Speech-to-Text + RAG Voice Assistant)
- Deploy serverless / AWS / GCP

## PHỤ LỤC

- A. Code mẫu RAG thuần Python
- B. API Documentation (Swagger/OpenAPI)
- C. Dataset mẫu (PDF, JSON sản phẩm, thuốc)
- D. Kết quả test, câu hỏi – đáp thực tế
- E. Tài liệu tham khảo