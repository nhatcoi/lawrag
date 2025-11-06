"""
RAG System - Luật với python
Sử dụng sentence-transformers/all-MiniLM-L12-v2 và Groq llama-3.3-70b-versatile
"""

import os
import re
import json
import glob
from typing import List, Dict, Tuple, Optional

import numpy as np
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Cấu hình
EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
BATCH_SIZE = 32


# ==================== PDF Processing ====================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Trích xuất text từ PDF"""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def split_articles(lines: List[str]) -> List[Tuple[str, str]]:
    """Tách các điều luật từ text"""
    articles = []
    current_id = ""
    current_lines = []
    article_regex = re.compile(r"^\s*Điều\s+(\d+)\b", re.UNICODE)
    
    for line in lines:
        line = line.rstrip("\n")
        m = article_regex.match(line)
        if m:
            if current_id:
                articles.append((current_id, "\n".join(current_lines)))
            number = m.group(1)
            current_id = f"Điều {number}"
            current_lines = [line]
        else:
            if current_id:
                current_lines.append(line)
    
    if current_id:
        articles.append((current_id, "\n".join(current_lines)))
    
    return articles


def sanitize_filename(name: str) -> str:
    """Làm sạch tên file"""
    return re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")


# ==================== Embedding & Index ====================

def create_embeddings(texts: List[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    """Tạo embeddings bằng sentence-transformers"""
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vectors.astype(np.float32)


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Xây dựng FAISS index với cosine similarity"""
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(index: faiss.Index, metadata: List[Dict], index_dir: str):
    """Lưu index và metadata"""
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_index(index_dir: str) -> Tuple[faiss.Index, List[Dict]]:
    """Tải index và metadata"""
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# ==================== Retrieval ====================

def retrieve(query: str, index_dir: str, top_k: int = 5) -> List[Dict]:
    """Truy xuất tài liệu liên quan"""
    index, metadata = load_index(index_dir)
    
    # Embed query
    model = SentenceTransformer(EMBED_MODEL)
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    q_vec = q_vec.astype(np.float32)
    faiss.normalize_L2(q_vec)
    
    # Search
    D, I = index.search(q_vec, top_k)
    
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx]
        path = item.get("path", "")
        text = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except:
            pass
        results.append({
            "rank": rank + 1,
            "score": float(score),
            "id": item.get("id", ""),
            "path": path,
            "text": text,
        })
    return results


def get_article_by_number(index_dir: str, article_number: str) -> Optional[str]:
    """Lấy nội dung điều luật theo số"""
    try:
        with open(os.path.join(index_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except:
        return None
    
    target_name = f"điều_{article_number}.txt".lower()
    for item in metadata:
        file_id = str(item.get("id", "")).lower()
        path = item.get("path", "")
        if target_name == file_id or target_name in file_id:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except:
                pass
    return None


# ==================== Generation ====================

def generate_answer(query: str, contexts: List[str], model: str = GROQ_MODEL) -> str:
    """Tạo câu trả lời bằng Groq"""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY chưa được thiết lập")
    
    client = Groq(api_key=api_key)
    context_block = "\n\n".join(contexts)[:20000]
    
    system_prompt = (
        "Bạn là trợ lý trả lời câu hỏi dựa trên ngữ cảnh pháp luật Việt Nam. "
        "Chỉ dùng thông tin trong ngữ cảnh. Nếu thiếu thông tin, hãy nói không đủ dữ liệu."
    )
    user_prompt = (
        f"Ngữ cảnh:\n{context_block}\n\nCâu hỏi: {query}\n"
        "Yêu cầu: Trả lời ngắn gọn, kèm trích dẫn điều luật (nếu có)."
    )
    
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content or ""


# ==================== Main Functions ====================

def process_pdf(pdf_path: str, output_dir: str):
    """Tách PDF thành các file điều luật"""
    print(f"Đọc PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    lines = text.splitlines()
    articles = split_articles(lines)
    print(f"Phát hiện {len(articles)} điều luật")
    
    os.makedirs(output_dir, exist_ok=True)
    for article_id, content in articles:
        filename = sanitize_filename(article_id.lower()) + ".txt"
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
    print(f"Đã ghi {len(articles)} file vào: {output_dir}")


def build_embeddings(split_dir: str, index_dir: str):
    """Tạo embeddings và lưu index"""
    # Đọc documents
    paths = sorted(glob.glob(os.path.join(split_dir, "*.txt")))
    documents = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            doc_id = os.path.basename(path)
            documents.append((doc_id, content, path))
    
    if not documents:
        raise FileNotFoundError("Không tìm thấy tài liệu .txt nào")
    
    print(f"Tạo embeddings cho {len(documents)} tài liệu...")
    texts = [content for _, content, _ in documents]
    vectors = create_embeddings(texts)
    
    print("Xây dựng FAISS index...")
    index = build_index(vectors)
    metadata = [{"id": doc_id, "path": path} for doc_id, _, path in documents]
    save_index(index, metadata, index_dir)
    print(f"Đã lưu index vào: {index_dir}")


def ask_question(query: str, index_dir: str, top_k: int = 5) -> str:
    """Hỏi câu hỏi và nhận câu trả lời"""
    contexts = []
    
    # Nếu query chứa "Điều <số>", lấy trực tiếp
    m = re.search(r"(?i)(điều)\s+(\d+)", query)
    if m:
        direct_text = get_article_by_number(index_dir, m.group(2))
        if direct_text:
            contexts.append(direct_text)
    
    # Truy xuất tài liệu liên quan
    results = retrieve(query, index_dir, top_k)
    contexts.extend([r["text"] for r in results])
    
    # Tạo câu trả lời
    answer = generate_answer(query, contexts)
    return answer


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System")
    sub = parser.add_subparsers(dest="command", required=True)
    
    # Split PDF
    p_split = sub.add_parser("split", help="Tách PDF thành các file điều luật")
    p_split.add_argument("--pdf", required=True, help="Đường dẫn file PDF")
    p_split.add_argument("--output", default="output_dieu_luat", help="Thư mục output")
    
    # Build embeddings
    p_embed = sub.add_parser("embed", help="Tạo embeddings và lưu index")
    p_embed.add_argument("--split-dir", required=True, help="Thư mục chứa các file .txt")
    p_embed.add_argument("--index-dir", default="faiss_index", help="Thư mục lưu index")
    
    # Ask question
    p_ask = sub.add_parser("ask", help="Đặt câu hỏi")
    p_ask.add_argument("--query", required=True, help="Câu hỏi")
    p_ask.add_argument("--index-dir", default="faiss_index", help="Thư mục chứa index")
    p_ask.add_argument("--top-k", type=int, default=5, help="Số lượng tài liệu truy xuất")
    
    # All in one
    p_all = sub.add_parser("all", help="Chạy split rồi embed")
    p_all.add_argument("--pdf", required=True, help="Đường dẫn file PDF")
    p_all.add_argument("--split-dir", default="output_dieu_luat", help="Thư mục output split")
    p_all.add_argument("--index-dir", default="faiss_index", help="Thư mục lưu index")
    
    args = parser.parse_args()
    
    if args.command == "split":
        process_pdf(args.pdf, args.output)
    elif args.command == "embed":
        build_embeddings(args.split_dir, args.index_dir)
    elif args.command == "ask":
        answer = ask_question(args.query, args.index_dir, args.top_k)
        print(answer)
    elif args.command == "all":
        process_pdf(args.pdf, args.split_dir)
        build_embeddings(args.split_dir, args.index_dir)

