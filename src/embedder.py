"""
Module xử lý tạo embeddings và lưu trữ FAISS index
Hỗ trợ cả OpenAI embeddings và local sentence-transformers
"""

import os
import json
import glob
from typing import List, Dict, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:
    raise RuntimeError(
        "faiss-cpu is required. Please install dependencies (pip install -r requirements.txt)."
    ) from exc

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

# Cấu hình mặc định
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))


def read_documents(split_dir: str) -> List[Tuple[str, str]]:
    """Đọc tất cả file .txt từ thư mục đã tách"""
    paths = sorted(glob.glob(os.path.join(split_dir, "*.txt")))
    documents: List[Tuple[str, str]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        doc_id = os.path.basename(path)
        if content:  # Chỉ lưu file có nội dung
            documents.append((doc_id, content))
    return documents


def get_embeddings_openai(client: "OpenAI", texts: List[str]) -> np.ndarray:
    """Tạo embeddings bằng OpenAI API"""
    embeddings: List[List[float]] = []
    # Xử lý theo batch để tránh rate limit
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_vectors = [d.embedding for d in resp.data]
        embeddings.extend(batch_vectors)    
    return np.array(embeddings, dtype=np.float32)


def get_embeddings_local(model_name: str, texts: List[str]) -> np.ndarray:
    """Tạo embeddings bằng sentence-transformers local model"""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed. Please install requirements.")
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=max(1, BATCH_SIZE // 4),  # Giảm batch size cho local model
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vectors.astype(np.float32)


def build_faiss_index(vectors: np.ndarray) -> "faiss.Index":
    """Xây dựng FAISS index với cosine similarity (Inner Product)"""
    # Chuẩn hóa L2 để sử dụng inner product cho cosine similarity
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine similarity khi đã normalize
    index.add(vectors)
    return index


def save_index(index: "faiss.Index", metadata: List[Dict[str, str]], index_dir: str) -> None:
    """Lưu FAISS index và metadata vào thư mục"""
    os.makedirs(index_dir, exist_ok=True)
    # Lưu index
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    # Lưu metadata
    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)




