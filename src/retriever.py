"""
Module xử lý truy xuất thông tin từ FAISS index
Tìm kiếm các tài liệu liên quan đến câu hỏi của người dùng
"""

import os
import json
from typing import List, Dict, Tuple, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:
    raise RuntimeError(
        "faiss-cpu is required. Please install dependencies (pip install -r requirements.txt)."
    ) from exc

from . import embedder


def load_index(index_dir: str):
    """Tải FAISS index và metadata từ thư mục"""
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata: List[Dict[str, str]] = json.load(f)
    return index, metadata


def embed_query(query: str, provider: str, local_model: str) -> np.ndarray:
    """Tạo embedding cho câu hỏi của người dùng"""
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY chưa được thiết lập trong biến môi trường")
        client = embedder.OpenAI(api_key=api_key)
        vecs = embedder.get_embeddings_openai(client, [query])
    elif provider == "local":
        vecs = embedder.get_embeddings_local(local_model, [query])
    else:
        raise ValueError("provider phải là 'openai' hoặc 'local'")
    # Chuẩn hóa L2 để khớp với index đã được normalize
    faiss.normalize_L2(vecs)
    return vecs


def retrieve(query: str, index_dir: str, top_k: int, provider: str, local_model: str) -> List[Dict[str, str]]:
    """
    Tìm kiếm top_k tài liệu liên quan nhất đến câu hỏi
    Trả về danh sách các tài liệu với điểm số similarity
    """
    index, metadata = load_index(index_dir)
    q = embed_query(query, provider, local_model)
    
    # Tìm kiếm trong FAISS index
    D, I = index.search(q, top_k)  # D = distances, I = indices
    
    results: List[Dict[str, str]] = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        # Kiểm tra index hợp lệ
        if idx < 0 or idx >= len(metadata):
            continue
            
        item = metadata[idx]
        path = item.get("path", "")
        text = ""
        
        # Đọc nội dung file
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception:
            text = ""
            
        results.append({
            "rank": str(rank + 1),
            "score": f"{float(score):.4f}",
            "id": item.get("id", str(idx)),
            "path": path,
            "text": text,
        })
    return results


def try_get_article_by_number(index_dir: str, article_number: str) -> Optional[str]:
    """
    Tìm và đọc trực tiếp nội dung file điều_<số>.txt từ metadata
    Đây là heuristic để cải thiện độ chính xác khi hỏi về điều luật cụ thể
    """
    try:
        with open(os.path.join(index_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata: List[Dict[str, str]] = json.load(f)
    except Exception:
        return None

    # Tìm file điều luật theo số
    target_name = f"điều_{article_number}.txt".lower()
    for item in metadata:
        file_id = str(item.get("id", "")).lower()
        path = item.get("path", "")
        if target_name == file_id or target_name in file_id:
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    return fp.read().strip()
            except Exception:
                return None
    return None


