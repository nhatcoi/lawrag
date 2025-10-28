"""
Module tạo và quản lý vector store
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any


class VectorStore:
    """Quản lý vector store với FAISS"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Tạo embeddings cho danh sách text"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def build_index(self, texts: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Xây dựng FAISS index"""
        print(f"Đang tạo embeddings cho {len(texts)} documents...")
        
        # Tạo embeddings
        embeddings = self.create_embeddings(texts)
        
        # Chuẩn hóa L2 cho cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Tạo FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        # Lưu metadata
        self.metadata = metadata
        
        print("✅ Đã tạo vector store thành công!")
    
    def save_index(self, index_dir: str) -> None:
        """Lưu index và metadata"""
        os.makedirs(index_dir, exist_ok=True)
        
        # Lưu FAISS index
        faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))
        
        # Lưu metadata
        with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Đã lưu vector store vào: {index_dir}")
    
    def load_index(self, index_dir: str) -> None:
        """Tải index và metadata"""
        # Tải FAISS index
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        
        # Tải metadata
        with open(os.path.join(index_dir, "metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        print(f"Đã tải vector store từ: {index_dir}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Tìm kiếm trong vector store"""
        # Tạo embedding cho query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Tìm kiếm
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format kết quả
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                results.append({
                    "rank": i + 1,
                    "score": float(score),
                    "content": self.metadata[idx].get("content", ""),
                    "article_id": self.metadata[idx].get("article_id", ""),
                    "source": self.metadata[idx].get("source", "")
                })
        
        return results
