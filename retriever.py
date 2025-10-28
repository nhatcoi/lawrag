"""
Module truy xuất thông tin từ vector store
"""

import os
import json
from typing import List, Dict, Any
from vectorstore import VectorStore


class LegalRetriever:
    """Truy xuất thông tin pháp luật"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Truy xuất thông tin liên quan"""
        # Thêm heuristic cho điều luật cụ thể
        if "điều" in query.lower():
            article_results = self._get_article_by_number(query)
            if article_results:
                return article_results
        
        # Tìm kiếm trong vector store
        results = self.vector_store.search(query, top_k)
        return results
    
    def _get_article_by_number(self, query: str) -> List[Dict[str, Any]]:
        """Tìm điều luật theo số"""
        import re
        match = re.search(r"điều\s+(\d+)", query.lower())
        if not match:
            return []
        
        article_number = match.group(1)
        target_file = f"điều_{article_number}.txt"
        
        # Tìm trong metadata
        for item in self.vector_store.metadata:
            if target_file in item.get("source", ""):
                return [{
                    "rank": 0,
                    "score": 1.0,
                    "content": item.get("content", ""),
                    "article_id": item.get("article_id", ""),
                    "source": item.get("source", ""),
                    "method": "article_heuristic"
                }]
        
        return []
