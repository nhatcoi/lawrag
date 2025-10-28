"""
Module retriever sử dụng LangChain
Kết hợp vector search với heuristic tìm kiếm điều luật cụ thể
"""

import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from .langchain_vectorstore import VectorStoreManager


class LegalRetriever(BaseRetriever):
    """Custom retriever cho tài liệu pháp luật với LangChain"""
    
    vector_store_manager: VectorStoreManager
    top_k: int
    enable_article_heuristic: bool
    article_pattern: re.Pattern
    
    def __init__(self, 
                 vector_store_manager: VectorStoreManager,
                 top_k: int = 5,
                 enable_article_heuristic: bool = True):
        """
        Khởi tạo LegalRetriever
        
        Args:
            vector_store_manager: VectorStoreManager instance
            top_k: Số lượng kết quả trả về
            enable_article_heuristic: Bật heuristic tìm điều luật cụ thể
        """
        # Regex để tìm "Điều X" trong query
        article_pattern = re.compile(r"(?i)(điều)\s+(\d+)")
        
        super().__init__(
            vector_store_manager=vector_store_manager,
            top_k=top_k,
            enable_article_heuristic=enable_article_heuristic,
            article_pattern=article_pattern
        )
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Lấy documents liên quan (implement abstract method)
        
        Args:
            query: Câu truy vấn
            
        Returns:
            List các Document liên quan
        """
        return self.retrieve(query)
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Tìm kiếm documents liên quan
        
        Args:
            query: Câu truy vấn
            
        Returns:
            List các Document với metadata bổ sung
        """
        results = []
        
        # Heuristic: Tìm điều luật cụ thể
        if self.enable_article_heuristic:
            article_docs = self._find_article_by_number(query)
            if article_docs:
                results.extend(article_docs)
                print(f"Tìm thấy {len(article_docs)} điều luật cụ thể")
        
        # Vector search cho các documents còn lại
        remaining_k = max(1, self.top_k - len(results))
        vector_docs = self._vector_search(query, remaining_k)
        
        # Thêm metadata cho vector search results
        for i, doc in enumerate(vector_docs):
            doc.metadata.update({
                "retrieval_method": "vector_search",
                "rank": len(results) + i + 1
            })
        
        results.extend(vector_docs)
        
        print(f"Tổng cộng {len(results)} documents được trả về")
        return results[:self.top_k]
    
    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """
        Tìm kiếm với điểm số similarity
        
        Args:
            query: Câu truy vấn
            
        Returns:
            List các tuple (Document, score)
        """
        results_with_scores = []
        
        # Heuristic: Tìm điều luật cụ thể (score cao)
        if self.enable_article_heuristic:
            article_docs = self._find_article_by_number(query)
            if article_docs:
                for doc in article_docs:
                    # Gán score cao cho heuristic results
                    doc.metadata.update({
                        "retrieval_method": "article_heuristic",
                        "similarity_score": 0.95
                    })
                    results_with_scores.append((doc, 0.95))
        
        # Vector search với scores
        remaining_k = max(1, self.top_k - len(results_with_scores))
        vector_results = self._vector_search_with_scores(query, remaining_k)
        
        # Thêm metadata cho vector search results
        for i, (doc, score) in enumerate(vector_results):
            doc.metadata.update({
                "retrieval_method": "vector_search",
                "similarity_score": score,
                "rank": len(results_with_scores) + i + 1
            })
            results_with_scores.append((doc, score))
        
        return results_with_scores[:self.top_k]
    
    def _find_article_by_number(self, query: str) -> List[Document]:
        """
        Tìm điều luật cụ thể bằng số
        
        Args:
            query: Câu truy vấn
            
        Returns:
            List các Document điều luật tìm thấy
        """
        match = self.article_pattern.search(query)
        if not match:
            return []
        
        article_number = match.group(2)
        
        # Tìm kiếm trong vector store với query cụ thể
        search_query = f"Điều {article_number}"
        
        try:
            # Tìm kiếm với query cụ thể về điều luật
            docs = self.vector_store_manager.similarity_search(search_query, k=3)
            
            # Lọc chỉ lấy documents có chứa số điều luật
            filtered_docs = []
            for doc in docs:
                content = doc.page_content.lower()
                metadata = doc.metadata
                
                # Kiểm tra trong content hoặc metadata
                if (f"điều {article_number}" in content or 
                    metadata.get("article_number") == article_number or
                    metadata.get("article_id", "").lower() == f"điều {article_number}"):
                    filtered_docs.append(doc)
            
            return filtered_docs[:1]  # Chỉ lấy 1 kết quả tốt nhất
            
        except Exception as e:
            print(f"Lỗi khi tìm điều luật {article_number}: {e}")
            return []
    
    def _vector_search(self, query: str, k: int) -> List[Document]:
        """
        Tìm kiếm vector similarity
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả
            
        Returns:
            List các Document
        """
        try:
            return self.vector_store_manager.similarity_search(query, k=k)
        except Exception as e:
            print(f"Lỗi vector search: {e}")
            return []
    
    def _vector_search_with_scores(self, query: str, k: int) -> List[tuple]:
        """
        Tìm kiếm vector similarity với scores
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả
            
        Returns:
            List các tuple (Document, score)
        """
        try:
            return self.vector_store_manager.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Lỗi vector search with scores: {e}")
            return []
    
    def get_retrieval_info(self, query: str) -> Dict[str, Any]:
        """
        Lấy thông tin về quá trình retrieval
        
        Args:
            query: Câu truy vấn
            
        Returns:
            Dict chứa thông tin retrieval
        """
        results = self.retrieve_with_scores(query)
        
        info = {
            "query": query,
            "total_results": len(results),
            "methods_used": [],
            "results_by_method": {}
        }
        
        for doc, score in results:
            method = doc.metadata.get("retrieval_method", "unknown")
            
            if method not in info["methods_used"]:
                info["methods_used"].append(method)
                info["results_by_method"][method] = []
            
            info["results_by_method"][method].append({
                "article_id": doc.metadata.get("article_id", "unknown"),
                "score": score,
                "content_preview": doc.page_content[:100] + "..."
            })
        
        return info


def create_legal_retriever(
    vector_store_manager: VectorStoreManager,
    top_k: int = 5,
    enable_article_heuristic: bool = True
) -> LegalRetriever:
    """
    Factory function để tạo LegalRetriever
    
    Args:
        vector_store_manager: VectorStoreManager instance
        top_k: Số lượng kết quả
        enable_article_heuristic: Bật heuristic
        
    Returns:
        LegalRetriever instance
    """
    return LegalRetriever(
        vector_store_manager=vector_store_manager,
        top_k=top_k,
        enable_article_heuristic=enable_article_heuristic
    )
