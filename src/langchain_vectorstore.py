"""
Module tạo và quản lý vector store sử dụng LangChain
Hỗ trợ nhiều loại embeddings và vector stores
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma as LangChainChroma


class VectorStoreManager:
    """Quản lý vector store với LangChain"""
    
    def __init__(self, 
                 vector_store_type: str = "faiss",
                 embeddings_type: str = "huggingface",
                 embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 persist_directory: Optional[str] = None):
        """
        Khởi tạo VectorStoreManager
        
        Args:
            vector_store_type: Loại vector store ("faiss", "chroma")
            embeddings_type: Loại embeddings ("openai", "huggingface")
            embeddings_model: Tên model embeddings
            persist_directory: Thư mục lưu trữ persistent
        """
        self.vector_store_type = vector_store_type
        self.embeddings_type = embeddings_type
        self.embeddings_model = embeddings_model
        self.persist_directory = persist_directory or "./vector_store"
        
        # Tạo thư mục lưu trữ
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo embeddings
        self.embeddings = self._create_embeddings()
        
        # Vector store sẽ được khởi tạo khi cần
        self.vector_store: Optional[VectorStore] = None
    
    def _create_embeddings(self) -> Embeddings:
        """Tạo embeddings theo loại được chỉ định"""
        if self.embeddings_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY không được thiết lập")
            
            return OpenAIEmbeddings(
                model=self.embeddings_model or "text-embedding-3-small",
                openai_api_key=api_key
            )
        
        elif self.embeddings_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=self.embeddings_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        else:
            raise ValueError(f"Loại embeddings không hỗ trợ: {self.embeddings_type}")
    
    def _create_vector_store(self, documents: List[Document]) -> VectorStore:
        """Tạo vector store từ documents"""
        if self.vector_store_type == "faiss":
            return FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        
        elif self.vector_store_type == "chroma":
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=os.path.join(self.persist_directory, "chroma")
            )
        
        else:
            raise ValueError(f"Loại vector store không hỗ trợ: {self.vector_store_type}")
    
    def build_vector_store(self, documents: List[Document]) -> None:
        """
        Xây dựng vector store từ documents
        
        Args:
            documents: List các Document để embed
        """
        print(f"Đang tạo embeddings cho {len(documents)} documents...")
        print(f"Sử dụng model: {self.embeddings_model}")
        
        # Tạo vector store
        self.vector_store = self._create_vector_store(documents)
        
        # Lưu vector store
        self.save_vector_store()
        
        print(f"Đã tạo vector store thành công!")
    
    def save_vector_store(self) -> None:
        """Lưu vector store vào disk"""
        if not self.vector_store:
            raise ValueError("Vector store chưa được khởi tạo")
        
        if self.vector_store_type == "faiss":
            save_path = os.path.join(self.persist_directory, "faiss_index")
            self.vector_store.save_local(save_path)
            print(f"Đã lưu FAISS index vào: {save_path}")
        
        elif self.vector_store_type == "chroma":
            # Chroma tự động persist
            print(f"Đã lưu Chroma index vào: {self.persist_directory}")
    
    def load_vector_store(self) -> None:
        """Tải vector store từ disk"""
        if self.vector_store_type == "faiss":
            save_path = os.path.join(self.persist_directory, "faiss_index")
            if os.path.exists(save_path):
                self.vector_store = FAISS.load_local(
                    save_path, 
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Đã tải FAISS index từ: {save_path}")
            else:
                raise FileNotFoundError(f"Không tìm thấy FAISS index tại: {save_path}")
        
        elif self.vector_store_type == "chroma":
            chroma_path = os.path.join(self.persist_directory, "chroma")
            if os.path.exists(chroma_path):
                self.vector_store = Chroma(
                    persist_directory=chroma_path,
                    embedding_function=self.embeddings
                )
                print(f"Đã tải Chroma index từ: {chroma_path}")
            else:
                raise FileNotFoundError(f"Không tìm thấy Chroma index tại: {chroma_path}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Tìm kiếm documents tương tự
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả trả về
            
        Returns:
            List các Document tương tự
        """
        if not self.vector_store:
            raise ValueError("Vector store chưa được khởi tạo hoặc tải")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Tìm kiếm với điểm số similarity
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả trả về
            
        Returns:
            List các tuple (Document, score)
        """
        if not self.vector_store:
            raise ValueError("Vector store chưa được khởi tạo hoặc tải")
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_vector_store(self) -> VectorStore:
        """Lấy vector store hiện tại"""
        if not self.vector_store:
            raise ValueError("Vector store chưa được khởi tạo")
        return self.vector_store


def create_vector_store_manager(
    vector_store_type: str = "faiss",
    embeddings_type: str = "huggingface", 
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    persist_directory: Optional[str] = None
) -> VectorStoreManager:
    """
    Factory function để tạo VectorStoreManager
    
    Args:
        vector_store_type: Loại vector store
        embeddings_type: Loại embeddings  
        embeddings_model: Tên model embeddings
        persist_directory: Thư mục lưu trữ
        
    Returns:
        VectorStoreManager instance
    """
    return VectorStoreManager(
        vector_store_type=vector_store_type,
        embeddings_type=embeddings_type,
        embeddings_model=embeddings_model,
        persist_directory=persist_directory
    )
