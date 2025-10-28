"""
Module RAG chính - kết hợp tất cả components
"""

from typing import List, Dict, Any
from splitter import LegalSplitter
from vectorstore import VectorStore
from retriever import LegalRetriever
from generator import AnswerGenerator


class LegalRAG:
    """Hệ thống RAG cho pháp luật"""
    
    def __init__(self):
        self.splitter = LegalSplitter()
        self.vector_store = VectorStore()
        self.retriever = None
        self.generator = AnswerGenerator()
    
    def build_knowledge_base(self, pdf_path: str, output_dir: str, index_dir: str) -> None:
        """Xây dựng knowledge base từ PDF"""
        print("🚀 Bắt đầu xây dựng knowledge base...")
        
        # Bước 1: Tách PDF thành điều luật
        print("📄 Bước 1: Tách PDF thành điều luật...")
        articles = self.splitter.process_pdf(pdf_path, output_dir)
        
        # Bước 2: Tạo vector store
        print("🔍 Bước 2: Tạo vector store...")
        texts = [content for _, content in articles]
        metadata = [
            {
                "article_id": article_id,
                "content": content,
                "source": f"{output_dir}/{article_id.lower().replace(' ', '_')}.txt"
            }
            for article_id, content in articles
        ]
        
        self.vector_store.build_index(texts, metadata)
        self.vector_store.save_index(index_dir)
        
        # Bước 3: Khởi tạo retriever
        self.retriever = LegalRetriever(self.vector_store)
        
        print("✅ Hoàn thành xây dựng knowledge base!")
    
    def load_knowledge_base(self, index_dir: str) -> None:
        """Tải knowledge base đã có"""
        print("📚 Đang tải knowledge base...")
        self.vector_store.load_index(index_dir)
        self.retriever = LegalRetriever(self.vector_store)
        print("✅ Đã tải knowledge base!")
    
    def ask_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Đặt câu hỏi và nhận câu trả lời"""
        print(f"❓ Câu hỏi: {question}")
        
        # Truy xuất thông tin liên quan
        contexts = self.retriever.retrieve(question, top_k)
        
        # Tạo câu trả lời
        answer = self.generator.generate_answer(question, contexts)
        
        return {
            "question": question,
            "answer": answer,
            "sources": contexts,
            "total_sources": len(contexts)
        }
