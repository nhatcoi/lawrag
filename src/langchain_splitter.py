"""
Module tách PDF sử dụng LangChain Document Loaders và Text Splitters
Tách văn bản pháp luật thành các chunk phù hợp cho RAG
"""

import os
import re
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class LegalDocumentSplitter:
    """Tách tài liệu pháp luật thành các chunk dựa trên điều luật"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Khởi tạo splitter
        
        Args:
            chunk_size: Kích thước chunk (số ký tự)
            chunk_overlap: Số ký tự chồng lấp giữa các chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Regex để tìm tiêu đề điều luật
        self.article_regex = re.compile(r"^\s*Điều\s+(\d+)\b", re.UNICODE | re.IGNORECASE)
        
        # Text splitter cho các trường hợp không tách được theo điều luật
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Tải PDF và tách thành các Document
        
        Args:
            pdf_path: Đường dẫn file PDF
            
        Returns:
            List các Document
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")
        
        # Sử dụng PyPDFLoader của LangChain
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"Đã tải PDF: {len(documents)} trang")
        return documents
    
    def split_by_articles(self, documents: List[Document]) -> List[Document]:
        """
        Tách documents theo điều luật
        
        Args:
            documents: List các Document từ PDF
            
        Returns:
            List các Document đã tách theo điều luật
        """
        article_docs = []
        current_article = []
        current_article_id = None
        
        # Ghép tất cả text từ các trang
        full_text = "\n".join([doc.page_content for doc in documents])
        lines = full_text.splitlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Kiểm tra có phải tiêu đề điều luật không
            match = self.article_regex.match(line)
            if match:
                # Lưu điều luật trước đó
                if current_article and current_article_id:
                    article_text = "\n".join(current_article)
                    article_doc = Document(
                        page_content=article_text,
                        metadata={
                            "source": documents[0].metadata.get("source", ""),
                            "article_id": current_article_id,
                            "article_number": match.group(1),
                            "type": "legal_article"
                        }
                    )
                    article_docs.append(article_doc)
                
                # Bắt đầu điều luật mới
                article_number = match.group(1)
                current_article_id = f"Điều {article_number}"
                current_article = [line]
            else:
                # Thêm dòng vào điều luật hiện tại
                if current_article_id:
                    current_article.append(line)
        
        # Lưu điều luật cuối cùng
        if current_article and current_article_id:
            article_text = "\n".join(current_article)
            article_doc = Document(
                page_content=article_text,
                metadata={
                    "source": documents[0].metadata.get("source", ""),
                    "article_id": current_article_id,
                    "type": "legal_article"
                }
            )
            article_docs.append(article_doc)
        
        print(f"Đã tách thành {len(article_docs)} điều luật")
        return article_docs
    
    def split_large_articles(self, documents: List[Document]) -> List[Document]:
        """
        Tách các điều luật quá dài thành chunk nhỏ hơn
        
        Args:
            documents: List các Document điều luật
            
        Returns:
            List các Document đã được tách chunk
        """
        final_docs = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata.copy()
            
            # Nếu điều luật quá dài, tách thành chunk
            if len(content) > self.chunk_size:
                chunks = self.text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            **metadata,
                            "chunk_id": f"{metadata.get('article_id', 'unknown')}_chunk_{i}",
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    final_docs.append(chunk_doc)
            else:
                # Điều luật ngắn, giữ nguyên
                final_docs.append(doc)
        
        print(f"Sau khi tách chunk: {len(final_docs)} documents")
        return final_docs
    
    def save_articles_to_files(self, article_docs: List[Document], output_dir: str) -> None:
        """
        Lưu các điều luật ra file txt riêng biệt
        
        Args:
            article_docs: List các Document điều luật
            output_dir: Thư mục lưu file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for doc in article_docs:
            article_id = doc.metadata.get("article_id", "unknown")
            filename = f"{article_id.lower().replace(' ', '_')}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(doc.page_content)
        
        print(f"Đã lưu {len(article_docs)} điều luật vào: {output_dir}")

    def process_pdf(self, pdf_path: str, save_articles: bool = False, output_dir: str = "./output_dieu_luat") -> List[Document]:
        """
        Xử lý PDF hoàn chỉnh: tải -> tách theo điều luật -> tách chunk
        
        Args:
            pdf_path: Đường dẫn file PDF
            save_articles: Có lưu các điều luật ra file txt không
            output_dir: Thư mục lưu file điều luật
            
        Returns:
            List các Document cuối cùng
        """
        # Bước 1: Tải PDF
        documents = self.load_pdf(pdf_path)
        
        # Bước 2: Tách theo điều luật
        article_docs = self.split_by_articles(documents)
        
        # Bước 2.5: Lưu điều luật ra file (nếu được yêu cầu)
        if save_articles:
            self.save_articles_to_files(article_docs, output_dir)
        
        # Bước 3: Tách chunk cho các điều luật dài
        final_docs = self.split_large_articles(article_docs)
        
        return final_docs


def create_legal_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> LegalDocumentSplitter:
    """
    Factory function để tạo LegalDocumentSplitter
    
    Args:
        chunk_size: Kích thước chunk
        chunk_overlap: Số ký tự chồng lấp
        
    Returns:
        LegalDocumentSplitter instance
    """
    return LegalDocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
