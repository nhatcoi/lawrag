"""
Module tách văn bản luật thành các điều luật riêng biệt
"""

import os
import re
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader


class LegalSplitter:
    """Tách văn bản luật thành các điều luật riêng biệt"""
    
    def __init__(self):
        self.article_regex = re.compile(r"^\s*Điều\s+(\d+)\b", re.UNICODE | re.IGNORECASE)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Trích xuất text từ PDF"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    
    def split_articles(self, text: str) -> List[Tuple[str, str]]:
        """Tách text thành các điều luật"""
        articles = []
        lines = text.splitlines()
        current_article = []
        current_article_id = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = self.article_regex.match(line)
            if match:
                # Lưu điều luật trước đó
                if current_article and current_article_id:
                    articles.append((current_article_id, "\n".join(current_article)))
                
                # Bắt đầu điều luật mới
                article_number = match.group(1)
                current_article_id = f"Điều {article_number}"
                current_article = [line]
            else:
                if current_article_id:
                    current_article.append(line)
        
        # Lưu điều luật cuối cùng
        if current_article and current_article_id:
            articles.append((current_article_id, "\n".join(current_article)))
        
        return articles
    
    def save_articles(self, articles: List[Tuple[str, str]], output_dir: str) -> None:
        """Lưu các điều luật vào file txt"""
        os.makedirs(output_dir, exist_ok=True)
        
        for article_id, content in articles:
            filename = re.sub(r"[^\w\-]+", "_", article_id.lower()) + ".txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content.strip() + "\n")
        
        print(f"Đã lưu {len(articles)} điều luật vào: {output_dir}")
    
    def process_pdf(self, pdf_path: str, output_dir: str) -> List[Tuple[str, str]]:
        """Xử lý PDF hoàn chỉnh"""
        print(f"Đang tách PDF: {pdf_path}")
        
        # Trích xuất text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Tách điều luật
        articles = self.split_articles(text)
        
        # Lưu file
        self.save_articles(articles, output_dir)
        
        return articles
