"""
Module RAG chain sử dụng LangChain
Kết hợp retrieval và generation với prompt templates
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from .langchain_retriever import LegalRetriever


class LegalRAGChain:
    """RAG chain cho tài liệu pháp luật với LangChain"""
    
    def __init__(self, 
                 retriever: LegalRetriever,
                 llm_provider: str = "groq",
                 llm_model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.2):
        """
        Khởi tạo LegalRAGChain
        
        Args:
            retriever: LegalRetriever instance
            llm_provider: Provider LLM ("groq", "openai")
            llm_model: Tên model LLM
            temperature: Temperature cho generation
        """
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        
        # Khởi tạo LLM
        self.llm = self._create_llm()
        
        # Tạo prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Xây dựng RAG chain
        self.rag_chain = self._build_rag_chain()
    
    def _create_llm(self):
        """Tạo LLM theo provider"""
        if self.llm_provider == "groq":
            return ChatGroq(
                model=self.llm_model,
                temperature=self.temperature,
                groq_api_key=self._get_groq_api_key()
            )
        
        elif self.llm_provider == "openai":
            return ChatOpenAI(
                model=self.llm_model,
                temperature=self.temperature,
                openai_api_key=self._get_openai_api_key()
            )
        
        else:
            raise ValueError(f"LLM provider không hỗ trợ: {self.llm_provider}")
    
    def _get_groq_api_key(self) -> str:
        """Lấy Groq API key"""
        import os
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY không được thiết lập")
        return api_key
    
    def _get_openai_api_key(self) -> str:
        """Lấy OpenAI API key"""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY không được thiết lập")
        return api_key
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Tạo prompt template cho pháp luật"""
        system_prompt = """Bạn là trợ lý chuyên gia về pháp luật Việt Nam. 
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin pháp luật được cung cấp.

Quy tắc:
1. Chỉ sử dụng thông tin trong ngữ cảnh được cung cấp
2. Nếu không đủ thông tin, hãy nói rõ "Không đủ dữ liệu để trả lời"
3. Trả lời ngắn gọn, chính xác, dễ hiểu
4. Trích dẫn điều luật cụ thể khi có thể
5. Sử dụng tiếng Việt tự nhiên"""

        human_prompt = """Ngữ cảnh pháp luật:
{context}

Câu hỏi: {question}

Hãy trả lời câu hỏi dựa trên thông tin pháp luật trên:"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    def _format_documents(self, docs: List[Document]) -> str:
        """Format documents thành context string"""
        if not docs:
            return "Không có thông tin liên quan."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            # Lấy thông tin metadata
            article_id = doc.metadata.get("article_id", f"Tài liệu {i}")
            source = doc.metadata.get("source", "")
            method = doc.metadata.get("retrieval_method", "unknown")
            
            # Format content
            content = doc.page_content.strip()
            
            # Tạo header cho document
            header = f"--- {article_id} ---"
            if source:
                header += f" (Nguồn: {source})"
            if method:
                header += f" [Tìm bằng: {method}]"
            
            formatted_doc = f"{header}\n{content}"
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    def _build_rag_chain(self):
        """Xây dựng RAG chain"""
        from langchain_core.runnables import RunnableLambda
        
        # Tạo parallel chain để lấy context và question
        retrieval_chain = RunnableParallel({
            "context": RunnableLambda(lambda x: self._format_documents(self.retriever.retrieve(x))),
            "question": RunnablePassthrough()
        })
        
        # Kết hợp với prompt và LLM
        rag_chain = (
            retrieval_chain
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def invoke(self, question: str) -> str:
        """
        Thực thi RAG chain
        
        Args:
            question: Câu hỏi
            
        Returns:
            Câu trả lời
        """
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            return f"Lỗi khi tạo câu trả lời: {str(e)}"
    
    def invoke_with_sources(self, question: str) -> Dict[str, Any]:
        """
        Thực thi RAG chain với thông tin sources
        
        Args:
            question: Câu hỏi
            
        Returns:
            Dict chứa answer và sources
        """
        try:
            # Lấy documents
            docs = self.retriever.retrieve_with_scores(question)
            
            # Format context
            context = self._format_documents([doc for doc, _ in docs])
            
            # Tạo prompt
            prompt = self.prompt_template.format(context=context, question=question)
            
            # Gọi LLM
            messages = self.prompt_template.format_messages(context=context, question=question)
            answer = self.llm.invoke(messages).content
            
            # Format sources
            sources = []
            for doc, score in docs:
                source_info = {
                    "article_id": doc.metadata.get("article_id", "unknown"),
                    "article_number": doc.metadata.get("article_number", ""),
                    "source": doc.metadata.get("source", ""),
                    "similarity_score": score,
                    "retrieval_method": doc.metadata.get("retrieval_method", "unknown"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "query": question,
                "total_sources": len(sources)
            }
            
        except Exception as e:
            return {
                "answer": f"Lỗi khi tạo câu trả lời: {str(e)}",
                "sources": [],
                "query": question,
                "total_sources": 0,
                "error": str(e)
            }
    
    def get_retrieval_info(self, question: str) -> Dict[str, Any]:
        """Lấy thông tin về quá trình retrieval"""
        return self.retriever.get_retrieval_info(question)


def create_legal_rag_chain(
    retriever: LegalRetriever,
    llm_provider: str = "groq",
    llm_model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2
) -> LegalRAGChain:
    """
    Factory function để tạo LegalRAGChain
    
    Args:
        retriever: LegalRetriever instance
        llm_provider: LLM provider
        llm_model: LLM model name
        temperature: Temperature
        
    Returns:
        LegalRAGChain instance
    """
    return LegalRAGChain(
        retriever=retriever,
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature
    )
