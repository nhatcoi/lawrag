"""
FastAPI REST API cho RAG System sử dụng LangChain
Cung cấp endpoint để chat và truy xuất thông tin pháp luật
"""

from typing import List, Optional, Dict, Any
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .langchain_vectorstore import create_vector_store_manager
from .langchain_retriever import create_legal_retriever
from .langchain_rag import create_legal_rag_chain


load_dotenv()
app = FastAPI(
    title="RAG API với LangChain - Hệ thống truy xuất pháp luật",
    version="2.0.0",
    description="API RAG sử dụng LangChain cho tài liệu pháp luật Việt Nam"
)

# CORS: Cho phép truy cập từ trình duyệt web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static front-end tại /app
app.mount("/app", StaticFiles(directory="/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/public", html=True), name="static")

# Global variables để cache components
vector_store_manager = None
retriever = None
rag_chain = None


def initialize_components(
    vector_store_type: str = "faiss",
    embeddings_type: str = "huggingface",
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    persist_directory: str = "./vector_store",
    llm_provider: str = "groq",
    llm_model: str = "llama-3.3-70b-versatile",
    top_k: int = 5,
    temperature: float = 0.2
) -> None:
    """Khởi tạo các components LangChain"""
    global vector_store_manager, retriever, rag_chain
    
    try:
        # Tạo vector store manager
        vector_store_manager = create_vector_store_manager(
            vector_store_type=vector_store_type,
            embeddings_type=embeddings_type,
            embeddings_model=embeddings_model,
            persist_directory=persist_directory
        )
        
        # Load vector store
        vector_store_manager.load_vector_store()
        
        # Tạo retriever
        retriever = create_legal_retriever(
            vector_store_manager=vector_store_manager,
            top_k=top_k,
            enable_article_heuristic=True
        )
        
        # Tạo RAG chain
        rag_chain = create_legal_rag_chain(
            retriever=retriever,
            llm_provider=llm_provider,
            llm_model=llm_model,
            temperature=temperature
        )
        
        print("✅ Đã khởi tạo components LangChain thành công!")
        
    except Exception as e:
        print(f"❌ Lỗi khởi tạo components: {e}")
        raise


# Pydantic models cho API
class AskRequest(BaseModel):
    """Request model cho endpoint /ask"""
    query: str = Field(..., description="Câu hỏi người dùng")
    top_k: Optional[int] = Field(5, description="Số lượng documents trả về")
    include_sources: Optional[bool] = Field(True, description="Có bao gồm sources không")


class Source(BaseModel):
    """Model cho thông tin nguồn tài liệu"""
    article_id: str
    article_number: Optional[str] = None
    source: str
    similarity_score: float
    retrieval_method: str
    content_preview: str


class AskResponse(BaseModel):
    """Response model cho endpoint /ask"""
    answer: str
    sources: List[Source]
    query: str
    total_sources: int
    retrieval_info: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model cho health check"""
    status: str
    components_loaded: bool
    vector_store_type: Optional[str] = None
    embeddings_type: Optional[str] = None
    llm_provider: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Khởi tạo components khi start server"""
    try:
        initialize_components()
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo components: {e}")
        print("Server vẫn chạy nhưng sẽ trả về lỗi khi gọi /ask")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global vector_store_manager, retriever, rag_chain
    
    components_loaded = all([
        vector_store_manager is not None,
        retriever is not None,
        rag_chain is not None
    ])
    
    return HealthResponse(
        status="healthy" if components_loaded else "unhealthy",
        components_loaded=components_loaded,
        vector_store_type=getattr(vector_store_manager, 'vector_store_type', None),
        embeddings_type=getattr(vector_store_manager, 'embeddings_type', None),
        llm_provider=getattr(rag_chain, 'llm_provider', None)
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Endpoint chính cho RAG chat
    Nhận câu hỏi, truy xuất thông tin liên quan và tạo câu trả lời
    """
    global rag_chain, retriever
    
    if not rag_chain or not retriever:
        raise HTTPException(
            status_code=503,
            detail="RAG components chưa được khởi tạo. Vui lòng kiểm tra server logs."
        )
    
    try:
        # Cập nhật top_k nếu được chỉ định
        if request.top_k != retriever.top_k:
            retriever.top_k = request.top_k
        
        if request.include_sources:
            # Lấy câu trả lời với sources
            result = rag_chain.invoke_with_sources(request.query)
            
            # Format sources
            sources = [
                Source(
                    article_id=source["article_id"],
                    article_number=source.get("article_number"),
                    source=source["source"],
                    similarity_score=source["similarity_score"],
                    retrieval_method=source["retrieval_method"],
                    content_preview=source["content_preview"]
                )
                for source in result["sources"]
            ]
            
            return AskResponse(
                answer=result["answer"],
                sources=sources,
                query=result["query"],
                total_sources=result["total_sources"],
                retrieval_info=rag_chain.get_retrieval_info(request.query)
            )
        
        else:
            # Chỉ lấy câu trả lời
            answer = rag_chain.invoke(request.query)
            return AskResponse(
                answer=answer,
                sources=[],
                query=request.query,
                total_sources=0
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý câu hỏi: {str(e)}"
        )


@app.post("/reinitialize")
async def reinitialize_components(
    vector_store_type: str = "faiss",
    embeddings_type: str = "huggingface",
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    persist_directory: str = "./vector_store",
    llm_provider: str = "groq",
    llm_model: str = "llama-3.3-70b-versatile",
    top_k: int = 5,
    temperature: float = 0.2
):
    """Reinitialize components với tham số mới"""
    try:
        initialize_components(
            vector_store_type=vector_store_type,
            embeddings_type=embeddings_type,
            embeddings_model=embeddings_model,
            persist_directory=persist_directory,
            llm_provider=llm_provider,
            llm_model=llm_model,
            top_k=top_k,
            temperature=temperature
        )
        
        return {
            "message": "Components đã được khởi tạo lại thành công",
            "config": {
                "vector_store_type": vector_store_type,
                "embeddings_type": embeddings_type,
                "embeddings_model": embeddings_model,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "top_k": top_k,
                "temperature": temperature
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi khởi tạo lại components: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG API với LangChain - Hệ thống truy xuất pháp luật",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "chat": "/ask",
        "web_ui": "/app/index.html"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
