"""
FastAPI cho hệ thống RAG pháp luật
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import os

from rag import LegalRAG


# Khởi tạo FastAPI
app = FastAPI(title="Legal RAG API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/app", StaticFiles(directory="public", html=True), name="static")

# Global RAG instance
rag_system = None


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceResponse(BaseModel):
    rank: int
    score: float
    content: str
    article_id: str
    source: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceResponse]
    total_sources: int


@app.on_event("startup")
async def startup():
    """Khởi tạo RAG system khi start server"""
    global rag_system
    try:
        rag_system = LegalRAG()
        rag_system.load_knowledge_base("./vector_store")
        print("✅ RAG system đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo RAG system: {e}")


@app.get("/")
async def root():
    return {
        "message": "Legal RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "chat": "/ask"
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """API endpoint để đặt câu hỏi"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system chưa sẵn sàng")
    
    try:
        result = rag_system.ask_question(request.question, request.top_k)
        
        # Format sources
        sources = [
            SourceResponse(
                rank=src.get("rank", 0),
                score=src.get("score", 0.0),
                content=src.get("content", ""),
                article_id=src.get("article_id", ""),
                source=src.get("source", "")
            )
            for src in result["sources"]
        ]
        
        return QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            total_sources=result["total_sources"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if rag_system else "unhealthy",
        "rag_ready": rag_system is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
