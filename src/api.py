"""
FastAPI REST API cho RAG System
Cung cấp endpoint để chat và truy xuất thông tin pháp luật
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from . import retriever, generator
import re


load_dotenv()
app = FastAPI(title="RAG API - Hệ thống truy xuất pháp luật", version="1.0.0")

# CORS: Cho phép truy cập từ trình duyệt web
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static front-end tại /app để tránh xung đột với /ask endpoint
app.mount("/app", StaticFiles(directory="/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/public", html=True), name="static")


# Pydantic models cho API request/response
class AskRequest(BaseModel):
    """Request model cho endpoint /ask"""
    query: str = Field(..., description="Câu hỏi người dùng")
    index_dir: str = "/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/faiss_index"
    provider: str = Field("local", description="'openai' hoặc 'local'")
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    groq_model: str = "llama-3.3-70b-versatile"


class Source(BaseModel):
    """Model cho thông tin nguồn tài liệu"""
    rank: int
    score: float
    id: str
    path: str


class AskResponse(BaseModel):
    """Response model cho endpoint /ask"""
    answer: str
    sources: List[Source]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Endpoint chính cho RAG chat
    Nhận câu hỏi, truy xuất thông tin liên quan và tạo câu trả lời
    """
    try:
        contexts: list[str] = []
        
        # Heuristic: Nếu query chứa "Điều <số>", chèn trực tiếp nội dung điều đó
        m = re.search(r"(?i)(điều)\s+(\d+)", req.query or "")
        if m:
            direct_text = retriever.try_get_article_by_number(req.index_dir, m.group(2))
            if direct_text:
                contexts.append(direct_text)

        # Truy xuất tài liệu liên quan từ FAISS index
        results = retriever.retrieve(
            query=req.query,
            index_dir=req.index_dir,
            top_k=req.top_k,
            provider=req.provider,
            local_model=req.local_model,
        )
        contexts.extend([r.get("text", "") for r in results])
        
        # Tạo câu trả lời bằng LLM
        answer = generator.generate_answer(
            query=req.query,
            contexts=contexts,
            model=req.groq_model,
        )
        
        # Chuẩn bị thông tin nguồn cho response
        sources: List[Source] = [
            Source(
                rank=int(r.get("rank", 0)),
                score=float(r.get("score", 0.0)),
                id=str(r.get("id", "")),
                path=str(r.get("path", "")),
            )
            for r in results
        ]
        return AskResponse(answer=answer or "", sources=sources)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


