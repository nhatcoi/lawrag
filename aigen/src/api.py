"""
FastAPI REST API cho RAG System
"""

import os
import re
from typing import List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from . import retriever, generator

load_dotenv()

# Project paths
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent
_default_public_dir = _project_root / "public"
_default_index_dir = _project_root / "faiss_index"

app = FastAPI(title="RAG API - Hệ thống truy xuất pháp luật", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static front-end tại /app nếu thư mục tồn tại
_public_dir = Path(os.environ.get("PUBLIC_DIR", str(_default_public_dir)))
if _public_dir.is_dir():
    app.mount("/app", StaticFiles(directory=str(_public_dir), html=True), name="static")


class AskRequest(BaseModel):
    """Request model cho endpoint /ask"""
    query: str = Field(..., description="Câu hỏi người dùng")
    index_dir: str = Field(default=str(_default_index_dir))
    provider: str = Field(default="local", description="'openai' hoặc 'local'")
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
    """Endpoint chính cho RAG chat"""
    try:
        contexts = []
        
        # Heuristic: Nếu query chứa "Điều <số>", chèn trực tiếp nội dung điều đó
        match = re.search(r"(?i)(điều)\s+(\d+)", req.query)
        if match:
            direct_text = retriever.try_get_article_by_number(req.index_dir, match.group(2))
            if direct_text:
                contexts.append(direct_text)

        # Truy xuất tài liệu liên quan từ FAISS index
        results = retriever.retrieve(req.query, req.index_dir, req.top_k, req.provider, req.local_model)
        contexts.extend([r.get("text", "") for r in results])
        
        # Tạo câu trả lời bằng LLM
        answer = generator.generate_answer(req.query, contexts, req.groq_model)
        
        # Chuẩn bị sources từ results đã truy xuất
        sources = [
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


