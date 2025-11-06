from pydantic import BaseModel, Field
from typing import List, Optional

class Source(BaseModel):
    rank: int
    score: float
    id: str
    path: str
    text: str

class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi người dùng")
    conversation_id: Optional[str] = Field(None, description="ID conversation hiện tại")
    provider: str = Field("local", description="'openai' hoặc 'local'")
    law_type_filter: Optional[str] = Field(None, description="Lọc theo loại luật, hoặc 'Tổng quan' để không lọc")

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    query: str
    conversation_id: str

class ConversationInfo(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int

class HistoryItem(BaseModel):
    id: str
    query: str
    answer: str
    timestamp: str
    sources_count: int

class ChunkInfo(BaseModel):
    filename: str
    path: str
    size: int
    created_at: str

class DocumentInfo(BaseModel):
    id: str
    filename: str
    original_name: str
    type: str
    size: int
    uploaded_at: str
    status: str
    chunks_count: int
    chunks: List[ChunkInfo] = []
    law_type: str = "Khác"

class SourceInfo(BaseModel):
    id: str
    filename: str
    path: str
    size: int
    type: str
    uploaded_at: str
    status: str
