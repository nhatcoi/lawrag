from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse, Source, ConversationInfo
from app.core import ask
from app.core.conversation import (
    create_conversation,
    get_conversation,
    add_message,
    get_conversation_context,
    get_all_conversations,
    delete_conversation,
    update_conversation_law_type_filter
)
from app.core.storage import add_to_history
from app.config import INDEX_DIR

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        import traceback
        import logging
        
        logger = logging.getLogger(__name__)
        
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query không được để trống")
        
        law_type_filter = req.law_type_filter or "Tổng quan"
        
        if not req.conversation_id:
            conv = create_conversation(law_type_filter=law_type_filter)
            conv_id = conv["id"]
        else:
            conv = get_conversation(req.conversation_id)
            if not conv:
                conv = create_conversation(law_type_filter=law_type_filter)
                conv_id = conv["id"]
            else:
                conv_id = req.conversation_id
                if req.law_type_filter:
                    update_conversation_law_type_filter(conv_id, law_type_filter)
                else:
                    law_type_filter = conv.get("law_type_filter", "Tổng quan")
        
        add_message(conv_id, "user", req.query)
        
        result = ask(req.query, INDEX_DIR, req.provider, law_type_filter)
        
        if not result or "answer" not in result:
            raise HTTPException(status_code=500, detail="Lỗi khi sinh câu trả lời từ LLM")
        
        sources = [
            Source(
                rank=s.get("rank", i+1),
                score=s.get("score", 0.0),
                id=s.get("id", ""),
                path=s.get("path", ""),
                text=s.get("text", ""),
            )
            for i, s in enumerate(result.get("sources", []))
        ]
        
        sources_data = [
            {
                "id": s.get("id", ""),
                "text": s.get("text", ""),
                "rank": s.get("rank", i+1)
            }
            for i, s in enumerate(result.get("sources", []))
        ]
        
        add_message(conv_id, "assistant", result["answer"], sources_data)
        add_to_history(req.query, result["answer"], len(sources))
        
        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            query=req.query,
            conversation_id=conv_id
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Lỗi xử lý chat: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in chat_endpoint: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")


@router.get("/conversations", response_model=List[ConversationInfo])
def get_conversations():
    try:
        conversations = get_all_conversations()
        return [
            ConversationInfo(
                id=conv["id"],
                title=conv["title"],
                created_at=conv["created_at"],
                updated_at=conv["updated_at"],
                message_count=len(conv.get("messages", []))
            )
            for conv in sorted(conversations, key=lambda x: x["updated_at"], reverse=True)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conv_id}")
def get_conversation_detail(conv_id: str):
    try:
        conv = get_conversation(conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conv_id}")
def delete_conversation_endpoint(conv_id: str):
    try:
        success = delete_conversation(conv_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/law-types")
def get_law_types():
    try:
        from core.document_manager import get_all_documents
        
        documents = get_all_documents()
        law_types = set()
        
        for doc in documents:
            law_type = doc.get("law_type")
            if law_type:
                law_types.add(law_type)
        
        law_types_list = sorted(list(law_types))
        law_types_list.insert(0, "Tổng quan")
        
        return {"law_types": law_types_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
