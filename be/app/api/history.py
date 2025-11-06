from typing import List
from fastapi import APIRouter, HTTPException
from app.models import HistoryItem
from app.core.storage import load_history, add_to_history

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=List[HistoryItem])
def get_history(limit: int = 50):
    try:
        history = load_history()
        return [HistoryItem(**item) for item in history[-limit:]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=HistoryItem)
def create_history_item(query: str, answer: str, sources_count: int = 0):
    try:
        item = add_to_history(query, answer, sources_count)
        return HistoryItem(**item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
