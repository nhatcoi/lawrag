import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from app.config import DATA_DIR

HISTORY_FILE = DATA_DIR / "history.json"


def load_history() -> List[Dict[str, Any]]:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history: List[Dict[str, Any]]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_to_history(query: str, answer: str, sources_count: int) -> Dict[str, Any]:
    history = load_history()
    new_item = {
        "id": str(len(history) + 1),
        "query": query,
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
        "sources_count": sources_count,
    }
    history.append(new_item)
    save_history(history)
    return new_item

