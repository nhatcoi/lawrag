import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import uuid
from config import DATA_DIR

CONVERSATIONS_FILE = DATA_DIR / "conversations.json"
MAX_CONTEXT_LENGTH = 8000


def load_conversations() -> List[Dict]:
    if CONVERSATIONS_FILE.exists():
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_conversations(conversations: List[Dict]) -> None:
    CONVERSATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def create_conversation(title: str = "New Conversation", law_type_filter: str = "Tổng quan") -> Dict:
    conversations = load_conversations()
    conv_id = str(uuid.uuid4())
    
    new_conv = {
        "id": conv_id,
        "title": title,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "law_type_filter": law_type_filter,
        "messages": []
    }
    
    conversations.append(new_conv)
    save_conversations(conversations)
    return new_conv


def get_conversation(conv_id: str) -> Optional[Dict]:
    conversations = load_conversations()
    return next((c for c in conversations if c["id"] == conv_id), None)


def add_message(conv_id: str, role: str, content: str, sources: List[Dict] = None) -> Dict:
    conversations = load_conversations()
    conv = next((c for c in conversations if c["id"] == conv_id), None)
    
    if not conv:
        return None
    
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "sources": sources or []
    }
    
    conv["messages"].append(message)
    conv["updated_at"] = datetime.now().isoformat()
    
    if len(conv["messages"]) > 0 and conv["messages"][0]["role"] == "user":
        user_content = conv["messages"][0]["content"]
        conv["title"] = user_content[:50] + ("..." if len(user_content) > 50 else "")
    
    save_conversations(conversations)
    return message


def get_conversation_context(conv_id: str, max_length: int = MAX_CONTEXT_LENGTH) -> List[Dict]:
    conv = get_conversation(conv_id)
    if not conv:
        return []
    
    messages = conv["messages"]
    total_length = sum(len(msg["content"]) for msg in messages)
    
    if total_length <= max_length:
        return messages
    
    context = []
    current_length = 0
    for msg in reversed(messages):
        msg_length = len(msg["content"])
        if current_length + msg_length <= max_length:
            context.insert(0, msg)
            current_length += msg_length
        else:
            break
    
    return context


def get_all_conversations() -> List[Dict]:
    return load_conversations()


def update_conversation_law_type_filter(conv_id: str, law_type_filter: str) -> bool:
    conversations = load_conversations()
    conv = next((c for c in conversations if c["id"] == conv_id), None)
    
    if not conv:
        return False
    
    conv["law_type_filter"] = law_type_filter
    conv["updated_at"] = datetime.now().isoformat()
    save_conversations(conversations)
    return True


def delete_conversation(conv_id: str) -> bool:
    conversations = load_conversations()
    initial_count = len(conversations)
    conversations = [c for c in conversations if c["id"] != conv_id]
    
    if len(conversations) < initial_count:
        save_conversations(conversations)
        return True
    return False

