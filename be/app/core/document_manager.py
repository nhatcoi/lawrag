import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import uuid
from app.config import DATA_DIR, FILES_DIR

DOCUMENTS_FILE = DATA_DIR / "documents.json"


def load_documents() -> List[Dict]:
    if DOCUMENTS_FILE.exists():
        try:
            with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_documents(documents: List[Dict]) -> None:
    DOCUMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


def create_document(filename: str, file_type: str, size: int, law_type: str = None) -> Dict:
    documents = load_documents()
    doc_id = str(uuid.uuid4())
    safe_name = Path(filename).stem.replace(" ", "_")
    
    counter = 1
    base_name = safe_name
    while (FILES_DIR / safe_name).exists():
        safe_name = f"{base_name}({counter})"
        counter += 1
    
    doc_dir = FILES_DIR / safe_name
    output_dir = doc_dir / f"output_{safe_name}"
    
    if not law_type:
        if "doanh-nghiep" in filename.lower() or "doanh-nghiep" in safe_name.lower():
            law_type = "Luật Doanh nghiệp"
        elif "lao-dong" in filename.lower() or "lao-dong" in safe_name.lower():
            law_type = "Bộ luật Lao động"
        else:
            law_type = "Khác"
    
    new_doc = {
        "id": doc_id,
        "filename": filename,
        "original_name": filename,
        "doc_dir": str(doc_dir),
        "output_dir": str(output_dir),
        "type": file_type,
        "size": size,
        "law_type": law_type,
        "uploaded_at": datetime.now().isoformat(),
        "status": "pending",
        "chunks": []
    }
    
    documents.append(new_doc)
    save_documents(documents)
    return new_doc


def add_chunk_to_document(doc_id: str, chunk_filename: str, chunk_path: Path, chunk_size: int) -> bool:
    documents = load_documents()
    doc = next((d for d in documents if d["id"] == doc_id), None)
    if not doc:
        return False
    
    chunk_info = {
        "filename": chunk_filename,
        "path": str(chunk_path.absolute()),
        "size": chunk_size,
        "created_at": datetime.now().isoformat()
    }
    
    if "chunks" not in doc:
        doc["chunks"] = []
    
    doc["chunks"].append(chunk_info)
    doc["status"] = "indexed"
    save_documents(documents)
    return True


def add_document(filename: str, file_path: Path, file_type: str, size: int) -> Dict:
    documents = load_documents()
    doc_id = f"{filename}_{datetime.now().timestamp()}"
    
    new_doc = {
        "id": doc_id,
        "filename": filename,
        "path": str(file_path.absolute()),
        "type": file_type,
        "size": size,
        "uploaded_at": datetime.now().isoformat(),
        "status": "indexed"
    }
    
    documents.append(new_doc)
    save_documents(documents)
    return new_doc


def delete_document(doc_id: str) -> bool:
    documents = load_documents()
    doc = next((d for d in documents if d["id"] == doc_id), None)
    if not doc:
        return False
    
    if "doc_dir" in doc:
        doc_dir = Path(doc["doc_dir"])
        if doc_dir.exists():
            import shutil
            try:
                shutil.rmtree(doc_dir)
            except Exception:
                pass
    elif "path" in doc:
        file_path = Path(doc["path"])
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
    
    documents = [d for d in documents if d["id"] != doc_id]
    save_documents(documents)
    return True


def get_document(doc_id: str) -> Optional[Dict]:
    documents = load_documents()
    return next((d for d in documents if d["id"] == doc_id), None)


def get_all_documents() -> List[Dict]:
    return load_documents()


def update_document_status(doc_id: str, status: str) -> None:
    documents = load_documents()
    for doc in documents:
        if doc["id"] == doc_id:
            doc["status"] = status
            break
    save_documents(documents)

