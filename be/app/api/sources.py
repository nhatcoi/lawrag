from typing import List
from fastapi import APIRouter, HTTPException
from pathlib import Path
from models import DocumentInfo, ChunkInfo, SourceInfo
from core.document_manager import get_all_documents, delete_document, get_document
from core import build_vector_store
from config import FILES_DIR, INDEX_DIR

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("", response_model=List[DocumentInfo])
def get_sources():
    try:
        documents = get_all_documents()
        result = []
        
        for doc in documents:
            if "doc_dir" in doc and "chunks" in doc:
                doc_dir = Path(doc["doc_dir"])
                if doc_dir.exists():
                    chunks = []
                    for chunk in doc.get("chunks", []):
                        chunk_path = Path(chunk["path"])
                        if chunk_path.exists():
                            chunks.append(ChunkInfo(
                                filename=chunk["filename"],
                                path=str(chunk_path.absolute()),
                                size=chunk.get("size", 0),
                                created_at=chunk.get("created_at", "")
                            ))
                    
                    result.append(DocumentInfo(
                        id=doc["id"],
                        filename=doc["filename"],
                        original_name=doc.get("original_name", doc["filename"]),
                        type=doc.get("type", "pdf"),
                        size=doc.get("size", 0),
                        uploaded_at=doc.get("uploaded_at", ""),
                        status=doc.get("status", "indexed"),
                        chunks_count=len(chunks),
                        chunks=chunks,
                        law_type=doc.get("law_type", "Khác")
                    ))
            elif "path" in doc:
                file_path = Path(doc["path"])
                if file_path.exists():
                    result.append(DocumentInfo(
                        id=doc["id"],
                        filename=doc["filename"],
                        original_name=doc["filename"],
                        type=doc.get("type", "txt"),
                        size=file_path.stat().st_size,
                        uploaded_at=doc.get("uploaded_at", ""),
                        status=doc.get("status", "indexed"),
                        chunks_count=1,
                        chunks=[ChunkInfo(
                            filename=doc["filename"],
                            path=str(file_path.absolute()),
                            size=file_path.stat().st_size,
                            created_at=doc.get("uploaded_at", "")
                        )],
                        law_type=doc.get("law_type", "Khác")
                    ))
        
        return sorted(result, key=lambda x: x.uploaded_at, reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_id}", response_model=DocumentInfo)
def get_source(source_id: str):
    try:
        doc = get_document(source_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if "doc_dir" in doc and "chunks" in doc:
            doc_dir = Path(doc["doc_dir"])
            if not doc_dir.exists():
                raise HTTPException(status_code=404, detail="Document folder not found")
            
            chunks = []
            for chunk in doc.get("chunks", []):
                chunk_path = Path(chunk["path"])
                if chunk_path.exists():
                    chunks.append(ChunkInfo(
                        filename=chunk["filename"],
                        path=str(chunk_path.absolute()),
                        size=chunk.get("size", 0),
                        created_at=chunk.get("created_at", "")
                    ))
            
            return DocumentInfo(
                id=doc["id"],
                filename=doc["filename"],
                original_name=doc.get("original_name", doc["filename"]),
                type=doc.get("type", "pdf"),
                size=doc.get("size", 0),
                uploaded_at=doc.get("uploaded_at", ""),
                status=doc.get("status", "indexed"),
                chunks_count=len(chunks),
                chunks=chunks,
                law_type=doc.get("law_type", "Khác")
            )
        else:
            # Legacy
            file_path = Path(doc["path"])
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="File not found")
            
            return DocumentInfo(
                id=doc["id"],
                filename=doc["filename"],
                original_name=doc["filename"],
                type=doc.get("type", "txt"),
                size=file_path.stat().st_size,
                uploaded_at=doc.get("uploaded_at", ""),
                status=doc.get("status", "indexed"),
                chunks_count=1,
                chunks=[ChunkInfo(
                    filename=doc["filename"],
                    path=str(file_path.absolute()),
                    size=file_path.stat().st_size,
                    created_at=doc.get("uploaded_at", "")
                )],
                law_type=doc.get("law_type", "Khác")
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{source_id}/chunks/{chunk_filename}")
def get_chunk_content(source_id: str, chunk_filename: str):
    try:
        doc = get_document(source_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        chunk_path = None
        if "chunks" in doc:
            for chunk in doc.get("chunks", []):
                if chunk["filename"] == chunk_filename:
                    chunk_path = Path(chunk["path"])
                    break
        
        if not chunk_path or not chunk_path.exists():
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        with open(chunk_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return {
            "id": source_id,
            "filename": chunk_filename,
            "path": str(chunk_path.absolute()),
            "size": chunk_path.stat().st_size,
            "content": content,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{source_id}")
def delete_source(source_id: str):
    try:
        success = delete_document(source_id)
        if not success:
            raise HTTPException(status_code=404, detail="Source not found")
        
        build_vector_store(FILES_DIR, INDEX_DIR)
        
        return {
            "message": "Document deleted successfully",
            "id": source_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
