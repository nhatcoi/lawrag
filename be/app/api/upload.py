from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional
from core import build_vector_store
from core.pdf_processor import process_pdf_to_txt
from core.document_manager import create_document, add_chunk_to_document
from config import INDEX_DIR, FILES_DIR
import shutil
import uuid

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("")
async def upload_endpoint(file: UploadFile = File(...), law_type: Optional[str] = Form(None)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        if not file.filename.endswith(('.txt', '.pdf')):
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .txt hoặc .pdf")
        
        FILES_DIR.mkdir(parents=True, exist_ok=True)
        
        if file.filename.endswith('.pdf'):
            doc = create_document(file.filename, "pdf", file.size or 0, law_type)
            doc_dir = Path(doc["doc_dir"])
            output_dir = Path(doc["output_dir"])
            safe_pdf_name = file.filename.replace(" ", "_")
            pdf_path = doc_dir / safe_pdf_name
            
            doc_dir.mkdir(parents=True, exist_ok=True)
            content = await file.read()
            with open(pdf_path, "wb") as buffer:
                buffer.write(content)
            
            try:
                created_files = process_pdf_to_txt(pdf_path, output_dir)
                if not created_files:
                    raise HTTPException(status_code=400, detail="Không thể tách PDF thành các file")
                
                for txt_file in created_files:
                    txt_path = Path(txt_file)
                    if txt_path.exists():
                        add_chunk_to_document(
                            doc["id"],
                            txt_path.name,
                            txt_path,
                            txt_path.stat().st_size
                        )
            except Exception as e:
                if pdf_path.exists():
                    pdf_path.unlink()
                if doc_dir.exists():
                    shutil.rmtree(doc_dir)
                raise HTTPException(status_code=400, detail=f"Lỗi xử lý PDF: {str(e)}")
        else:
            safe_filename = file.filename.replace(" ", "_")
            safe_stem = Path(safe_filename).stem or "txt_file"
            doc_dir = FILES_DIR / safe_stem
            doc_dir.mkdir(parents=True, exist_ok=True)
            file_path = doc_dir / safe_filename
            
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            doc = create_document(file.filename, "txt", file.size or 0, law_type)
            add_chunk_to_document(
                doc["id"],
                safe_filename,
                file_path,
                file_path.stat().st_size
            )
        
        build_vector_store(FILES_DIR, INDEX_DIR)
        
        return {
            "message": "File uploaded and indexed successfully",
            "filename": file.filename,
            "document_id": doc["id"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi upload: {str(e)}")


@router.post("/multiple")
async def upload_multiple_endpoint(files: List[UploadFile] = File(...)):
    try:
        results = []
        FILES_DIR.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if not file.filename.endswith(('.txt', '.pdf')):
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Chỉ hỗ trợ file .txt hoặc .pdf"
                })
                continue
            
            try:
                if file.filename.endswith('.pdf'):
                    doc = create_document(file.filename, "pdf", file.size or 0, law_type)
                    doc_dir = Path(doc["doc_dir"])
                    output_dir = Path(doc["output_dir"])
                    safe_pdf_name = file.filename.replace(" ", "_")
                    pdf_path = doc_dir / safe_pdf_name
                    
                    doc_dir.mkdir(parents=True, exist_ok=True)
                    content = await file.read()
                    with open(pdf_path, "wb") as buffer:
                        buffer.write(content)
                    
                    created_files = process_pdf_to_txt(pdf_path, output_dir)
                    for txt_file in created_files:
                        txt_path = Path(txt_file)
                        if txt_path.exists():
                            add_chunk_to_document(
                                doc["id"],
                                txt_path.name,
                                txt_path,
                                txt_path.stat().st_size
                            )
                else:
                    safe_filename = file.filename.replace(" ", "_")
                    safe_stem = Path(safe_filename).stem or "txt_file"
                    doc_dir = FILES_DIR / safe_stem
                    doc_dir.mkdir(parents=True, exist_ok=True)
                    file_path = doc_dir / safe_filename
                    
                    content = await file.read()
                    with open(file_path, "wb") as buffer:
                        buffer.write(content)
                    
                    doc = create_document(file.filename, "txt", file.size or 0, law_type)
                    add_chunk_to_document(
                        doc["id"],
                        safe_filename,
                        file_path,
                        file_path.stat().st_size
                    )
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "message": "Upload thành công"
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e)
                })
        
        build_vector_store(FILES_DIR, INDEX_DIR)
        
        return {
            "message": "Upload completed",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi upload: {str(e)}")
