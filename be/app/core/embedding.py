import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import FILES_DIR, INDEX_DIR, EMBEDDING_MODEL


def get_embeddings(provider: str = "local"):
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_documents(docs_dir: Path = FILES_DIR) -> List:
    """Load tất cả file .txt từ docs_dir và các subfolder output_*"""
    all_docs = []
    
    # Load từ root (legacy files)
    if docs_dir.exists():
        root_loader = DirectoryLoader(
            str(docs_dir), 
            glob="*.txt", 
            loader_cls=TextLoader, 
            loader_kwargs={"encoding": "utf-8"}
        )
        all_docs.extend(root_loader.load())
    
    # Load từ các folder output_* (recursive)
    import glob
    output_pattern = str(docs_dir / "*/output_*/*.txt")
    output_files = glob.glob(output_pattern, recursive=True)
    
    from core.document_manager import get_all_documents
    documents_meta = {Path(doc.get("doc_dir", "")).name: doc.get("law_type") for doc in get_all_documents() if "doc_dir" in doc}
    
    for txt_file in output_files:
        try:
            doc = TextLoader(txt_file, encoding="utf-8").load()
            for d in doc:
                source_path = str(txt_file)
                doc_folder = Path(txt_file).parent.parent.name
                
                law_type = documents_meta.get(doc_folder)
                if not law_type:
                    if "doanh-nghiep" in doc_folder.lower() or "doanh-nghiep" in source_path.lower():
                        law_type = "Luật Doanh nghiệp"
                    elif "lao-dong" in doc_folder.lower() or "lao-dong" in source_path.lower():
                        law_type = "Bộ luật Lao động"
                    else:
                        law_type = "Khác"
                
                d.metadata["law_type"] = law_type
                all_docs.append(d)
        except Exception:
            pass
    
    return all_docs


def build_vector_store(docs_dir: Path = FILES_DIR, index_dir: Path = INDEX_DIR, provider: str = "local"):
    texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(load_documents(docs_dir))
    embeddings = get_embeddings(provider)
    vectorstore = FAISS.from_documents(texts, embeddings)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def load_vector_store(index_dir: Path = INDEX_DIR, provider: str = "local"):
    return FAISS.load_local(str(index_dir), get_embeddings(provider), allow_dangerous_deserialization=True)
