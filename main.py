"""
LangChain RAG (đơn giản, tất cả trong một file)
- build: PDF -> tách Điều (regex) -> embed (HuggingFace) -> lưu FAISS
- ask: load FAISS -> retrieve -> Groq LLM (llama) sinh câu trả lời
"""

import os
import re
import argparse
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# Heading dạng phổ biến: "Điều <số>." (chấp nhận "." hoặc ")")
ARTICLE_REGEX = re.compile(r"(?im)^[\t ]*[Đđ]iều[\t ]+(\d+)[\t ]*[\.)]", re.UNICODE)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Đọc toàn bộ text từ PDF (ngắn gọn, dùng pdfplumber)."""
    import pdfplumber  # type: ignore
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join((page.extract_text(x_tolerance=1, y_tolerance=1) or "") for page in pdf.pages)


def split_articles(text: str) -> List[Document]:
    """Tách văn bản thành các Điều dựa theo regex tiêu đề."""
    matches = list(ARTICLE_REGEX.finditer(text))
    if not matches:
        return [Document(page_content=text.strip())]
    
    docs: List[Document] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            docs.append(Document(page_content=content, metadata={"article": m.group(1)}))
    return docs


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE)
    return sanitized.strip("_")


def write_articles_to_dir(articles: List[Document], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for doc in articles:
        art = str(doc.metadata.get("article", "unknown")).lower()
        fname = _sanitize_filename(f"điều_{art}") + ".txt"
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc.page_content.strip() + "\n")


def build_index(pdf_path: str, index_dir: str, model_name: str, output_dir: str | None = None) -> None:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")


    print(f"1. Đọc PDF: {pdf_path}")
    full_text = extract_text_from_pdf(pdf_path)

    print(f"2. Tách điều luật")
    articles = split_articles(full_text)
    print(f"Tách được {len(articles)} điều luật")

    if output_dir:
        print(f"3. Ghi các điều luật ra thư mục: {output_dir}")
        write_articles_to_dir(articles, output_dir)

    print("4. Tạo embeddings và FAISS store...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vs = FAISS.from_documents(articles, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)
    print(f"Đã lưu FAISS index tại: {index_dir}")


def ask_question(query: str, index_dir: str, top_k: int, groq_model: str, model_name: str) -> str:
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục FAISS index: {index_dir}")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    llm = ChatGroq(model=groq_model)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
    )

    result = chain.invoke({"query": query})
    return (result.get("result") or result.get("output_text") or "").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG CLI (regex chunk + FAISS + HuggingFace + Groq)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Tạo FAISS index từ PDF bằng regex Điều")
    p_build.add_argument("--pdf-path", default="luat_lao_dong.pdf")
    p_build.add_argument("--index-dir", default="vector_store/faiss_index")
    p_build.add_argument("--local-model", default="sentence-transformers/all-MiniLM-L12-v2")
    p_build.add_argument("--output-dir", default=None, help="Nếu đặt, sẽ lưu mỗi Điều thành 1 file .txt")

    p_ask = sub.add_parser("ask", help="Hỏi đáp dựa trên FAISS + Groq")
    p_ask.add_argument("--query", required=True)
    p_ask.add_argument("--index-dir", default="vector_store/faiss_index")
    p_ask.add_argument("--top-k", type=int, default=20)
    p_ask.add_argument("--groq-model", default="llama-3.3-70b-versatile")
    p_ask.add_argument("--local-model", default="sentence-transformers/all-MiniLM-L12-v2")

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        build_index(
            pdf_path=args.pdf_path,
            index_dir=args.index_dir,
            model_name=args.local_model,
            output_dir=args.output_dir,
        )
    elif args.command == "ask":
        answer = ask_question(
            query=args.query,
            index_dir=args.index_dir,
            top_k=args.top_k,
            groq_model=args.groq_model,
            model_name=args.local_model,
        )
        print(answer)
    else:
        parser.error("Lệnh không hợp lệ")


if __name__ == "__main__":
    main()


