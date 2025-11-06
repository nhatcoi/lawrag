"""
RAG System - Hệ thống truy xuất và tạo sinh văn bản
"""

import os
import re
import argparse
from dotenv import load_dotenv
from pathlib import Path

from src import splitter, embedder, retriever, generator

# Project root directory
PROJECT_ROOT = Path(__file__).parent
DEFAULT_PDF = PROJECT_ROOT / "luat_lao_dong.pdf"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "output_dieu_luat"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "faiss_index"


def cmd_split(pdf_path: str, output_dir: str) -> None:
    """Tách PDF thành các file điều luật riêng biệt"""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")
    
    print(f"Đọc PDF: {pdf_path}")
    text = splitter.extract_text_from_pdf(pdf_path)
    articles = splitter.split_articles(text.splitlines())
    print(f"Phát hiện {len(articles)} điều luật")
    
    splitter.write_articles(articles, output_dir)
    print(f"Đã ghi {len(articles)} file vào: {output_dir}")


def cmd_embed(split_dir: str, index_dir: str, provider: str, model: str, batch_size: int, local_model: str) -> None:
    """Tạo embeddings và lưu vào FAISS index"""
    documents = embedder.read_documents(split_dir)
    if not documents:
        raise FileNotFoundError("Không tìm thấy tài liệu .txt nào để embed. Hãy chạy split trước.")

    doc_ids, texts = zip(*documents)
    print(f"Tạo embeddings cho {len(texts)} tài liệu bằng provider: {provider}")

    embedder.EMBED_MODEL = model
    embedder.BATCH_SIZE = batch_size

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY chưa được thiết lập")
        client = embedder.OpenAI(api_key=api_key)
        vectors = embedder.get_embeddings_openai(client, list(texts))
    elif provider == "local":
        vectors = embedder.get_embeddings_local(local_model, list(texts))
    else:
        raise ValueError("provider phải là 'openai' hoặc 'local'")

    print("Xây dựng FAISS index...")
    index = embedder.build_faiss_index(vectors)
    metadata = [{"id": d, "path": os.path.join(split_dir, d)} for d in doc_ids]
    embedder.save_index(index, metadata, index_dir)
    print(f"Đã lưu index và metadata vào: {index_dir}")


def _add_embed_args(parser: argparse.ArgumentParser) -> None:
    """Thêm arguments chung cho embed và all"""
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--provider", choices=["openai", "local"], default="local")
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-model", default="sentence-transformers/all-MiniLM-L6-v2")


def build_parser() -> argparse.ArgumentParser:
    """Xây dựng CLI parser với các lệnh con"""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="Tách PDF thành các file Điều luật .txt")
    p_split.add_argument("--pdf-path", default=str(DEFAULT_PDF))
    p_split.add_argument("--output-dir", default=str(DEFAULT_SPLIT_DIR))

    p_embed = sub.add_parser("embed", help="Tạo embeddings và lưu FAISS index")
    _add_embed_args(p_embed)

    p_all = sub.add_parser("all", help="Chạy split rồi embed trong một lệnh")
    p_all.add_argument("--pdf-path", default=str(DEFAULT_PDF))
    _add_embed_args(p_all)

    p_ask = sub.add_parser("ask", help="Đặt câu hỏi (RAG)")
    p_ask.add_argument("--query", required=True)
    p_ask.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p_ask.add_argument("--provider", choices=["openai", "local"], default="local")
    p_ask.add_argument("--local-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--groq-model", default="llama-3.3-70b-versatile")

    return parser


def cmd_ask(query: str, index_dir: str, provider: str, local_model: str, top_k: int, groq_model: str) -> None:
    """RAG: Truy xuất thông tin và tạo câu trả lời"""
    contexts = []
    
    # Heuristic: Nếu query chứa "Điều <số>", chèn trực tiếp nội dung điều đó
    match = re.search(r"(?i)(điều)\s+(\d+)", query)
    if match:
        direct_text = retriever.try_get_article_by_number(index_dir, match.group(2))
        if direct_text:
            contexts.append(direct_text)

    # Truy xuất tài liệu liên quan
    results = retriever.retrieve(query, index_dir, top_k, provider, local_model)
    contexts.extend([r["text"] for r in results])
    
    # Tạo câu trả lời bằng LLM
    answer = generator.generate_answer(query, contexts, groq_model)
    print(answer)


def main() -> None:
    """Hàm chính xử lý các lệnh CLI"""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "split":
        cmd_split(args.pdf_path, args.output_dir)
    elif args.command == "embed":
        cmd_embed(args.split_dir, args.index_dir, args.provider, args.model, args.batch_size, args.local_model)
    elif args.command == "all":
        cmd_split(args.pdf_path, args.split_dir)
        cmd_embed(args.split_dir, args.index_dir, args.provider, args.model, args.batch_size, args.local_model)
    elif args.command == "ask":
        cmd_ask(args.query, args.index_dir, args.provider, args.local_model, args.top_k, args.groq_model)
    else:
        parser.error("Lệnh không hợp lệ")


if __name__ == "__main__":
    main()

