"""
RAG System - Hệ thống truy xuất và tạo sinh văn bản
Chương trình chính để xử lý PDF, tạo embeddings và trả lời câu hỏi
"""

import os
import argparse
from dotenv import load_dotenv

from src import splitter
from src import embedder
from src import retriever
from src import generator


def cmd_split(pdf_path: str, output_dir: str) -> None:
    """Tách PDF thành các file điều luật riêng biệt"""
    text = splitter.extract_text_from_pdf(pdf_path)
    lines = text.splitlines()
    print("Tiến hành tách theo điều luật...")
    articles = splitter.split_articles(lines)
    print(f"Phát hiện {len(articles)} điều luật")
    splitter.write_articles(articles, output_dir)
    print(f"Đã ghi {len(articles)} file vào: {output_dir}")


def cmd_embed(split_dir: str, index_dir: str, provider: str, model: str, batch_size: int, local_model: str) -> None:
    """Tạo embeddings và lưu vào FAISS index"""
    # Khởi tạo client OpenAI nếu cần
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY chưa được thiết lập trong biến môi trường")
        client = embedder.OpenAI(api_key=api_key)
    else:
        client = None

    # Đọc tài liệu từ thư mục đã tách
    documents = embedder.read_documents(split_dir)
    if not documents:
        raise FileNotFoundError("Không tìm thấy tài liệu .txt nào để embed. Hãy chạy split trước.")

    doc_ids = [doc_id for doc_id, _ in documents]
    texts = [content for _, content in documents]
    print(f"Tạo embeddings cho {len(texts)} tài liệu bằng provider: {provider}")

    # Cấu hình tham số runtime
    embedder.EMBED_MODEL = model
    embedder.BATCH_SIZE = batch_size

    # Tạo embeddings theo provider
    if provider == "openai":
        vectors = embedder.get_embeddings_openai(client, texts)  # type: ignore[arg-type]
    elif provider == "local":
        vectors = embedder.get_embeddings_local(local_model, texts)
    else:
        raise ValueError("provider phải là 'openai' hoặc 'local'")

    # Xây dựng FAISS index với cosine similarity
    print("Xây dựng FAISS index (cosine similarity)...")
    index = embedder.build_faiss_index(vectors)
    metadata = [{"id": d, "path": os.path.join(split_dir, d)} for d in doc_ids]
    embedder.save_index(index, metadata, index_dir)
    print(f"Đã lưu index và metadata vào: {index_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Xây dựng CLI parser với các lệnh con"""
    parser = argparse.ArgumentParser(description="RAG System CLI: Tách PDF và xây dựng FAISS index")
    sub = parser.add_subparsers(dest="command", required=True)

    # Lệnh split: Tách PDF thành các file điều luật
    p_split = sub.add_parser("split", help="Tách PDF thành các file Điều luật .txt")
    p_split.add_argument("--pdf-path", default="/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/luat_lao_dong.pdf")
    p_split.add_argument("--output-dir", default="/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/output_dieu_luat")

    # Lệnh embed: Tạo embeddings và lưu FAISS index
    p_embed = sub.add_parser("embed", help="Tạo embeddings và lưu FAISS index")
    p_embed.add_argument("--split-dir", default="/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/output_dieu_luat")
    p_embed.add_argument("--index-dir", default="/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/faiss_index")
    p_embed.add_argument("--provider", choices=["openai", "local"], default="openai")
    p_embed.add_argument("--model", default="text-embedding-3-small")
    p_embed.add_argument("--batch-size", type=int, default=64)
    p_embed.add_argument("--local-model", default="sentence-transformers/all-MiniLM-L6-v2")

    # Lệnh all: Chạy cả split và embed
    p_all = sub.add_parser("all", help="Chạy split rồi embed trong một lệnh")
    p_all.add_argument("--pdf-path", default="/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/luat_lao_dong.pdf")
    p_all.add_argument("--split-dir", default="/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/output_dieu_luat")
    p_all.add_argument("--index-dir", default="/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/faiss_index")
    p_all.add_argument("--provider", choices=["openai", "local"], default="openai")
    p_all.add_argument("--model", default="text-embedding-3-small")
    p_all.add_argument("--batch-size", type=int, default=64)
    p_all.add_argument("--local-model", default="sentence-transformers/all-MiniLM-L6-v2")

    # Lệnh ask: Đặt câu hỏi sử dụng RAG
    p_ask = sub.add_parser("ask", help="Đặt câu hỏi (RAG)")
    p_ask.add_argument("--query", required=True)
    p_ask.add_argument("--index-dir", default="/Users/coinhat/Documents/PROJECT/AI/RAG/bai6/faiss_index")
    p_ask.add_argument("--provider", choices=["openai", "local"], default="local")
    p_ask.add_argument("--local-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--groq-model", default="llama-3.3-70b-versatile")

    return parser


def main() -> None:
    """Hàm chính xử lý các lệnh CLI"""
    load_dotenv()  # Tải biến môi trường từ .env
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "split":
        # Kiểm tra file PDF tồn tại
        if not os.path.isfile(args.pdf_path):
            raise FileNotFoundError(f"Không tìm thấy file PDF: {args.pdf_path}")
        print(f"Đọc PDF: {args.pdf_path}")
        cmd_split(args.pdf_path, args.output_dir)
        
    elif args.command == "embed":
        cmd_embed(args.split_dir, args.index_dir, args.provider, args.model, args.batch_size, args.local_model)
        
    elif args.command == "all":
        # Chạy cả split và embed
        if not os.path.isfile(args.pdf_path):
            raise FileNotFoundError(f"Không tìm thấy file PDF: {args.pdf_path}")
        print(f"Đọc PDF: {args.pdf_path}")
        cmd_split(args.pdf_path, args.split_dir)
        cmd_embed(args.split_dir, args.index_dir, args.provider, args.model, args.batch_size, args.local_model)
        
    elif args.command == "ask":
        # RAG: Truy xuất thông tin và tạo câu trả lời
        import re
        contexts = []
        
        # Heuristic: Nếu query chứa "Điều <số>", chèn trực tiếp nội dung điều đó
        m = re.search(r"(?i)(điều)\s+(\d+)", args.query)
        if m:
            article_num = m.group(2)
            direct_text = retriever.try_get_article_by_number(args.index_dir, article_num)
            if direct_text:
                contexts.append(direct_text)

        # Truy xuất tài liệu liên quan
        results = retriever.retrieve(
            query=args.query,
            index_dir=args.index_dir,
            top_k=args.top_k,
            provider=args.provider,
            local_model=args.local_model,
        )
        contexts.extend([r["text"] for r in results])
        
        # Tạo câu trả lời bằng LLM
        answer = generator.generate_answer(
            query=args.query,
            contexts=contexts,
            model=args.groq_model,
        )
        print(answer)
    else:
        parser.error("Lệnh không hợp lệ")


if __name__ == "__main__":
    main()

