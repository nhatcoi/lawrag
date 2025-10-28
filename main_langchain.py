"""
CLI chính cho RAG System sử dụng LangChain
Tích hợp tất cả các module LangChain
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from src.langchain_splitter import create_legal_splitter
from src.langchain_vectorstore import create_vector_store_manager
from src.langchain_retriever import create_legal_retriever
from src.langchain_rag import create_legal_rag_chain


def cmd_split_pdf(pdf_path: str, output_dir: str, chunk_size: int, chunk_overlap: int) -> None:
    """Tách PDF thành documents sử dụng LangChain"""
    print(f"Đang tách PDF: {pdf_path}")
    
    # Tạo splitter
    splitter = create_legal_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Xử lý PDF và lưu điều luật ra file
    documents = splitter.process_pdf(pdf_path, save_articles=True, output_dir=output_dir)
    
    print(f"Đã tách thành {len(documents)} documents")
    print(f"Lưu điều luật vào thư mục: {output_dir}")


def cmd_build_vectorstore(
    pdf_path: str, 
    vector_store_type: str, 
    embeddings_type: str, 
    embeddings_model: str,
    persist_directory: str,
    chunk_size: int,
    chunk_overlap: int
) -> None:
    """Xây dựng vector store từ PDF"""
    print(f"Đang xây dựng vector store từ: {pdf_path}")
    
    # Bước 1: Tách PDF
    print("Bước 1: Tách PDF...")
    splitter = create_legal_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = splitter.process_pdf(pdf_path, save_articles=True, output_dir="./output_dieu_luat")
    
    # Bước 2: Tạo vector store
    print("Bước 2: Tạo vector store...")
    vector_store_manager = create_vector_store_manager(
        vector_store_type=vector_store_type,
        embeddings_type=embeddings_type,
        embeddings_model=embeddings_model,
        persist_directory=persist_directory
    )
    
    # Bước 3: Build vector store
    print("Bước 3: Embedding và lưu trữ...")
    vector_store_manager.build_vector_store(documents)
    
    print("✅ Hoàn thành xây dựng vector store!")


def cmd_ask_question(
    question: str,
    vector_store_type: str,
    embeddings_type: str,
    embeddings_model: str,
    persist_directory: str,
    llm_provider: str,
    llm_model: str,
    top_k: int,
    temperature: float
) -> None:
    """Đặt câu hỏi sử dụng RAG"""
    print(f"Câu hỏi: {question}")
    
    # Bước 1: Load vector store
    print("Đang tải vector store...")
    vector_store_manager = create_vector_store_manager(
        vector_store_type=vector_store_type,
        embeddings_type=embeddings_type,
        embeddings_model=embeddings_model,
        persist_directory=persist_directory
    )
    vector_store_manager.load_vector_store()
    
    # Bước 2: Tạo retriever
    print("Tạo retriever...")
    retriever = create_legal_retriever(
        vector_store_manager=vector_store_manager,
        top_k=top_k,
        enable_article_heuristic=True
    )
    
    # Bước 3: Tạo RAG chain
    print("Tạo RAG chain...")
    rag_chain = create_legal_rag_chain(
        retriever=retriever,
        llm_provider=llm_provider,
        llm_model=llm_model,
        temperature=temperature
    )
    
    # Bước 4: Thực thi
    print("Đang tạo câu trả lời...")
    result = rag_chain.invoke_with_sources(question)
    
    # Hiển thị kết quả
    print("\n" + "="*50)
    print("CÂU TRẢ LỜI:")
    print("="*50)
    print(result["answer"])
    
    print(f"\nSố nguồn tài liệu: {result['total_sources']}")
    
    if result["sources"]:
        print("\nNGUỒN TÀI LIỆU:")
        print("-"*50)
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source['article_id']}")
            print(f"   Score: {source['similarity_score']:.3f}")
            print(f"   Method: {source['retrieval_method']}")
            print(f"   Preview: {source['content_preview']}")
            print()


def build_parser() -> argparse.ArgumentParser:
    """Xây dựng CLI parser"""
    parser = argparse.ArgumentParser(
        description="RAG System với LangChain - Hệ thống truy xuất pháp luật"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Lệnh split-pdf
    split_parser = subparsers.add_parser("split-pdf", help="Tách PDF thành documents")
    split_parser.add_argument("--pdf-path", required=True, help="Đường dẫn file PDF")
    split_parser.add_argument("--output-dir", default="./output_documents", help="Thư mục lưu documents")
    split_parser.add_argument("--chunk-size", type=int, default=1000, help="Kích thước chunk")
    split_parser.add_argument("--chunk-overlap", type=int, default=200, help="Số ký tự chồng lấp")
    
    # Lệnh build-vectorstore
    build_parser = subparsers.add_parser("build-vectorstore", help="Xây dựng vector store")
    build_parser.add_argument("--pdf-path", required=True, help="Đường dẫn file PDF")
    build_parser.add_argument("--vector-store-type", choices=["faiss", "chroma"], default="faiss", help="Loại vector store")
    build_parser.add_argument("--embeddings-type", choices=["openai", "huggingface"], default="huggingface", help="Loại embeddings")
    build_parser.add_argument("--embeddings-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Model embeddings")
    build_parser.add_argument("--persist-directory", default="./vector_store", help="Thư mục lưu vector store")
    build_parser.add_argument("--chunk-size", type=int, default=1000, help="Kích thước chunk")
    build_parser.add_argument("--chunk-overlap", type=int, default=200, help="Số ký tự chồng lấp")
    
    # Lệnh ask
    ask_parser = subparsers.add_parser("ask", help="Đặt câu hỏi")
    ask_parser.add_argument("--question", required=True, help="Câu hỏi")
    ask_parser.add_argument("--vector-store-type", choices=["faiss", "chroma"], default="faiss", help="Loại vector store")
    ask_parser.add_argument("--embeddings-type", choices=["openai", "huggingface"], default="huggingface", help="Loại embeddings")
    ask_parser.add_argument("--embeddings-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Model embeddings")
    ask_parser.add_argument("--persist-directory", default="./vector_store", help="Thư mục vector store")
    ask_parser.add_argument("--llm-provider", choices=["groq", "openai"], default="groq", help="LLM provider")
    ask_parser.add_argument("--llm-model", default="llama-3.3-70b-versatile", help="LLM model")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Số lượng documents trả về")
    ask_parser.add_argument("--temperature", type=float, default=0.2, help="Temperature cho LLM")
    
    # Lệnh all (build + test)
    all_parser = subparsers.add_parser("all", help="Build vector store và test")
    all_parser.add_argument("--pdf-path", required=True, help="Đường dẫn file PDF")
    all_parser.add_argument("--question", default="Nội dung Điều 1 Bộ luật Lao động quy định gì?", help="Câu hỏi test")
    all_parser.add_argument("--vector-store-type", choices=["faiss", "chroma"], default="faiss", help="Loại vector store")
    all_parser.add_argument("--embeddings-type", choices=["openai", "huggingface"], default="huggingface", help="Loại embeddings")
    all_parser.add_argument("--embeddings-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Model embeddings")
    all_parser.add_argument("--persist-directory", default="./vector_store", help="Thư mục lưu vector store")
    all_parser.add_argument("--llm-provider", choices=["groq", "openai"], default="groq", help="LLM provider")
    all_parser.add_argument("--llm-model", default="llama-3.3-70b-versatile", help="LLM model")
    all_parser.add_argument("--top-k", type=int, default=5, help="Số lượng documents trả về")
    all_parser.add_argument("--temperature", type=float, default=0.2, help="Temperature cho LLM")
    
    return parser


def main() -> None:
    """Hàm chính"""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        if args.command == "split-pdf":
            cmd_split_pdf(
                pdf_path=args.pdf_path,
                output_dir=args.output_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
        
        elif args.command == "build-vectorstore":
            cmd_build_vectorstore(
                pdf_path=args.pdf_path,
                vector_store_type=args.vector_store_type,
                embeddings_type=args.embeddings_type,
                embeddings_model=args.embeddings_model,
                persist_directory=args.persist_directory,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
        
        elif args.command == "ask":
            cmd_ask_question(
                question=args.question,
                vector_store_type=args.vector_store_type,
                embeddings_type=args.embeddings_type,
                embeddings_model=args.embeddings_model,
                persist_directory=args.persist_directory,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                top_k=args.top_k,
                temperature=args.temperature
            )
        
        elif args.command == "all":
            print("🚀 Chạy pipeline hoàn chỉnh...")
            
            # Build vector store
            print("\n📚 Bước 1: Xây dựng vector store...")
            cmd_build_vectorstore(
                pdf_path=args.pdf_path,
                vector_store_type=args.vector_store_type,
                embeddings_type=args.embeddings_type,
                embeddings_model=args.embeddings_model,
                persist_directory=args.persist_directory,
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Test với câu hỏi
            print(f"\n❓ Bước 2: Test với câu hỏi...")
            cmd_ask_question(
                question=args.question,
                vector_store_type=args.vector_store_type,
                embeddings_type=args.embeddings_type,
                embeddings_model=args.embeddings_model,
                persist_directory=args.persist_directory,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                top_k=args.top_k,
                temperature=args.temperature
            )
            
            print("\n✅ Hoàn thành pipeline!")
        
        else:
            parser.error(f"Lệnh không hợp lệ: {args.command}")
    
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
