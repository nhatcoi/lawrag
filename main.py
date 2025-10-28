"""
Main CLI cho hệ thống RAG pháp luật
"""

import argparse
import os
from dotenv import load_dotenv
from rag import LegalRAG


def main():
    """Hàm main"""
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Legal RAG System")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Lệnh build
    build_parser = subparsers.add_parser("build", help="Xây dựng knowledge base")
    build_parser.add_argument("--pdf-path", required=True, help="Đường dẫn file PDF")
    build_parser.add_argument("--output-dir", default="./output_dieu_luat", help="Thư mục lưu điều luật")
    build_parser.add_argument("--index-dir", default="./vector_store", help="Thư mục lưu vector store")
    
    # Lệnh ask
    ask_parser = subparsers.add_parser("ask", help="Đặt câu hỏi")
    ask_parser.add_argument("--question", required=True, help="Câu hỏi")
    ask_parser.add_argument("--index-dir", default="./vector_store", help="Thư mục vector store")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Số lượng kết quả")
    
    # Lệnh all
    all_parser = subparsers.add_parser("all", help="Build và test")
    all_parser.add_argument("--pdf-path", required=True, help="Đường dẫn file PDF")
    all_parser.add_argument("--question", default="Nội dung Điều 1 quy định gì?", help="Câu hỏi test")
    
    # Lệnh server
    server_parser = subparsers.add_parser("server", help="Chạy API server")
    
    args = parser.parse_args()
    
    if args.command == "build":
        rag = LegalRAG()
        rag.build_knowledge_base(args.pdf_path, args.output_dir, args.index_dir)
    
    elif args.command == "ask":
        rag = LegalRAG()
        rag.load_knowledge_base(args.index_dir)
        result = rag.ask_question(args.question, args.top_k)
        
        print(f"\n❓ Câu hỏi: {result['question']}")
        print(f"💬 Trả lời: {result['answer']}")
        print(f"\n📚 Nguồn tài liệu ({result['total_sources']}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['article_id']} (Score: {source['score']:.3f})")
    
    elif args.command == "all":
        rag = LegalRAG()
        rag.build_knowledge_base(args.pdf_path, "./output_dieu_luat", "./vector_store")
        
        print(f"\n🧪 Test với câu hỏi: {args.question}")
        result = rag.ask_question(args.question)
        
        print(f"\n❓ Câu hỏi: {result['question']}")
        print(f"💬 Trả lời: {result['answer']}")
        print(f"\n📚 Nguồn tài liệu ({result['total_sources']}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['article_id']} (Score: {source['score']:.3f})")
    
    elif args.command == "server":
        import uvicorn
        from api import app
        print("🚀 Khởi động API server...")
        print("📱 Web UI: http://localhost:8000/app")
        print("📖 API Docs: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
