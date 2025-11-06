source .venv/bin/activate
python main.py -h
# Build data and index (local):
python main.py all --pdf-path /Users/coinhat/Documents/AI/lawrag/luat_lao_dong.pdf --split-dir /Users/coinhat/Documents/AI/lawrag/output_dieu_luat --index-dir /Users/coinhat/Documents/AI/lawrag/faiss_index --provider local --local-model sentence-transformers/all-MiniLM-L6-v2
# Ask a question (needs GROQ_API_KEY):
python main.py ask --query "Xin tóm tắt nội dung chính của Điều 10" --index-dir /Users/coinhat/Documents/AI/lawrag/faiss_index --provider local --local-model sentence-transformers/all-MiniLM-L6-v2 --top-k 5 --groq-model llama-3.3-70b-versatile

