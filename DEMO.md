Ví dụ prompt terminal : python3 main.py ask --query "Nội dung Điều 1 Bộ luật Lao động quy định gì?" --provider local --local-model sentence-transformers/all-MiniLM-L12-v2 --top-k 20

Kết quả : Điều 1 Bộ luật Lao động quy định về phạm vi điều chỉnh, bao gồm tiêu chuẩn lao động, quyền, nghĩa vụ, trách nhiệm của người lao động, người sử dụng lao động và quản lý nhà nước về lao động. (Điều 1)


------
Giải thích prompt terminal : 

query: Câu hỏi của bạn. Model sẽ dùng câu này để embed truy vấn và tạo câu trả lời.
--provider local: Chọn cách embed truy vấn bằng mô hình cục bộ (không dùng OpenAI), giúp tránh quota/API cost.
--local-model sentence-transformers/all-MiniLM-L6-v2: Tên mô hình embed cục bộ dùng cho truy vấn. all-MiniLM-L6-v2 mạnh cho tiếng Anh; với tiếng Việt nên cân nhắc mô hình đa ngôn ngữ như sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
--top-k 20: Lấy 20 đoạn gần nhất từ FAISS để làm ngữ cảnh cho bước sinh (tăng cơ hội bao phủ Điều 1).
Mặc định nếu không chỉ định:
--index-dir: dùng đường dẫn mặc định tới thư mục FAISS (faiss_index).
--groq-model: dùng llama-3.3-70b-versatile.
GROQ_API_KEY đọc từ .env để gọi ChatGroq sinh câu trả lời.
