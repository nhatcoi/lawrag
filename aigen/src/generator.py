"""
Module tạo sinh câu trả lời sử dụng Groq LLM
Kết hợp ngữ cảnh được truy xuất với câu hỏi để tạo câu trả lời
"""

import os
from typing import List

from dotenv import load_dotenv

try:
    from groq import Groq  # type: ignore
except Exception as exc:
    raise RuntimeError(
        "groq SDK is required. Please install dependencies (pip install -r requirements.txt)."
    ) from exc


def format_context(chunks: List[str], max_chars: int = 20000) -> str:
    """Ghép các đoạn văn bản thành ngữ cảnh, giới hạn độ dài"""
    joined = "\n\n".join(chunks)
    return joined[:max_chars]


def generate_answer(query: str, contexts: List[str], model: str = "llama-3.3-70b-versatile") -> str:
    """
    Tạo câu trả lời dựa trên câu hỏi và ngữ cảnh được truy xuất
    Sử dụng Groq LLM với prompt được thiết kế cho pháp luật Việt Nam
    """
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY chưa được thiết lập trong biến môi trường")

    client = Groq(api_key=api_key)

    # Chuẩn bị ngữ cảnh
    context_block = format_context(contexts)
    
    # Thiết kế prompt phù hợp với pháp luật
    system_prompt = (
        "Bạn là trợ lý trả lời câu hỏi dựa trên ngữ cảnh pháp luật Việt Nam. "
        "Chỉ dùng thông tin trong ngữ cảnh. Nếu thiếu thông tin, hãy nói không đủ dữ liệu."
    )
    user_prompt = (
        f"Ngữ cảnh:\n{context_block}\n\nCâu hỏi: {query}\n"
        "Yêu cầu: Trả lời ngắn gọn, kèm trích dẫn điều luật (nếu có)."
    )

    # Gọi Groq API
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # Thấp để có câu trả lời ổn định
        max_tokens=800,
    )
    return resp.choices[0].message.content or ""


