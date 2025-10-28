"""
Module tạo câu trả lời sử dụng GROQ LLM
"""

import os
from groq import Groq
from typing import List, Dict, Any


class AnswerGenerator:
    """Tạo câu trả lời từ thông tin truy xuất"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Tạo câu trả lời từ query và contexts"""
        # Giới hạn độ dài context để tránh lỗi token limit
        max_context_length = 3000
        context_text = ""
        
        for ctx in contexts:
            content = ctx.get("content", "")
            if len(context_text + content) > max_context_length:
                break
            context_text += content + "\n\n"
        
        # Prompt cho pháp luật
        system_prompt = """Bạn là trợ lý pháp luật chuyên nghiệp. 
        Trả lời câu hỏi dựa trên thông tin pháp luật được cung cấp.
        Nếu không đủ thông tin, hãy nói rõ điều đó."""
        
        user_prompt = f"""Ngữ cảnh pháp luật:
{context_text.strip()}

Câu hỏi: {query}

Hãy trả lời ngắn gọn, chính xác dựa trên thông tin pháp luật trên."""
        
        # Gọi GROQ API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        return response.choices[0].message.content or "Không thể tạo câu trả lời."
