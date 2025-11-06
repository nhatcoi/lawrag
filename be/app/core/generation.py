from typing import List, Dict
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.config import GROQ_MODEL, GROQ_API_KEY, TOP_K


def get_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    return ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.7)


def format_sources_for_prompt(sources: List[Dict]) -> str:
    if not sources:
        return ""
    
    formatted = []
    for i, source in enumerate(sources[:5], 1):
        source_id = source.get("id", "")
        source_text = source.get("text", "")[:300]
        
        if "điều" in source_id.lower():
            article_num = source_id.replace("điều_", "").replace(".txt", "")
            formatted.append(f"[{i}] Nguồn: Điều {article_num}\n{source_text}")
        else:
            formatted.append(f"[{i}] {source_id}\n{source_text}")
    
    return "\n\n".join(formatted)


def create_qa_chain(vectorstore, return_source_documents: bool = True):
    llm = get_llm()
    
    prompt_template = """Bạn là một luật sư AI chuyên nghiệp tên MrLawyerAI, có kiến thức sâu về pháp luật Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên ngữ cảnh pháp luật được cung cấp.

QUY TẮC TRẢ LỜI:
1. Trả lời theo cách tự nhiên, dễ hiểu, như một luật sư đang tư vấn cho khách hàng
2. KHÔNG copy-paste nguyên văn từ ngữ cảnh, mà diễn giải lại theo cách riêng
3. Xác định loại luật được hỏi trong câu hỏi (Luật Doanh nghiệp, Bộ luật Lao động, Luật Dân sự, Luật Hình sự, v.v.)
4. CHỈ sử dụng thông tin từ loại luật được hỏi trong ngữ cảnh, KHÔNG trộn lẫn các loại luật khác
5. Nếu câu hỏi không chỉ định rõ loại luật, hãy xác định dựa trên nội dung câu hỏi và chỉ sử dụng thông tin phù hợp
6. Luôn trích dẫn điều luật cụ thể khi có thông tin, theo format: "Theo Điều X – [Tên Bộ luật/Luật] [Nguồn số]"
7. Nếu không có thông tin phù hợp trong ngữ cảnh, hãy nói rõ ràng
8. Trả lời ngắn gọn, rõ ràng, dễ hiểu, có cấu trúc

Ngữ cảnh:
{context}

Câu hỏi: {question}

Hãy trả lời như một luật sư chuyên nghiệp, xác định đúng loại luật được hỏi và chỉ sử dụng thông tin từ loại luật đó."""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=return_source_documents,
        chain_type_kwargs={"prompt": prompt},
    )
