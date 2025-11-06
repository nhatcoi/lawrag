from pathlib import Path
from typing import Dict, Any, List
from .embedding import load_vector_store
from .generation import create_qa_chain
from config import INDEX_DIR


def ask(query: str, index_dir: Path = INDEX_DIR, provider: str = "local", law_type_filter: str = None) -> Dict[str, Any]:
    import re
    
    law_keywords = {
        "doanh nghiệp": "doanh-nghiep",
        "lao động": "lao-dong",
        "dân sự": "dan-su",
        "hình sự": "hinh-su",
        "thương mại": "thuong-mai",
        "đất đai": "dat-dai",
    }
    
    query_lower = query.lower()
    enhanced_query = query
    
    if law_type_filter and law_type_filter != "Tổng quan":
        enhanced_query = f"{query} {law_type_filter}"
    else:
        for law_name, keyword in law_keywords.items():
            if law_name in query_lower:
                enhanced_query = f"{query} {law_name}"
                break
    
    vectorstore = load_vector_store(index_dir, provider)
    
    from .generation import get_llm, TOP_K
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
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
    
    if law_type_filter and law_type_filter != "Tổng quan":
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        use_filtered_retriever = False
        
        try:
            from langchain_core.retrievers import BaseRetriever
            from langchain_core.documents import Document as LC_Document
            
            class FilteredRetriever(BaseRetriever):
                def __init__(self, base_retriever, filter_value):
                    super().__init__()
                    self.base_retriever = base_retriever
                    self.filter_value = filter_value
                
                def get_relevant_documents(self, query: str) -> List[LC_Document]:
                    try:
                        docs = self.base_retriever.get_relevant_documents(query)
                        filtered = [doc for doc in docs 
                                  if doc.metadata and doc.metadata.get("law_type") == self.filter_value]
                        if filtered:
                            return filtered[:5]
                        return docs[:5]
                    except Exception as e:
                        print(f"Error in FilteredRetriever.get_relevant_documents: {e}")
                        return docs[:5] if 'docs' in locals() else []
                
                async def aget_relevant_documents(self, query: str) -> List[LC_Document]:
                    try:
                        docs = await self.base_retriever.aget_relevant_documents(query)
                        filtered = [doc for doc in docs 
                                  if doc.metadata and doc.metadata.get("law_type") == self.filter_value]
                        if filtered:
                            return filtered[:5]
                        return docs[:5]
                    except Exception as e:
                        print(f"Error in FilteredRetriever.aget_relevant_documents: {e}")
                        return docs[:5] if 'docs' in locals() else []
            
            retriever = FilteredRetriever(base_retriever, law_type_filter)
            use_filtered_retriever = True
        except (ImportError, Exception) as e:
            print(f"Warning: Could not create FilteredRetriever, using post-filter: {e}")
            retriever = base_retriever
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        
        result = qa_chain({"query": enhanced_query})
        
        if "source_documents" in result and not use_filtered_retriever:
            filtered_docs = [doc for doc in result["source_documents"]
                           if doc.metadata and doc.metadata.get("law_type") == law_type_filter]
            if filtered_docs:
                result["source_documents"] = filtered_docs[:5]
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        
        result = qa_chain({"query": enhanced_query})
    
    answer = result["result"]
    
    sources = []
    if "source_documents" in result:
        for i, doc in enumerate(result["source_documents"], 1):
            source_id = doc.metadata.get("source", "").split("/")[-1]
            sources.append({
                "rank": i,
                "score": 0.0,
                "id": source_id,
                "path": doc.metadata.get("source", ""),
                "text": doc.page_content[:500],
            })
    
    return {"answer": answer, "sources": sources}
