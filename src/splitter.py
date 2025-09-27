"""
Module xử lý tách PDF thành các file điều luật riêng biệt
"""

import os
import re
from typing import List, Tuple

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore


def extract_text_from_pdf(pdf_path: str) -> str:
    """Trích xuất text từ file PDF"""
    if pdfplumber is None:
        raise RuntimeError(
            "pdfplumber is required to read PDFs. Please install dependencies (pip install -r requirements.txt)."
        )

    text_parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Trích xuất text với tolerance cho việc căn chỉnh
            page_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)

# Regex để tìm tiêu đề điều luật: "Điều [số]"
ARTICLE_HEADING_REGEX = re.compile(r"^\s*Điều\s+(\d+)\b", re.UNICODE)

def split_articles(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Tách các dòng text thành các điều luật dựa trên regex pattern
    Trả về list các tuple (id_điều_luật, [các_dòng_text])
    """
    articles: List[Tuple[str, List[str]]] = []
    current_id: str = ""
    current_lines: List[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        m = ARTICLE_HEADING_REGEX.match(line)
        if m:
            # Tìm thấy tiêu đề điều luật mới
            if current_id:
                # Lưu điều luật trước đó
                articles.append((current_id, current_lines))
            number = m.group(1)
            current_id = f"Điều {number}"
            current_lines = [line]
        else:
            # Thêm dòng vào điều luật hiện tại
            if current_id:
                current_lines.append(line)
            else:
                # Bỏ qua dòng trước khi tìm thấy điều luật đầu tiên
                continue

    # Lưu điều luật cuối cùng
    if current_id:
        articles.append((current_id, current_lines))

    return articles


def sanitize_filename(name: str) -> str:
    """Làm sạch tên file, thay thế ký tự đặc biệt bằng underscore"""
    sanitized = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE)
    return sanitized.strip("_")


def write_articles(articles: List[Tuple[str, List[str]]], output_dir: str) -> None:
    """Ghi các điều luật vào file .txt riêng biệt"""
    os.makedirs(output_dir, exist_ok=True)
    for article_id, article_lines in articles:
        # Tạo tên file an toàn
        filename = sanitize_filename(article_id.lower()) + ".txt"
        out_path = os.path.join(output_dir, filename)
        # Ghi nội dung điều luật
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(article_lines).strip() + "\n")
