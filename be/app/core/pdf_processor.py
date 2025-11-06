import os
import re
from pathlib import Path
from typing import List, Tuple

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def extract_text_from_pdf(pdf_path: Path) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required. Install with: pip install pdfplumber")
    
    text_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def split_articles(lines: List[str]) -> List[Tuple[str, List[str]]]:
    article_pattern = re.compile(r"^\s*Điều\s+(\d+)\b", re.UNICODE)
    articles = []
    current_id = ""
    current_lines = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        match = article_pattern.match(line)
        if match:
            if current_id:
                articles.append((current_id, current_lines))
            number = match.group(1)
            current_id = f"Điều {number}"
            current_lines = [line]
        else:
            if current_id:
                current_lines.append(line)

    if current_id:
        articles.append((current_id, current_lines))
    return articles


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE)
    return sanitized.strip("_")


def process_pdf_to_txt(pdf_path: Path, output_dir: Path) -> List[str]:
    """Process PDF và tách thành các file txt trong output_dir"""
    output_dir.mkdir(parents=True, exist_ok=True)
    text = extract_text_from_pdf(pdf_path)
    lines = text.splitlines()
    articles = split_articles(lines)
    
    created_files = []
    for article_id, article_lines in articles:
        filename = sanitize_filename(article_id.lower()) + ".txt"
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(article_lines).strip() + "\n")
        created_files.append(str(out_path))
    
    return created_files

