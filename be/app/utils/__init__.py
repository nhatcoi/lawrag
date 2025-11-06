import hashlib
import re
from pathlib import Path
from typing import Optional

def hash_file(file_path: Path) -> str:
    """Tính hash của file để kiểm tra duplicate"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def clean_text(text: str) -> str:
    """Làm sạch text: remove extra whitespace, normalize"""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def sanitize_filename(filename: str) -> str:
    """Sanitize filename để tránh path injection"""
    sanitized = re.sub(r'[^\w\-_\.]+', '_', filename)
    return sanitized.strip('_')

def get_file_version(file_path: Path) -> int:
    """Lấy version number từ filename (nếu có pattern <id>_v<num>)"""
    name = file_path.stem
    match = re.search(r'_v(\d+)$', name)
    return int(match.group(1)) if match else 1

def generate_versioned_filename(base_path: Path, version: int) -> Path:
    """Tạo filename với version: <id>_v<version>.<ext>"""
    stem = base_path.stem.rstrip('_v' + str(version))
    return base_path.parent / f"{stem}_v{version}{base_path.suffix}"

