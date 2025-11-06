from pathlib import Path
import glob
from config import FILES_DIR


def find_article_by_number(article_number: str, docs_dir: Path = FILES_DIR) -> str:
    patterns = [
        docs_dir / f"điều_{article_number}.txt",
        docs_dir / f"điều_{article_number.zfill(2)}.txt",
    ]
    
    for pattern in patterns:
        if pattern.exists():
            return pattern.read_text(encoding="utf-8").strip()
    
    output_pattern = str(docs_dir / "*/output_*" / f"điều_{article_number}.txt")
    files = glob.glob(output_pattern)
    if files:
        return Path(files[0]).read_text(encoding="utf-8").strip()
    
    output_pattern2 = str(docs_dir / "*/output_*" / f"điều_{article_number.zfill(2)}.txt")
    files = glob.glob(output_pattern2)
    if files:
        return Path(files[0]).read_text(encoding="utf-8").strip()
    
    return ""
