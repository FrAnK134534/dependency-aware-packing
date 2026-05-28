from __future__ import annotations

from pathlib import Path


def read_pdf_pages(path: str | Path) -> list[str]:
    pdf_path = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError:
        return [_fallback_text(pdf_path)]

    try:
        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        return [_fallback_text(pdf_path)]
    return pages


def _fallback_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")
