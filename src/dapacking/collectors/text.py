from __future__ import annotations

from pathlib import Path


def read_text_file(path: str | Path) -> str:
    data = Path(path).read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")
