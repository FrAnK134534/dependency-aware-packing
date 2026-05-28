from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    section_id: str
    title: str
    content: str
    index: int
    metadata: dict[str, object] = field(default_factory=dict)


def section_markdown(text: str, default_title: str = "") -> list[Section]:
    sections: list[Section] = []
    current_title = default_title or "Document"
    current_lines: list[str] = []
    index = 0

    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading and current_lines:
            sections.append(_make_section(index, current_title, "\n".join(current_lines)))
            index += 1
            current_lines = []
        if heading:
            current_title = heading.group(2).strip()
        current_lines.append(line)

    if current_lines:
        sections.append(_make_section(index, current_title, "\n".join(current_lines)))
    return sections or [_make_section(0, current_title, text)]


def section_text(text: str, default_title: str = "", max_chars: int = 4000) -> list[Section]:
    paragraphs = [
        paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()
    ]
    if not paragraphs:
        return []

    sections: list[Section] = []
    current: list[str] = []
    current_chars = 0
    for paragraph in paragraphs:
        if current and current_chars + len(paragraph) > max_chars:
            sections.append(
                _make_section(len(sections), default_title or "Text", "\n\n".join(current))
            )
            current = []
            current_chars = 0
        current.append(paragraph)
        current_chars += len(paragraph)

    if current:
        sections.append(_make_section(len(sections), default_title or "Text", "\n\n".join(current)))
    return sections


def section_pages(pages: list[str], default_title: str = "") -> list[Section]:
    sections: list[Section] = []
    for index, page in enumerate(pages):
        page = page.strip()
        if not page:
            continue
        sections.append(
            Section(
                section_id=f"page-{index + 1}",
                title=f"{default_title or 'PDF'} page {index + 1}",
                content=page,
                index=index,
                metadata={"page": index + 1},
            )
        )
    return sections


def _make_section(index: int, title: str, content: str) -> Section:
    section_id = _slug(title) or f"section-{index + 1}"
    return Section(
        section_id=f"{index:04d}-{section_id}",
        title=title,
        content=content.strip(),
        index=index,
    )


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:80]
