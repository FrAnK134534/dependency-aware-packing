from __future__ import annotations

import re
from dataclasses import dataclass

from dapacking.collectors.sectioning import Section, section_text


@dataclass(frozen=True)
class HtmlExtract:
    title: str
    text: str
    links: list[str]
    sections: list[Section]


def extract_html(html: str, default_title: str = "") -> HtmlExtract:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _extract_html_without_bs4(html, default_title)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = default_title
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    links = [str(tag.get("href")) for tag in soup.find_all("a") if tag.get("href")]

    sections: list[Section] = []
    current_title = title or "HTML"
    current_parts: list[str] = []
    for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li"]):
        if node.name in {"h1", "h2", "h3", "h4"}:
            if current_parts:
                sections.append(
                    Section(
                        section_id=f"{len(sections):04d}-{_slug(current_title)}",
                        title=current_title,
                        content="\n".join(current_parts).strip(),
                        index=len(sections),
                    )
                )
                current_parts = []
            current_title = node.get_text(" ", strip=True) or current_title
        else:
            text = node.get_text(" ", strip=True)
            if text:
                current_parts.append(text)

    if current_parts:
        sections.append(
            Section(
                section_id=f"{len(sections):04d}-{_slug(current_title)}",
                title=current_title,
                content="\n".join(current_parts).strip(),
                index=len(sections),
            )
        )

    text = soup.get_text("\n", strip=True)
    return HtmlExtract(
        title=title,
        text=text,
        links=links,
        sections=sections or section_text(text, title),
    )


def _extract_html_without_bs4(html: str, default_title: str) -> HtmlExtract:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else default_title
    links = re.findall(r"href=[\"']([^\"']+)[\"']", html, flags=re.IGNORECASE)
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return HtmlExtract(title=title, text=text, links=links, sections=section_text(text, title))


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:80] or "section"
