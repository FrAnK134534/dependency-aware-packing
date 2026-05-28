from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from dapacking.collectors.html import extract_html
from dapacking.collectors.manifest import ManifestEntry, read_manifest
from dapacking.collectors.pdf import read_pdf_pages
from dapacking.collectors.sectioning import Section, section_markdown, section_pages, section_text
from dapacking.collectors.text import read_text_file
from dapacking.documents import Document


@dataclass(frozen=True)
class ExternalCorpusConfig:
    base_dir: Path | None = None
    fetch_urls: bool = False
    follow_same_domain_once: bool = False
    max_follow_links: int = 20
    request_timeout: float = 20.0


def build_external_documents(
    manifest_path: str | Path,
    config: ExternalCorpusConfig | None = None,
) -> list[Document]:
    manifest_path = Path(manifest_path)
    config = config or ExternalCorpusConfig(base_dir=manifest_path.parent)
    entries = read_manifest(manifest_path)

    documents: list[Document] = []
    for entry in entries:
        documents.extend(_documents_for_entry(entry, config))
    return documents


def _documents_for_entry(entry: ManifestEntry, config: ExternalCorpusConfig) -> list[Document]:
    if entry.source_kind == "local_markdown":
        text = read_text_file(_local_path(entry, config))
        sections = section_markdown(text, entry.title)
        return _sections_to_documents(entry, sections)
    if entry.source_kind == "local_text":
        text = read_text_file(_local_path(entry, config))
        sections = section_text(text, entry.title)
        return _sections_to_documents(entry, sections)
    if entry.source_kind == "local_html":
        html = read_text_file(_local_path(entry, config))
        extracted = extract_html(html, entry.title)
        metadata = {"links": extracted.links}
        title = extracted.title or entry.title
        return _sections_to_documents(
            entry,
            extracted.sections,
            extra_metadata=metadata,
            title=title,
        )
    if entry.source_kind == "local_pdf":
        pages = read_pdf_pages(_local_path(entry, config))
        sections = section_pages(pages, entry.title)
        return _sections_to_documents(entry, sections)
    if entry.source_kind == "url_html":
        if not config.fetch_urls:
            raise ValueError("url_html requires --fetch-urls so network access is explicit")
        return _documents_for_url(entry, config)
    raise ValueError(f"Unsupported source_kind: {entry.source_kind}")


def _documents_for_url(entry: ManifestEntry, config: ExternalCorpusConfig) -> list[Document]:
    html = _fetch_url(entry.location, config.request_timeout)
    extracted = extract_html(html, entry.title)
    documents = _sections_to_documents(
        entry,
        extracted.sections,
        extra_metadata={"links": _absolute_links(entry.location, extracted.links)},
        title=extracted.title or entry.title,
        url=entry.location,
    )

    if not config.follow_same_domain_once:
        return documents

    source_domain = urlparse(entry.location).netloc
    followed = 0
    for link in _absolute_links(entry.location, extracted.links):
        if followed >= config.max_follow_links:
            break
        if urlparse(link).netloc != source_domain:
            continue
        followed += 1
        child_entry = ManifestEntry(
            source_id=f"{entry.source_id}_linked_{followed}",
            source_kind="url_html",
            location=link,
            collection=entry.collection,
            license=entry.license,
            title=entry.title,
            source_type=entry.source_type,
            document_id=f"{entry.document_id or entry.source_id}_linked_{followed}",
            metadata=entry.metadata,
        )
        try:
            child_html = _fetch_url(link, config.request_timeout)
        except Exception:
            continue
        child_extract = extract_html(child_html, entry.title)
        documents.extend(
            _sections_to_documents(
                child_entry,
                child_extract.sections,
                extra_metadata={"links": _absolute_links(link, child_extract.links)},
                title=child_extract.title or entry.title,
                url=link,
            )
        )
    return documents


def _sections_to_documents(
    entry: ManifestEntry,
    sections: list[Section],
    extra_metadata: dict[str, object] | None = None,
    title: str | None = None,
    url: str = "",
) -> list[Document]:
    documents: list[Document] = []
    document_id = entry.document_id or entry.source_id
    source_type = entry.source_type or _default_source_type(entry.source_kind)
    base_metadata = {
        "collection": entry.collection,
        "source_id": entry.source_id,
        "document_id": document_id,
        "source_kind": entry.source_kind,
        "source_type": source_type,
        "license": entry.license,
        "title": title or entry.title,
        "location": entry.location,
    }
    if url:
        base_metadata["url"] = url
    if entry.metadata:
        base_metadata.update(entry.metadata)
    if extra_metadata:
        base_metadata.update(extra_metadata)

    for section in sections:
        if not section.content.strip():
            continue
        section_id = section.section_id
        metadata = dict(base_metadata)
        metadata.update(section.metadata)
        metadata.update(
            {
                "path": f"{document_id}/{section_id}",
                "section_id": section_id,
                "section_title": section.title,
                "section_index": section.index,
            }
        )
        docid = f"{entry.collection}:{document_id}#{section_id}"
        documents.append(Document(docid=docid, content=section.content, metadata=metadata))
    return documents


def _local_path(entry: ManifestEntry, config: ExternalCorpusConfig) -> Path:
    path = Path(entry.location)
    if not path.is_absolute() and config.base_dir:
        path = config.base_dir / path
    return path


def _default_source_type(source_kind: str) -> str:
    return {
        "local_html": "web_page",
        "url_html": "web_page",
        "local_markdown": "technical_doc",
        "local_pdf": "paper_section",
        "local_text": "text_section",
    }.get(source_kind, "text_section")


def _fetch_url(url: str, timeout: float) -> str:
    try:
        import requests
    except ImportError as exc:
        message = "URL collection requires installing dependency-aware-packing[collectors]"
        raise ImportError(message) from exc
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def _absolute_links(base_url: str, links: list[str]) -> list[str]:
    absolute: list[str] = []
    for link in links:
        if not link:
            continue
        absolute.append(urljoin(base_url, link))
    return absolute
