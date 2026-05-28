from dapacking.collectors import ExternalCorpusConfig, build_external_documents
from dapacking.edges import build_dependency_edges


def test_external_collectors_build_section_documents(tmp_path) -> None:
    (tmp_path / "guide.md").write_text(
        "# Intro\nDefinition: Context gain measures loss reduction.\n\n"
        "# Example\nContext gain improves when context helps.",
        encoding="utf-8",
    )
    (tmp_path / "page.html").write_text(
        "<html><head><title>API</title></head><body>"
        "<h1>Trainer.fit</h1><p>Call Trainer.fit in examples.</p>"
        "<a href='guide.html'>Guide</a></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("Plain notes\n\nAnother paragraph", encoding="utf-8")
    (tmp_path / "paper.pdf").write_text("Figure 1 shows the system.", encoding="utf-8")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "source_id\tsource_kind\tlocation\tcollection\tlicense\ttitle\tsource_type\n"
        "guide\tlocal_markdown\tguide.md\tmanual\tMIT\tGuide\ttechnical_doc\n"
        "api\tlocal_html\tpage.html\tmanual\tMIT\tAPI\tapi_doc\n"
        "notes\tlocal_text\tnotes.txt\tmanual\tMIT\tNotes\ttext_section\n"
        "paper\tlocal_pdf\tpaper.pdf\tpapers\tCC-BY\tPaper\tpaper_section\n",
        encoding="utf-8",
    )

    documents = build_external_documents(
        manifest,
        ExternalCorpusConfig(base_dir=tmp_path),
    )

    assert len(documents) >= 5
    assert {document.metadata["source_kind"] for document in documents} == {
        "local_markdown",
        "local_html",
        "local_text",
        "local_pdf",
    }
    assert all(document.metadata.get("collection") for document in documents)
    assert all(document.metadata.get("document_id") for document in documents)


def test_external_collector_links_can_build_hyperlink_edges(tmp_path) -> None:
    (tmp_path / "a.html").write_text(
        "<html><head><title>A</title></head><body><a href='b.html'>B</a></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "b.html").write_text(
        "<html><head><title>B</title></head><body><p>Target page</p></body></html>",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "source_id\tsource_kind\tlocation\tcollection\tlicense\ttitle\tsource_type\n"
        "a\tlocal_html\ta.html\tweb\tMIT\tA\tweb_page\n"
        "b\tlocal_html\tb.html\tweb\tMIT\tB\tweb_page\n",
        encoding="utf-8",
    )

    documents = build_external_documents(manifest, ExternalCorpusConfig(base_dir=tmp_path))
    edges = build_dependency_edges(documents)

    assert any("hyperlink_relation" in edge.relation for edge in edges)
