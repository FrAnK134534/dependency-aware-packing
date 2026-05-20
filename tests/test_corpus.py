from pathlib import Path

from dapacking.corpus import build_documents_from_repos, classify_source_type


def test_classify_source_type() -> None:
    assert classify_source_type("README.md") == "readme"
    assert classify_source_type("tests/test_model.py") == "test"
    assert classify_source_type("docs/usage.md") == "docs"
    assert classify_source_type("config/train.yaml") == "config"
    assert classify_source_type("examples/demo.py") == "example"
    assert classify_source_type("src/model.py") == "source"


def test_build_documents_from_repo(tmp_path: Path) -> None:
    repo = tmp_path / "toy_repo"
    repo.mkdir()
    (repo / "README.md").write_text("Use src/model.py", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "model.py").write_text("class Model: pass", encoding="utf-8")
    (repo / "data.bin").write_bytes(b"\x00\x01")

    documents = build_documents_from_repos([repo])

    docids = {document.docid for document in documents}
    assert "toy_repo:README.md" in docids
    assert "toy_repo:src/model.py" in docids
    assert "toy_repo:data.bin" not in docids
