from dapacking.documents import Document
from dapacking.edges import build_dependency_edges


def test_build_dependency_edges_skips_same_repo_only_by_default() -> None:
    docs = [
        Document(
            "repo:src/a.py",
            "def a(): pass",
            {"repo": "repo", "path": "src/a.py", "source_type": "source"},
        ),
        Document(
            "repo:tools/b.py",
            "def b(): pass",
            {"repo": "repo", "path": "tools/b.py", "source_type": "source"},
        ),
    ]

    edges = build_dependency_edges(docs)

    assert edges == []


def test_build_dependency_edges_detects_readme_code() -> None:
    docs = [
        Document(
            "repo:README.md",
            "Use src/model.py to train.",
            {"repo": "repo", "path": "README.md", "source_type": "readme"},
        ),
        Document(
            "repo:src/model.py",
            "class Model: pass",
            {"repo": "repo", "path": "src/model.py", "source_type": "source"},
        ),
    ]

    edges = build_dependency_edges(docs)

    assert len(edges) == 1
    assert edges[0].source_docid == "repo:README.md"
    assert edges[0].target_docid == "repo:src/model.py"
    assert "readme_code_relation" in edges[0].relation


def test_build_dependency_edges_adds_non_code_relations_and_weak_labels() -> None:
    docs = [
        Document(
            "manual:guide#definition",
            "Definition: Context gain is the loss reduction from useful context.",
            {
                "collection": "manual",
                "document_id": "guide",
                "path": "guide/definition",
                "source_type": "technical_doc",
                "role": "definition",
                "term": "Context gain",
                "section_index": 0,
            },
        ),
        Document(
            "manual:guide#example",
            "Context gain should be positive when dependency context helps.",
            {
                "collection": "manual",
                "document_id": "guide",
                "path": "guide/example",
                "source_type": "technical_doc",
                "section_index": 1,
            },
        ),
    ]

    edges = build_dependency_edges(docs)

    assert len(edges) >= 1
    labels = set(edges[0].metadata["labels"])
    assert "definition_usage_relation" in labels
    assert "same_document" in labels
    assert "section_neighbor" in labels
