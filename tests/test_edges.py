from dapacking.documents import Document
from dapacking.edges import build_dependency_edges


def test_build_dependency_edges_skips_same_repo_only_by_default() -> None:
    docs = [
        Document("repo:src/a.py", "def a(): pass", {"repo": "repo", "path": "src/a.py", "source_type": "source"}),
        Document("repo:tools/b.py", "def b(): pass", {"repo": "repo", "path": "tools/b.py", "source_type": "source"}),
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
