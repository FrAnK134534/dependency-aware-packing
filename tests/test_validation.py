from dapacking.documents import Document
from dapacking.edges import DependencyEdge
from dapacking.validation import (
    DependencyValidationConfig,
    build_dependency_validation_records,
)


def test_build_dependency_validation_records_uses_strong_edges_by_default() -> None:
    documents = [
        Document("repo:README.md", "Use src/model.py", {"repo": "repo", "path": "README.md", "source_type": "readme"}),
        Document("repo:src/model.py", "class Model: pass", {"repo": "repo", "path": "src/model.py", "source_type": "source"}),
        Document("repo:src/utils.py", "def helper(): pass", {"repo": "repo", "path": "src/utils.py", "source_type": "source"}),
    ]
    edges = [
        DependencyEdge(
            source_docid="repo:README.md",
            target_docid="repo:src/model.py",
            relation="readme_code_relation+same_repo",
            weight=0.7,
            metadata={"labels": ["readme_code_relation", "same_repo"]},
        ),
        DependencyEdge(
            source_docid="repo:src/model.py",
            target_docid="repo:src/utils.py",
            relation="same_directory+same_repo",
            weight=0.35,
            metadata={"labels": ["same_directory", "same_repo"]},
        ),
    ]

    records = build_dependency_validation_records(
        documents,
        edges,
        DependencyValidationConfig(max_examples_per_relation=5),
    )

    assert len(records) == 1
    assert records[0]["primary_relation"] == "readme_code_relation"
    assert records[0]["context_with_dependency"].startswith("<doc id=\"repo:README.md\"")
    assert records[0]["context_without_dependency"].startswith("<doc id=\"repo:src/model.py\"")
    assert records[0]["target_text"] == "class Model: pass"


def test_build_dependency_validation_records_can_include_weak_edges() -> None:
    documents = [
        Document("repo:src/model.py", "class Model: pass", {"repo": "repo", "path": "src/model.py", "source_type": "source"}),
        Document("repo:src/utils.py", "def helper(): pass", {"repo": "repo", "path": "src/utils.py", "source_type": "source"}),
    ]
    edges = [
        DependencyEdge(
            source_docid="repo:src/model.py",
            target_docid="repo:src/utils.py",
            relation="same_directory+same_repo",
            weight=0.35,
            metadata={"labels": ["same_directory", "same_repo"]},
        )
    ]

    records = build_dependency_validation_records(
        documents,
        edges,
        DependencyValidationConfig(max_examples_per_relation=5, include_weak=True),
    )

    assert len(records) == 1
    assert records[0]["primary_relation"] == "same_directory"
