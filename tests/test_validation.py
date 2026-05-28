from dapacking.documents import Document
from dapacking.edges import DependencyEdge
from dapacking.validation import (
    ControlValidationConfig,
    DependencyValidationConfig,
    build_control_validation_records,
    build_dependency_validation_records,
    read_review_annotations,
)


def test_build_dependency_validation_records_uses_strong_edges_by_default() -> None:
    documents = [
        Document(
            "repo:README.md",
            "Use src/model.py",
            {"repo": "repo", "path": "README.md", "source_type": "readme"},
        ),
        Document(
            "repo:src/model.py",
            "class Model: pass",
            {"repo": "repo", "path": "src/model.py", "source_type": "source"},
        ),
        Document(
            "repo:src/utils.py",
            "def helper(): pass",
            {"repo": "repo", "path": "src/utils.py", "source_type": "source"},
        ),
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
        Document(
            "repo:src/model.py",
            "class Model: pass",
            {"repo": "repo", "path": "src/model.py", "source_type": "source"},
        ),
        Document(
            "repo:src/utils.py",
            "def helper(): pass",
            {"repo": "repo", "path": "src/utils.py", "source_type": "source"},
        ),
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


def test_build_dependency_validation_records_filters_review_annotations() -> None:
    documents = [
        Document(
            "repo:README.md",
            "Use src/model.py",
            {"repo": "repo", "path": "README.md", "source_type": "readme"},
        ),
        Document(
            "repo:src/model.py",
            "class Model: pass",
            {"repo": "repo", "path": "src/model.py", "source_type": "source"},
        ),
        Document(
            "repo:docs/api.md",
            "Use src/utils.py",
            {"repo": "repo", "path": "docs/api.md", "source_type": "docs"},
        ),
        Document(
            "repo:src/utils.py",
            "def helper(): pass",
            {"repo": "repo", "path": "src/utils.py", "source_type": "source"},
        ),
    ]
    edges = [
        DependencyEdge(
            source_docid="repo:README.md",
            target_docid="repo:src/model.py",
            relation="readme_code_relation+same_repo",
            weight=0.75,
            metadata={"labels": ["readme_code_relation", "same_repo"]},
        ),
        DependencyEdge(
            source_docid="repo:docs/api.md",
            target_docid="repo:src/utils.py",
            relation="docs_code_relation+same_repo",
            weight=0.75,
            metadata={"labels": ["docs_code_relation", "same_repo"]},
        ),
    ]
    annotations = {
        ("repo:README.md", "repo:src/model.py", "readme_code_relation"): {
            "review_label": "partial",
            "review_confidence": 0.8,
        },
        ("repo:docs/api.md", "repo:src/utils.py", "docs_code_relation"): {
            "review_label": "partial",
            "review_confidence": 0.4,
        },
    }

    records = build_dependency_validation_records(
        documents,
        edges,
        DependencyValidationConfig(
            max_examples_per_relation=5,
            review_annotations=annotations,
            min_review_confidence=0.6,
        ),
    )

    assert len(records) == 1
    assert records[0]["source_docid"] == "repo:README.md"
    assert records[0]["metadata"]["review_label"] == "partial"
    assert records[0]["metadata"]["review_confidence"] == 0.8


def test_read_review_annotations_csv_supports_manual_reasonable(tmp_path) -> None:
    review_csv = tmp_path / "review.csv"
    review_csv.write_text(
        "source_docid,target_docid,relation,manual_reasonable,manual_confidence\n"
        "a,b,import_relation,yes,0.9\n",
        encoding="utf-8",
    )

    annotations = read_review_annotations(review_csv)

    assert annotations[("a", "b", "import_relation")]["review_label"] == "yes"
    assert annotations[("a", "b", "import_relation")]["review_confidence"] == 0.9


def test_build_control_validation_records_creates_non_edge_and_cross_group_controls() -> None:
    documents = [
        Document("repo_a:a.py", "A", {"repo": "repo_a", "path": "a.py", "source_type": "source"}),
        Document("repo_a:b.py", "B", {"repo": "repo_a", "path": "b.py", "source_type": "source"}),
        Document("repo_b:c.py", "C", {"repo": "repo_b", "path": "c.py", "source_type": "source"}),
    ]
    edges = [
        DependencyEdge(
            source_docid="repo_a:a.py",
            target_docid="repo_a:b.py",
            relation="import_relation+same_repo",
            weight=1.1,
            metadata={"labels": ["import_relation", "same_repo"]},
        )
    ]

    records = build_control_validation_records(
        documents,
        edges,
        ControlValidationConfig(max_examples_per_control=2, seed=1),
    )

    relation_counts = {record["primary_relation"] for record in records}
    assert "random_cross_group" in relation_counts
    assert all(record["relation"] != "import_relation+same_repo" for record in records)
    assert all("control_type" in record["metadata"] for record in records)
