from dapacking.documents import Document
from dapacking.edges import DependencyEdge
from dapacking.review import EdgeReviewConfig, sample_edge_review_records


def test_sample_edge_review_records_adds_manual_columns() -> None:
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
    ]
    edges = [
        DependencyEdge(
            source_docid="repo:README.md",
            target_docid="repo:src/model.py",
            relation="readme_code_relation+same_repo",
            weight=0.7,
            metadata={"labels": ["readme_code_relation", "same_repo"]},
        )
    ]

    records = sample_edge_review_records(documents, edges, EdgeReviewConfig(sample_size=1))

    assert len(records) == 1
    assert records[0]["is_strong"] is True
    assert records[0]["primary_relation"] == "readme_code_relation"
    assert records[0]["manual_reasonable"] == ""
    assert records[0]["manual_confidence"] == ""
    assert records[0]["source_path"] == "README.md"


def test_sample_edge_review_records_can_balance_by_relation() -> None:
    documents = [
        Document(
            "repo:README.md",
            "Use src/model.py",
            {"repo": "repo", "path": "README.md", "source_type": "readme"},
        ),
        Document(
            "repo:docs/api.md",
            "Use src/utils.py",
            {"repo": "repo", "path": "docs/api.md", "source_type": "docs"},
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
            source_docid="repo:docs/api.md",
            target_docid="repo:src/utils.py",
            relation="docs_code_relation+same_repo",
            weight=0.7,
            metadata={"labels": ["docs_code_relation", "same_repo"]},
        ),
    ]

    records = sample_edge_review_records(
        documents,
        edges,
        EdgeReviewConfig(per_relation_sample_size=1),
    )

    assert {record["primary_relation"] for record in records} == {
        "docs_code_relation",
        "readme_code_relation",
    }
