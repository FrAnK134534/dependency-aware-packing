from dapacking.documents import Document
from dapacking.edges import DependencyEdge
from dapacking.review import EdgeReviewConfig, sample_edge_review_records


def test_sample_edge_review_records_adds_manual_columns() -> None:
    documents = [
        Document("repo:README.md", "Use src/model.py", {"repo": "repo", "path": "README.md", "source_type": "readme"}),
        Document("repo:src/model.py", "class Model: pass", {"repo": "repo", "path": "src/model.py", "source_type": "source"}),
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
    assert records[0]["manual_reasonable"] == ""
    assert records[0]["source_path"] == "README.md"
