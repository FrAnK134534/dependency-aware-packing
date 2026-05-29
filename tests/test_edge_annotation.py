from dapacking.documents import Document
from dapacking.edge_annotation import annotate_edge_review_record


def test_assisted_annotation_accepts_explicit_readme_code_path() -> None:
    documents = {
        "repo:README.md": Document(
            "repo:README.md",
            "Use src/model.py to customize the Model class.",
            {"repo": "repo", "path": "README.md", "source_type": "readme"},
        ),
        "repo:src/model.py": Document(
            "repo:src/model.py",
            "class Model: pass",
            {"repo": "repo", "path": "src/model.py", "source_type": "source"},
        ),
    }
    record = {
        "primary_relation": "readme_code_relation",
        "source_docid": "repo:README.md",
        "target_docid": "repo:src/model.py",
    }

    annotation = annotate_edge_review_record(record, documents)

    assert annotation.label == "yes"
    assert annotation.confidence >= 0.8


def test_assisted_annotation_rejects_generic_doc_code_without_evidence() -> None:
    documents = {
        "repo:docs/guide.md": Document(
            "repo:docs/guide.md",
            "This page describes common utilities in broad terms.",
            {"repo": "repo", "path": "docs/guide.md", "source_type": "docs"},
        ),
        "repo:src/config.py": Document(
            "repo:src/config.py",
            "VALUE = 1",
            {"repo": "repo", "path": "src/config.py", "source_type": "source"},
        ),
    }
    record = {
        "primary_relation": "docs_code_relation",
        "source_docid": "repo:docs/guide.md",
        "target_docid": "repo:src/config.py",
    }

    annotation = annotate_edge_review_record(record, documents)

    assert annotation.label == "no"
    assert annotation.error_type == "missing_doc_code_evidence"
