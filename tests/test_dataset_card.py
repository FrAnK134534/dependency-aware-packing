from dapacking.dataset_card import render_dataset_card
from dapacking.documents import Document
from dapacking.edges import DependencyEdge


def test_render_dataset_card_includes_sources_edges_and_leakage_policy() -> None:
    documents = [
        Document(
            "repo:README.md",
            "Use src/model.py",
            {"repo": "repo", "path": "README.md", "source_type": "readme", "license": "MIT"},
        ),
        Document(
            "repo:src/model.py",
            "class Model: pass",
            {"repo": "repo", "path": "src/model.py", "source_type": "source", "license": "MIT"},
        ),
    ]
    edges = [
        DependencyEdge(
            "repo:README.md",
            "repo:src/model.py",
            "readme_code_relation+same_repo",
            0.75,
            {"labels": ["readme_code_relation", "same_repo"]},
        )
    ]

    card = render_dataset_card("toy", documents, edges)

    assert "# Dataset Card: toy" in card
    assert "readme_code_relation" in card
    assert "Leakage Policy" in card
