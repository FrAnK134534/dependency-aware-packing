from dapacking.edge_filter import filter_dependency_edges, relation_counts
from dapacking.edges import DependencyEdge
from dapacking.relation_config import RelationConfig


def test_filter_dependency_edges_keeps_allowed_labels_and_reweights() -> None:
    edge = DependencyEdge(
        source_docid="repo:a.py",
        target_docid="repo:b.py",
        relation="import_relation+same_repo+readme_code_relation",
        weight=1.75,
        metadata={"labels": ["import_relation", "same_repo", "readme_code_relation"]},
    )
    config = RelationConfig(
        allowed_relations=("import_relation",),
        relation_reliability={"import_relation": 0.5},
    )

    filtered = filter_dependency_edges([edge], config)

    assert len(filtered) == 1
    assert filtered[0].relation == "import_relation"
    assert filtered[0].weight == 0.5
    assert filtered[0].metadata["original_relation"] == edge.relation
    assert filtered[0].metadata["original_weight"] == edge.weight
    assert filtered[0].metadata["original_labels"] == [
        "import_relation",
        "same_repo",
        "readme_code_relation",
    ]
    assert filtered[0].metadata["labels"] == ["import_relation"]


def test_filter_dependency_edges_drops_edges_without_allowed_labels() -> None:
    edge = DependencyEdge(
        source_docid="repo:README.md",
        target_docid="repo:src/model.py",
        relation="readme_code_relation+same_repo",
        weight=0.75,
        metadata={"labels": ["readme_code_relation", "same_repo"]},
    )
    config = RelationConfig(allowed_relations=("import_relation",))

    assert filter_dependency_edges([edge], config) == []
    assert relation_counts([edge])["readme_code_relation"] == 1
