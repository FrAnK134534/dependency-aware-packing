from pathlib import Path

from dapacking.stats import summarize_packed_file


def test_summarize_packed_file(tmp_path: Path) -> None:
    path = tmp_path / "packed.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"method": "demo", "stats": {"tokens": 10, "num_docs": 2, "dependency_score": 0.5, "token_utilization": 0.25, "truncation_rate": 0.0}}',
                '{"method": "demo", "stats": {"tokens": 20, "num_docs": 3, "dependency_score": 1.0, "token_utilization": 0.5, "truncation_rate": 0.1}}',
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_packed_file(path)

    assert summary.method == "demo"
    assert summary.samples == 2
    assert summary.total_tokens == 30
    assert summary.avg_dependency_score == 0.75


def test_summarize_packed_file_with_edges(tmp_path: Path) -> None:
    packed = tmp_path / "packed.jsonl"
    edges = tmp_path / "edges.jsonl"
    packed.write_text(
        '{"method": "demo", "docids": ["r:a.md", "r:b.py"], "stats": {"tokens": 10, "num_docs": 2, "dependency_score": 0.5, "token_utilization": 0.25, "truncation_rate": 0.0}}\n',
        encoding="utf-8",
    )
    edges.write_text(
        '{"source_docid": "r:a.md", "target_docid": "r:b.py", "relation": "readme_code", "weight": 0.6, "metadata": {"repo": "r"}}\n',
        encoding="utf-8",
    )

    summary = summarize_packed_file(packed, edges)

    assert summary.avg_order_dependency == 0.6
    assert summary.edge_coverage == 1.0
    assert summary.weighted_edge_coverage == 1.0
    assert summary.same_repo_pair_ratio == 1.0
