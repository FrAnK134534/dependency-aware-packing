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
