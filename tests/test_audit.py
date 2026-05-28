from dapacking.audit import render_review_markdown, summarize_review_records


def test_summarize_review_records_reports_precision_by_relation() -> None:
    records = [
        {"primary_relation": "import_relation", "manual_reasonable": "yes"},
        {"primary_relation": "import_relation", "manual_reasonable": "partial"},
        {"primary_relation": "import_relation", "manual_reasonable": "no"},
        {
            "primary_relation": "docs_code_relation",
            "manual_reasonable": "no",
            "manual_error_type": "generic_name",
        },
    ]

    summaries = summarize_review_records(records)
    rows = {summary.relation: summary for summary in summaries}

    assert rows["import_relation"].reviewed == 3
    assert rows["import_relation"].strict_precision == 1 / 3
    assert rows["import_relation"].supportive_rate == 2 / 3
    assert rows["docs_code_relation"].no == 1

    report = render_review_markdown(summaries, records)
    assert "generic_name" in report
