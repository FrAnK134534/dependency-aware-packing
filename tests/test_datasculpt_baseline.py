import csv
import json
import subprocess
import sys
from pathlib import Path


def test_import_datasculpt_output(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        json.dumps(
            {
                "docid": "repo:README.md",
                "content": "Use src/model.py.",
                "metadata": {"repo": "repo", "path": "README.md", "source_type": "readme"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "docid": "repo:src/model.py",
                "content": "class Model: pass",
                "metadata": {"repo": "repo", "path": "src/model.py", "source_type": "source"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_folder = tmp_path / "data_sculpt"
    output_folder.mkdir()
    (output_folder / "part-00000").write_text(
        json.dumps(
            {
                "total_token_num": 8,
                "docs": [
                    {"source_id": "repo:README.md", "chunk": "Use src/model.py.", "token_len": 4},
                    {"source_id": "repo:src/model.py", "chunk": "class Model: pass", "token_len": 4},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    packed = tmp_path / "packed.jsonl"
    summary = tmp_path / "summary.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/baselines/import_datasculpt_output.py",
            "--datasculpt-output-folder",
            str(output_folder),
            "--documents",
            str(documents),
            "--output",
            str(packed),
            "--summary",
            str(summary),
            "--max-tokens",
            "128",
        ],
        check=True,
    )

    sample = json.loads(packed.read_text(encoding="utf-8").splitlines()[0])
    assert sample["method"] == "datasculpt_original"
    assert sample["docids"] == ["repo:README.md", "repo:src/model.py"]
    with summary.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["method"] == "datasculpt_original"
