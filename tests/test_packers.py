from dapacking.documents import Document
from dapacking.packers import PackingConfig, build_packer


def test_dependency_aware_packer_groups_related_docs() -> None:
    docs = [
        Document("repo:src/utils.py", "def normalize(x): return x", {"repo": "repo", "path": "src/utils.py"}),
        Document(
            "repo:src/train.py",
            "from utils import normalize\nprint(normalize(1))",
            {"repo": "repo", "path": "src/train.py"},
        ),
        Document("other:README.md", "unrelated notes", {"repo": "other", "path": "README.md"}),
    ]

    packer = build_packer(PackingConfig(method="dependency_aware", max_tokens=256))
    samples = packer.pack(docs)

    grouped_docids = [set(sample.docids) for sample in samples]
    assert {"repo:src/utils.py", "repo:src/train.py"} in grouped_docids
