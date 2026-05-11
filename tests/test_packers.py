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


def test_bm25_packer_groups_lexically_related_docs() -> None:
    docs = [
        Document("repo:src/model.py", "tiny classifier hidden size predict", {"repo": "repo", "path": "src/model.py"}),
        Document(
            "repo:README.md",
            "This tiny classifier uses hidden size and predict for examples",
            {"repo": "repo", "path": "README.md"},
        ),
        Document("other:notes.md", "database migration schedule", {"repo": "other", "path": "notes.md"}),
    ]

    packer = build_packer(PackingConfig(method="bm25", max_tokens=256))
    samples = packer.pack(docs)

    grouped_docids = [set(sample.docids) for sample in samples]
    assert {"repo:src/model.py", "repo:README.md"} in grouped_docids
