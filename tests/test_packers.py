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
    assert samples[0].stats["tokenizer"] == "simple"


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


def test_semantic_packer_groups_related_docs() -> None:
    docs = [
        Document("repo:a.md", "optimizer learning rate warmup schedule", {"repo": "repo", "path": "a.md"}),
        Document("repo:b.md", "learning rate schedule uses warmup steps", {"repo": "repo", "path": "b.md"}),
        Document("repo:c.md", "release packaging wheel metadata", {"repo": "repo", "path": "c.md"}),
    ]

    packer = build_packer(PackingConfig(method="semantic", max_tokens=256))
    samples = packer.pack(docs)

    grouped_docids = [set(sample.docids) for sample in samples]
    assert any({"repo:a.md", "repo:b.md"}.issubset(group) for group in grouped_docids)


def test_datasculpt_lite_reports_semantic_stats() -> None:
    docs = [
        Document("repo:a.md", "token budget context packing", {"repo": "repo", "path": "a.md"}),
        Document("repo:b.md", "context packing token utilization", {"repo": "repo", "path": "b.md"}),
    ]

    packer = build_packer(PackingConfig(method="datasculpt_lite", max_tokens=256))
    samples = packer.pack(docs)

    assert samples[0].stats["semantic_similarity"] > 0
    assert "redundant_pair_rate" in samples[0].stats


def test_dependency_aware_v2_token_fit_adds_same_repo_fillers() -> None:
    docs = [
        Document("repo:src/utils.py", "def normalize(x): return x", {"repo": "repo", "path": "src/utils.py"}),
        Document(
            "repo:tests/test_utils.py",
            "from utils import normalize\ndef test_normalize(): assert normalize(1)",
            {"repo": "repo", "path": "tests/test_utils.py", "source_type": "test"},
        ),
        Document(
            "repo:docs/notes.md",
            "release notes migration guide compatibility matrix",
            {"repo": "repo", "path": "docs/notes.md"},
        ),
        Document("other:README.md", "unrelated project", {"repo": "other", "path": "README.md"}),
    ]

    packer = build_packer(PackingConfig(method="dependency_aware_v2_token_fit", max_tokens=256))
    samples = packer.pack(docs)

    grouped_docids = [set(sample.docids) for sample in samples]
    assert any(
        {"repo:src/utils.py", "repo:tests/test_utils.py", "repo:docs/notes.md"}.issubset(group)
        for group in grouped_docids
    )


def test_dependency_aware_v2_strong_first_adds_strong_edges_before_fillers() -> None:
    docs = [
        Document("repo:src/utils.py", "def normalize(x): return x", {"repo": "repo", "path": "src/utils.py"}),
        Document(
            "repo:tests/test_utils.py",
            "from utils import normalize\ndef test_normalize(): assert normalize(1)",
            {"repo": "repo", "path": "tests/test_utils.py", "source_type": "test"},
        ),
        Document("repo:src/peer.py", "def peer(): return 2", {"repo": "repo", "path": "src/peer.py"}),
    ]

    packer = build_packer(PackingConfig(method="dependency_aware_v2_strong_first", max_tokens=256))
    samples = packer.pack(docs)

    assert samples[0].docids[:2] == ["repo:src/utils.py", "repo:tests/test_utils.py"]


def test_dependency_aware_no_same_repo_still_uses_import_relation() -> None:
    docs = [
        Document("repo:src/utils.py", "def normalize(x): return x", {"repo": "repo", "path": "src/utils.py"}),
        Document(
            "repo:src/train.py",
            "from utils import normalize\nprint(normalize(1))",
            {"repo": "repo", "path": "src/train.py"},
        ),
    ]

    packer = build_packer(PackingConfig(method="dependency_aware_no_same_repo", max_tokens=256))
    samples = packer.pack(docs)

    assert set(samples[0].docids) == {"repo:src/utils.py", "repo:src/train.py"}


def test_dependency_aware_strong_edges_only_ignores_same_directory_only() -> None:
    docs = [
        Document("repo:src/a.py", "def alpha(): return 1", {"repo": "repo", "path": "src/a.py"}),
        Document("repo:src/b.py", "def beta(): return 2", {"repo": "repo", "path": "src/b.py"}),
    ]

    packer = build_packer(PackingConfig(method="dependency_aware_strong_edges_only", max_tokens=256))
    samples = packer.pack(docs)

    grouped_docids = [set(sample.docids) for sample in samples]
    assert not any({"repo:src/a.py", "repo:src/b.py"}.issubset(group) for group in grouped_docids)
