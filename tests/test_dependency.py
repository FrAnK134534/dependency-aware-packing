from dapacking.dependency import dependency_score
from dapacking.documents import Document


def test_import_dependency_score_detects_python_import() -> None:
    source = Document(
        docid="repo:src/utils.py",
        content="def normalize(x): return x",
        metadata={"repo": "repo", "path": "src/utils.py"},
    )
    target = Document(
        docid="repo:src/train.py",
        content="from utils import normalize\nprint(normalize(1))",
        metadata={"repo": "repo", "path": "src/train.py"},
    )

    evidence = dependency_score(source, target)

    assert evidence.score > 0
    assert "import_relation" in evidence.labels


def test_cross_repo_documents_only_get_no_dependency() -> None:
    source = Document("a:README.md", "hello", {"repo": "a", "path": "README.md"})
    target = Document("b:src/app.py", "print('hello')", {"repo": "b", "path": "src/app.py"})

    evidence = dependency_score(source, target)

    assert evidence.score == 0


def test_config_script_requires_explicit_call_or_read() -> None:
    config = Document(
        "repo:configs/train.yaml",
        "learning_rate: 1e-4",
        {"repo": "repo", "path": "configs/train.yaml", "source_type": "config"},
    )
    script = Document(
        "repo:scripts/train.py",
        "def main(): pass",
        {"repo": "repo", "path": "scripts/train.py", "source_type": "script"},
    )

    assert "config_script_relation" not in dependency_score(config, script).labels

    calling_config = Document(
        "repo:.github/workflows/train.yml",
        "steps:\n  - run: python scripts/train.py --config configs/train.yaml",
        {"repo": "repo", "path": ".github/workflows/train.yml", "source_type": "config"},
    )
    reading_script = Document(
        "repo:scripts/train.py",
        "import yaml\nconfig = yaml.safe_load(open('configs/train.yaml'))",
        {"repo": "repo", "path": "scripts/train.py", "source_type": "script"},
    )

    assert "config_script_relation" in dependency_score(calling_config, script).labels
    assert "config_script_relation" in dependency_score(config, reading_script).labels


def test_readme_code_requires_explicit_path_module_or_symbol() -> None:
    readme = Document(
        "repo:README.md",
        "The model is trained with a clean architecture.",
        {"repo": "repo", "path": "README.md", "source_type": "readme"},
    )
    target = Document(
        "repo:src/model.py",
        "class Model: pass",
        {"repo": "repo", "path": "src/model.py", "source_type": "source"},
    )

    assert "readme_code_relation" not in dependency_score(readme, target).labels

    explicit_readme = Document(
        "repo:README.md",
        "Use src/model.py and instantiate Model for training.",
        {"repo": "repo", "path": "README.md", "source_type": "readme"},
    )

    assert "readme_code_relation" in dependency_score(explicit_readme, target).labels


def test_generic_test_source_stem_requires_import_or_path_evidence() -> None:
    source = Document(
        "repo:src/config.py",
        "class TrainConfig: pass",
        {"repo": "repo", "path": "src/config.py", "source_type": "source"},
    )
    weak_test = Document(
        "repo:tests/test_config.py",
        "def test_defaults(): assert config",
        {"repo": "repo", "path": "tests/test_config.py", "source_type": "test"},
    )
    import_test = Document(
        "repo:tests/test_config.py",
        "from src.config import TrainConfig\n\ndef test_defaults(): assert TrainConfig",
        {"repo": "repo", "path": "tests/test_config.py", "source_type": "test"},
    )

    assert "test_source_relation" not in dependency_score(source, weak_test).labels
    assert "test_source_relation" in dependency_score(source, import_test).labels


def test_non_code_explicit_relations_are_strong() -> None:
    api_doc = Document(
        "docs:api#fit",
        "The Trainer.fit API runs optimization.",
        {
            "collection": "docs",
            "document_id": "api",
            "path": "api/fit",
            "source_type": "api_doc",
            "api_name": "Trainer.fit",
        },
    )
    usage = Document(
        "docs:tutorial#train",
        "trainer = Trainer()\ntrainer.fit(data)",
        {
            "collection": "docs",
            "document_id": "tutorial",
            "path": "tutorial/train",
            "source_type": "tutorial",
        },
    )
    citation_source = Document(
        "papers:paper-a#related",
        "We follow the method in Smith2024.",
        {
            "collection": "papers",
            "document_id": "paper-a",
            "path": "paper-a/related",
            "source_type": "paper_section",
        },
    )
    cited = Document(
        "papers:paper-b#abstract",
        "Abstract",
        {
            "collection": "papers",
            "document_id": "paper-b",
            "path": "paper-b/abstract",
            "source_type": "paper_section",
            "citation_key": "Smith2024",
        },
    )

    assert "api_doc_usage_relation" in dependency_score(api_doc, usage).labels
    assert "citation_relation" in dependency_score(citation_source, cited).labels
