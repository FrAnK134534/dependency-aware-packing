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
