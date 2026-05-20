from __future__ import annotations

import re
from typing import Any

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_MODEL_TOKENIZER: Any | None = None
_TOKENIZER_NAME = "simple"


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def configure_tokenizer(
    tokenizer_name: str | None = None,
    *,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> None:
    """Configure the tokenizer used for counting and truncation.

    `tokenize` intentionally remains a lightweight lexical tokenizer for BM25
    and TF-IDF features. This function only affects `count_tokens` and
    `truncate_to_tokens`, which need to match the model context window before
    training.
    """

    global _MODEL_TOKENIZER, _TOKENIZER_NAME

    if not tokenizer_name or tokenizer_name == "simple":
        _MODEL_TOKENIZER = None
        _TOKENIZER_NAME = "simple"
        return

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Using a model tokenizer requires the optional 'transformers' package. "
            "Install it on the training server or use --tokenizer simple."
        ) from exc

    _MODEL_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    _TOKENIZER_NAME = tokenizer_name


def active_tokenizer_name() -> str:
    return _TOKENIZER_NAME


def count_tokens(text: str) -> int:
    """Count tokens for packing statistics.

    By default this uses the lightweight tokenizer. Call `configure_tokenizer`
    with a HuggingFace tokenizer name/path before training-grade packing.
    """

    if _MODEL_TOKENIZER is not None:
        return len(_MODEL_TOKENIZER.encode(text, add_special_tokens=False))
    return len(tokenize(text))


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    if _MODEL_TOKENIZER is not None:
        token_ids = _MODEL_TOKENIZER.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text, 0
        return _MODEL_TOKENIZER.decode(token_ids[:max_tokens]), len(token_ids) - max_tokens

    tokens = TOKEN_PATTERN.findall(text)
    if len(tokens) <= max_tokens:
        return text, 0
    return " ".join(tokens[:max_tokens]), len(tokens) - max_tokens
