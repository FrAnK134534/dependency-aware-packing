from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Lightweight tokenizer for packing statistics.

    Model-specific tokenizers should replace this before final training runs.
    The lightweight counter keeps early packing experiments dependency-light.
    """

    return len(TOKEN_PATTERN.findall(text))


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, int]:
    tokens = TOKEN_PATTERN.findall(text)
    if len(tokens) <= max_tokens:
        return text, 0
    return " ".join(tokens[:max_tokens]), len(tokens) - max_tokens
