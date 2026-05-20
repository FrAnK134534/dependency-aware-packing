from __future__ import annotations

import math
from collections import Counter

from dapacking.documents import Document
from dapacking.tokenization import tokenize


class TfidfIndex:
    """Small dependency-free semantic proxy for early packing experiments."""

    def __init__(
        self,
        documents: list[Document],
        max_terms_per_document: int = 256,
    ) -> None:
        self.documents = documents
        self.max_terms_per_document = max_terms_per_document
        self.term_counts = [self._term_counts(document) for document in documents]
        self.document_frequencies = self._document_frequencies()
        self.vectors = [self._tfidf_vector(counts) for counts in self.term_counts]
        self.norms = [
            math.sqrt(sum(value * value for value in vector.values())) for vector in self.vectors
        ]

    def cosine(self, left_index: int, right_index: int) -> float:
        left = self.vectors[left_index]
        right = self.vectors[right_index]
        left_norm = self.norms[left_index]
        right_norm = self.norms[right_index]
        if not left or not right or left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        if len(left) > len(right):
            left, right = right, left
        dot = sum(value * right.get(term, 0.0) for term, value in left.items())
        return dot / (left_norm * right_norm)

    def _term_counts(self, document: Document) -> Counter[str]:
        path_hint = document.path.replace("/", " ").replace("_", " ").replace("-", " ")
        terms = [
            term for term in tokenize(f"{path_hint} {document.content}") if _is_semantic_term(term)
        ]
        return Counter(terms)

    def _document_frequencies(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for counts in self.term_counts:
            for term in counts:
                frequencies[term] = frequencies.get(term, 0) + 1
        return frequencies

    def _tfidf_vector(self, counts: Counter[str]) -> dict[str, float]:
        n_docs = max(len(self.documents), 1)
        weighted: dict[str, float] = {}
        for term, count in counts.items():
            idf = math.log((n_docs + 1) / (self.document_frequencies.get(term, 0) + 1)) + 1
            weighted[term] = (1 + math.log(count)) * idf

        if len(weighted) > self.max_terms_per_document:
            weighted = dict(
                sorted(weighted.items(), key=lambda item: item[1], reverse=True)[
                    : self.max_terms_per_document
                ]
            )
        return weighted


def token_jaccard(left: Document, right: Document) -> float:
    left_terms = set(term for term in tokenize(left.content) if _is_semantic_term(term))
    right_terms = set(term for term in tokenize(right.content) if _is_semantic_term(term))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _is_semantic_term(term: str) -> bool:
    return len(term) >= 2 and any(char.isalnum() for char in term)
