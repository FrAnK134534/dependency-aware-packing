from __future__ import annotations

import math
from collections import Counter

from dapacking.documents import Document
from dapacking.tokenization import tokenize


class BM25Index:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document.content) for document in documents]
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_freqs = self._document_frequencies()

    def score(self, query: str, document_index: int) -> float:
        query_terms = Counter(tokenize(query))
        freqs = self.term_freqs[document_index]
        doc_length = self.doc_lengths[document_index]
        score = 0.0

        for term, query_count in query_terms.items():
            term_frequency = freqs.get(term, 0)
            if term_frequency == 0:
                continue

            idf = self._idf(term)
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
            )
            score += query_count * idf * (term_frequency * (self.k1 + 1)) / denominator

        return score

    def _document_frequencies(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                frequencies[term] = frequencies.get(term, 0) + 1
        return frequencies

    def _idf(self, term: str) -> float:
        n_docs = len(self.documents)
        doc_frequency = self.doc_freqs.get(term, 0)
        return math.log(1 + (n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
