from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from dapacking.bm25 import BM25Index
from dapacking.dependency import DEFAULT_WEIGHTS, dependency_score
from dapacking.documents import Document, PackedSample
from dapacking.tokenization import count_tokens, truncate_to_tokens


@dataclass
class PackingConfig:
    method: str
    max_tokens: int
    seed: int = 42
    dependency_weights: dict[str, float] | None = None


class BasePacker(ABC):
    def __init__(self, config: PackingConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    @abstractmethod
    def pack(self, documents: list[Document]) -> list[PackedSample]:
        raise NotImplementedError

    def _make_sample(self, index: int, documents: list[Document], truncated_tokens: int = 0) -> PackedSample:
        content = "\n\n".join(
            f"<doc id=\"{doc.docid}\" path=\"{doc.path}\">\n{doc.content}\n</doc>" for doc in documents
        )
        token_count = count_tokens(content)
        dep_score = average_dependency_score(documents, self.config.dependency_weights)
        return PackedSample(
            sample_id=f"{self.config.method}_{index:06d}",
            method=self.config.method,
            docids=[doc.docid for doc in documents],
            content=content,
            stats={
                "tokens": token_count,
                "num_docs": len(documents),
                "dependency_score": round(dep_score, 4),
                "token_utilization": round(token_count / self.config.max_tokens, 4),
                "truncated_tokens": truncated_tokens,
                "truncation_rate": round(truncated_tokens / max(token_count + truncated_tokens, 1), 4),
            },
        )


class SequentialFillMixin:
    config: PackingConfig

    def _pack_in_order(self, documents: Iterable[Document]) -> list[PackedSample]:
        samples: list[PackedSample] = []
        current: list[Document] = []
        current_tokens = 0
        truncated_tokens = 0

        for document in documents:
            doc_tokens = count_tokens(document.content)
            if doc_tokens > self.config.max_tokens:
                content, overflow = truncate_to_tokens(document.content, self.config.max_tokens)
                document = Document(document.docid, content, document.metadata)
                doc_tokens = count_tokens(document.content)
                truncated_tokens += overflow

            if current and current_tokens + doc_tokens > self.config.max_tokens:
                samples.append(self._make_sample(len(samples), current, truncated_tokens))
                current = []
                current_tokens = 0
                truncated_tokens = 0

            current.append(document)
            current_tokens += doc_tokens

        if current:
            samples.append(self._make_sample(len(samples), current, truncated_tokens))

        return samples


class RandomPacker(SequentialFillMixin, BasePacker):
    def pack(self, documents: list[Document]) -> list[PackedSample]:
        ordered = list(documents)
        self.rng.shuffle(ordered)
        return self._pack_in_order(ordered)


class LengthAwarePacker(SequentialFillMixin, BasePacker):
    def pack(self, documents: list[Document]) -> list[PackedSample]:
        ordered = sorted(documents, key=lambda doc: count_tokens(doc.content), reverse=True)
        return self._pack_in_order(ordered)


class SameRepoPacker(SequentialFillMixin, BasePacker):
    def pack(self, documents: list[Document]) -> list[PackedSample]:
        buckets: dict[str, list[Document]] = defaultdict(list)
        for document in documents:
            buckets[document.repo or "__unknown__"].append(document)

        ordered: list[Document] = []
        for repo in sorted(buckets):
            repo_docs = buckets[repo]
            self.rng.shuffle(repo_docs)
            ordered.extend(repo_docs)
        return self._pack_in_order(ordered)


class BM25Packer(BasePacker):
    def pack(self, documents: list[Document]) -> list[PackedSample]:
        remaining = {document.docid: document for document in documents}
        samples: list[PackedSample] = []

        while remaining:
            anchor = self._select_anchor(list(remaining.values()))
            del remaining[anchor.docid]
            current = [anchor]
            current_tokens = count_tokens(anchor.content)

            while remaining:
                candidate, score = self._best_candidate(anchor, list(remaining.values()))
                if candidate is None or score <= 0:
                    break

                candidate_tokens = count_tokens(candidate.content)
                if current_tokens + candidate_tokens > self.config.max_tokens:
                    break

                current.append(candidate)
                current_tokens += candidate_tokens
                del remaining[candidate.docid]

            samples.append(self._make_sample(len(samples), current))

        return samples

    def _select_anchor(self, documents: list[Document]) -> Document:
        return max(documents, key=lambda doc: count_tokens(doc.content))

    def _best_candidate(
        self,
        anchor: Document,
        candidates: list[Document],
    ) -> tuple[Document | None, float]:
        index = BM25Index(candidates)
        best_doc: Document | None = None
        best_score = -1.0
        for candidate_index, candidate in enumerate(candidates):
            score = index.score(anchor.content, candidate_index)
            if score > best_score:
                best_doc = candidate
                best_score = score
        return best_doc, best_score


class DependencyAwarePacker(BasePacker):
    def pack(self, documents: list[Document]) -> list[PackedSample]:
        remaining = {document.docid: document for document in documents}
        samples: list[PackedSample] = []

        while remaining:
            anchor = self._select_anchor(list(remaining.values()))
            del remaining[anchor.docid]
            current = [anchor]
            current_tokens = count_tokens(anchor.content)

            while remaining:
                candidate, score = self._best_candidate(current, list(remaining.values()))
                if candidate is None or score <= 0:
                    break

                candidate_tokens = count_tokens(candidate.content)
                if current_tokens + candidate_tokens > self.config.max_tokens:
                    break

                current.append(candidate)
                current_tokens += candidate_tokens
                del remaining[candidate.docid]

            samples.append(self._make_sample(len(samples), current))

        return samples

    def _select_anchor(self, documents: list[Document]) -> Document:
        return max(documents, key=lambda doc: (len(_candidate_edges(doc, documents)), count_tokens(doc.content)))

    def _best_candidate(
        self,
        current: list[Document],
        candidates: list[Document],
    ) -> tuple[Document | None, float]:
        best_doc: Document | None = None
        best_score = -1.0

        for candidate in candidates:
            dep = max(
                dependency_score(existing, candidate, self.config.dependency_weights or DEFAULT_WEIGHTS).score
                for existing in current
            )
            if dep <= 0:
                continue
            capacity_bonus = 1.0 - min(
                count_tokens(candidate.content) / max(self.config.max_tokens, 1),
                1.0,
            )
            score = dep + 0.05 * capacity_bonus
            if score > best_score:
                best_doc = candidate
                best_score = score

        return best_doc, best_score


def average_dependency_score(
    documents: list[Document],
    weights: dict[str, float] | None = None,
) -> float:
    if len(documents) < 2:
        return 0.0

    total = 0.0
    count = 0
    for i, source in enumerate(documents):
        for j, target in enumerate(documents):
            if i == j:
                continue
            total += dependency_score(source, target, weights or DEFAULT_WEIGHTS).score
            count += 1
    return total / max(count, 1)


def build_packer(config: PackingConfig) -> BasePacker:
    packers = {
        "random": RandomPacker,
        "length_aware": LengthAwarePacker,
        "same_repo": SameRepoPacker,
        "bm25": BM25Packer,
        "dependency_aware": DependencyAwarePacker,
    }
    try:
        return packers[config.method](config)
    except KeyError as exc:
        available = ", ".join(sorted(packers))
        raise ValueError(f"Unknown packing method '{config.method}'. Available: {available}") from exc


def _candidate_edges(anchor: Document, documents: list[Document]) -> list[Document]:
    return [
        document
        for document in documents
        if document.docid != anchor.docid and dependency_score(anchor, document).score > 0
    ]
