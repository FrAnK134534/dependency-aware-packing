from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from dapacking.bm25 import BM25Index
from dapacking.dependency import DEFAULT_WEIGHTS, WEAK_DEPENDENCY_LABELS, dependency_score
from dapacking.documents import Document, PackedSample
from dapacking.edges import DependencyEdge, build_dependency_edges
from dapacking.semantic import TfidfIndex, token_jaccard
from dapacking.tokenization import active_tokenizer_name, configure_tokenizer, count_tokens, truncate_to_tokens


@dataclass
class PackingConfig:
    method: str
    max_tokens: int
    seed: int = 42
    dependency_weights: dict[str, float] | None = None
    min_dependency_score: float = 0.11
    min_similarity_score: float = 0.01
    candidate_pool_size: int = 80
    redundancy_threshold: float = 0.72
    tokenizer_name: str = "simple"
    tokenizer_local_files_only: bool = False
    tokenizer_trust_remote_code: bool = False


class BasePacker(ABC):
    def __init__(self, config: PackingConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        configure_tokenizer(
            config.tokenizer_name,
            local_files_only=config.tokenizer_local_files_only,
            trust_remote_code=config.tokenizer_trust_remote_code,
        )

    @abstractmethod
    def pack(self, documents: list[Document]) -> list[PackedSample]:
        raise NotImplementedError

    def _make_sample(
        self,
        index: int,
        documents: list[Document],
        truncated_tokens: int = 0,
    ) -> PackedSample:
        content = "\n\n".join(format_document_for_window(doc) for doc in documents)
        token_count = count_tokens(content)
        dep_score = average_dependency_score(documents, self.config.dependency_weights)
        semantic_similarity, redundant_pair_rate = average_semantic_metrics(documents)
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
                "semantic_similarity": round(semantic_similarity, 4),
                "redundant_pair_rate": round(redundant_pair_rate, 4),
                "tokenizer": active_tokenizer_name(),
            },
        )

    def _prepare_documents(self, documents: list[Document]) -> tuple[list[Document], dict[str, int]]:
        prepared: list[Document] = []
        truncated_by_docid: dict[str, int] = {}
        for document in documents:
            prepared_document, overflow = truncate_document_for_window(
                document,
                self.config.max_tokens,
            )
            prepared.append(prepared_document)
            truncated_by_docid[prepared_document.docid] = overflow
        return prepared, truncated_by_docid

    def _sample_truncated_tokens(
        self,
        documents: list[Document],
        truncated_by_docid: dict[str, int],
    ) -> int:
        return sum(truncated_by_docid.get(document.docid, 0) for document in documents)


class SequentialFillMixin:
    config: PackingConfig

    def _pack_in_order(self, documents: Iterable[Document]) -> list[PackedSample]:
        documents, truncated_by_docid = self._prepare_documents(list(documents))
        samples: list[PackedSample] = []
        current: list[Document] = []
        current_tokens = 0

        for document in documents:
            doc_tokens = document_window_tokens(document)

            if current and current_tokens + doc_tokens > self.config.max_tokens:
                samples.append(
                    self._make_sample(
                        len(samples),
                        current,
                        self._sample_truncated_tokens(current, truncated_by_docid),
                    )
                )
                current = []
                current_tokens = 0

            current.append(document)
            current_tokens += doc_tokens

        if current:
            samples.append(
                self._make_sample(
                    len(samples),
                    current,
                    self._sample_truncated_tokens(current, truncated_by_docid),
                )
            )

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
        documents, truncated_by_docid = self._prepare_documents(documents)
        remaining = {document.docid: document for document in documents}
        index = BM25Index(documents)
        doc_indices = {document.docid: idx for idx, document in enumerate(documents)}
        token_counts = {document.docid: document_window_tokens(document) for document in documents}
        samples: list[PackedSample] = []

        while remaining:
            anchor = self._select_anchor(list(remaining.values()), token_counts)
            del remaining[anchor.docid]
            current = [anchor]
            current_tokens = token_counts[anchor.docid]

            candidates = same_repo_candidates(anchor, remaining.values())
            ranked_candidates = self._rank_candidates(anchor, candidates, index, doc_indices)
            for candidate, score in ranked_candidates:
                if score <= 0:
                    break
                if candidate.docid not in remaining:
                    continue
                candidate_tokens = token_counts[candidate.docid]
                if current_tokens + candidate_tokens > self.config.max_tokens:
                    continue

                current.append(candidate)
                current_tokens += candidate_tokens
                del remaining[candidate.docid]

            samples.append(
                self._make_sample(
                    len(samples),
                    current,
                    self._sample_truncated_tokens(current, truncated_by_docid),
                )
            )

        return samples

    def _select_anchor(self, documents: list[Document], token_counts: dict[str, int]) -> Document:
        return max(documents, key=lambda doc: token_counts[doc.docid])

    def _rank_candidates(
        self,
        anchor: Document,
        candidates: list[Document],
        index: BM25Index,
        doc_indices: dict[str, int],
    ) -> list[tuple[Document, float]]:
        query_terms = Counter(index.doc_tokens[doc_indices[anchor.docid]])
        ranked = [
            (candidate, index.score_terms(query_terms, doc_indices[candidate.docid]))
            for candidate in candidates
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        if self.config.candidate_pool_size > 0:
            return ranked[: self.config.candidate_pool_size]
        return ranked


class SemanticPacker(BasePacker):
    def pack(self, documents: list[Document]) -> list[PackedSample]:
        documents, truncated_by_docid = self._prepare_documents(documents)
        remaining = {document.docid: document for document in documents}
        index = TfidfIndex(documents)
        doc_indices = {document.docid: idx for idx, document in enumerate(documents)}
        token_counts = {document.docid: document_window_tokens(document) for document in documents}
        samples: list[PackedSample] = []

        while remaining:
            anchor = self._select_anchor(list(remaining.values()), token_counts)
            del remaining[anchor.docid]
            current = [anchor]
            current_tokens = token_counts[anchor.docid]

            candidates = same_repo_candidates(anchor, remaining.values())
            ranked_candidates = self._rank_candidates(anchor, candidates, index, doc_indices)
            for candidate, score in ranked_candidates:
                if score < self.config.min_similarity_score:
                    break
                if candidate.docid not in remaining:
                    continue
                candidate_tokens = token_counts[candidate.docid]
                if current_tokens + candidate_tokens > self.config.max_tokens:
                    continue

                current.append(candidate)
                current_tokens += candidate_tokens
                del remaining[candidate.docid]

            samples.append(
                self._make_sample(
                    len(samples),
                    current,
                    self._sample_truncated_tokens(current, truncated_by_docid),
                )
            )

        return samples

    def _select_anchor(self, documents: list[Document], token_counts: dict[str, int]) -> Document:
        return max(documents, key=lambda doc: token_counts[doc.docid])

    def _rank_candidates(
        self,
        anchor: Document,
        candidates: list[Document],
        index: TfidfIndex,
        doc_indices: dict[str, int],
    ) -> list[tuple[Document, float]]:
        anchor_index = doc_indices[anchor.docid]
        ranked = [
            (candidate, index.cosine(anchor_index, doc_indices[candidate.docid]))
            for candidate in candidates
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        if self.config.candidate_pool_size > 0:
            return ranked[: self.config.candidate_pool_size]
        return ranked


class DataSculptLitePacker(BasePacker):
    """Semantic packing with lightweight integrity, efficiency, and redundancy terms."""

    def pack(self, documents: list[Document]) -> list[PackedSample]:
        documents, truncated_by_docid = self._prepare_documents(documents)
        remaining = {document.docid: document for document in documents}
        index = TfidfIndex(documents)
        doc_indices = {document.docid: idx for idx, document in enumerate(documents)}
        token_counts = {document.docid: document_window_tokens(document) for document in documents}
        samples: list[PackedSample] = []

        while remaining:
            anchor = self._select_anchor(list(remaining.values()), token_counts)
            del remaining[anchor.docid]
            current = [anchor]
            current_tokens = token_counts[anchor.docid]

            while remaining:
                candidates = same_repo_candidates(anchor, remaining.values())
                if not candidates:
                    break
                candidate, score = self._best_candidate(
                    current,
                    candidates,
                    current_tokens,
                    index,
                    doc_indices,
                    token_counts,
                )
                if candidate is None or score < self.config.min_similarity_score:
                    break

                current.append(candidate)
                current_tokens += token_counts[candidate.docid]
                del remaining[candidate.docid]

            samples.append(
                self._make_sample(
                    len(samples),
                    current,
                    self._sample_truncated_tokens(current, truncated_by_docid),
                )
            )

        return samples

    def _select_anchor(self, documents: list[Document], token_counts: dict[str, int]) -> Document:
        return max(documents, key=lambda doc: token_counts[doc.docid])

    def _best_candidate(
        self,
        current: list[Document],
        candidates: list[Document],
        current_tokens: int,
        index: TfidfIndex,
        doc_indices: dict[str, int],
        token_counts: dict[str, int],
    ) -> tuple[Document | None, float]:
        best_doc: Document | None = None
        best_score = -1.0
        anchor = current[0]
        anchor_repo = anchor.repo
        current_indices = [doc_indices[document.docid] for document in current]
        if self.config.candidate_pool_size > 0 and len(candidates) > self.config.candidate_pool_size:
            anchor_index = doc_indices[anchor.docid]
            candidates = sorted(
                candidates,
                key=lambda candidate: index.cosine(anchor_index, doc_indices[candidate.docid]),
                reverse=True,
            )[: self.config.candidate_pool_size]

        for candidate in candidates:
            candidate_tokens = token_counts[candidate.docid]
            if current_tokens + candidate_tokens > self.config.max_tokens:
                continue

            candidate_index = doc_indices[candidate.docid]
            similarities = [index.cosine(candidate_index, existing_index) for existing_index in current_indices]
            anchor_similarity = similarities[0]
            avg_similarity = sum(similarities) / max(len(similarities), 1)
            max_similarity = max(similarities) if similarities else 0.0
            if (
                anchor_similarity < self.config.min_similarity_score
                and avg_similarity < self.config.min_similarity_score
            ):
                continue

            utilization_after = (current_tokens + candidate_tokens) / max(self.config.max_tokens, 1)
            fit_bonus = 1.0 - abs(0.92 - min(utilization_after, 1.0))
            same_repo_bonus = 1.0 if anchor_repo and anchor_repo == candidate.repo else 0.0
            redundancy_penalty = max(0.0, max_similarity - self.config.redundancy_threshold)
            semantic_score = 0.7 * anchor_similarity + 0.3 * avg_similarity
            score = (
                semantic_score
                + 0.08 * fit_bonus
                + 0.05 * same_repo_bonus
                - 0.35 * redundancy_penalty
            )

            if score > best_score:
                best_doc = candidate
                best_score = score

        return best_doc, best_score


class DependencyAwarePacker(BasePacker):
    excluded_dependency_labels: frozenset[str] = frozenset()
    token_fit_fill: bool = False
    token_fit_target_utilization: float = 0.9
    strong_first: bool = False

    def pack(self, documents: list[Document]) -> list[PackedSample]:
        documents, truncated_by_docid = self._prepare_documents(documents)
        remaining = {document.docid: document for document in documents}
        token_counts = {document.docid: document_window_tokens(document) for document in documents}
        dependency_edges = build_dependency_edges(
            documents,
            weights=self.config.dependency_weights or DEFAULT_WEIGHTS,
            min_score=self.config.min_dependency_score,
        )
        dependency_scores = self._dependency_scores_from_edges(dependency_edges)
        strong_dependency_scores = (
            self._dependency_scores_from_edges(dependency_edges, excluded_labels=WEAK_DEPENDENCY_LABELS)
            if self.strong_first
            else dependency_scores
        )
        samples: list[PackedSample] = []

        while remaining:
            anchor_scores = strong_dependency_scores if self.strong_first else dependency_scores
            anchor = self._select_anchor(list(remaining.values()), token_counts, anchor_scores)
            del remaining[anchor.docid]
            current = [anchor]
            current_tokens = token_counts[anchor.docid]
            current_repo = anchor.repo

            phase_scores = [strong_dependency_scores, dependency_scores] if self.strong_first else [dependency_scores]
            for scores in phase_scores:
                current, current_tokens = self._fill_by_dependency(
                    current,
                    current_tokens,
                    remaining,
                    current_repo,
                    token_counts,
                    scores,
                )

            if self.token_fit_fill:
                current, current_tokens = self._fill_by_token_fit(
                    current,
                    current_tokens,
                    remaining,
                    token_counts,
                    dependency_scores,
                )

            samples.append(
                self._make_sample(
                    len(samples),
                    current,
                    self._sample_truncated_tokens(current, truncated_by_docid),
                )
            )

        return samples

    def _build_dependency_scores(
        self,
        documents: list[Document],
        excluded_labels: frozenset[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        edges = build_dependency_edges(
            documents,
            weights=self.config.dependency_weights or DEFAULT_WEIGHTS,
            min_score=self.config.min_dependency_score,
        )
        return self._dependency_scores_from_edges(edges, excluded_labels)

    def _dependency_scores_from_edges(
        self,
        edges: list[DependencyEdge],
        excluded_labels: frozenset[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        scores: dict[str, dict[str, float]] = defaultdict(dict)
        for edge in edges:
            labels = tuple(str(label) for label in edge.metadata.get("labels", []))
            if not labels:
                labels = tuple(label for label in edge.relation.split("+") if label)
            score = self._filtered_dependency_score(labels, excluded_labels)
            if score >= self.config.min_dependency_score:
                scores[edge.source_docid][edge.target_docid] = score
        return scores

    def _filtered_dependency_score(
        self,
        labels: tuple[str, ...],
        excluded_labels: frozenset[str] | None = None,
    ) -> float:
        weights = self.config.dependency_weights or DEFAULT_WEIGHTS
        excluded_labels = self.excluded_dependency_labels | (excluded_labels or frozenset())
        filtered_labels = [
            label for label in labels if label not in excluded_labels
        ]
        return sum(weights.get(label, 0.0) for label in filtered_labels)

    def _select_anchor(
        self,
        documents: list[Document],
        token_counts: dict[str, int],
        dependency_scores: dict[str, dict[str, float]],
    ) -> Document:
        return max(
            documents,
            key=lambda doc: (len(dependency_scores.get(doc.docid, {})), token_counts[doc.docid]),
        )

    def _fill_by_dependency(
        self,
        current: list[Document],
        current_tokens: int,
        remaining: dict[str, Document],
        current_repo: str,
        token_counts: dict[str, int],
        dependency_scores: dict[str, dict[str, float]],
    ) -> tuple[list[Document], int]:
        while remaining:
            candidates = [
                document
                for document in remaining.values()
                if not current_repo or document.repo == current_repo
            ]
            if not candidates:
                break

            candidate, score = self._best_candidate(
                current,
                candidates,
                current_tokens,
                token_counts,
                dependency_scores,
            )
            if candidate is None or score <= 0:
                break

            current.append(candidate)
            current_tokens += token_counts[candidate.docid]
            del remaining[candidate.docid]

        return current, current_tokens

    def _best_candidate(
        self,
        current: list[Document],
        candidates: list[Document],
        current_tokens: int,
        token_counts: dict[str, int],
        dependency_scores: dict[str, dict[str, float]],
    ) -> tuple[Document | None, float]:
        best_doc: Document | None = None
        best_score = -1.0

        for candidate in candidates:
            candidate_tokens = token_counts[candidate.docid]
            if current_tokens + candidate_tokens > self.config.max_tokens:
                continue

            dep = max(
                dependency_scores.get(existing.docid, {}).get(candidate.docid, 0.0)
                for existing in current
            )
            if dep < self.config.min_dependency_score:
                continue
            capacity_bonus = 1.0 - min(
                candidate_tokens / max(self.config.max_tokens, 1),
                1.0,
            )
            score = dep + 0.05 * capacity_bonus
            if score > best_score:
                best_doc = candidate
                best_score = score

        return best_doc, best_score

    def _fill_by_token_fit(
        self,
        current: list[Document],
        current_tokens: int,
        remaining: dict[str, Document],
        token_counts: dict[str, int],
        dependency_scores: dict[str, dict[str, float]],
    ) -> tuple[list[Document], int]:
        anchor_repo = current[0].repo

        while (
            remaining
            and current_tokens / max(self.config.max_tokens, 1) < self.token_fit_target_utilization
        ):
            candidates = [
                document
                for document in remaining.values()
                if not anchor_repo or document.repo == anchor_repo
            ]
            if not candidates:
                break

            candidate, score = self._best_token_fit_candidate(
                current,
                candidates,
                current_tokens,
                token_counts,
                dependency_scores,
            )
            if candidate is None or score <= 0:
                break

            current.append(candidate)
            current_tokens += token_counts[candidate.docid]
            del remaining[candidate.docid]

        return current, current_tokens

    def _best_token_fit_candidate(
        self,
        current: list[Document],
        candidates: list[Document],
        current_tokens: int,
        token_counts: dict[str, int],
        dependency_scores: dict[str, dict[str, float]],
    ) -> tuple[Document | None, float]:
        remaining_budget = self.config.max_tokens - current_tokens
        if remaining_budget <= 0:
            return None, -1.0

        anchor_repo = current[0].repo
        best_doc: Document | None = None
        best_score = -1.0

        for candidate in candidates:
            candidate_tokens = token_counts[candidate.docid]
            if candidate_tokens > remaining_budget:
                continue

            dependency_bonus = max(
                dependency_scores.get(existing.docid, {}).get(candidate.docid, 0.0)
                for existing in current
            )
            same_parent_bonus = (
                1.0 if any(_same_parent(existing, candidate) for existing in current) else 0.0
            )
            same_repo_bonus = 1.0 if anchor_repo and anchor_repo == candidate.repo else 0.0
            fit_score = candidate_tokens / max(remaining_budget, 1)
            redundancy_penalty = max(token_jaccard(existing, candidate) for existing in current)
            score = (
                0.65 * fit_score
                + 0.20 * dependency_bonus
                + 0.10 * same_parent_bonus
                + 0.08 * same_repo_bonus
                - 0.18 * redundancy_penalty
            )

            if score > best_score:
                best_doc = candidate
                best_score = score

        return best_doc, best_score


class DependencyAwareV2TokenFitPacker(DependencyAwarePacker):
    token_fit_fill = True


class DependencyAwareV2StrongFirstPacker(DependencyAwarePacker):
    token_fit_fill = True
    strong_first = True


class DependencyAwareNoSameDirectoryPacker(DependencyAwarePacker):
    excluded_dependency_labels = frozenset({"same_directory"})


class DependencyAwareNoSameRepoPacker(DependencyAwarePacker):
    excluded_dependency_labels = frozenset({"same_repo"})


class DependencyAwareStrongEdgesOnlyPacker(DependencyAwarePacker):
    excluded_dependency_labels = WEAK_DEPENDENCY_LABELS


def _same_parent(left: Document, right: Document) -> bool:
    return bool(left.repo and left.repo == right.repo and left.parent == right.parent)


def same_repo_candidates(anchor: Document, candidates: Iterable[Document]) -> list[Document]:
    if not anchor.repo:
        return list(candidates)
    return [candidate for candidate in candidates if candidate.repo == anchor.repo]


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


def format_document_for_window(document: Document) -> str:
    return f"<doc id=\"{document.docid}\" path=\"{document.path}\">\n{document.content}\n</doc>"


def document_window_tokens(document: Document) -> int:
    return count_tokens(format_document_for_window(document))


def truncate_document_for_window(document: Document, max_tokens: int) -> tuple[Document, int]:
    if document_window_tokens(document) <= max_tokens:
        return document, 0

    wrapper_tokens = count_tokens(format_document_for_window(Document(document.docid, "", document.metadata)))
    content_budget = max(max_tokens - wrapper_tokens, 1)
    content, overflow = truncate_to_tokens(document.content, content_budget)
    return Document(document.docid, content, document.metadata), overflow


def average_semantic_metrics(documents: list[Document]) -> tuple[float, float]:
    if len(documents) < 2:
        return 0.0, 0.0

    total = 0.0
    redundant = 0
    count = 0
    for i, source in enumerate(documents):
        for target in documents[i + 1 :]:
            similarity = token_jaccard(source, target)
            total += similarity
            if similarity >= 0.72:
                redundant += 1
            count += 1
    return total / max(count, 1), redundant / max(count, 1)


def build_packer(config: PackingConfig) -> BasePacker:
    packers = {
        "random": RandomPacker,
        "length_aware": LengthAwarePacker,
        "same_repo": SameRepoPacker,
        "bm25": BM25Packer,
        "semantic": SemanticPacker,
        "datasculpt_lite": DataSculptLitePacker,
        "dependency_aware": DependencyAwarePacker,
        "dependency_aware_v2_token_fit": DependencyAwareV2TokenFitPacker,
        "dependency_aware_v2_strong_first": DependencyAwareV2StrongFirstPacker,
        "dependency_aware_no_same_directory": DependencyAwareNoSameDirectoryPacker,
        "dependency_aware_no_same_repo": DependencyAwareNoSameRepoPacker,
        "dependency_aware_strong_edges_only": DependencyAwareStrongEdgesOnlyPacker,
    }
    try:
        return packers[config.method](config)
    except KeyError as exc:
        available = ", ".join(sorted(packers))
        raise ValueError(f"Unknown packing method '{config.method}'. Available: {available}") from exc
