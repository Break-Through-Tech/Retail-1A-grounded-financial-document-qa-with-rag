"""Retrieve-and-rerank for the FinanceBench pipeline.

A first stage (BM25, dense, or both fused with reciprocal rank fusion) produces
``candidate_k`` candidates; a cross-encoder then scores each ``(question, chunk)`` pair
and reorders them. ``sentence_transformers`` is imported only inside
:meth:`CrossEncoderScorer.load`.
"""

from __future__ import annotations

import platform
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from src.retrieval.embeddings import REVISION_PATTERN

__all__ = [
    "Candidate",
    "CrossEncoderScorer",
    "FusedResult",
    "PairScorer",
    "RRF_K",
    "RerankResult",
    "RetrieveAndRerank",
    "SearchOutput",
    "rerank",
    "rrf_fuse",
]

RERANK_CONFIG_VERSION = "1.0.0"
RRF_K = 60


class PairScorer(Protocol):
    """Scores (question, chunk text) pairs; higher means more relevant."""

    def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray: ...

    def describe(self) -> dict[str, Any]: ...


def _check_int(name: str, value: object, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")


def _check_query(query: object) -> None:
    if not isinstance(query, str):
        raise TypeError(f"query must be str, got {type(query).__name__}")
    if not query.strip():
        raise ValueError("query must contain non-whitespace text")


def _check_methods(obj: object, label: str, methods: Sequence[str]) -> None:
    missing = [m for m in methods if not callable(getattr(obj, m, None))]
    if missing:
        raise TypeError(f"{label} is missing required method(s): {missing}")


@dataclass(frozen=True)
class Candidate:
    """A first-stage hit: its text and where the first stage ranked it."""

    chunk_id: str
    text: str
    retrieval_rank: int
    retrieval_score: float


@dataclass(frozen=True)
class RerankResult:
    """A reranked chunk with its first-stage rank and score kept for the audit trail."""

    rank: int
    chunk_id: str
    score: float
    retrieval_rank: int
    retrieval_score: float


@dataclass(frozen=True)
class FusedResult:
    """A chunk in a reciprocal-rank-fusion ranking."""

    rank: int
    chunk_id: str
    score: float


def rrf_fuse(rankings: Sequence[Sequence[Any]], *, rrf_k: int = RRF_K) -> list[FusedResult]:
    """Fuse ranked lists: a chunk scores the sum of ``1 / (rrf_k + rank)`` over lists.

    Items need a ``chunk_id``; rank comes from list order. Ties break on ``chunk_id``.
    """
    _check_int("rrf_k", rrf_k)
    if isinstance(rankings, str) or not rankings:
        raise ValueError("rankings must be a non-empty sequence of ranked lists")
    scores: dict[str, float] = {}
    for position, ranking in enumerate(rankings):
        ids = [item.chunk_id for item in ranking]
        repeated = sorted(c for c, n in Counter(ids).items() if n > 1)
        if repeated:
            raise ValueError(f"ranking {position} repeats chunk_id values: {repeated}")
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [FusedResult(r, c, s) for r, (c, s) in enumerate(ordered, start=1)]


def rerank(
    query: str, candidates: Sequence[Candidate], scorer: PairScorer, top_k: int
) -> list[RerankResult]:
    """Reorder ``candidates`` by the scorer's relevance to ``query`` and keep ``top_k``.

    The scorer is called once with every candidate text, and not at all if there are none.
    Ties keep first-stage order, then break on ``chunk_id``.
    """
    _check_query(query)
    _check_int("top_k", top_k)
    _check_methods(scorer, "scorer", ("score_pairs", "describe"))
    if not candidates:
        return []
    repeated = sorted(c for c, n in Counter(x.chunk_id for x in candidates).items() if n > 1)
    if repeated:
        raise ValueError(f"candidates repeat chunk_id values: {repeated}")
    if any(not c.text.strip() for c in candidates):
        raise ValueError("candidates must not have blank text")

    raw = scorer.score_pairs(query, [c.text for c in candidates])
    if not isinstance(raw, np.ndarray) or not (
        np.issubdtype(raw.dtype, np.floating) or np.issubdtype(raw.dtype, np.integer)
    ):
        raise TypeError("scorer must return a real-valued numpy ndarray")
    if raw.shape != (len(candidates),):
        raise ValueError(f"scorer must return shape ({len(candidates)},), got {raw.shape}")
    scores = raw.astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("scorer returned NaN or infinite scores")

    paired = sorted(
        zip(scores.tolist(), candidates),
        key=lambda p: (-p[0], p[1].retrieval_rank, p[1].chunk_id),
    )
    return [
        RerankResult(rank, c.chunk_id, score, c.retrieval_rank, c.retrieval_score)
        for rank, (score, c) in enumerate(paired[:top_k], start=1)
    ]


@dataclass(frozen=True)
class SearchOutput:
    """Reranked results plus the cost of producing them."""

    results: list[RerankResult]
    n_candidates: int
    scorer_calls: int
    retrieve_ms: float
    rerank_ms: float

    @property
    def latency_ms(self) -> float:
        return self.retrieve_ms + self.rerank_ms


class RetrieveAndRerank:
    """First-stage retrieval, RRF fusion when there are several retrievers, then reranking."""

    def __init__(
        self,
        retrievers: Mapping[str, Any],
        texts: Mapping[str, str],
        scorer: PairScorer,
        *,
        candidate_k: int = 50,
        rrf_k: int = RRF_K,
        per_retriever_k: int | None = None,
    ) -> None:
        """Assemble a pipeline.

        Args:
            retrievers: Name -> object with ``retrieve(query, k)`` returning items that
                have ``chunk_id`` and ``score`` (``BM25Index`` and ``DenseRetriever`` fit).
                One retriever means no fusion; two or more are fused with RRF.
            texts: ``chunk_id`` -> chunk text for every chunk a retriever can return.
            scorer: A :class:`PairScorer`, usually :class:`CrossEncoderScorer`.
            candidate_k: Candidates handed to the reranker (untuned default).
            rrf_k: RRF constant.
            per_retriever_k: Depth taken from each retriever before fusion;
                defaults to ``candidate_k``.
        """
        if not isinstance(retrievers, Mapping) or not retrievers:
            raise ValueError("retrievers must be a non-empty mapping of name -> retriever")
        for name, retriever in retrievers.items():
            _check_methods(retriever, f"retriever {name!r}", ("retrieve",))
        _check_methods(scorer, "scorer", ("score_pairs", "describe"))
        _check_int("candidate_k", candidate_k)
        _check_int("rrf_k", rrf_k)
        if per_retriever_k is not None:
            _check_int("per_retriever_k", per_retriever_k)
        self._retrievers = dict(retrievers)
        self._texts = texts
        self._scorer = scorer
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._depth = per_retriever_k or candidate_k

    def candidates(self, query: str) -> list[Candidate]:
        """Run the first stage only; use it to measure recall@``candidate_k``."""
        _check_query(query)
        if len(self._retrievers) == 1:
            (retriever,) = self._retrievers.values()
            hits = retriever.retrieve(query, self._candidate_k)
            ranked = [(h.chunk_id, float(h.score)) for h in hits][: self._candidate_k]
        else:
            lists = [
                list(self._retrievers[name].retrieve(query, self._depth))
                for name in sorted(self._retrievers)
            ]
            lists = [lst for lst in lists if lst]
            fused = rrf_fuse(lists, rrf_k=self._rrf_k) if lists else []
            ranked = [(f.chunk_id, f.score) for f in fused[: self._candidate_k]]

        out = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            if chunk_id not in self._texts:
                raise ValueError(f"no text for retrieved chunk_id {chunk_id!r}")
            out.append(Candidate(chunk_id, self._texts[chunk_id], rank, score))
        return out

    def search(self, query: str, top_k: int = 10) -> SearchOutput:
        """Retrieve candidates, rerank them, and return the top ``top_k`` with timings."""
        _check_query(query)
        _check_int("top_k", top_k)
        started = time.perf_counter()
        candidates = self.candidates(query)
        retrieved = time.perf_counter()
        results = rerank(query, candidates, self._scorer, top_k)
        finished = time.perf_counter()
        return SearchOutput(
            results=results,
            n_candidates=len(candidates),
            scorer_calls=1 if candidates else 0,
            retrieve_ms=(retrieved - started) * 1000.0,
            rerank_ms=(finished - retrieved) * 1000.0,
        )

    def config_record(self) -> dict[str, Any]:
        """Return the configuration and scorer provenance as a JSON-serializable dict."""
        fused = len(self._retrievers) > 1
        return {
            "config_version": RERANK_CONFIG_VERSION,
            "first_stage": sorted(self._retrievers),
            "fusion": "reciprocal_rank_fusion" if fused else "none",
            "rrf_k": self._rrf_k if fused else None,
            "per_retriever_k": self._depth if fused else None,
            "candidate_k": self._candidate_k,
            "tie_order": "(-rerank_score, retrieval_rank, chunk_id)",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scorer": dict(self._scorer.describe()),
        }


class CrossEncoderScorer:
    """A :class:`PairScorer` backed by a pinned Sentence Transformers ``CrossEncoder``.

    The model ID and a full 40-character commit revision are required. All pairs are
    scored in one ``predict`` batch and the model's own activation is kept.

    Cautions: text beyond the model's ``max_seq_length`` is silently truncated, and
    scores are comparable only within one model.
    """

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        revision: str,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int | None = None,
    ) -> None:
        """Wrap a loaded ``CrossEncoder`` (or any object with ``predict``)."""
        if not REVISION_PATTERN.fullmatch(revision):
            raise ValueError(
                f"revision must be a full 40-character lowercase commit hash, got {revision!r}"
            )
        _check_int("batch_size", batch_size)
        _check_methods(model, "model", ("predict",))
        if max_seq_length is not None:
            # sentence-transformers renamed ``max_length`` to ``max_seq_length`` in 5.4;
            # set whichever the installed version exposes.
            attribute = "max_seq_length" if hasattr(model, "max_seq_length") else "max_length"
            setattr(model, attribute, max_seq_length)
        self._model = model
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._batch_size = batch_size

    @classmethod
    def load(
        cls,
        model_id: str,
        revision: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int | None = None,
    ) -> CrossEncoderScorer:
        """Load a pinned cross-encoder; a bad revision fails before any import."""
        if not REVISION_PATTERN.fullmatch(revision):
            raise ValueError(
                f"revision must be a full 40-character lowercase commit hash, got {revision!r}"
            )
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_id, revision=revision, device=device, trust_remote_code=False)
        return cls(
            model,
            model_id=model_id,
            revision=revision,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )

    def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Return a float64 array with one score per text, in input order."""
        if not texts:
            raise ValueError("texts must contain at least one chunk text")
        raw = self._model.predict(
            [(query, t) for t in texts],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw, dtype=np.float64)
        if scores.shape != (len(texts),):
            raise ValueError(f"model must return shape ({len(texts)},), got {scores.shape}")
        return scores

    def describe(self) -> dict[str, Any]:
        """Return the model identity and settings, for provenance."""
        return {
            "scorer": "sentence-transformers-cross-encoder",
            "model_id": self._model_id,
            "revision": self._revision,
            "device": self._device,
            "batch_size": self._batch_size,
            "max_seq_length": getattr(
                self._model, "max_seq_length", getattr(self._model, "max_length", None)
            ),
        }
