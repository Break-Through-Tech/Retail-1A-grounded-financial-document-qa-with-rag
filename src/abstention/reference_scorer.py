"""A model-free reranker for development and CI: IDF-weighted query-term coverage.

``score = sum(idf(t) for distinct query terms t found in the chunk) / sum(idf(t) for
all distinct query terms)``, so it lies in [0, 1]: 1.0 when the chunk contains every
question term, 0.0 when it contains none. A term that appears nowhere in the corpus
gets the highest IDF in the index, so a question about something the corpus never
mentions scores low everywhere.

It exists so the whole tuning path can run without downloading a model. It is **not** a
substitute for the cross-encoder: a threshold tuned on it applies only to it, and the
policy's scorer check enforces that.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.retrieval.bm25 import tokenize

__all__ = ["IdfCoverageScorer"]

VERSION = "1.0.0"


class IdfCoverageScorer:
    """A :class:`~src.retrieval.rerank.PairScorer` that needs no model."""

    def __init__(self, idf: Mapping[str, float]) -> None:
        """Build from a term -> IDF map, e.g. ``BM25Index.idf``.

        Raises:
            ValueError: If ``idf`` is empty.
        """
        if not idf:
            raise ValueError("idf must be non-empty")
        self._idf = idf
        self._unseen = max(idf.values())

    def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Score every chunk text against ``query``; one value per text, in [0, 1]."""
        terms = set(tokenize(query))
        total = sum(self._idf.get(t, self._unseen) for t in terms)
        if total <= 0:
            return np.zeros(len(texts), dtype=np.float64)
        scores = []
        for text in texts:
            present = terms & set(tokenize(text))
            scores.append(sum(self._idf.get(t, self._unseen) for t in present) / total)
        return np.asarray(scores, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        """Identity recorded in the policy; a different version is a different scorer."""
        return {"scorer": "idf-coverage", "version": VERSION}
