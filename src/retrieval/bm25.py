"""Lucene-style Okapi BM25 keyword retrieval for the FinanceBench baseline.

Issue #9 of the Retail-1A challenge project. Implements the fixed configuration in
the Issue #9 BM25 decision record (v1.0.0). This module scores and ranks chunks; it
does not build them and it does not evaluate them.

Design notes that the final report depends on:

* The configuration is fixed and deliberately untuned. ``k1``, ``b`` and the IDF
  formula follow Apache Lucene's ``BM25Similarity`` defaults, so the baseline is a
  fixed reference point for the +10 pp Hit@5 gate rather than a tuned retriever.
  There is no parameter to sweep.
* IDF is ``ln(1 + (N - df + 0.5) / (df + 0.5))``. It is positive for every
  ``df <= N`` and strictly decreasing in ``df``, so no floor is applied. This is not
  ``rank_bm25``'s Okapi IDF, which goes negative and is then floored.
* The corpus is whatever ``(chunk_id, text)`` pairs the caller supplies. Which
  chunks exist, how they are built, and which of them a question may search are
  Issue #8 decisions; N, df and avgdl are computed over the supplied input only.
* Results carry ``rank``, ``chunk_id`` and ``score`` and nothing else. Chunk
  metadata is joined downstream by ``chunk_id``. ``retrieve`` takes the question
  text and ``k`` only, so no answer or gold evidence can reach the scorer.
* Ranking is deterministic: scores accumulate in question-token order, chunks
  scoring 0 are dropped, and ties break on ascending ``chunk_id``.
"""

from __future__ import annotations

import math
import platform
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

__all__ = [
    "B",
    "BM25Index",
    "BM25_CONFIG_VERSION",
    "IDF_FORMULA",
    "K1",
    "RankedResult",
    "THOUSANDS_SEPARATOR",
    "TOKEN_PATTERN",
    "UNICODE_NORMALIZATION",
    "VARIANT",
    "bm25_idf",
    "config_record",
    "tokenize",
]

#: Version of the fixed configuration below. Any change to a rule is a new version.
BM25_CONFIG_VERSION = "1.0.0"
VARIANT = "lucene-style-okapi-bm25"

#: Term-frequency saturation. Apache Lucene ``BM25Similarity`` default.
K1 = 1.2
#: Document-length normalization. Apache Lucene ``BM25Similarity`` default.
B = 0.75
IDF_FORMULA = "ln(1 + (N - df + 0.5) / (df + 0.5))"

UNICODE_NORMALIZATION = "NFKC"
#: A comma between digit groups, removed before tokenizing: "1,234.5" -> "1234.5".
THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
#: A decimal number, or a maximal run of letters and digits. Everything else
#: separates tokens and is dropped.
TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)+|[^\W_]+")


def config_record() -> dict[str, Any]:
    """Return the fixed retrieval configuration as a serializable record.

    The interpreter and Unicode database versions are included because NFKC,
    ``str.casefold`` and the regex ``\\w`` class all depend on the Unicode database.

    Returns:
        A JSON-serializable dict with no timestamp, identical across calls.
    """
    return {
        "config_version": BM25_CONFIG_VERSION,
        "variant": VARIANT,
        "k1": K1,
        "b": B,
        "idf": IDF_FORMULA,
        "idf_floor": "none",
        "unicode_normalization": UNICODE_NORMALIZATION,
        "case": "casefold",
        "thousands_separator_pattern": THOUSANDS_SEPARATOR.pattern,
        "token_pattern": TOKEN_PATTERN.pattern,
        "stopwords": "none",
        "stemming": "none",
        "ngrams": 1,
        "query_term_repeats": "counted",
        "zero_scores": "excluded",
        "tie_order": "(-score, chunk_id)",
        "tuning": "none",
        "python_version": platform.python_version(),
        "unicode_version": unicodedata.unidata_version,
    }


def tokenize(text: str) -> list[str]:
    """Tokenize ``text`` with the fixed baseline rules.

    The same function tokenizes chunks and questions. Steps: NFKC normalization,
    ``casefold``, removal of commas between digit groups, then every match of
    :data:`TOKEN_PATTERN`. No stopwords are removed and nothing is stemmed.

    Args:
        text: Chunk or question text.

    Returns:
        Tokens in text order, repeats kept. Empty if ``text`` has no token.

    Raises:
        TypeError: If ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    folded = unicodedata.normalize(UNICODE_NORMALIZATION, text).casefold()
    return TOKEN_PATTERN.findall(THOUSANDS_SEPARATOR.sub("", folded))


def bm25_idf(df: int, n_docs: int) -> float:
    """Return the Lucene BM25 IDF of a term found in ``df`` of ``n_docs`` chunks.

    Args:
        df: Number of chunks containing the term.
        n_docs: Number of chunks in the index.

    Returns:
        ``ln(1 + (n_docs - df + 0.5) / (df + 0.5))``, always positive.

    Raises:
        TypeError: If either argument is not an ``int`` (``bool`` is rejected).
        ValueError: If ``df`` does not lie in ``[1, n_docs]``.
    """
    for name, value in (("df", df), ("n_docs", n_docs)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if not 1 <= df <= n_docs:
        raise ValueError(f"df must lie in [1, n_docs={n_docs}]; got {df}")
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


@dataclass(frozen=True)
class RankedResult:
    """One retrieved chunk.

    Attributes:
        rank: 1-based position in the ranking.
        chunk_id: Identifier of the retrieved chunk, as supplied to the index.
        score: BM25 score, always greater than 0.
    """

    rank: int
    chunk_id: str
    score: float


@dataclass(frozen=True, eq=False)
class BM25Index:
    """An immutable BM25 index over a fixed set of chunks. Build it with :meth:`build`.

    Attributes:
        chunk_ids: Chunk identifiers in ascending order; position is the internal
            document index.
        doc_lengths: Token count of each chunk, aligned with ``chunk_ids``.
        avgdl: Mean token count over all chunks, including chunks with no token.
        postings: Term -> ``(document index, term frequency)`` pairs, ascending by
            document index.
        idf: Term -> :func:`bm25_idf` over this index.
    """

    chunk_ids: tuple[str, ...]
    doc_lengths: tuple[int, ...]
    avgdl: float
    postings: Mapping[str, tuple[tuple[int, int], ...]] = field(repr=False)
    idf: Mapping[str, float] = field(repr=False)

    @property
    def n_docs(self) -> int:
        """Number of chunks in the index, the N of the IDF formula."""
        return len(self.chunk_ids)

    @classmethod
    def build(cls, documents: Iterable[tuple[str, str]]) -> BM25Index:
        """Index ``(chunk_id, text)`` pairs.

        Chunks are ordered by ``chunk_id`` before indexing, so the input order never
        affects the result. A chunk whose text has no token is still indexed: it
        counts toward N and avgdl but can never be retrieved.

        Args:
            documents: ``(chunk_id, text)`` tuples. A mapping is rejected because
                iterating it yields keys, which would also hide duplicate IDs.

        Returns:
            The built index.

        Raises:
            TypeError: If an item is not a tuple, or a ``chunk_id`` or text is not a
                ``str``.
            ValueError: If a tuple does not have exactly two items, a ``chunk_id`` is
                empty or repeated, there are no chunks, or no chunk has a token.
        """
        pairs: list[tuple[str, str]] = []
        for item in documents:
            if not isinstance(item, tuple):
                raise TypeError(
                    f"each document must be a (chunk_id, text) tuple, got {type(item).__name__}"
                )
            if len(item) != 2:
                raise ValueError(
                    f"each document must be a (chunk_id, text) pair, got {len(item)} items"
                )
            chunk_id, text = item
            if not isinstance(chunk_id, str):
                raise TypeError(f"chunk_id must be str, got {type(chunk_id).__name__}")
            if not chunk_id:
                raise ValueError("chunk_id must be a non-empty string")
            if not isinstance(text, str):
                raise TypeError(
                    f"text of chunk {chunk_id!r} must be str, got {type(text).__name__}"
                )
            pairs.append((chunk_id, text))

        if not pairs:
            raise ValueError("cannot build a BM25 index from an empty corpus")
        id_counts = Counter(chunk_id for chunk_id, _ in pairs)
        repeated = sorted(chunk_id for chunk_id, count in id_counts.items() if count > 1)
        if repeated:
            raise ValueError(f"chunk_id values must be unique; repeated: {repeated}")

        pairs.sort(key=lambda pair: pair[0])
        term_counts = [Counter(tokenize(text)) for _, text in pairs]
        doc_lengths = tuple(sum(counts.values()) for counts in term_counts)
        if sum(doc_lengths) == 0:
            raise ValueError("cannot build a BM25 index: no chunk contains a token")

        postings: dict[str, list[tuple[int, int]]] = {}
        for doc_index, counts in enumerate(term_counts):
            for term in sorted(counts):
                postings.setdefault(term, []).append((doc_index, counts[term]))
        n_docs = len(pairs)
        terms = sorted(postings)
        return cls(
            chunk_ids=tuple(chunk_id for chunk_id, _ in pairs),
            doc_lengths=doc_lengths,
            avgdl=sum(doc_lengths) / n_docs,
            postings=MappingProxyType({term: tuple(postings[term]) for term in terms}),
            idf=MappingProxyType({term: bm25_idf(len(postings[term]), n_docs) for term in terms}),
        )

    def retrieve(self, query: str, k: int) -> list[RankedResult]:
        """Rank chunks against ``query`` and return the top ``k``.

        Each question token adds ``IDF * tf * (k1 + 1) / (tf + k1 * (1 - b + b *
        len / avgdl))`` to every chunk containing it, so a repeated question word
        counts each time. Chunks scoring 0 are dropped, then results sort by
        ``(-score, chunk_id)``.

        Args:
            query: Question text, tokenized with :func:`tokenize`.
            k: Maximum number of results.

        Returns:
            At most ``k`` results with ranks ``1..n``. Fewer than ``k`` when fewer
            chunks share a token with ``query``; empty when none does.

        Raises:
            TypeError: If ``query`` is not a ``str`` or ``k`` is not an ``int``
                (``bool`` is rejected).
            ValueError: If ``k`` is below 1 or ``query`` is empty or whitespace.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be int, got {type(k).__name__}")
        if k < 1:
            raise ValueError(f"k must be at least 1; got {k}")
        if not query.strip():
            raise ValueError("query must contain non-whitespace text")

        scores: dict[int, float] = {}
        for term in tokenize(query):
            term_postings = self.postings.get(term)
            if term_postings is None:
                continue
            weight = self.idf[term]
            for doc_index, tf in term_postings:
                length_norm = K1 * (1 - B + B * self.doc_lengths[doc_index] / self.avgdl)
                contribution = weight * tf * (K1 + 1) / (tf + length_norm)
                scores[doc_index] = scores.get(doc_index, 0.0) + contribution

        ranked = sorted(
            (
                (score, self.chunk_ids[doc_index])
                for doc_index, score in scores.items()
                if score > 0
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [
            RankedResult(rank=rank, chunk_id=chunk_id, score=score)
            for rank, (score, chunk_id) in enumerate(ranked[:k], start=1)
        ]
