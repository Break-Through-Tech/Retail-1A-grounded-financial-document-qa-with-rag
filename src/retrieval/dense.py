"""Exact dense retrieval over L2-normalized embeddings for the FinanceBench baseline.

Issues #18 (dense retrieval) and #19 (vector index) of the Retail-1A challenge
project, Phase A. This module stores chunk vectors in an exact FAISS index, ranks them
by cosine similarity, and turns question text into a ranking through a pluggable
encoder. It does not build chunks, choose an embedding model, or score rankings.

Design notes that the final report depends on:

* Similarity is cosine, computed as the inner product of L2-normalized vectors in a
  ``faiss.IndexFlatIP``. The index is exact: every query is compared with every
  chunk, so there is no approximation to tune.
* Normalization happens in one place, :func:`normalize_rows`, for chunk vectors at
  build time and for query vectors at search time. Encoders are never asked to
  normalize. Zero, NaN and infinite vectors are rejected rather than kept, because
  cosine similarity is undefined for them.
* FAISS does not order equal scores by chunk ID (identical rows come back in reverse
  insertion order), so every search covers the whole index and is re-sorted here by
  ``(-score, chunk_id)``. Ties at the ``k`` boundary are resolved correctly too.
* There is no score threshold. Every chunk has a defined similarity, negative ones
  included, so fewer than ``k`` results come back only when the index holds fewer
  than ``k`` chunks.
* Chunks are ordered by ``chunk_id`` before they are encoded or indexed, so input
  order changes neither what an encoder sees nor the ranking.
* Reproducible: identical input vectors and configuration give identical scores and
  rankings. Not promised: bit-identical neural embeddings across hardware or library
  versions; the encoder's ``describe()`` records what produced them.
* This module never imports ``sentence_transformers`` or ``torch``. The real model
  adapter lives in :mod:`src.retrieval.embeddings`.
"""

from __future__ import annotations

import platform
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

import faiss
import numpy as np

__all__ = [
    "DENSE_CONFIG_VERSION",
    "DenseResult",
    "DenseRetriever",
    "INDEX_TYPE",
    "SIMILARITY",
    "TextEncoder",
    "VectorIndex",
    "normalize_rows",
]

#: Version of the fixed configuration below. Any change to a rule is a new version.
DENSE_CONFIG_VERSION = "1.0.0"
SIMILARITY = "cosine"
INDEX_TYPE = "faiss.IndexFlatIP"

_ENCODER_METHODS = ("encode_documents", "encode_queries", "describe")


class TextEncoder(Protocol):
    """Turns text into embedding rows: a real model adapter, or a fixed fake in tests."""

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return one embedding row per chunk text, in input order."""
        ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Return one embedding row per question text, in input order."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable record identifying the encoder and its settings."""
        ...


def _require_real_array(vectors: object, name: str) -> np.ndarray:
    """Return ``vectors`` if it is an ndarray of real numbers; raise ``TypeError`` if not."""
    if not isinstance(vectors, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(vectors).__name__}")
    if not (np.issubdtype(vectors.dtype, np.floating) or np.issubdtype(vectors.dtype, np.integer)):
        raise TypeError(f"{name} must have a real numeric dtype, got {vectors.dtype}")
    return vectors


def _check_k(k: int) -> None:
    """Raise unless ``k`` is an ``int`` (not ``bool``) of at least 1."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    if k < 1:
        raise ValueError(f"k must be at least 1; got {k}")


def normalize_rows(
    vectors: np.ndarray,
    *,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
) -> np.ndarray:
    """Validate embedding rows and scale each one to unit L2 length.

    This is the single normalization path for chunk vectors and query vectors. Norms
    are computed in float64; the result is stored as float32, which FAISS requires.

    Args:
        vectors: A 2-D array with one embedding per row.
        expected_rows: If given, the exact number of rows required.
        expected_dim: If given, the exact number of columns required.

    Returns:
        A new C-contiguous float32 array of unit-length rows, same shape as ``vectors``.

    Raises:
        TypeError: If ``vectors`` is not an ndarray, or its dtype is not a real number
            type (``bool``, complex and ``object`` are rejected).
        ValueError: If it is not 2-D, has no rows or columns, has the wrong row count
            or dimension, contains NaN or infinity, or has a row of zero or non-finite
            length.
    """
    _require_real_array(vectors, "vectors")
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-dimensional, got {vectors.ndim} dimension(s)")
    rows, dim = vectors.shape
    if rows == 0 or dim == 0:
        raise ValueError(f"vectors must have at least one row and one column, got {vectors.shape}")
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"expected {expected_rows} vector row(s), got {rows}")
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"expected vectors of dimension {expected_dim}, got {dim}")

    values = vectors.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("vectors must not contain NaN or infinite values")
    norms = np.linalg.norm(values, axis=1)
    invalid = np.flatnonzero(~np.isfinite(norms) | (norms == 0))
    if invalid.size:
        raise ValueError(
            f"rows {invalid[:5].tolist()} have zero or non-finite length; "
            "cosine similarity is undefined for them"
        )
    return np.ascontiguousarray(values / norms[:, np.newaxis], dtype=np.float32)


@dataclass(frozen=True)
class DenseResult:
    """One retrieved chunk.

    Attributes:
        rank: 1-based position in the ranking.
        chunk_id: Identifier of the retrieved chunk, as supplied to the index.
        score: Cosine similarity to the query, in ``[-1, 1]`` up to float32 rounding.
    """

    rank: int
    chunk_id: str
    score: float


@dataclass(frozen=True, eq=False)
class VectorIndex:
    """An exact cosine-similarity index over chunk vectors. Build it with :meth:`build`.

    Attributes:
        chunk_ids: Chunk identifiers in ascending order; row ``i`` of ``vectors`` and of
            the FAISS index belongs to ``chunk_ids[i]``.
        vectors: Unit-length float32 rows, read-only.
        faiss_index: The ``faiss.IndexFlatIP`` holding ``vectors``.
    """

    chunk_ids: tuple[str, ...]
    vectors: np.ndarray = field(repr=False)
    faiss_index: Any = field(repr=False)

    @property
    def dimension(self) -> int:
        """Length of every stored vector."""
        return int(self.vectors.shape[1])

    @property
    def n_chunks(self) -> int:
        """Number of chunks in the index."""
        return len(self.chunk_ids)

    @classmethod
    def build(cls, chunk_ids: Sequence[str], vectors: np.ndarray) -> VectorIndex:
        """Index one vector per chunk ID.

        Rows are reordered so ``chunk_ids`` ascend, then normalized with
        :func:`normalize_rows` and added to an exact inner-product index.

        Args:
            chunk_ids: Unique, non-empty chunk identifiers, aligned with ``vectors``.
            vectors: One embedding row per chunk ID.

        Returns:
            The built index.

        Raises:
            TypeError: If ``chunk_ids`` is a string or holds a non-string, or
                ``vectors`` fails :func:`normalize_rows`'s type checks.
            ValueError: If there are no chunk IDs, an ID is empty or repeated, the row
                count differs from the ID count, or ``vectors`` fails
                :func:`normalize_rows`'s value checks.
        """
        if isinstance(chunk_ids, str):
            raise TypeError("chunk_ids must be a sequence of strings, not a single string")
        ids = list(chunk_ids)
        for chunk_id in ids:
            if not isinstance(chunk_id, str):
                raise TypeError(f"chunk_id must be str, got {type(chunk_id).__name__}")
            if not chunk_id:
                raise ValueError("chunk_id must be a non-empty string")
        if not ids:
            raise ValueError("cannot build a vector index from an empty corpus")
        repeated = sorted(chunk_id for chunk_id, count in Counter(ids).items() if count > 1)
        if repeated:
            raise ValueError(f"chunk_id values must be unique; repeated: {repeated}")

        unit = normalize_rows(vectors, expected_rows=len(ids))
        order = sorted(range(len(ids)), key=ids.__getitem__)
        unit = np.ascontiguousarray(unit[order])
        index = faiss.IndexFlatIP(unit.shape[1])
        index.add(unit)
        unit.flags.writeable = False
        return cls(chunk_ids=tuple(ids[i] for i in order), vectors=unit, faiss_index=index)

    def search(self, query_vector: np.ndarray, k: int) -> list[DenseResult]:
        """Return the ``k`` chunks most similar to ``query_vector``.

        The whole index is searched and the hits are re-sorted by
        ``(-score, chunk_id)``, because FAISS does not order equal scores by chunk ID.

        Args:
            query_vector: A 1-D embedding of length :attr:`dimension`; it is
                normalized here, so its length does not matter.
            k: Maximum number of results.

        Returns:
            ``min(k, n_chunks)`` results with ranks ``1..n``.

        Raises:
            TypeError: If ``k`` is not an ``int`` (``bool`` is rejected) or
                ``query_vector`` is not a real numeric ndarray.
            ValueError: If ``k`` is below 1, or ``query_vector`` is not 1-D, has the
                wrong length, or fails :func:`normalize_rows`'s value checks.
        """
        _check_k(k)
        _require_real_array(query_vector, "query_vector")
        if query_vector.ndim != 1:
            raise ValueError(
                f"query_vector must be 1-dimensional, got {query_vector.ndim} dimension(s)"
            )
        query = normalize_rows(query_vector[np.newaxis, :], expected_dim=self.dimension)
        scores, rows = self.faiss_index.search(query, self.n_chunks)
        hits = sorted(
            (
                (float(score), self.chunk_ids[row])
                for score, row in zip(scores[0], rows[0])
                if row >= 0
            ),
            key=lambda hit: (-hit[0], hit[1]),
        )
        return [
            DenseResult(rank=rank, chunk_id=chunk_id, score=score)
            for rank, (score, chunk_id) in enumerate(hits[:k], start=1)
        ]


@dataclass(frozen=True, eq=False)
class DenseRetriever:
    """Question text -> ranked chunks, through an encoder and a :class:`VectorIndex`.

    Build it with :meth:`build`.

    Attributes:
        index: The vector index over the encoded chunks.
        encoder: The encoder used for chunks at build time and for every question.
    """

    index: VectorIndex
    encoder: TextEncoder = field(repr=False)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Chunk identifiers in ascending order."""
        return self.index.chunk_ids

    @property
    def n_chunks(self) -> int:
        """Number of chunks in the index."""
        return self.index.n_chunks

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.index.dimension

    @classmethod
    def build(cls, documents: Iterable[tuple[str, str]], encoder: TextEncoder) -> DenseRetriever:
        """Encode ``(chunk_id, text)`` pairs and index them.

        Pairs are ordered by ``chunk_id`` and encoded with one ``encode_documents`` call,
        so input order never changes what the encoder sees.

        Args:
            documents: ``(chunk_id, text)`` tuples. A mapping is rejected because
                iterating it yields keys, which would also hide duplicate IDs.
            encoder: Any object with ``encode_documents``, ``encode_queries`` and
                ``describe`` methods.

        Returns:
            The built retriever.

        Raises:
            TypeError: If the encoder lacks a required method, an item is not a tuple,
                a ``chunk_id`` or text is not a ``str``, or the encoder's output fails
                :func:`normalize_rows`'s type checks.
            ValueError: If a tuple does not have exactly two items, a ``chunk_id`` is
                empty or repeated, a text is empty or whitespace, there are no chunks,
                or the encoder's output has the wrong row count or fails
                :func:`normalize_rows`'s value checks.
        """
        missing = [name for name in _ENCODER_METHODS if not callable(getattr(encoder, name, None))]
        if missing:
            raise TypeError(f"encoder is missing required method(s): {missing}")

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
            if not text.strip():
                raise ValueError(f"text of chunk {chunk_id!r} is empty or whitespace")
            pairs.append((chunk_id, text))

        if not pairs:
            raise ValueError("cannot build a dense retriever from an empty corpus")
        id_counts = Counter(chunk_id for chunk_id, _ in pairs)
        repeated = sorted(chunk_id for chunk_id, count in id_counts.items() if count > 1)
        if repeated:
            raise ValueError(f"chunk_id values must be unique; repeated: {repeated}")

        pairs.sort(key=lambda pair: pair[0])
        vectors = encoder.encode_documents([text for _, text in pairs])
        index = VectorIndex.build([chunk_id for chunk_id, _ in pairs], vectors)
        return cls(index=index, encoder=encoder)

    def retrieve(self, query: str, k: int) -> list[DenseResult]:
        """Rank chunks by cosine similarity to ``query`` and return the top ``k``.

        Args:
            query: Question text, encoded alone with ``encode_queries``.
            k: Maximum number of results.

        Returns:
            ``min(k, n_chunks)`` results with ranks ``1..n``.

        Raises:
            TypeError: If ``query`` is not a ``str``, ``k`` is not an ``int`` (``bool`` is
                rejected), or the query embedding is not a real numeric ndarray.
            ValueError: If ``k`` is below 1, ``query`` is empty or whitespace, or the
                query embedding is not shaped ``(1, dimension)`` or fails
                :func:`normalize_rows`'s value checks.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        _check_k(k)
        if not query.strip():
            raise ValueError("query must contain non-whitespace text")

        embedding = _require_real_array(self.encoder.encode_queries([query]), "query embedding")
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(
                f"query encoder must return shape (1, {self.dimension}), got {embedding.shape}"
            )
        return self.index.search(embedding[0], k)

    def config_record(self) -> dict[str, Any]:
        """Return the retrieval configuration and encoder provenance.

        Returns:
            A JSON-serializable dict with no timestamp.
        """
        return {
            "config_version": DENSE_CONFIG_VERSION,
            "similarity": SIMILARITY,
            "index_type": INDEX_TYPE,
            "search": "exhaustive, re-sorted",
            "normalization": "L2 per row in float64, stored as float32, chunk and query",
            "score_threshold": "none",
            "tie_order": "(-score, chunk_id)",
            "n_chunks": self.n_chunks,
            "dimension": self.dimension,
            "faiss_version": faiss.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "encoder": dict(self.encoder.describe()),
        }
