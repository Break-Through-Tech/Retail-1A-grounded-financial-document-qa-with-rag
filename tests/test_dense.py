"""Tests for the exact dense retrieval core and vector index (issues #18 and #19).

Every fixture is a synthetic vector or a fixed lookup-table encoder; no model is
loaded and no FinanceBench row is used. Expected cosine values are worked out by hand
from the vectors, never by running the index.

The load-bearing tests are :func:`test_ties_break_by_ascending_chunk_id`, because
FAISS itself returns tied rows in reverse insertion order, and
:func:`test_dense_core_does_not_import_sentence_transformers_or_torch`, which keeps the
neural-network stack out of the core.
"""

from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from src.retrieval.dense import (
    DENSE_CONFIG_VERSION,
    INDEX_TYPE,
    SIMILARITY,
    DenseResult,
    DenseRetriever,
    VectorIndex,
    normalize_rows,
)

ROOT = Path(__file__).resolve().parents[1]

#: Compass-point vectors whose cosines to (2, 1) are known exactly:
#: northeast 3/sqrt(10), east 2/sqrt(5), north 1/sqrt(5), west -2/sqrt(5).
COMPASS = {
    "east": [1.0, 0.0],
    "north": [0.0, 1.0],
    "northeast": [1.0, 1.0],
    "west": [-1.0, 0.0],
}


class FakeEncoder:
    """Maps known texts to fixed vectors and records every call."""

    def __init__(self, table: dict[str, Sequence[float]]) -> None:
        self.table = {text: list(vector) for text, vector in table.items()}
        self.document_calls: list[list[str]] = []
        self.query_calls: list[list[str]] = []

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_calls.append(list(texts))
        return np.array([self.table[text] for text in texts], dtype=np.float64)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        self.query_calls.append(list(texts))
        return np.array([self.table[text] for text in texts], dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"encoder": "fake", "texts": len(self.table)}


class FixedOutputEncoder(FakeEncoder):
    """Returns preset arrays, to exercise output validation."""

    def __init__(self, documents: Any = None, query: Any = None) -> None:
        super().__init__({})
        self.documents = documents
        self.query = query

    def encode_documents(self, texts: Sequence[str]) -> Any:
        self.document_calls.append(list(texts))
        return self.documents

    def encode_queries(self, texts: Sequence[str]) -> Any:
        self.query_calls.append(list(texts))
        return self.query


def compass_index() -> VectorIndex:
    """The compass vectors indexed under their names."""
    return VectorIndex.build(list(COMPASS), np.array(list(COMPASS.values())))


def compass_retriever() -> tuple[DenseRetriever, FakeEncoder]:
    """A retriever whose chunk texts are the compass names, plus a query "q" = (2, 1)."""
    encoder = FakeEncoder({**COMPASS, "q": [2.0, 1.0]})
    documents = [(f"chunk-{name}", name) for name in COMPASS]
    return DenseRetriever.build(documents, encoder), encoder


# --- normalization --------------------------------------------------------------


def test_normalize_rows_returns_unit_float32_rows() -> None:
    unit = normalize_rows(np.array([[3, 4], [0, -2]]))
    assert unit.dtype == np.float32
    assert unit.flags["C_CONTIGUOUS"]
    assert unit.tolist() == [[pytest.approx(0.6), pytest.approx(0.8)], [0.0, -1.0]]


def test_normalize_rows_checks_expected_rows_and_dimension() -> None:
    vectors = np.ones((2, 3))
    with pytest.raises(ValueError, match="row"):
        normalize_rows(vectors, expected_rows=3)
    with pytest.raises(ValueError, match="dimension 4"):
        normalize_rows(vectors, expected_dim=4)


@pytest.mark.parametrize(
    "bad",
    [
        [[1.0, 0.0]],
        np.array([[1, 0]], dtype=object),
        np.array([[True, False]]),
        np.array([[1 + 1j, 0]]),
    ],
)
def test_vectors_must_be_a_real_numeric_ndarray(bad: object) -> None:
    with pytest.raises(TypeError):
        normalize_rows(bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        VectorIndex.build(["a"], bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", [(3,), (1, 2, 2), (0, 2), (2, 0)])
def test_vectors_must_be_a_non_empty_matrix(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        normalize_rows(np.ones(shape))


# --- index construction and mapping ---------------------------------------------


def test_vector_index_maps_rows_to_sorted_chunk_ids() -> None:
    ids = ["c", "a", "b"]
    vectors = np.array([[0.0, 3.0], [2.0, 0.0], [1.0, 1.0]])
    index = VectorIndex.build(ids, vectors)
    assert index.chunk_ids == ("a", "b", "c")
    assert index.n_chunks == 3
    assert index.dimension == 2
    s = 1 / math.sqrt(2)
    expected = {"a": [1.0, 0.0], "b": [s, s], "c": [0.0, 1.0]}
    for row, chunk_id in enumerate(index.chunk_ids):
        assert index.vectors[row].tolist() == pytest.approx(expected[chunk_id])


def test_chunk_vectors_are_normalized() -> None:
    small = VectorIndex.build(["a", "b"], np.array([[3.0, 4.0], [0.0, 2.0]]))
    large = VectorIndex.build(["a", "b"], np.array([[300.0, 400.0], [0.0, 0.5]]))
    assert np.linalg.norm(small.vectors, axis=1).tolist() == pytest.approx([1.0, 1.0])
    query = np.array([1.0, 1.0])
    assert small.search(query, 2) == large.search(query, 2)


def test_duplicate_chunk_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        VectorIndex.build(["a", "b", "a"], np.eye(3))


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        VectorIndex.build([], np.zeros((0, 2)))
    with pytest.raises(ValueError, match="empty corpus"):
        DenseRetriever.build([], FakeEncoder({}))


@pytest.mark.parametrize("chunk_ids", [[None], [5], [b"a"]])
def test_chunk_ids_must_be_strings(chunk_ids: list[object]) -> None:
    with pytest.raises(TypeError):
        VectorIndex.build(chunk_ids, np.ones((1, 2)))  # type: ignore[arg-type]


def test_chunk_ids_must_be_non_empty_and_not_a_single_string() -> None:
    with pytest.raises(ValueError):
        VectorIndex.build([""], np.ones((1, 2)))
    with pytest.raises(TypeError):
        VectorIndex.build("ab", np.ones((2, 2)))  # type: ignore[arg-type]


def test_chunk_id_count_must_match_vector_rows() -> None:
    with pytest.raises(ValueError, match="row"):
        VectorIndex.build(["a", "b"], np.eye(3))


def test_stored_vectors_are_read_only() -> None:
    index = compass_index()
    assert not index.vectors.flags.writeable
    with pytest.raises(ValueError):
        index.vectors[0, 0] = 5.0


# --- invalid vector values ------------------------------------------------------


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nan_and_inf_chunk_vectors_are_rejected(bad_value: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        VectorIndex.build(["a", "b"], np.array([[1.0, 0.0], [bad_value, 1.0]]))


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nan_and_inf_query_vectors_are_rejected(bad_value: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        compass_index().search(np.array([bad_value, 1.0]), 2)


def test_zero_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero or non-finite length"):
        VectorIndex.build(["a", "b"], np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="zero or non-finite length"):
        compass_index().search(np.array([0.0, 0.0]), 2)


def test_vector_dimension_mismatch_is_rejected() -> None:
    index = compass_index()
    with pytest.raises(ValueError, match="dimension 2"):
        index.search(np.array([1.0, 0.0, 0.0]), 2)


@pytest.mark.parametrize("query", [np.ones((1, 2)), np.ones((2, 2)), np.array(1.0)])
def test_query_vector_must_be_one_dimensional(query: np.ndarray) -> None:
    with pytest.raises(ValueError, match="1-dimensional"):
        compass_index().search(query, 2)


def test_query_vector_must_be_a_real_numeric_ndarray() -> None:
    with pytest.raises(TypeError):
        compass_index().search([1.0, 0.0], 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compass_index().search(np.array([True, False]), 2)


# --- similarity and ranking -----------------------------------------------------


def test_search_returns_exact_nearest_neighbours_in_order() -> None:
    results = compass_index().search(np.array([2.0, 1.0]), 4)
    assert [r.chunk_id for r in results] == ["northeast", "east", "north", "west"]
    expected = [3 / math.sqrt(10), 2 / math.sqrt(5), 1 / math.sqrt(5), -2 / math.sqrt(5)]
    assert [r.score for r in results] == pytest.approx(expected, abs=1e-6)
    assert [r.rank for r in results] == [1, 2, 3, 4]


def test_scores_are_cosine_similarity() -> None:
    index = VectorIndex.build(["a", "b"], np.array([[3, 4], [0, 5]]))
    results = index.search(np.array([10, 0]), 2)
    # cos((3,4),(1,0)) = 3/5; cos((0,5),(1,0)) = 0.
    assert [(r.chunk_id, r.score) for r in results] == [
        ("a", pytest.approx(0.6, abs=1e-6)),
        ("b", pytest.approx(0.0, abs=1e-6)),
    ]


def test_query_vector_is_normalized() -> None:
    # A power-of-two scale keeps normalization exact, so the results must be identical.
    index = compass_index()
    assert index.search(np.array([2.0, 1.0]), 4) == index.search(np.array([16.0, 8.0]), 4)


def test_negative_similarities_are_still_ranked() -> None:
    results = compass_index().search(np.array([1.0, 0.0]), 10)
    assert len(results) == 4
    assert results[-1].chunk_id == "west"
    assert results[-1].score == pytest.approx(-1.0, abs=1e-6)


def test_ties_break_by_ascending_chunk_id() -> None:
    """FAISS returns identical rows in reverse insertion order; the index must not."""
    vectors = np.array([[1.0, 0.0]] * 4 + [[0.0, 1.0]])
    for ids in (["d", "c", "b", "a", "e"], ["a", "b", "c", "d", "e"]):
        results = VectorIndex.build(ids, vectors).search(np.array([1.0, 0.0]), 5)
        assert [r.chunk_id for r in results] == ["a", "b", "c", "d", "e"]
        assert len({r.score for r in results[:4]}) == 1


def test_ties_at_the_k_boundary_keep_the_lowest_chunk_ids() -> None:
    vectors = np.array([[1.0, 0.0]] * 4)
    results = VectorIndex.build(["d", "c", "b", "a"], vectors).search(np.array([1.0, 0.0]), 2)
    assert [r.chunk_id for r in results] == ["a", "b"]


def test_k_one_returns_only_the_nearest() -> None:
    results = compass_index().search(np.array([2.0, 1.0]), 1)
    assert [(r.rank, r.chunk_id) for r in results] == [(1, "northeast")]


def test_k_larger_than_corpus_returns_every_chunk_without_padding() -> None:
    results = compass_index().search(np.array([2.0, 1.0]), 100)
    assert sorted(r.chunk_id for r in results) == sorted(COMPASS)
    assert all(-1.0 - 1e-6 <= r.score <= 1.0 + 1e-6 for r in results)


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_is_rejected(k: int) -> None:
    with pytest.raises(ValueError):
        compass_index().search(np.array([1.0, 0.0]), k)
    with pytest.raises(ValueError):
        compass_retriever()[0].retrieve("q", k)


@pytest.mark.parametrize("k", [1.5, "5", True, None])
def test_non_integer_k_is_rejected(k: object) -> None:
    with pytest.raises(TypeError):
        compass_index().search(np.array([1.0, 0.0]), k)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compass_retriever()[0].retrieve("q", k)  # type: ignore[arg-type]


# --- retriever and encoder boundary ---------------------------------------------


def test_retriever_ranks_by_query_embedding() -> None:
    retriever, _ = compass_retriever()
    results = retriever.retrieve("q", 4)
    assert [r.chunk_id for r in results] == [
        "chunk-northeast", "chunk-east", "chunk-north", "chunk-west",
    ]


def test_documents_are_encoded_once_in_chunk_id_order() -> None:
    encoder = FakeEncoder({"text-a": [1, 0], "text-b": [0, 1], "text-c": [1, 1]})
    DenseRetriever.build([("c", "text-c"), ("a", "text-a"), ("b", "text-b")], encoder)
    assert encoder.document_calls == [["text-a", "text-b", "text-c"]]
    assert encoder.query_calls == []


def test_query_is_encoded_alone_via_encode_queries() -> None:
    retriever, encoder = compass_retriever()
    retriever.retrieve("q", 2)
    assert encoder.query_calls == [["q"]]
    assert len(encoder.document_calls) == 1


def test_encoder_returning_wrong_row_count_is_rejected() -> None:
    encoder = FixedOutputEncoder(documents=np.ones((1, 2)))
    with pytest.raises(ValueError, match="row"):
        DenseRetriever.build([("a", "x"), ("b", "y")], encoder)


def test_encoder_returning_a_list_is_rejected() -> None:
    encoder = FixedOutputEncoder(documents=[[1.0, 0.0]])
    with pytest.raises(TypeError):
        DenseRetriever.build([("a", "x")], encoder)


def _retriever_with_query_output(query: Any) -> DenseRetriever:
    encoder = FixedOutputEncoder(documents=np.array([[1.0, 0.0], [0.0, 1.0]]), query=query)
    return DenseRetriever.build([("a", "x"), ("b", "y")], encoder)


@pytest.mark.parametrize(
    "query_output", [np.ones((2, 2)), np.ones(2), np.ones((1, 3)), np.ones((1, 2, 1))]
)
def test_query_encoder_shape_is_validated(query_output: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _retriever_with_query_output(query_output).retrieve("question", 2)


def test_query_encoder_must_return_a_numeric_ndarray() -> None:
    with pytest.raises(TypeError):
        _retriever_with_query_output([[1.0, 0.0]]).retrieve("question", 2)
    with pytest.raises(TypeError):
        _retriever_with_query_output(np.array([[True, False]])).retrieve("question", 2)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_chunk_text_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="empty or whitespace"):
        DenseRetriever.build([("a", text)], FakeEncoder({text: [1, 0]}))


def test_document_types_are_validated() -> None:
    encoder = FakeEncoder({"x": [1, 0]})
    with pytest.raises(TypeError):
        DenseRetriever.build(["x"], encoder)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        DenseRetriever.build({"a": "x"}, encoder)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DenseRetriever.build([("a", "x", "extra")], encoder)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        DenseRetriever.build([(5, "x")], encoder)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        DenseRetriever.build([("", "x")], encoder)
    with pytest.raises(TypeError):
        DenseRetriever.build([("a", None)], encoder)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="unique"):
        DenseRetriever.build([("a", "x"), ("a", "x")], encoder)


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_blank_question_is_rejected(query: str) -> None:
    with pytest.raises(ValueError):
        compass_retriever()[0].retrieve(query, 2)


@pytest.mark.parametrize("query", [None, b"q", 3])
def test_non_string_question_is_rejected(query: object) -> None:
    with pytest.raises(TypeError):
        compass_retriever()[0].retrieve(query, 2)  # type: ignore[arg-type]


def test_encoder_without_required_methods_is_rejected() -> None:
    class NoDescribe:
        def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
            return np.ones((len(texts), 2))

        def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
            return np.ones((len(texts), 2))

    for encoder in (object(), NoDescribe()):
        with pytest.raises(TypeError, match="missing required method"):
            DenseRetriever.build([("a", "x")], encoder)  # type: ignore[arg-type]


# --- determinism and boundary ---------------------------------------------------


def test_repeated_runs_are_identical() -> None:
    def run() -> list[DenseResult]:
        return compass_retriever()[0].retrieve("q", 4)

    first, second = run(), run()
    assert first == second
    serialized = [json.dumps([asdict(r) for r in results]) for results in (first, second)]
    assert serialized[0].encode("utf-8") == serialized[1].encode("utf-8")


def test_input_order_does_not_change_rankings() -> None:
    encoder_table = {**COMPASS, "q": [2.0, 1.0]}
    documents = [(f"chunk-{name}", name) for name in COMPASS]
    orders = [documents, list(reversed(documents)), [documents[i] for i in (2, 0, 3, 1)]]
    built = [DenseRetriever.build(order, FakeEncoder(encoder_table)) for order in orders]
    rankings = [retriever.retrieve("q", 4) for retriever in built]
    assert rankings[0] == rankings[1] == rankings[2]
    assert all(np.array_equal(built[0].index.vectors, r.index.vectors) for r in built[1:])


def test_result_has_only_rank_chunk_id_score() -> None:
    assert [f.name for f in fields(DenseResult)] == ["rank", "chunk_id", "score"]


def test_retrieve_accepts_only_query_and_k() -> None:
    assert list(inspect.signature(DenseRetriever.retrieve).parameters) == ["self", "query", "k"]
    assert list(inspect.signature(VectorIndex.search).parameters) == [
        "self", "query_vector", "k",
    ]


def test_config_record_is_complete_and_timestamp_free() -> None:
    retriever, _ = compass_retriever()
    record = retriever.config_record()
    assert record == retriever.config_record()
    assert record["config_version"] == DENSE_CONFIG_VERSION == "1.0.0"
    assert record["similarity"] == SIMILARITY == "cosine"
    assert record["index_type"] == INDEX_TYPE == "faiss.IndexFlatIP"
    assert record["score_threshold"] == "none"
    assert record["tie_order"] == "(-score, chunk_id)"
    assert record["n_chunks"] == 4
    assert record["dimension"] == 2
    assert record["encoder"] == {"encoder": "fake", "texts": 5}
    blob = json.dumps(record).lower()
    for forbidden in ("timestamp", "created_at", "generated_at", "datetime"):
        assert forbidden not in blob


def test_dense_core_does_not_import_sentence_transformers_or_torch() -> None:
    code = (
        "import sys, src.retrieval.dense; "
        "print(sorted(m for m in ('sentence_transformers', 'torch') if m in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]"
