"""Tests for retrieve-and-rerank (``src/retrieval/rerank.py``).

No model is loaded and no FinanceBench row is used. First stages are tiny hand-built
retrievers, the scorers are lookup tables, and the cross-encoder wrapper is exercised
with a stub that has a ``predict`` method. Expected RRF scores are worked out by hand.

A few tests also run the real ``BM25Index`` and ``DenseRetriever`` through the pipeline,
so a change to their ``retrieve`` interface breaks here rather than in an experiment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseRetriever
from src.retrieval.rerank import (
    RRF_K,
    Candidate,
    CrossEncoderScorer,
    RetrieveAndRerank,
    rerank,
    rrf_fuse,
)

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40

DOCS = [
    ("c1", "apple revenue grew in 2022 to 394 billion"),
    ("c2", "microsoft net income was 72 billion"),
    ("c3", "apple net income 2022 was 99.8 billion"),
    ("c4", "risk factors include supply chain disruption"),
    ("c5", "apple capital expenditure in fiscal 2022"),
]
TEXTS = dict(DOCS)


# --------------------------------------------------------------------------- fakes


class Hit:
    """Minimal first-stage result: ``chunk_id`` and ``score``."""

    def __init__(self, chunk_id: str, score: float = 1.0) -> None:
        self.chunk_id = chunk_id
        self.score = score


class FakeRetriever:
    """Returns a fixed ranked list, truncated to ``k``, and records each call."""

    def __init__(self, chunk_ids: Sequence[str]) -> None:
        self._hits = [Hit(c, 10.0 - i) for i, c in enumerate(chunk_ids)]
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[Hit]:
        self.calls.append((query, k))
        return self._hits[:k]


class TableScorer:
    """Scores a text by lookup, and records every call."""

    def __init__(self, table: dict[str, float], default: float = 0.0) -> None:
        self.table = table
        self.default = default
        self.calls: list[tuple[str, list[str]]] = []

    def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
        self.calls.append((query, list(texts)))
        return np.array([self.table.get(t, self.default) for t in texts], dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"scorer": "table"}


class ConstantScorer:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
        return np.full(len(texts), self.value)

    def describe(self) -> dict[str, Any]:
        return {}


class BagEncoder:
    """Deterministic bag-of-words encoder over a fixed vocabulary."""

    VOCAB = sorted({w for _, t in DOCS for w in t.split()} | {"apple", "net", "income"})

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self.VOCAB)))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                if word in self.VOCAB:
                    out[row, self.VOCAB.index(word)] += 1.0
        return out

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def describe(self) -> dict[str, Any]:
        return {"encoder": "bag-of-words"}


class StubCrossEncoder:
    """Mimics the parts of ``CrossEncoder`` the wrapper touches."""

    def __init__(self, scores: Any = None) -> None:
        self.max_seq_length = 512
        self.scores = scores
        self.last_call: dict[str, Any] = {}

    def predict(self, pairs: Sequence[tuple[str, str]], **kwargs: Any) -> Any:
        self.last_call = {"pairs": list(pairs), **kwargs}
        if self.scores is not None:
            return self.scores
        return np.array([float(len(text)) for _, text in pairs], dtype=np.float32)


def cand(chunk_id: str, rank: int, text: str | None = None, score: float = 0.0) -> Candidate:
    return Candidate(chunk_id, text if text is not None else f"text of {chunk_id}", rank, score)


# --------------------------------------------------------------------------- rrf_fuse


def test_rrf_scores_match_hand_calculation() -> None:
    fused = rrf_fuse([[Hit("a"), Hit("b")], [Hit("b"), Hit("c")]])
    scores = {f.chunk_id: f.score for f in fused}
    assert scores["a"] == pytest.approx(1 / (RRF_K + 1))
    assert scores["b"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert scores["c"] == pytest.approx(1 / (RRF_K + 2))
    assert [f.chunk_id for f in fused] == ["b", "a", "c"]
    assert [f.rank for f in fused] == [1, 2, 3]


def test_rrf_ties_break_on_ascending_chunk_id() -> None:
    fused = rrf_fuse([[Hit("b"), Hit("a")], [Hit("a"), Hit("b")]])
    assert [f.chunk_id for f in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(fused[1].score)


def test_rrf_respects_custom_rrf_k() -> None:
    fused = rrf_fuse([[Hit("a")]], rrf_k=1)
    assert fused[0].score == pytest.approx(0.5)


def test_rrf_single_list_keeps_order() -> None:
    fused = rrf_fuse([[Hit("z"), Hit("m"), Hit("a")]])
    assert [f.chunk_id for f in fused] == ["z", "m", "a"]


@pytest.mark.parametrize("rankings", [[], "abc"])
def test_rrf_rejects_empty_or_string_rankings(rankings: Any) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        rrf_fuse(rankings)


def test_rrf_rejects_repeated_chunk_id_within_a_list() -> None:
    with pytest.raises(ValueError, match="repeats"):
        rrf_fuse([[Hit("a"), Hit("a")]])


@pytest.mark.parametrize("bad, error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)])
def test_rrf_validates_rrf_k(bad: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        rrf_fuse([[Hit("a")]], rrf_k=bad)


# --------------------------------------------------------------------------- rerank()


def test_rerank_orders_by_score_and_keeps_first_stage_fields() -> None:
    cands = [cand("a", 1, "ta", 9.0), cand("b", 2, "tb", 8.0), cand("c", 3, "tc", 7.0)]
    scorer = TableScorer({"ta": 0.1, "tb": 0.9, "tc": 0.5})
    results = rerank("q", cands, scorer, top_k=3)
    assert [r.chunk_id for r in results] == ["b", "c", "a"]
    assert [r.rank for r in results] == [1, 2, 3]
    top = results[0]
    assert (top.score, top.retrieval_rank, top.retrieval_score) == (0.9, 2, 8.0)


def test_rerank_truncates_to_top_k() -> None:
    cands = [cand("a", 1, "ta"), cand("b", 2, "tb"), cand("c", 3, "tc")]
    results = rerank("q", cands, TableScorer({"ta": 1, "tb": 2, "tc": 3}), top_k=2)
    assert [r.chunk_id for r in results] == ["c", "b"]


def test_rerank_top_k_larger_than_candidates_returns_all() -> None:
    results = rerank("q", [cand("a", 1)], ConstantScorer(), top_k=50)
    assert len(results) == 1


def test_rerank_ties_keep_first_stage_order_then_chunk_id() -> None:
    # First-stage rank order (z, m, a) is the reverse of chunk_id order, so a tie that
    # fell through to chunk_id alone would give a, m, z.
    cands = [cand("m", 2), cand("a", 3), cand("z", 1)]
    results = rerank("q", cands, ConstantScorer(), top_k=3)
    assert [r.chunk_id for r in results] == ["z", "m", "a"]

    same_rank = [cand("y", 1), cand("x", 1)]
    results = rerank("q", same_rank, ConstantScorer(), top_k=2)
    assert [r.chunk_id for r in results] == ["x", "y"]


def test_rerank_calls_scorer_once_with_all_texts_in_input_order() -> None:
    scorer = TableScorer({})
    rerank("the question", [cand("a", 1, "ta"), cand("b", 2, "tb")], scorer, top_k=1)
    assert scorer.calls == [("the question", ["ta", "tb"])]


def test_rerank_with_no_candidates_never_calls_scorer() -> None:
    scorer = TableScorer({})
    assert rerank("q", [], scorer, top_k=3) == []
    assert scorer.calls == []


def test_rerank_accepts_integer_scores() -> None:
    class IntScorer(ConstantScorer):
        def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
            return np.array([1, 3], dtype=np.int64)

    results = rerank("q", [cand("a", 1), cand("b", 2)], IntScorer(), top_k=2)
    assert [r.chunk_id for r in results] == ["b", "a"]
    assert isinstance(results[0].score, float)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_rerank_rejects_blank_query(query: str) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        rerank(query, [cand("a", 1)], ConstantScorer(), top_k=1)


def test_rerank_rejects_non_string_query() -> None:
    with pytest.raises(TypeError, match="query"):
        rerank(None, [cand("a", 1)], ConstantScorer(), top_k=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad, error", [(0, ValueError), (-3, ValueError), (2.0, TypeError), (True, TypeError)])
def test_rerank_validates_top_k(bad: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        rerank("q", [cand("a", 1)], ConstantScorer(), top_k=bad)


def test_rerank_rejects_scorer_missing_methods() -> None:
    class NoDescribe:
        def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
            return np.zeros(len(texts))

    with pytest.raises(TypeError, match="describe"):
        rerank("q", [cand("a", 1)], NoDescribe(), top_k=1)  # type: ignore[arg-type]


def test_rerank_rejects_duplicate_chunk_ids() -> None:
    with pytest.raises(ValueError, match="repeat"):
        rerank("q", [cand("a", 1), cand("a", 2)], ConstantScorer(), top_k=1)


def test_rerank_rejects_blank_candidate_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        rerank("q", [cand("a", 1, "  ")], ConstantScorer(), top_k=1)


@pytest.mark.parametrize(
    "output, error, match",
    [
        ([0.1, 0.2], TypeError, "ndarray"),  # list, not ndarray
        (np.array(["a", "b"]), TypeError, "real-valued"),
        (np.array([0.1]), ValueError, "shape"),
        (np.array([[0.1], [0.2]]), ValueError, "shape"),
        (np.array([np.nan, 0.2]), ValueError, "NaN"),
        (np.array([np.inf, 0.2]), ValueError, "NaN or infinite"),
    ],
)
def test_rerank_rejects_malformed_scorer_output(output: Any, error: type[Exception], match: str) -> None:
    class BadScorer:
        def score_pairs(self, query: str, texts: Sequence[str]) -> Any:
            return output

        def describe(self) -> dict[str, Any]:
            return {}

    with pytest.raises(error, match=match):
        rerank("q", [cand("a", 1), cand("b", 2)], BadScorer(), top_k=2)  # type: ignore[arg-type]


def test_result_dataclasses_are_frozen() -> None:
    result = rerank("q", [cand("a", 1)], ConstantScorer(), top_k=1)[0]
    with pytest.raises(FrozenInstanceError):
        result.score = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- RetrieveAndRerank


def make_pipeline(**kwargs: Any) -> tuple[RetrieveAndRerank, dict[str, FakeRetriever], TableScorer]:
    retrievers = {
        "bm25": FakeRetriever(["c1", "c2", "c3"]),
        "dense": FakeRetriever(["c3", "c4", "c1"]),
    }
    scorer = TableScorer({TEXTS["c4"]: 5.0, TEXTS["c3"]: 3.0})
    return RetrieveAndRerank(retrievers, TEXTS, scorer, **kwargs), retrievers, scorer


def test_single_retriever_skips_fusion_and_keeps_native_scores() -> None:
    retriever = FakeRetriever(["c2", "c1"])
    pipe = RetrieveAndRerank({"only": retriever}, TEXTS, ConstantScorer(), candidate_k=2)
    cands = pipe.candidates("q")
    assert [(c.chunk_id, c.retrieval_rank, c.retrieval_score) for c in cands] == [
        ("c2", 1, 10.0),
        ("c1", 2, 9.0),
    ]
    assert retriever.calls == [("q", 2)]
    assert pipe.config_record()["fusion"] == "none"


def test_multiple_retrievers_are_fused_with_rrf() -> None:
    pipe, _, _ = make_pipeline(candidate_k=10)
    cands = pipe.candidates("q")
    # c3: rank 3 in bm25 + rank 1 in dense; c1: rank 1 + rank 3; c1 and c3 tie on RRF.
    tie = 1 / (RRF_K + 1) + 1 / (RRF_K + 3)
    assert [c.chunk_id for c in cands] == ["c1", "c3", "c2", "c4"]
    assert cands[0].retrieval_score == pytest.approx(tie)
    assert cands[1].retrieval_score == pytest.approx(tie)
    assert [c.retrieval_rank for c in cands] == [1, 2, 3, 4]


def test_fusion_depth_defaults_to_candidate_k_and_can_be_overridden() -> None:
    pipe, retrievers, _ = make_pipeline(candidate_k=4)
    pipe.candidates("q")
    assert all(r.calls == [("q", 4)] for r in retrievers.values())

    pipe, retrievers, _ = make_pipeline(candidate_k=4, per_retriever_k=2)
    pipe.candidates("q")
    assert all(r.calls == [("q", 2)] for r in retrievers.values())


def test_candidate_k_caps_the_fused_list() -> None:
    pipe, _, _ = make_pipeline(candidate_k=2)
    assert len(pipe.candidates("q")) == 2


def test_fusion_ignores_a_retriever_that_returns_nothing() -> None:
    retrievers = {"bm25": FakeRetriever([]), "dense": FakeRetriever(["c3", "c1"])}
    pipe = RetrieveAndRerank(retrievers, TEXTS, ConstantScorer(), candidate_k=5)
    assert [c.chunk_id for c in pipe.candidates("q")] == ["c3", "c1"]


def test_fusion_with_every_retriever_empty_gives_no_candidates() -> None:
    retrievers = {"bm25": FakeRetriever([]), "dense": FakeRetriever([])}
    pipe = RetrieveAndRerank(retrievers, TEXTS, ConstantScorer())
    assert pipe.candidates("q") == []


def test_search_reranks_and_reports_cost() -> None:
    pipe, _, scorer = make_pipeline(candidate_k=10)
    out = pipe.search("q", top_k=2)
    assert [r.chunk_id for r in out.results] == ["c4", "c3"]
    assert out.n_candidates == 4
    assert out.scorer_calls == 1
    assert len(scorer.calls) == 1
    assert out.retrieve_ms >= 0 and out.rerank_ms >= 0
    assert out.latency_ms == pytest.approx(out.retrieve_ms + out.rerank_ms)
    # The audit trail points back to the first-stage position.
    assert out.results[0].retrieval_rank == 4


def test_search_with_no_candidates_reports_zero_scorer_calls() -> None:
    pipe = RetrieveAndRerank({"r": FakeRetriever([])}, TEXTS, ConstantScorer())
    out = pipe.search("q")
    assert out.results == []
    assert (out.n_candidates, out.scorer_calls) == (0, 0)


def test_missing_chunk_text_raises() -> None:
    pipe = RetrieveAndRerank({"r": FakeRetriever(["c1", "ghost"])}, TEXTS, ConstantScorer())
    with pytest.raises(ValueError, match="ghost"):
        pipe.candidates("q")


@pytest.mark.parametrize("method", ["candidates", "search"])
def test_blank_query_is_rejected_before_retrieval(method: str) -> None:
    retriever = FakeRetriever(["c1"])
    pipe = RetrieveAndRerank({"r": retriever}, TEXTS, ConstantScorer())
    with pytest.raises(ValueError, match="non-whitespace"):
        getattr(pipe, method)("   ")
    assert retriever.calls == []


def test_search_rejects_bad_top_k_before_retrieval() -> None:
    retriever = FakeRetriever(["c1"])
    pipe = RetrieveAndRerank({"r": retriever}, TEXTS, ConstantScorer())
    with pytest.raises(ValueError):
        pipe.search("q", top_k=0)
    assert retriever.calls == []


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"retrievers": {}}, ValueError),
        ({"retrievers": [FakeRetriever([])]}, ValueError),
        ({"retrievers": {"r": object()}}, TypeError),
        ({"scorer": object()}, TypeError),
        ({"candidate_k": 0}, ValueError),
        ({"candidate_k": 2.5}, TypeError),
        ({"rrf_k": 0}, ValueError),
        ({"per_retriever_k": 0}, ValueError),
    ],
)
def test_constructor_validation(kwargs: dict[str, Any], error: type[Exception]) -> None:
    base: dict[str, Any] = {
        "retrievers": {"r": FakeRetriever(["c1"])},
        "texts": TEXTS,
        "scorer": ConstantScorer(),
    }
    base.update(kwargs)
    with pytest.raises(error):
        RetrieveAndRerank(**base)


def test_config_record_is_json_serializable_and_complete() -> None:
    pipe, _, _ = make_pipeline(candidate_k=7, rrf_k=30, per_retriever_k=5)
    record = pipe.config_record()
    json.dumps(record)
    assert record["first_stage"] == ["bm25", "dense"]
    assert record["fusion"] == "reciprocal_rank_fusion"
    assert (record["rrf_k"], record["per_retriever_k"], record["candidate_k"]) == (30, 5, 7)
    assert record["tie_order"] == "(-rerank_score, retrieval_rank, chunk_id)"
    assert record["scorer"] == {"scorer": "table"}
    assert "timestamp" not in record  # records must be reproducible


def test_config_record_hides_fusion_settings_for_one_retriever() -> None:
    pipe = RetrieveAndRerank({"r": FakeRetriever(["c1"])}, TEXTS, ConstantScorer())
    record = pipe.config_record()
    assert (record["fusion"], record["rrf_k"], record["per_retriever_k"]) == ("none", None, None)


# --------------------------------------------------------------------------- real retrievers


def test_pipeline_runs_over_real_bm25_index() -> None:
    pipe = RetrieveAndRerank(
        {"bm25": BM25Index.build(DOCS)},
        TEXTS,
        TableScorer({TEXTS["c3"]: 1.0}),
        candidate_k=4,
    )
    out = pipe.search("apple net income 2022", top_k=2)
    assert out.results[0].chunk_id == "c3"
    assert out.n_candidates <= 4


def test_pipeline_fuses_real_bm25_and_dense_retrievers() -> None:
    pipe = RetrieveAndRerank(
        {"bm25": BM25Index.build(DOCS), "dense": DenseRetriever.build(DOCS, BagEncoder())},
        TEXTS,
        TableScorer({TEXTS["c3"]: 1.0}),
        candidate_k=5,
    )
    out = pipe.search("apple net income 2022", top_k=3)
    assert len(out.results) == 3
    assert out.results[0].chunk_id == "c3"
    # Results are deterministic across runs.
    again = pipe.search("apple net income 2022", top_k=3)
    assert [r.chunk_id for r in again.results] == [r.chunk_id for r in out.results]


def test_bm25_miss_yields_empty_result_without_scoring() -> None:
    scorer = TableScorer({})
    pipe = RetrieveAndRerank({"bm25": BM25Index.build(DOCS)}, TEXTS, scorer)
    out = pipe.search("zzzzqqqq")
    assert out.results == [] and out.scorer_calls == 0
    assert scorer.calls == []


# --------------------------------------------------------------------------- CrossEncoderScorer


def make_scorer(model: Any = None, **kwargs: Any) -> CrossEncoderScorer:
    return CrossEncoderScorer(
        model or StubCrossEncoder(), model_id="org/model", revision=REVISION, **kwargs
    )


def test_score_pairs_batches_every_pair_in_one_predict_call() -> None:
    model = StubCrossEncoder()
    scores = make_scorer(model, batch_size=8).score_pairs("q", ["aa", "bbbb"])
    assert model.last_call["pairs"] == [("q", "aa"), ("q", "bbbb")]
    assert model.last_call["batch_size"] == 8
    assert model.last_call["show_progress_bar"] is False
    assert model.last_call["convert_to_numpy"] is True
    assert scores.dtype == np.float64
    assert scores.tolist() == [2.0, 4.0]


def test_score_pairs_rejects_empty_texts() -> None:
    with pytest.raises(ValueError, match="at least one"):
        make_scorer().score_pairs("q", [])


def test_score_pairs_rejects_wrong_output_shape() -> None:
    scorer = make_scorer(StubCrossEncoder(scores=np.array([[0.1], [0.2]])))
    with pytest.raises(ValueError, match="shape"):
        scorer.score_pairs("q", ["a", "b"])


@pytest.mark.parametrize("revision", ["main", "a" * 7, "A" * 40, "a" * 41, ""])
def test_constructor_requires_full_lowercase_commit_hash(revision: str) -> None:
    with pytest.raises(ValueError, match="40-character"):
        CrossEncoderScorer(StubCrossEncoder(), model_id="m", revision=revision)


def test_constructor_requires_predict_method() -> None:
    with pytest.raises(TypeError, match="predict"):
        CrossEncoderScorer(object(), model_id="m", revision=REVISION)


def test_constructor_validates_batch_size() -> None:
    with pytest.raises(ValueError):
        make_scorer(batch_size=0)


def test_max_seq_length_is_set_on_models_that_expose_it() -> None:
    model = StubCrossEncoder()
    scorer = make_scorer(model, max_seq_length=256)
    assert model.max_seq_length == 256
    assert scorer.describe()["max_seq_length"] == 256


def test_max_seq_length_falls_back_to_legacy_max_length_attribute() -> None:
    class LegacyModel:
        max_length = 512

        def predict(self, pairs: Any, **kwargs: Any) -> np.ndarray:
            return np.zeros(len(pairs))

    model = LegacyModel()
    scorer = make_scorer(model, max_seq_length=128)
    assert model.max_length == 128
    assert not hasattr(model, "max_seq_length")
    assert scorer.describe()["max_seq_length"] == 128


def test_describe_records_identity_and_settings() -> None:
    desc = make_scorer(device="cuda", batch_size=4).describe()
    json.dumps(desc)
    assert desc == {
        "scorer": "sentence-transformers-cross-encoder",
        "model_id": "org/model",
        "revision": REVISION,
        "device": "cuda",
        "batch_size": 4,
        "max_seq_length": 512,
    }


def test_load_rejects_unpinned_revision_before_importing_anything() -> None:
    with pytest.raises(ValueError, match="40-character"):
        CrossEncoderScorer.load("org/model", "main")


def test_cross_encoder_scorer_plugs_into_the_pipeline() -> None:
    model = StubCrossEncoder(scores=np.array([0.2, 0.9], dtype=np.float32))
    pipe = RetrieveAndRerank({"r": FakeRetriever(["c1", "c2"])}, TEXTS, make_scorer(model))
    out = pipe.search("q", top_k=2)
    assert [r.chunk_id for r in out.results] == ["c2", "c1"]
    assert pipe.config_record()["scorer"]["model_id"] == "org/model"


# --------------------------------------------------------------------------- imports


def test_importing_rerank_does_not_load_sentence_transformers_or_torch() -> None:
    code = (
        "import sys; import src.retrieval.rerank; "
        "bad = [m for m in ('sentence_transformers', 'torch') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
