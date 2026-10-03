"""Tests for the fixed Lucene-style Okapi BM25 baseline (issue #9).

Every fixture here is a synthetic string; no FinanceBench row is used. Expected
values are derived by hand from the formula in :mod:`src.retrieval.bm25`, never by
running the scorer, so a regression in the scorer cannot also move its expectation.

The load-bearing tests are :func:`test_scores_match_hand_computed_values`, which pins
the formula, and :func:`test_token_pattern_is_the_approved_alternation`, which pins
the tokenizer regex against the escaped-pipe mistake.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import asdict, fields

import pytest

from src.retrieval.bm25 import (
    B,
    BM25Index,
    K1,
    RankedResult,
    THOUSANDS_SEPARATOR,
    TOKEN_PATTERN,
    bm25_idf,
    config_record,
    tokenize,
)

#: Three chunks small enough to score by hand. Lengths 2, 3, 2; avgdl = 7/3.
HAND_CORPUS = [
    ("d1", "revenue grew"),
    ("d2", "revenue fell sharply"),
    ("d3", "costs grew"),
]


@pytest.fixture(scope="module")
def hand_index() -> BM25Index:
    """The hand-computable index."""
    return BM25Index.build(HAND_CORPUS)


# --- tokenizer ------------------------------------------------------------------


def test_tokenize_is_deterministic_and_pure() -> None:
    text = "Capital expenditure for FY2022 was $1,234.5 million (10-K)."
    first = tokenize(text)
    first.append("mutated")
    assert tokenize(text) == tokenize(text) == [
        "capital", "expenditure", "for", "fy2022", "was", "1234.5", "million", "10", "k",
    ]


def test_tokenize_applies_nfkc() -> None:
    assert tokenize("ﬁnancial") == ["financial"]  # "fi" ligature from PDF text
    assert tokenize("１２３") == ["123"]  # full-width digits
    assert tokenize("ＦＹ２０２２") == ["fy2022"]  # full-width FY2022


def test_tokenize_casefolds() -> None:
    assert tokenize("Revenue REVENUE revenue") == ["revenue", "revenue", "revenue"]
    assert tokenize("Straße") == ["strasse"]


def test_tokenize_distinguishes_10k_and_10q() -> None:
    assert tokenize("10-K") == ["10", "k"]
    assert tokenize("10-Q") == ["10", "q"]
    assert tokenize("10-K") != tokenize("10-Q")
    assert tokenize("Form 10-K405") == ["form", "10", "k405"]


def test_tokenize_keeps_fiscal_tokens_whole() -> None:
    assert tokenize("FY2022") == ["fy2022"]
    assert tokenize("Q4FY22") == ["q4fy22"]
    assert tokenize("H1-2023") == ["h1", "2023"]


def test_tokenize_normalizes_formatted_numbers() -> None:
    assert tokenize("$1,234.5") == ["1234.5"]
    assert tokenize("(1,577)") == ["1577"]
    assert tokenize("$1,234,567.89") == ["1234567.89"]
    assert tokenize("1.2.3") == ["1.2.3"]
    assert tokenize("0.5x") == ["0.5", "x"]
    # Only a comma followed by exactly three digits is a digit-group separator.
    assert tokenize("2019, 2020") == ["2019", "2020"]
    assert tokenize("2019,2020") == ["2019", "2020"]
    assert tokenize("1,2") == ["1", "2"]


def test_tokenize_drops_percent_sign() -> None:
    assert tokenize("12.5%") == ["12.5"]
    assert tokenize("margin of 12.5 %") == ["margin", "of", "12.5"]


def test_tokenize_splits_hyphenated_terms() -> None:
    assert tokenize("capital-expenditure") == ["capital", "expenditure"]
    assert tokenize("Capital-Expenditure") == tokenize("capital expenditure")
    assert tokenize("year-over-year") == ["year", "over", "year"]
    assert tokenize("snake_case") == ["snake", "case"]


def test_tokenize_keeps_one_letter_tokens() -> None:
    assert tokenize("U.S. PP&E") == ["u", "s", "pp", "e"]
    assert tokenize("3M’s") == ["3m", "s"]


def test_tokenize_pins_nfkc_superscript_behavior() -> None:
    """NFKC turns a footnote superscript into a digit; pinned so a change is noticed."""
    assert tokenize("EBITDA¹") == ["ebitda1"]


def test_tokenize_returns_empty_for_text_without_tokens() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("?! % $") == []


@pytest.mark.parametrize("bad", [None, b"bytes", 12, ["revenue"]])
def test_tokenize_rejects_non_string(bad: object) -> None:
    with pytest.raises(TypeError):
        tokenize(bad)  # type: ignore[arg-type]


def test_token_pattern_is_the_approved_alternation() -> None:
    """The alternation pipe must be a real ``|``, not an escaped literal ``\\|``."""
    assert TOKEN_PATTERN.pattern == r"\d+(?:\.\d+)+|[^\W_]+"
    assert "\\|" not in TOKEN_PATTERN.pattern
    assert THOUSANDS_SEPARATOR.pattern == r"(?<=\d),(?=\d{3}(?!\d))"


# --- IDF ------------------------------------------------------------------------


def test_idf_matches_lucene_formula() -> None:
    # ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1 + 1.5 / 2.5) = ln(1.6)
    assert bm25_idf(2, 3) == pytest.approx(math.log(1.6), abs=1e-12)
    # ln(1 + (3 - 1 + 0.5) / (1 + 0.5)) = ln(8 / 3)
    assert bm25_idf(1, 3) == pytest.approx(math.log(8 / 3), abs=1e-12)


@pytest.mark.parametrize("n_docs", [1, 7, 125, 10_000])
def test_idf_is_positive_when_every_chunk_has_the_term(n_docs: int) -> None:
    assert bm25_idf(n_docs, n_docs) > 0


def test_idf_strictly_decreases_with_df() -> None:
    values = [bm25_idf(df, 125) for df in range(1, 126)]
    assert all(earlier > later for earlier, later in zip(values, values[1:]))


@pytest.mark.parametrize("df, n_docs", [(0, 5), (6, 5), (1, 0), (-1, 5)])
def test_idf_rejects_out_of_range_df(df: int, n_docs: int) -> None:
    with pytest.raises(ValueError):
        bm25_idf(df, n_docs)


@pytest.mark.parametrize("df, n_docs", [(1.0, 5), (True, 5), (1, "5"), (1, None)])
def test_idf_rejects_non_integer_arguments(df: object, n_docs: object) -> None:
    with pytest.raises(TypeError):
        bm25_idf(df, n_docs)  # type: ignore[arg-type]


# --- index construction ---------------------------------------------------------


def test_index_statistics_match_hand_counts(hand_index: BM25Index) -> None:
    assert hand_index.n_docs == 3
    assert hand_index.chunk_ids == ("d1", "d2", "d3")
    assert hand_index.doc_lengths == (2, 3, 2)
    assert hand_index.avgdl == pytest.approx(7 / 3)
    document_frequency = {term: len(entries) for term, entries in hand_index.postings.items()}
    assert document_frequency == {"costs": 1, "fell": 1, "grew": 2, "revenue": 2, "sharply": 1}
    assert hand_index.postings["revenue"] == ((0, 1), (1, 1))
    assert hand_index.idf["revenue"] == pytest.approx(math.log(1.6), abs=1e-12)


def test_fixed_parameters_are_lucene_defaults() -> None:
    assert K1 == 1.2
    assert B == 0.75


# --- scoring --------------------------------------------------------------------


def test_scores_match_hand_computed_values(hand_index: BM25Index) -> None:
    """Hand derivation, k1 = 1.2, b = 0.75, avgdl = 7/3, IDF(revenue) = IDF(grew) = ln 1.6.

    A term with tf = 1 in a chunk of length L contributes
    ln(1.6) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * L / (7/3))).
    d1 (L = 2) matches both terms: 2 * 0.4991764 = 0.998353.
    d3 (L = 2) matches "grew":            0.499176.
    d2 (L = 3) matches "revenue":         0.420817.
    d3 outranks d2 only because it is shorter: same tf, same IDF.
    """
    results = hand_index.retrieve("Revenue grew?", k=10)
    assert [r.chunk_id for r in results] == ["d1", "d3", "d2"]
    expected = {"d1": 0.998353, "d3": 0.499176, "d2": 0.420817}
    for result in results:
        assert result.score == pytest.approx(expected[result.chunk_id], abs=1e-6)

    def contribution(length: int) -> float:
        return math.log(1.6) * (K1 + 1) / (1 + K1 * (1 - B + B * length / (7 / 3)))

    by_id = {r.chunk_id: r.score for r in results}
    assert by_id["d1"] == pytest.approx(2 * contribution(2), abs=1e-12)
    assert by_id["d3"] == pytest.approx(contribution(2), abs=1e-12)
    assert by_id["d2"] == pytest.approx(contribution(3), abs=1e-12)


def test_repeated_query_word_counts_each_time(hand_index: BM25Index) -> None:
    once = {r.chunk_id: r.score for r in hand_index.retrieve("grew", k=10)}
    twice = {r.chunk_id: r.score for r in hand_index.retrieve("grew grew", k=10)}
    assert set(once) == set(twice) == {"d1", "d3"}
    for chunk_id in once:
        assert twice[chunk_id] == 2 * once[chunk_id]


def test_no_match_query_returns_empty(hand_index: BM25Index) -> None:
    assert hand_index.retrieve("dividend payout", k=5) == []


def test_question_without_tokens_returns_empty(hand_index: BM25Index) -> None:
    assert hand_index.retrieve("?!", k=5) == []


def test_zero_score_chunks_are_excluded_even_when_k_exceeds_corpus() -> None:
    index = BM25Index.build([("a", "revenue grew"), ("b", "costs fell"), ("c", "margin")])
    results = index.retrieve("revenue", k=100)
    assert [r.chunk_id for r in results] == ["a"]
    assert all(r.score > 0 for r in results)


def test_k_one_returns_only_the_top_result(hand_index: BM25Index) -> None:
    results = hand_index.retrieve("revenue grew", k=1)
    assert [(r.rank, r.chunk_id) for r in results] == [(1, "d1")]


def test_k_larger_than_matches_returns_all_matches_without_padding(
    hand_index: BM25Index,
) -> None:
    assert len(hand_index.retrieve("revenue grew", k=50)) == 3
    assert len(hand_index.retrieve("grew", k=50)) == 2


def test_ties_break_by_ascending_chunk_id() -> None:
    documents = [("chunk-b", "net revenue"), ("chunk-c", "costs"), ("chunk-a", "net revenue")]
    for ordering in (documents, list(reversed(documents))):
        results = BM25Index.build(ordering).retrieve("revenue", k=10)
        assert [r.chunk_id for r in results] == ["chunk-a", "chunk-b"]
        assert results[0].score == results[1].score


def test_ranks_are_contiguous_from_one(hand_index: BM25Index) -> None:
    results = hand_index.retrieve("revenue grew", k=10)
    assert [r.rank for r in results] == [1, 2, 3]


def test_zero_token_chunk_is_indexed_but_never_returned() -> None:
    index = BM25Index.build([("a", "revenue grew"), ("empty", ""), ("punct", "?!")])
    assert index.n_docs == 3
    assert index.doc_lengths == (2, 0, 0)
    assert index.avgdl == pytest.approx(2 / 3)
    # N = 3 includes the empty chunks: ln(1 + (3 - 1 + 0.5) / (1 + 0.5)) = ln(8/3).
    assert index.idf["revenue"] == pytest.approx(math.log(8 / 3), abs=1e-12)
    assert [r.chunk_id for r in index.retrieve("revenue grew", k=10)] == ["a"]


# --- input validation -----------------------------------------------------------


def test_duplicate_chunk_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        BM25Index.build([("a", "revenue"), ("b", "costs"), ("a", "margin")])


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Index.build([])


def test_corpus_with_no_tokens_is_rejected() -> None:
    with pytest.raises(ValueError, match="no chunk contains a token"):
        BM25Index.build([("a", ""), ("b", "?! %")])


@pytest.mark.parametrize("text", [None, b"revenue", 5, ["revenue"]])
def test_chunk_text_must_be_a_string(text: object) -> None:
    with pytest.raises(TypeError):
        BM25Index.build([("a", text)])  # type: ignore[list-item]


@pytest.mark.parametrize("chunk_id", [None, 5, b"a"])
def test_chunk_id_must_be_a_string(chunk_id: object) -> None:
    with pytest.raises(TypeError):
        BM25Index.build([(chunk_id, "revenue")])  # type: ignore[list-item]


def test_chunk_id_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        BM25Index.build([("", "revenue")])


def test_documents_must_be_chunk_id_text_pairs() -> None:
    with pytest.raises(TypeError):
        BM25Index.build(["revenue grew"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        BM25Index.build({"a": "revenue grew"})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BM25Index.build([("a", "revenue", "extra")])  # type: ignore[list-item]


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_is_rejected(hand_index: BM25Index, k: int) -> None:
    with pytest.raises(ValueError):
        hand_index.retrieve("revenue", k=k)


@pytest.mark.parametrize("k", [1.5, "5", True, None])
def test_non_integer_k_is_rejected(hand_index: BM25Index, k: object) -> None:
    with pytest.raises(TypeError):
        hand_index.retrieve("revenue", k=k)  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_question_is_rejected(hand_index: BM25Index, query: str) -> None:
    with pytest.raises(ValueError):
        hand_index.retrieve(query, k=5)


@pytest.mark.parametrize("query", [None, b"revenue", 3])
def test_non_string_question_is_rejected(hand_index: BM25Index, query: object) -> None:
    with pytest.raises(TypeError):
        hand_index.retrieve(query, k=5)  # type: ignore[arg-type]


# --- determinism and boundary ---------------------------------------------------


def test_input_order_does_not_change_rankings() -> None:
    documents = [
        ("p3", "capital expenditure rose"),
        ("p1", "capital expenditure fell in FY2022"),
        ("p4", "revenue grew"),
        ("p2", "capital-expenditure guidance"),
    ]
    permutations = [documents, list(reversed(documents)), [documents[i] for i in (2, 0, 3, 1)]]
    rankings = [
        BM25Index.build(order).retrieve("capital expenditure FY2022", k=10) for order in permutations
    ]
    assert rankings[0] == rankings[1] == rankings[2]
    assert [r.chunk_id for r in rankings[0]][0] == "p1"


def test_repeated_builds_and_queries_are_identical() -> None:
    def run() -> list[RankedResult]:
        return BM25Index.build(HAND_CORPUS).retrieve("revenue grew", k=10)

    first, second = run(), run()
    assert first == second
    serialized = [json.dumps([asdict(r) for r in run_], sort_keys=True) for run_ in (first, second)]
    assert serialized[0].encode("utf-8") == serialized[1].encode("utf-8")


def test_config_record_is_fixed_and_timestamp_free() -> None:
    record = config_record()
    assert record == config_record()
    assert record["k1"] == 1.2
    assert record["b"] == 0.75
    assert record["variant"] == "lucene-style-okapi-bm25"
    assert record["idf"] == "ln(1 + (N - df + 0.5) / (df + 0.5))"
    assert record["idf_floor"] == "none"
    assert record["token_pattern"] == r"\d+(?:\.\d+)+|[^\W_]+"
    assert record["stopwords"] == "none"
    assert record["stemming"] == "none"
    assert record["ngrams"] == 1
    assert record["tie_order"] == "(-score, chunk_id)"
    assert record["tuning"] == "none"
    blob = json.dumps(record).lower()
    for forbidden in ("timestamp", "created_at", "generated_at", "datetime"):
        assert forbidden not in blob


def test_retrieve_accepts_only_query_and_k() -> None:
    assert list(inspect.signature(BM25Index.retrieve).parameters) == ["self", "query", "k"]


def test_ranked_result_has_no_gold_fields() -> None:
    assert [f.name for f in fields(RankedResult)] == ["rank", "chunk_id", "score"]
