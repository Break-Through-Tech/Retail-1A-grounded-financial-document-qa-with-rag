"""Tests for the response objects and the grounded pipeline (``response.py``, ``pipeline.py``).

Retrieval is a fixed ranked list and the scorer a lookup table, so every score is known.
The load-bearing tests are the ones that prove the generator is never reached when the
evidence is insufficient (:func:`test_the_generator_is_never_called_when_evidence_is_insufficient`)
and that an abstention is a well-formed response with a null answer.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

from src.abstention.pipeline import EvidenceChunk, GroundedQA
from src.abstention.policy import EvidencePolicy, PolicyMismatchError
from src.abstention.response import (
    ABSTENTION_REASONS,
    GeneratedAnswer,
    abstention_message,
    answer_response,
    insufficient_evidence_response,
    validate_response,
)
from src.retrieval.rerank import RetrieveAndRerank

META = {
    "m18a": {"doc_name": "3M_2018_10K", "page_num": 59, "company": "3M", "doc_type": "10k", "doc_period": 2018, "doc_link": "u1"},
    "m18b": {"doc_name": "3M_2018_10K", "page_num": 60, "company": "3M", "doc_type": "10k", "doc_period": 2018, "doc_link": "u1"},
    "m22": {"doc_name": "3M_2022_10K", "page_num": 7, "company": "3M", "doc_type": "10k", "doc_period": 2022, "doc_link": "u2"},
    "amd": {"doc_name": "AMD_2018_10K", "page_num": 3, "company": "AMD", "doc_type": "10k", "doc_period": 2018, "doc_link": "u3"},
}
TEXTS = {cid: f"text of {cid}" for cid in META}
Q = "What is 3M's FY2018 capital expenditure?"
ROW = {"financebench_id": "fb_1", "question_type": "metrics-generated", "question_reasoning": "Information extraction"}


class FakeRetriever:
    def __init__(self, chunk_ids: Sequence[str]) -> None:
        self._hits = [SimpleNamespace(chunk_id=c, score=10.0 - i) for i, c in enumerate(chunk_ids)]

    def retrieve(self, query: str, k: int) -> list[Any]:
        return self._hits[:k]


class TableScorer:
    def __init__(self, table: dict[str, float]) -> None:
        self.table = table

    def score_pairs(self, query: str, texts: Sequence[str]) -> np.ndarray:
        return np.array([self.table.get(t, 0.0) for t in texts], dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"scorer": "table"}


class RecordingGenerator:
    def __init__(self, result: GeneratedAnswer | None = None) -> None:
        self.calls: list[tuple[str, list[EvidenceChunk]]] = []
        self.result = result

    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> GeneratedAnswer:
        self.calls.append((question, list(evidence)))
        if self.result is not None:
            return self.result
        return GeneratedAnswer("$1,577 million", ["capex = purchases of PP&E"], [evidence[0].chunk_id], model_calls=2)


def make_qa(
    scores: dict[str, float],
    retrieved: Sequence[str] = ("m22", "amd", "m18a", "m18b"),
    *,
    threshold: float = 1.0,
    generator: RecordingGenerator | None = None,
    context_k: int = 5,
    policy: EvidencePolicy | None = None,
) -> tuple[GroundedQA, RecordingGenerator]:
    generator = generator or RecordingGenerator()
    search = RetrieveAndRerank(
        {"r": FakeRetriever(retrieved)},
        TEXTS,
        TableScorer({TEXTS[c]: s for c, s in scores.items()}),
        candidate_k=10,
    )
    policy = policy or EvidencePolicy(threshold=threshold, scorer={"scorer": "table"})
    return GroundedQA(search, policy, META, TEXTS, generator, context_k=context_k), generator


# --------------------------------------------------------------------------- the pipeline


def test_strong_in_scope_evidence_is_answered_with_citations() -> None:
    qa, generator = make_qa({"m18a": 3.0, "m18b": 2.0, "m22": 9.0, "amd": 8.0})
    response = qa.answer(Q, ROW)
    validate_response(response)
    assert response["abstained"] is False and response["answer"] == "$1,577 million"
    assert response["calculation"] == {"shown": True, "steps": ["capex = purchases of PP&E"]}
    assert [c["doc_name"] for c in response["citations"]] == ["3M_2018_10K"]
    assert response["citations"][0]["rerank_score"] == 3.0
    assert response["confidence"] >= 0.5
    assert len(generator.calls) == 1


def test_the_generator_sees_only_in_scope_chunks_best_first() -> None:
    qa, generator = make_qa({"m18a": 2.0, "m18b": 3.0, "m22": 9.0, "amd": 8.0})
    qa.answer(Q)
    _, evidence = generator.calls[0]
    assert [e.chunk_id for e in evidence] == ["m18b", "m18a"]
    assert evidence[0] == EvidenceChunk("m18b", "text of m18b", "3M_2018_10K", 60, 3.0)


def test_uncited_results_go_to_retrieved_but_unused() -> None:
    qa, _ = make_qa({"m18a": 3.0, "m18b": 2.0, "m22": 9.0, "amd": 8.0})
    unused = qa.answer(Q)["retrieved_but_unused"]
    assert {"doc_name": "3M_2022_10K", "evidence_page_num": 7} in unused
    assert {"doc_name": "3M_2018_10K", "evidence_page_num": 60} in unused
    assert {"doc_name": "3M_2018_10K", "evidence_page_num": 59} not in unused  # cited


def test_weak_evidence_returns_an_insufficient_evidence_response() -> None:
    qa, generator = make_qa({"m18a": 0.4, "m18b": 0.3, "m22": 9.0})
    response = qa.answer(Q, ROW)
    validate_response(response)
    assert response["abstained"] is True and response["answer"] is None
    assert response["citations"] == [] and response["confidence"] < 0.5
    assert response["abstention"]["reason"] == "below_threshold"
    assert response["abstention"]["top_score"] == 0.4 and response["abstention"]["threshold"] == 1.0
    assert "cannot be answered from the available documents" in response["abstention"]["message"]
    assert (response["financebench_id"], response["question_type"]) == ("fb_1", "metrics-generated")
    assert generator.calls == []


def test_a_strong_match_from_the_wrong_filing_is_still_a_refusal() -> None:
    qa, generator = make_qa({"m22": 9.0, "amd": 8.0}, retrieved=("m22", "amd"))
    response = qa.answer(Q)
    assert response["abstention"]["reason"] == "out_of_scope"
    assert response["abstention"]["named_companies"] == ["3M"] and response["abstention"]["named_years"] == [2018]
    assert response["confidence"] == 0.0 and generator.calls == []
    assert len(response["retrieved_but_unused"]) == 2


def test_no_results_is_a_refusal_that_costs_no_model_call() -> None:
    qa, generator = make_qa({}, retrieved=())
    response = qa.answer(Q)
    validate_response(response)
    assert response["abstention"]["reason"] == "no_results"
    assert response["model_calls"] == 0 and response["retrieved_but_unused"] == []
    assert generator.calls == []


@pytest.mark.parametrize("best", [-5.0, 0.0, 0.5, 0.999999])
def test_the_generator_is_never_called_when_evidence_is_insufficient(best: float) -> None:
    qa, generator = make_qa({"m18a": best, "m18b": best - 1})
    assert qa.answer(Q)["abstained"] is True
    assert generator.calls == []


def test_a_refusal_costs_one_scorer_call_and_no_generation() -> None:
    qa, _ = make_qa({"m18a": 0.1})
    assert qa.answer(Q)["model_calls"] == 1


def test_an_answer_counts_scorer_and_generator_calls() -> None:
    qa, _ = make_qa({"m18a": 3.0})
    assert qa.answer(Q)["model_calls"] == 1 + 2


def test_a_declining_generator_produces_an_abstention() -> None:
    qa, generator = make_qa({"m18a": 3.0}, generator=RecordingGenerator(GeneratedAnswer(None, model_calls=1)))
    response = qa.answer(Q)
    validate_response(response)
    assert response["abstention"]["reason"] == "generator_declined"
    assert response["answer"] is None and response["confidence"] == 0.0
    assert response["model_calls"] == 2 and len(generator.calls) == 1


@pytest.mark.parametrize("text", ["", "   \n"])
def test_a_blank_answer_counts_as_declining(text: str) -> None:
    qa, _ = make_qa({"m18a": 3.0}, generator=RecordingGenerator(GeneratedAnswer(text, cited_chunk_ids=["m18a"])))
    assert qa.answer(Q)["abstention"]["reason"] == "generator_declined"


def test_an_answer_that_cites_nothing_is_not_returned() -> None:
    qa, _ = make_qa({"m18a": 3.0}, generator=RecordingGenerator(GeneratedAnswer("$1,577 million")))
    response = qa.answer(Q)
    validate_response(response)
    assert response["abstention"]["reason"] == "uncited_answer" and response["answer"] is None


def test_an_answer_citing_a_chunk_outside_the_evidence_is_a_bug_not_an_answer() -> None:
    # m22 was dropped by the scope check, so the generator was never shown it.
    bad = GeneratedAnswer("x", cited_chunk_ids=["m22"])
    qa, _ = make_qa({"m18a": 3.0, "m22": 9.0}, generator=RecordingGenerator(bad))
    with pytest.raises(ValueError, match="outside the evidence"):
        qa.answer(Q)


def test_free_form_questions_have_null_benchmark_fields() -> None:
    qa, _ = make_qa({"m18a": 0.1})
    response = qa.answer(Q)
    assert (response["financebench_id"], response["question_type"], response["question_reasoning"]) == (None, None, None)


def test_responses_are_strict_json() -> None:
    for scores in ({"m18a": 3.0}, {"m18a": 0.1}):
        qa, _ = make_qa(scores)
        json.dumps(qa.answer(Q, ROW), allow_nan=False)


def test_latency_is_a_whole_number_of_milliseconds() -> None:
    qa, _ = make_qa({"m18a": 3.0})
    assert type(qa.answer(Q)["latency_ms"]) is int


def test_pipeline_refuses_a_policy_tuned_for_another_scorer() -> None:
    other = EvidencePolicy(threshold=1.0, scorer={"scorer": "cross-encoder", "revision": "a" * 40})
    with pytest.raises(PolicyMismatchError, match="different scorer"):
        make_qa({}, policy=other)


def test_pipeline_refuses_a_policy_tuned_with_another_context_size() -> None:
    tuned = {"first_stage": ["r"], "fusion": "none", "rrf_k": None, "per_retriever_k": None, "candidate_k": 10, "context_k": 5}
    policy = EvidencePolicy(threshold=1.0, scorer={"scorer": "table"}, tuning={"pipeline": tuned})
    make_qa({}, policy=policy, context_k=5)
    with pytest.raises(PolicyMismatchError, match="retrieval settings differ"):
        make_qa({}, policy=policy, context_k=3)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_context_k_must_be_a_positive_int(bad: Any) -> None:
    with pytest.raises(ValueError, match="context_k"):
        make_qa({}, context_k=bad)


# --------------------------------------------------------------------------- messages


def _decision(**kw: Any) -> Any:
    base = dict(top_score=0.4, threshold=1.0, named_companies=("3M",), named_years=(2018,))
    return SimpleNamespace(**{**base, **kw})


@pytest.mark.parametrize("reason", ABSTENTION_REASONS)
def test_every_message_states_the_insufficient_evidence_plainly(reason: str) -> None:
    message = abstention_message(reason, _decision())
    assert message.startswith("Insufficient evidence:")
    assert message.endswith("cannot be answered from the available documents.")


def test_below_threshold_message_reports_the_scores() -> None:
    assert "best score 0.40, required 1.00" in abstention_message("below_threshold", _decision())


def test_out_of_scope_message_names_what_the_question_asked_about() -> None:
    assert "3M / fiscal year 2018" in abstention_message("out_of_scope", _decision())
    many = _decision(named_companies=("3M", "AMD"), named_years=(2018, 2019))
    assert "3M and AMD / fiscal years 2018, 2019" in abstention_message("out_of_scope", many)


def test_unknown_reason_is_rejected() -> None:
    with pytest.raises(ValueError, match="reason must be one of"):
        abstention_message("because", _decision())


# --------------------------------------------------------------------------- builders


def test_answer_response_rejects_an_answer_with_no_citation() -> None:
    qa, _ = make_qa({"m18a": 3.0})
    out = qa._search.search(Q, top_k=5)
    decision = qa._policy.decide(Q, out.results, META)
    with pytest.raises(ValueError, match="at least one chunk"):
        answer_response(ROW, decision, GeneratedAnswer("x"), out.results, META, TEXTS, latency_ms=1, model_calls=1)


def test_insufficient_evidence_response_rejects_an_unknown_reason() -> None:
    qa, _ = make_qa({"m18a": 0.1})
    out = qa._search.search(Q, top_k=5)
    decision = qa._policy.decide(Q, out.results, META)
    with pytest.raises(ValueError, match="reason must be one of"):
        insufficient_evidence_response(ROW, decision, out.results, META, latency_ms=1, model_calls=1, reason="nope")


def test_duplicate_pages_are_listed_once_in_the_audit_trail() -> None:
    meta = {**META, "m18a2": dict(META["m18a"])}
    results = [SimpleNamespace(chunk_id="m18a", score=1.0), SimpleNamespace(chunk_id="m18a2", score=0.5)]
    decision = EvidencePolicy(threshold=5.0).decide(Q, results, meta)
    response = insufficient_evidence_response(None, decision, results, meta, latency_ms=0, model_calls=0)
    assert response["retrieved_but_unused"] == [{"doc_name": "3M_2018_10K", "evidence_page_num": 59}]


# --------------------------------------------------------------------------- validation


def _good_abstention() -> dict[str, Any]:
    qa, _ = make_qa({"m18a": 0.1})
    return qa.answer(Q, ROW)


def _good_answer() -> dict[str, Any]:
    qa, _ = make_qa({"m18a": 3.0})
    return qa.answer(Q, ROW)


def test_validate_accepts_both_good_responses() -> None:
    validate_response(_good_abstention())
    validate_response(_good_answer())


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda r: r.pop("confidence"), "missing key 'confidence'"),
        (lambda r: r.update(answer="a guess"), "answer null"),
        (lambda r: r.update(confidence=0.9), "below 0.5"),
        (lambda r: r.update(citations=[{}]), "no citations"),
        (lambda r: r.pop("abstention"), "abstention block"),
        (lambda r: r["abstention"].update(reason="whim"), "abstention block"),
        (lambda r: r["abstention"].update(message="I think so"), "state the insufficient evidence"),
        (lambda r: r.update(abstained="yes"), "abstained must be a bool"),
        (lambda r: r.update(latency_ms=1.5), "must be ints"),
        (lambda r: r.update(calculation={"shown": False}), "calculation"),
    ],
)
def test_validate_rejects_a_malformed_abstention(mutate: Any, expected: str) -> None:
    response = _good_abstention()
    mutate(response)
    with pytest.raises(ValueError, match=expected):
        validate_response(response)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda r: r.update(answer=None), "non-empty string"),
        (lambda r: r.update(answer="  "), "non-empty string"),
        (lambda r: r.update(citations=[]), "at least one citation"),
        (lambda r: r.update(confidence=0.2), "at least 0.5"),
        (lambda r: r["citations"][0].pop("rerank_score"), "missing"),
    ],
)
def test_validate_rejects_a_malformed_answer(mutate: Any, expected: str) -> None:
    response = _good_answer()
    mutate(response)
    with pytest.raises(ValueError, match=expected):
        validate_response(response)
