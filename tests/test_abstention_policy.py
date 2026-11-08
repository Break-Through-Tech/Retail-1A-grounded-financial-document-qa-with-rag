"""Tests for the insufficient-evidence rule (``src/abstention/policy.py``).

Chunks are stand-ins with a ``chunk_id`` and a ``score``; no model and no corpus is
involved. Expected confidences are worked out by hand: sigmoid(ln 3) is exactly 0.75.

The load-bearing tests are :func:`test_out_of_scope_chunks_never_raise_the_best_score`,
because the whole point of the scope check is that a strong match from the wrong
filing must not buy an answer, and :func:`test_an_abstention_never_has_confidence_of_half`,
which pins the invariant the evaluation relies on.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.abstention.policy import (
    POLICY_VERSION,
    REASONS,
    EvidencePolicy,
    PolicyMismatchError,
    confidence_from_score,
)


def hit(chunk_id: str, score: float) -> Any:
    return SimpleNamespace(chunk_id=chunk_id, score=score)


META = {
    "m18": {"company": "3M", "doc_period": 2018},
    "m22": {"company": "3M", "doc_period": 2022},
    "amd": {"company": "AMD", "doc_period": 2018},
}
Q = "What is 3M's FY2018 capital expenditure?"  # names 3M and 2018


def policy(**kwargs: Any) -> EvidencePolicy:
    kwargs.setdefault("threshold", 1.0)
    kwargs.setdefault("scorer", {"scorer": "stub"})
    return EvidencePolicy(**kwargs)


# --------------------------------------------------------------------------- decisions


def test_strong_in_scope_evidence_is_sufficient() -> None:
    decision = policy().decide(Q, [hit("m18", 3.0)], META)
    assert (decision.sufficient, decision.reason, decision.top_score) == (True, "supported", 3.0)
    assert decision.reason in REASONS


def test_weak_in_scope_evidence_is_below_threshold() -> None:
    decision = policy().decide(Q, [hit("m18", 0.5)], META)
    assert (decision.sufficient, decision.reason, decision.top_score) == (False, "below_threshold", 0.5)


def test_a_score_equal_to_the_threshold_is_sufficient() -> None:
    assert policy(threshold=2.0).decide(Q, [hit("m18", 2.0)], META).sufficient
    assert not policy(threshold=2.0).decide(Q, [hit("m18", 1.999999)], META).sufficient


def test_no_results_means_no_evidence() -> None:
    decision = policy().decide(Q, [], META)
    assert (decision.sufficient, decision.reason, decision.confidence, decision.top_score) == (
        False,
        "no_results",
        0.0,
        None,
    )


def test_evidence_from_other_filings_only_is_out_of_scope() -> None:
    decision = policy().decide(Q, [hit("m22", 9.0), hit("amd", 8.0)], META)
    assert (decision.sufficient, decision.reason, decision.confidence, decision.top_score) == (
        False,
        "out_of_scope",
        0.0,
        None,
    )
    assert decision.named_companies == ("3M",) and decision.named_years == (2018,)
    assert [c.chunk_id for c in decision.ineligible] == ["m22", "amd"]


def test_out_of_scope_chunks_never_raise_the_best_score() -> None:
    # The 2022 filing matches far better, but the question is about 2018.
    decision = policy().decide(Q, [hit("m22", 9.0), hit("m18", 0.5)], META)
    assert decision.top_score == 0.5
    assert (decision.sufficient, decision.reason) == (False, "below_threshold")


def test_chunks_are_partitioned_in_the_order_given() -> None:
    results = [hit("m22", 9.0), hit("m18", 3.0), hit("amd", 2.0), hit("m18", 1.5)]
    decision = policy().decide(Q, results, META)
    assert [c.chunk_id for c in decision.eligible] == ["m18", "m18"]
    assert [c.chunk_id for c in decision.ineligible] == ["m22", "amd"]
    assert decision.top_score == 3.0


def test_a_question_that_names_nothing_keeps_every_chunk() -> None:
    decision = policy().decide("What drove the change?", [hit("m22", 3.0), hit("amd", 2.0)], META)
    assert decision.sufficient and len(decision.eligible) == 2 and decision.ineligible == ()


def test_scope_check_off_ignores_company_and_period() -> None:
    decision = policy(scope_check=False).decide(Q, [hit("m22", 9.0)], META)
    assert decision.sufficient and decision.top_score == 9.0
    assert decision.named_companies == () and decision.named_years == ()


def test_known_companies_are_derived_from_metadata_when_omitted() -> None:
    derived = policy().decide(Q, [hit("m22", 9.0)], META)
    supplied = policy().decide(Q, [hit("m22", 9.0)], META, known_companies=["3M", "AMD"])
    assert derived == supplied and derived.reason == "out_of_scope"


def test_unknown_chunk_metadata_fails_a_stated_restriction() -> None:
    decision = policy().decide(Q, [hit("ghost", 9.0)], META)
    assert decision.reason == "out_of_scope"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "1.0", True])
def test_non_finite_or_non_numeric_scores_are_rejected(bad: Any) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        policy().decide(Q, [hit("m18", bad)], META)


def test_decision_is_deterministic() -> None:
    results = [hit("m18", 3.0), hit("m22", 2.0)]
    assert policy().decide(Q, results, META) == policy().decide(Q, results, META)


# --------------------------------------------------------------------------- confidence


def test_confidence_is_one_half_at_the_threshold() -> None:
    assert confidence_from_score(2.0, 2.0, 5.0) == 0.5


def test_confidence_matches_the_sigmoid_worked_by_hand() -> None:
    # sigmoid(ln 3) = 3 / 4 and sigmoid(-ln 3) = 1 / 4.
    scale, threshold = 2.0, 1.0
    assert confidence_from_score(threshold + scale * math.log(3), threshold, scale) == pytest.approx(0.75)
    assert confidence_from_score(threshold - scale * math.log(3), threshold, scale) == pytest.approx(0.25)


def test_confidence_is_stable_at_extreme_magnitudes() -> None:
    assert confidence_from_score(1e6, 0.0, 1.0) == 1.0
    assert confidence_from_score(-1e6, 0.0, 1.0) == 0.0


def test_confidence_rises_with_the_score() -> None:
    values = [confidence_from_score(s, 1.0, 0.7) for s in (-2.0, 0.0, 1.0, 1.5, 4.0)]
    assert values == sorted(values) and len(set(values)) == len(values)


def test_confidence_scale_never_changes_a_decision() -> None:
    for scale in (0.01, 1.0, 100.0):
        assert policy(confidence_scale=scale).decide(Q, [hit("m18", 1.0)], META).sufficient
        assert not policy(confidence_scale=scale).decide(Q, [hit("m18", 0.99)], META).sufficient


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e9])
@pytest.mark.parametrize("gap", [1e-300, 1e-17, 1e-12, 0.001, 1.0, 50.0])
def test_an_abstention_never_has_confidence_of_half(gap: float, scale: float) -> None:
    # With a wide scale, a score one ulp under the threshold gives sigmoid(-1e-17), which
    # rounds to exactly 0.5 unless the policy clamps it.
    just_below = 1.0 - gap if 1.0 - gap < 1.0 else math.nextafter(1.0, 0.0)
    decision = policy(threshold=1.0, confidence_scale=scale).decide(Q, [hit("m18", just_below)], META)
    assert not decision.sufficient
    assert decision.confidence < 0.5


def test_an_answer_always_has_confidence_of_at_least_half() -> None:
    for score in (1.0, math.nextafter(1.0, 2.0), 5.0, 1e9):
        decision = policy(threshold=1.0).decide(Q, [hit("m18", score)], META)
        assert decision.sufficient and decision.confidence >= 0.5


# --------------------------------------------------------------------------- construction


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1", None])
def test_threshold_must_be_a_finite_number(bad: Any) -> None:
    with pytest.raises(ValueError, match="threshold"):
        EvidencePolicy(threshold=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_confidence_scale_must_be_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence_scale"):
        EvidencePolicy(threshold=1.0, confidence_scale=bad)


def test_scope_check_must_be_a_bool() -> None:
    with pytest.raises(TypeError, match="scope_check"):
        EvidencePolicy(threshold=1.0, scope_check="yes")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- scorer binding


def test_check_scorer_accepts_the_tuned_scorer() -> None:
    policy(scorer={"scorer": "ce", "model_id": "m", "revision": "a" * 40}).check_scorer(
        {"scorer": "ce", "model_id": "m", "revision": "a" * 40}
    )


def test_check_scorer_ignores_device_and_batch_size() -> None:
    tuned = {"scorer": "ce", "revision": "a" * 40, "device": "cpu", "batch_size": 32}
    policy(scorer=tuned).check_scorer({"scorer": "ce", "revision": "a" * 40, "device": "cuda", "batch_size": 8})


@pytest.mark.parametrize(
    "running",
    [
        {"scorer": "ce", "revision": "b" * 40},  # another commit of the same model
        {"scorer": "other", "revision": "a" * 40},
        {"scorer": "ce", "revision": "a" * 40, "max_seq_length": 128},  # changes the scores
        {},
    ],
)
def test_check_scorer_rejects_any_other_scorer(running: dict[str, Any]) -> None:
    with pytest.raises(PolicyMismatchError, match="different scorer"):
        policy(scorer={"scorer": "ce", "revision": "a" * 40}).check_scorer(running)


def test_a_policy_with_no_recorded_scorer_cannot_be_checked() -> None:
    with pytest.raises(PolicyMismatchError, match="records no scorer"):
        EvidencePolicy(threshold=1.0).check_scorer({"scorer": "ce"})


def _search_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "first_stage": ["bm25"],
        "fusion": "none",
        "rrf_k": None,
        "per_retriever_k": None,
        "candidate_k": 50,
        "scorer": {"scorer": "stub"},
    }
    return {**config, **overrides}


def _tuned_pipeline(**overrides: Any) -> dict[str, Any]:
    return {
        "first_stage": ["bm25"],
        "fusion": "none",
        "rrf_k": None,
        "per_retriever_k": None,
        "candidate_k": 50,
        "context_k": 5,
        **overrides,
    }


def test_check_pipeline_accepts_the_tuned_setup() -> None:
    policy(tuning={"pipeline": _tuned_pipeline()}).check_pipeline(_search_config(), 5)


@pytest.mark.parametrize(
    "override, context_k",
    [
        ({"candidate_k": 20}, 5),
        ({"first_stage": ["bm25", "dense"], "fusion": "reciprocal_rank_fusion", "rrf_k": 60}, 5),
        ({}, 3),
    ],
)
def test_check_pipeline_rejects_changed_retrieval_settings(override: dict[str, Any], context_k: int) -> None:
    with pytest.raises(PolicyMismatchError, match="retrieval settings differ"):
        policy(tuning={"pipeline": _tuned_pipeline()}).check_pipeline(_search_config(**override), context_k)


def test_check_pipeline_checks_the_scorer_first() -> None:
    with pytest.raises(PolicyMismatchError, match="different scorer"):
        policy(tuning={"pipeline": _tuned_pipeline()}).check_pipeline(_search_config(scorer={"scorer": "x"}), 5)


def test_check_pipeline_without_a_tuning_record_checks_only_the_scorer() -> None:
    policy().check_pipeline(_search_config(candidate_k=7), 99)


# --------------------------------------------------------------------------- persistence


def test_record_round_trip() -> None:
    original = policy(threshold=0.25, confidence_scale=0.5, scope_check=False, tuning={"split": "dev"})
    assert EvidencePolicy.from_record(original.to_record()) == original


def test_record_is_strict_json_and_names_the_rule() -> None:
    record = policy().to_record()
    json.dumps(record, allow_nan=False)
    assert record["policy_version"] == POLICY_VERSION
    assert record["rule"]["signal"] == "best reranker score among in-scope chunks"


def test_load_reads_a_written_policy(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    original = policy(threshold=0.75)
    path.write_text(json.dumps(original.to_record()), encoding="utf-8")
    assert EvidencePolicy.load(path) == original


def test_from_record_rejects_another_version() -> None:
    record = policy().to_record()
    record["policy_version"] = "0.0.1"
    with pytest.raises(ValueError, match="unsupported policy_version"):
        EvidencePolicy.from_record(record)


@pytest.mark.parametrize("missing", ["rule", "scorer"])
def test_from_record_rejects_a_missing_section(missing: str) -> None:
    record = policy().to_record()
    del record[missing]
    with pytest.raises(ValueError, match="missing"):
        EvidencePolicy.from_record(record)
