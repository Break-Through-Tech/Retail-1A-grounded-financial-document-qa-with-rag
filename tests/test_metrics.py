"""Tests for FinanceBench retrieval metrics (issue #10)"""
from __future__ import annotations
import pytest
from src.evaluation.metrics import evaluate_rankings, hit_at_k, is_gold_match, reciprocal_rank_at_k

def test_exact_gold_page_is_relevant():
    result = {
        "doc_name": "APPLE_2022_10K",
        "evidence_page_num": 32,
    }
    gold = [
        {
            "doc_name": "APPLE_2022_10K",
            "evidence_page_num": 32,
        }
    ]
    assert is_gold_match(result, gold)


def test_wrong_page_is_not_relevant():
    result = {
        "doc_name": "APPLE_2022_10K",
        "evidence_page_num": 20,
    }
    gold = [
        {
            "doc_name": "APPLE_2022_10K",
            "evidence_page_num": 32,
        }
    ]
    assert not is_gold_match(result, gold)


def test_wrong_document_is_not_relevant():
    result = {
        "doc_name": "MICROSOFT_2022_10K",
        "evidence_page_num": 32,
    }
    gold = [
        {
            "doc_name": "APPLE_2022_10K",
            "evidence_page_num": 32,
        }
    ]
    assert not is_gold_match(result, gold)


def test_any_gold_evidence_can_match():
    result = {
        "doc_name": "APPLE_2022_10K",
        "evidence_page_num": 45,
    }
    gold = [
        {
            "doc_name": "APPLE_2022_10K",
            "evidence_page_num": 32,
        },
        {
            "doc_name": "APPLE_2022_10K",
            "evidence_page_num": 45,
        },
    ]
    assert is_gold_match(result, gold)


def test_hit_at_1_when_gold_is_first():
    results = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 20},
    ]
    gold = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32}
    ]
    assert hit_at_k(results, gold, 1) == 1


def test_hit_at_1_is_zero_when_gold_is_second():
    results = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 20},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
    ]
    gold = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32}
    ]
    assert hit_at_k(results, gold, 1) == 0


def test_hit_at_5_when_gold_is_within_top_five():
    results = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 10},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 20},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 40},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 50},
    ]
    gold = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32}
    ]
    assert hit_at_k(results, gold, 5) == 1


def test_hit_at_5_is_zero_when_gold_is_outside_top_five():
    results = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 10},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 20},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 30},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 40},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 50},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
    ]
    gold = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32}
    ]
    assert hit_at_k(results, gold, 5) == 0


def test_empty_ranking_has_no_hit():
    gold = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32}
    ]
    assert hit_at_k([], gold, 5) == 0


def test_reciprocal_rank_is_one_when_gold_is_first():
    results = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 20},
    ]
    gold = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32}
    ]

    assert reciprocal_rank_at_k(results, gold, 10) == 1.0


def test_reciprocal_rank_uses_gold_position():
    results = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 10},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 20},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
    ]
    gold = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32}
    ]

    assert reciprocal_rank_at_k(results, gold, 10) == 1 / 3


def test_reciprocal_rank_uses_first_gold_match():
    results = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 10},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 45},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 20},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
    ]
    gold = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 45},
    ]

    assert reciprocal_rank_at_k(results, gold, 10) == 0.5


def test_reciprocal_rank_is_zero_when_gold_is_outside_cutoff():
    results = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": page}
        for page in range(1, 11)
    ]
    results.append(
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32}
    )

    gold = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32}
    ]

    assert reciprocal_rank_at_k(results, gold, 10) == 0.0


def test_empty_ranking_has_zero_reciprocal_rank():
    gold = [
        {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32}
    ]

    assert reciprocal_rank_at_k([], gold, 10) == 0.0


def test_evaluate_rankings_aggregates_metrics():
    rankings = {
        "q1": [
            {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
        ],
        "q2": [
            {"doc_name": "APPLE_2022_10K", "evidence_page_num": 10},
            {"doc_name": "APPLE_2022_10K", "evidence_page_num": 45},
        ],
    }

    gold = {
        "q1": [
            {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
        ],
        "q2": [
            {"doc_name": "APPLE_2022_10K", "evidence_page_num": 45},
        ],
    }

    metrics = evaluate_rankings(rankings, gold)

    assert metrics["hit@1"] == 0.5
    assert metrics["hit@5"] == 1.0
    assert metrics["mrr@10"] == 0.75


def test_evaluate_rankings_scores_missing_ranking_as_zero():
    rankings = {
        "q1": [
            {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
        ],
    }

    gold = {
        "q1": [
            {"doc_name": "APPLE_2022_10K", "evidence_page_num": 32},
        ],
        "q2": [
            {"doc_name": "APPLE_2022_10K", "evidence_page_num": 45},
        ],
    }

    metrics = evaluate_rankings(rankings, gold)

    assert metrics["hit@1"] == 0.5
    assert metrics["hit@5"] == 0.5
    assert metrics["mrr@10"] == 0.5


def test_evaluate_rankings_rejects_empty_gold():
    with pytest.raises(ValueError, match="gold must contain at least one question"):
        evaluate_rankings({}, {})


def test_hit_at_k_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        hit_at_k([], [], 0)


def test_reciprocal_rank_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        reciprocal_rank_at_k([], [], 0)