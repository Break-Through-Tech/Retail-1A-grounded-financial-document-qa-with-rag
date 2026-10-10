"""Retrieval evaluation metrics for FinanceBench.

This module evaluates ranked retrieval results against the gold supporting
evidence associated with each FinanceBench question.
"""

from __future__ import annotations


def is_gold_match(result: dict, gold_items: list[dict]) -> bool:
    """Return whether a retrieved result matches any gold evidence page.

    A page-level match requires both the document name and evidence page
    number to match one of the question's gold evidence items.
    """
    return any(
        result["doc_name"] == gold["doc_name"]
        and result["evidence_page_num"] == gold["evidence_page_num"]
        for gold in gold_items
    )


def hit_at_k(results: list[dict], gold_items: list[dict], k: int) -> int:
    """Return 1 if any gold evidence page appears within the top-k results.

    Results are expected to be ordered from highest to lowest retrieval rank.
    Return 0 otherwise.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    return int(
        any(is_gold_match(result, gold_items) for result in results[:k])
    )


def reciprocal_rank_at_k(
    results: list[dict],
    gold_items: list[dict],
    k: int = 10,
) -> float:
    """Return the reciprocal rank of the first gold match within the top-k.

    Results are expected to be ordered from highest to lowest retrieval rank.
    Return 0.0 if no gold evidence appears within the top-k results.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    for rank, result in enumerate(results[:k], start=1):
        if is_gold_match(result, gold_items):
            return 1.0 / rank

    return 0.0


def evaluate_rankings(
    rankings: dict[str, list[dict]],
    gold: dict[str, list[dict]],
) -> dict[str, float]:
    """Compute aggregate retrieval metrics across a set of questions.

    Rankings and gold evidence are keyed by FinanceBench question ID.
    """
    if not gold:
        raise ValueError("gold must contain at least one question")

    hit_1_scores = []
    hit_5_scores = []
    reciprocal_ranks = []

    for financebench_id, gold_items in gold.items():
        results = rankings.get(financebench_id, [])

        hit_1_scores.append(hit_at_k(results, gold_items, 1))
        hit_5_scores.append(hit_at_k(results, gold_items, 5))
        reciprocal_ranks.append(
            reciprocal_rank_at_k(results, gold_items, 10)
        )

    question_count = len(gold)

    return {
        "hit@1": sum(hit_1_scores) / question_count,
        "hit@5": sum(hit_5_scores) / question_count,
        "mrr@10": sum(reciprocal_ranks) / question_count,
    }