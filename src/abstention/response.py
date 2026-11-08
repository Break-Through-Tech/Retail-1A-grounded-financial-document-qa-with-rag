"""Build the pipeline's response object, including the insufficient-evidence response.

The shape follows ``docs/response_structure.md``. An abstention is a normal response
with ``abstained`` true, ``answer`` null, no citations and a ``confidence`` below 0.5,
plus an ``abstention`` block that says in plain words why the system declined. The
``answer`` field stays null on purpose: an answer-accuracy scorer reading it can never
mistake the explanation for a guess, while a reader of the response sees the message
in ``abstention.message``.

Every chunk that was retrieved but not cited, which on an abstention is all of them,
is listed in ``retrieved_but_unused`` so the audit trail shows what the system looked
at before declining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.abstention.policy import REASONS, EvidenceDecision

__all__ = [
    "ABSTENTION_REASONS",
    "RESPONSE_KEYS",
    "GeneratedAnswer",
    "abstention_message",
    "answer_response",
    "insufficient_evidence_response",
    "validate_response",
]

#: Reasons an abstention can carry: the gate's four non-"supported" reasons, plus the
#: two ways the answer generator can decline after the gate let the evidence through.
ABSTENTION_REASONS = tuple(r for r in REASONS if r != "supported") + (
    "generator_declined",
    "uncited_answer",
)

RESPONSE_KEYS = (
    "financebench_id",
    "question_type",
    "question_reasoning",
    "answer",
    "confidence",
    "abstained",
    "calculation",
    "citations",
    "retrieved_but_unused",
    "latency_ms",
    "model_calls",
)

_CITATION_KEYS = (
    "doc_name",
    "company",
    "doc_type",
    "doc_period",
    "evidence_page_num",
    "evidence_text",
    "doc_link",
    "retrieval_score",
    "rerank_score",
)

_TAIL = "so it cannot be answered from the available documents."


@dataclass(frozen=True)
class GeneratedAnswer:
    """What an answer generator hands back.

    Attributes:
        text: The answer, or ``None`` if the generator declined to answer.
        calculation_steps: Steps of any calculation, in order; empty if none.
        cited_chunk_ids: IDs of the evidence chunks the answer rests on.
        model_calls: Model or API calls the generator made.
    """

    text: str | None
    calculation_steps: Sequence[str] = ()
    cited_chunk_ids: Sequence[str] = ()
    model_calls: int = 1


def _scope_text(decision: EvidenceDecision) -> str:
    parts = []
    if decision.named_companies:
        parts.append(" and ".join(decision.named_companies))
    if decision.named_years:
        years = ", ".join(str(y) for y in decision.named_years)
        parts.append(f"fiscal year {years}" if len(decision.named_years) == 1 else f"fiscal years {years}")
    return " / ".join(parts)


def abstention_message(reason: str, decision: EvidenceDecision) -> str:
    """Return the plain-language statement that the system cannot answer.

    Raises:
        ValueError: If ``reason`` is not in :data:`ABSTENTION_REASONS`.
    """
    if reason == "no_results":
        return f"Insufficient evidence: no passage in the indexed documents matched this question, {_TAIL}"
    if reason == "out_of_scope":
        return (
            f"Insufficient evidence: the question refers to {_scope_text(decision)}, but none of "
            f"the retrieved passages come from a matching filing, {_TAIL}"
        )
    if reason == "below_threshold":
        return (
            "Insufficient evidence: the best-matching passages are not relevant enough to support "
            f"an answer (best score {decision.top_score:.2f}, required {decision.threshold:.2f}), {_TAIL}"
        )
    if reason == "generator_declined":
        return (
            "Insufficient evidence: the retrieved passages were reviewed but do not contain "
            f"what is needed to answer, {_TAIL}"
        )
    if reason == "uncited_answer":
        return (
            "Insufficient evidence: an answer could not be tied to a specific passage, " + _TAIL
        )
    raise ValueError(f"reason must be one of {ABSTENTION_REASONS}, got {reason!r}")


def _question_fields(question_row: Mapping[str, Any] | None) -> dict[str, Any]:
    row = question_row or {}
    return {
        "financebench_id": row.get("financebench_id"),
        "question_type": row.get("question_type"),
        "question_reasoning": row.get("question_reasoning"),
    }


def _page_ref(chunk_id: str, chunk_meta: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    meta = chunk_meta.get(chunk_id, {})
    return {"doc_name": meta.get("doc_name"), "evidence_page_num": meta.get("page_num")}


def _unused(results: Sequence[Any], cited: set[str], chunk_meta: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    seen, out = set(), []
    for item in results:
        if item.chunk_id in cited:
            continue
        ref = _page_ref(item.chunk_id, chunk_meta)
        key = (ref["doc_name"], ref["evidence_page_num"])
        if key not in seen:
            seen.add(key)
            out.append(ref)
    return out


def insufficient_evidence_response(
    question_row: Mapping[str, Any] | None,
    decision: EvidenceDecision,
    results: Sequence[Any],
    chunk_meta: Mapping[str, Mapping[str, Any]],
    *,
    latency_ms: float,
    model_calls: int,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build the response for a question the system declines to answer.

    Args:
        question_row: Benchmark row (``financebench_id``, ``question_type``,
            ``question_reasoning``), or ``None`` for a free-form question.
        decision: The gate's decision.
        results: Every reranked chunk, in rank order, for the audit trail.
        chunk_meta: ``chunk_id`` -> metadata with ``doc_name`` and ``page_num``.
        latency_ms: Total time spent, rounded to a whole number in the response.
        model_calls: Model or API calls made.
        reason: Overrides ``decision.reason``; used when the generator declines after
            the gate passed the evidence.

    Returns:
        A response dict with ``abstained`` true and ``answer`` null.

    Raises:
        ValueError: If the reason is not one of :data:`ABSTENTION_REASONS`.
    """
    reason = reason or decision.reason
    message = abstention_message(reason, decision)
    # A generator-side abstention overrides the score: no usable evidence, confidence 0.
    confidence = decision.confidence if reason in REASONS else 0.0
    return {
        **_question_fields(question_row),
        "answer": None,
        "confidence": float(confidence),
        "abstained": True,
        "calculation": {"shown": False, "steps": []},
        "citations": [],
        "retrieved_but_unused": _unused(results, set(), chunk_meta),
        "abstention": {
            "reason": reason,
            "message": message,
            "top_score": decision.top_score,
            "threshold": decision.threshold,
            "named_companies": list(decision.named_companies),
            "named_years": list(decision.named_years),
        },
        "latency_ms": round(latency_ms),
        "model_calls": int(model_calls),
    }


def answer_response(
    question_row: Mapping[str, Any] | None,
    decision: EvidenceDecision,
    generated: GeneratedAnswer,
    results: Sequence[Any],
    chunk_meta: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    *,
    latency_ms: float,
    model_calls: int,
) -> dict[str, Any]:
    """Build the response for a question the system answers.

    Each cited chunk becomes one citation carrying its reranker and first-stage scores.

    Raises:
        ValueError: If the generated answer cites a chunk that is not in the evidence
            the gate passed, or cites nothing.
    """
    evidence = {item.chunk_id: item for item in decision.eligible}
    cited_ids = list(dict.fromkeys(generated.cited_chunk_ids))
    unknown = [c for c in cited_ids if c not in evidence]
    if unknown:
        raise ValueError(f"answer cites chunks outside the evidence: {unknown}")
    if not cited_ids:
        raise ValueError("an answer must cite at least one chunk")
    citations = []
    for chunk_id in cited_ids:
        meta, item = chunk_meta.get(chunk_id, {}), evidence[chunk_id]
        citations.append(
            {
                "doc_name": meta.get("doc_name"),
                "company": meta.get("company"),
                "doc_type": meta.get("doc_type"),
                "doc_period": meta.get("doc_period"),
                "evidence_page_num": meta.get("page_num"),
                "evidence_text": texts[chunk_id],
                "doc_link": meta.get("doc_link"),
                "retrieval_score": float(item.retrieval_score),
                "rerank_score": float(item.score),
            }
        )
    steps = [str(s) for s in generated.calculation_steps]
    return {
        **_question_fields(question_row),
        "answer": generated.text,
        "confidence": float(decision.confidence),
        "abstained": False,
        "calculation": {"shown": bool(steps), "steps": steps},
        "citations": citations,
        "retrieved_but_unused": _unused(results, set(cited_ids), chunk_meta),
        "latency_ms": round(latency_ms),
        "model_calls": int(model_calls),
    }


def validate_response(response: Mapping[str, Any]) -> None:
    """Check a response against ``docs/response_structure.md`` and the abstention rules.

    Beyond key presence it enforces the invariants the evaluation relies on: an
    abstention has a null answer, no citations and a confidence below 0.5, and carries
    an ``abstention`` block; an answer is non-empty, cited, and has confidence of at
    least 0.5.

    Raises:
        ValueError: Listing every problem found.
    """
    problems = [f"missing key {k!r}" for k in RESPONSE_KEYS if k not in response]
    if not problems:
        abstained = response["abstained"]
        if not isinstance(abstained, bool):
            problems.append("abstained must be a bool")
        confidence = response["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            problems.append("confidence must be a number in [0, 1]")
        elif abstained and confidence >= 0.5:
            problems.append("an abstention must have confidence below 0.5")
        elif abstained is False and confidence < 0.5:
            problems.append("an answer must have confidence of at least 0.5")
        if not isinstance(response["latency_ms"], int) or not isinstance(response["model_calls"], int):
            problems.append("latency_ms and model_calls must be ints")
        calc = response["calculation"]
        if not isinstance(calc, Mapping) or set(calc) != {"shown", "steps"}:
            problems.append("calculation must have exactly 'shown' and 'steps'")
        if abstained is True:
            if response["answer"] is not None:
                problems.append("an abstention must have answer null")
            if response["citations"]:
                problems.append("an abstention must have no citations")
            block = response.get("abstention")
            if not isinstance(block, Mapping) or block.get("reason") not in ABSTENTION_REASONS:
                problems.append("an abstention needs an abstention block with a known reason")
            elif not str(block.get("message", "")).startswith("Insufficient evidence"):
                problems.append("the abstention message must state the insufficient evidence")
        elif abstained is False:
            if not isinstance(response["answer"], str) or not response["answer"].strip():
                problems.append("an answer must be a non-empty string")
            if not response["citations"]:
                problems.append("an answer must have at least one citation")
            for i, citation in enumerate(response["citations"]):
                absent = [k for k in _CITATION_KEYS if k not in citation]
                if absent:
                    problems.append(f"citation {i} is missing {absent}")
    if problems:
        raise ValueError("invalid response: " + "; ".join(problems))
