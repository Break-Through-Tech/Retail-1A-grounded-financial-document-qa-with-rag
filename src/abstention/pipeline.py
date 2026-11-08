"""Question in, grounded response or insufficient-evidence response out.

:class:`GroundedQA` is the thin layer that connects retrieve-and-rerank, the
insufficient-evidence rule and an answer generator. The generator is injected so this
module needs no LLM; it is called only when the rule says the evidence suffices, so a
weak retrieval never reaches a model that could talk itself into an answer.

A second line of defence follows the first: if the generator declines, or returns an
answer that cites nothing, the response is an abstention too. An answer without a
citation cannot be checked against a source, so it is not returned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from src.abstention.policy import EvidencePolicy
from src.abstention.response import (
    GeneratedAnswer,
    answer_response,
    insufficient_evidence_response,
)
from src.retrieval.rerank import RetrieveAndRerank

__all__ = ["AnswerGenerator", "EvidenceChunk", "GroundedQA"]


@dataclass(frozen=True)
class EvidenceChunk:
    """One in-scope chunk handed to the generator, best-scoring first."""

    chunk_id: str
    text: str
    doc_name: str | None
    page_num: int | None
    score: float


class AnswerGenerator(Protocol):
    """Writes an answer from evidence, or declines by returning ``text=None``."""

    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> GeneratedAnswer: ...


class GroundedQA:
    """Retrieve, rerank, check the evidence, then answer or decline."""

    def __init__(
        self,
        search: RetrieveAndRerank,
        policy: EvidencePolicy,
        chunk_meta: Mapping[str, Mapping[str, Any]],
        texts: Mapping[str, str],
        generator: AnswerGenerator,
        *,
        context_k: int = 5,
    ) -> None:
        """Assemble the pipeline.

        Args:
            search: The retrieve-and-rerank pipeline.
            policy: A tuned policy. It must have been tuned for ``search``'s scorer.
            chunk_meta: ``chunk_id`` -> metadata (``doc_name``, ``page_num``,
                ``company``, ``doc_type``, ``doc_period``, ``doc_link``).
            texts: ``chunk_id`` -> chunk text.
            generator: Writes the answer; see :class:`AnswerGenerator`.
            context_k: Reranked chunks examined and passed on; use the value the
                policy was tuned with.

        Raises:
            PolicyMismatchError: If the policy was tuned for a different scorer, first
                stage, ``candidate_k`` or ``context_k``.
            ValueError: If ``context_k`` is below 1.
        """
        if isinstance(context_k, bool) or not isinstance(context_k, int) or context_k < 1:
            raise ValueError(f"context_k must be an int of at least 1, got {context_k!r}")
        policy.check_pipeline(search.config_record(), context_k)
        self._search = search
        self._policy = policy
        self._meta = chunk_meta
        self._texts = texts
        self._generator = generator
        self._context_k = context_k
        self._companies = sorted({m["company"] for m in chunk_meta.values() if isinstance(m.get("company"), str)})

    def answer(self, question: str, question_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Answer ``question`` from the indexed documents, or say it cannot be answered.

        Args:
            question: Question text.
            question_row: Benchmark row whose ``financebench_id``, ``question_type`` and
                ``question_reasoning`` are copied into the response; optional.

        Returns:
            A response dict. ``abstained`` is true, ``answer`` is null and
            ``abstention`` explains why whenever the evidence is insufficient.
        """
        out = self._search.search(question, top_k=self._context_k)
        decision = self._policy.decide(
            question, out.results, self._meta, known_companies=self._companies
        )
        if not decision.sufficient:
            return insufficient_evidence_response(
                question_row,
                decision,
                out.results,
                self._meta,
                latency_ms=out.latency_ms,
                model_calls=out.scorer_calls,
            )

        evidence = [
            EvidenceChunk(
                chunk_id=item.chunk_id,
                text=self._texts[item.chunk_id],
                doc_name=self._meta.get(item.chunk_id, {}).get("doc_name"),
                page_num=self._meta.get(item.chunk_id, {}).get("page_num"),
                score=item.score,
            )
            for item in decision.eligible
        ]
        started = time.perf_counter()
        generated = self._generator.generate(question, evidence)
        latency_ms = out.latency_ms + (time.perf_counter() - started) * 1000.0
        model_calls = out.scorer_calls + generated.model_calls

        declined = generated.text is None or not str(generated.text).strip()
        reason = "generator_declined" if declined else None
        if reason is None and not list(generated.cited_chunk_ids):
            reason = "uncited_answer"
        if reason is not None:
            return insufficient_evidence_response(
                question_row,
                decision,
                out.results,
                self._meta,
                latency_ms=latency_ms,
                model_calls=model_calls,
                reason=reason,
            )
        return answer_response(
            question_row,
            decision,
            generated,
            out.results,
            self._meta,
            self._texts,
            latency_ms=latency_ms,
            model_calls=model_calls,
        )
