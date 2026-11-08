"""The insufficient-evidence rule: when the pipeline must decline instead of answering.

The rule runs after reranking and before any answer is generated. Given a question and
its reranked chunks it asks three things, in order:

1. **Results.** If retrieval returned nothing, there is no evidence.
2. **Scope** (optional, on by default). Keep only the chunks whose filing matches the
   company and fiscal year the question names (:mod:`src.abstention.scope`). If the
   question names a company or year and no retrieved chunk comes from a matching
   filing, the evidence is out of scope. Chunks that fail the scope test are never
   handed to the answer generator.
3. **Threshold.** Take the best reranker score among the remaining chunks. If it is
   below ``threshold`` the evidence is too weak.

Otherwise the evidence is sufficient. ``threshold`` is tuned on development data only
(:mod:`src.abstention.tune`) and is valid for exactly one scorer: reranker scores are
comparable only within one model, so a policy refuses to run against a scorer other
than the one it was tuned for.

``confidence`` is ``sigmoid((top_score - threshold) / confidence_scale)``: 0.5 at the
threshold, and strictly below 0.5 whenever the pipeline abstains. It is a monotone
score for the decision, not a calibrated probability, and the scale never changes a
decision.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

from src.abstention.scope import chunk_in_scope, named_companies, named_years

__all__ = [
    "IGNORED_SCORER_KEYS",
    "PIPELINE_KEYS",
    "POLICY_VERSION",
    "REASONS",
    "EvidenceDecision",
    "EvidencePolicy",
    "PolicyMismatchError",
    "confidence_from_score",
    "evidence_signal",
]

POLICY_VERSION = "1.0.0"

#: Every value :attr:`EvidenceDecision.reason` can take.
REASONS = ("supported", "no_results", "out_of_scope", "below_threshold")

#: Scorer settings that do not change scores, so they may differ between the tuning run
#: and a deployment (a different device or batch size must not invalidate a threshold).
IGNORED_SCORER_KEYS = frozenset({"device", "batch_size"})

#: Retrieval settings that shape which chunks reach the threshold; they must match the
#: tuning run. They are the keys of ``RetrieveAndRerank.config_record()``.
PIPELINE_KEYS = ("first_stage", "fusion", "rrf_k", "per_retriever_k", "candidate_k")


class PolicyMismatchError(ValueError):
    """The policy was tuned for a different scorer than the one it is being used with."""


@dataclass(frozen=True)
class EvidenceDecision:
    """The outcome of applying a policy to one question's reranked chunks.

    Attributes:
        sufficient: ``True`` when the pipeline may go on to answer.
        reason: One of :data:`REASONS`.
        confidence: ``sigmoid((top_score - threshold) / scale)``; ``0.0`` when there is
            no in-scope chunk to score.
        top_score: Best reranker score among the in-scope chunks, or ``None``.
        threshold: The threshold that was applied.
        named_companies: Companies the question names, sorted; empty if none or the
            scope check is off.
        named_years: Years the question names, ascending; empty if none or the scope
            check is off.
        eligible: In-scope chunks, in the order given; the evidence for generation.
        ineligible: Chunks dropped by the scope check, in the order given.
    """

    sufficient: bool
    reason: str
    confidence: float
    top_score: float | None
    threshold: float
    named_companies: tuple[str, ...]
    named_years: tuple[int, ...]
    eligible: tuple[Any, ...]
    ineligible: tuple[Any, ...]


def confidence_from_score(score: float, threshold: float, scale: float) -> float:
    """Return ``sigmoid((score - threshold) / scale)``, stable for large magnitudes."""
    x = (score - threshold) / scale
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _check_scores(results: Sequence[Any]) -> None:
    for item in results:
        score = getattr(item, "score", None)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"chunk {getattr(item, 'chunk_id', '?')!r} has a non-finite score")


def _derive_companies(chunk_meta: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted({m["company"] for m in chunk_meta.values() if isinstance(m.get("company"), str)})


def evidence_signal(
    question: str,
    results: Sequence[Any],
    chunk_meta: Mapping[str, Mapping[str, Any]],
    *,
    scope_check: bool = True,
    known_companies: Collection[str] | None = None,
) -> tuple[float | None, list[Any], list[Any], frozenset[str], frozenset[int]]:
    """Split ``results`` by scope and return the score the threshold is applied to.

    This is the single place the scope test and the signal are computed, so tuning and
    run time cannot drift apart.

    Args:
        question: Question text.
        results: Reranked chunks with ``chunk_id`` and ``score``.
        chunk_meta: ``chunk_id`` -> metadata with ``company`` and ``doc_period``.
        scope_check: Whether to drop out-of-scope chunks.
        known_companies: Company names to look for in the question; derived from
            ``chunk_meta`` when omitted (slow for a large corpus, so pass it).

    Returns:
        ``(top_score, eligible, ineligible, companies, years)``. ``top_score`` is the
        best score among eligible chunks, ``None`` if there are none.

    Raises:
        ValueError: If a result has a non-finite score.
    """
    _check_scores(results)
    companies: frozenset[str] = frozenset()
    years: frozenset[int] = frozenset()
    if scope_check:
        if known_companies is None:
            known_companies = _derive_companies(chunk_meta)
        companies = named_companies(question, known_companies)
        years = named_years(question)
    eligible, ineligible = [], []
    for item in results:
        meta = chunk_meta.get(item.chunk_id, {})
        (eligible if chunk_in_scope(meta, companies, years) else ineligible).append(item)
    top = max((float(item.score) for item in eligible), default=None)
    return top, eligible, ineligible, companies, years


@dataclass(frozen=True)
class EvidencePolicy:
    """A tuned insufficient-evidence rule. See the module docstring for the rule.

    Attributes:
        threshold: Minimum best in-scope reranker score for the evidence to count as
            sufficient. Chosen on development data only.
        confidence_scale: Positive width of the confidence sigmoid; no effect on
            decisions.
        scope_check: Whether the company/period scope test is applied.
        scorer: The scorer's ``describe()`` record this threshold was tuned for.
        tuning: Provenance and development metrics written by the tuner; carried
            along, never read by the rule.
    """

    threshold: float
    confidence_scale: float = 1.0
    scope_check: bool = True
    scorer: Mapping[str, Any] = field(default_factory=dict)
    tuning: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("threshold", "confidence_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if self.confidence_scale <= 0:
            raise ValueError(f"confidence_scale must be positive, got {self.confidence_scale!r}")
        if not isinstance(self.scope_check, bool):
            raise TypeError(f"scope_check must be bool, got {type(self.scope_check).__name__}")

    def decide(
        self,
        question: str,
        results: Sequence[Any],
        chunk_meta: Mapping[str, Mapping[str, Any]],
        *,
        known_companies: Collection[str] | None = None,
    ) -> EvidenceDecision:
        """Decide whether ``results`` are enough evidence to answer ``question``.

        Args:
            question: Question text.
            results: Reranked chunks (``RerankResult``-like: ``chunk_id``, ``score``).
            chunk_meta: ``chunk_id`` -> metadata with ``company`` and ``doc_period``.
            known_companies: Company names to look for in the question; derived from
                ``chunk_meta`` when omitted.

        Returns:
            The decision, with the in-scope chunks as the evidence for generation.

        Raises:
            ValueError: If a result has a non-finite score.
        """
        top, eligible, ineligible, companies, years = evidence_signal(
            question,
            results,
            chunk_meta,
            scope_check=self.scope_check,
            known_companies=known_companies,
        )
        if not results:
            reason, sufficient, confidence = "no_results", False, 0.0
        elif top is None:
            reason, sufficient, confidence = "out_of_scope", False, 0.0
        else:
            confidence = confidence_from_score(top, self.threshold, self.confidence_scale)
            sufficient = top >= self.threshold
            if not sufficient:
                # A score a hair under the threshold can round to exactly 0.5.
                confidence = min(confidence, math.nextafter(0.5, 0.0))
            reason = "supported" if sufficient else "below_threshold"
        return EvidenceDecision(
            sufficient=sufficient,
            reason=reason,
            confidence=confidence,
            top_score=top,
            threshold=float(self.threshold),
            named_companies=tuple(sorted(companies)),
            named_years=tuple(sorted(years)),
            eligible=tuple(eligible),
            ineligible=tuple(ineligible),
        )

    def check_scorer(self, scorer_description: Mapping[str, Any]) -> None:
        """Require that ``scorer_description`` is the scorer this policy was tuned for.

        Settings in :data:`IGNORED_SCORER_KEYS` are not compared.

        Raises:
            PolicyMismatchError: If the policy has no recorded scorer, or any other
                recorded setting differs.
        """
        if not self.scorer:
            raise PolicyMismatchError("policy records no scorer, so it cannot be checked")
        tuned = {k: v for k, v in self.scorer.items() if k not in IGNORED_SCORER_KEYS}
        actual = {k: v for k, v in scorer_description.items() if k not in IGNORED_SCORER_KEYS}
        if tuned != actual:
            raise PolicyMismatchError(
                "threshold was tuned for a different scorer: "
                f"tuned for {json.dumps(tuned, sort_keys=True)}, "
                f"running {json.dumps(actual, sort_keys=True)}"
            )

    def check_pipeline(self, search_config: Mapping[str, Any], context_k: int) -> None:
        """Require that the pipeline matches the one this policy was tuned on.

        Checks the scorer (:meth:`check_scorer`) and, when the policy records the
        retrieval settings it was tuned with, that ``first_stage``, ``fusion``,
        ``rrf_k``, ``per_retriever_k``, ``candidate_k`` and ``context_k`` are equal. A
        different candidate pool changes the best score the threshold sees.

        Args:
            search_config: ``RetrieveAndRerank.config_record()``.
            context_k: Reranked chunks the pipeline examines.

        Raises:
            PolicyMismatchError: If the scorer or any retrieval setting differs.
        """
        self.check_scorer(search_config["scorer"])
        tuned = self.tuning.get("pipeline")
        if tuned is None:
            return
        keys = (*PIPELINE_KEYS, "context_k")
        expected = {k: tuned.get(k) for k in keys}
        actual = {**{k: search_config.get(k) for k in PIPELINE_KEYS}, "context_k": context_k}
        if expected != actual:
            raise PolicyMismatchError(
                "retrieval settings differ from the tuning run: "
                f"tuned with {json.dumps(expected, sort_keys=True)}, "
                f"running {json.dumps(actual, sort_keys=True)}"
            )

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable record: the rule, its scorer, and its provenance."""
        return {
            "policy_version": POLICY_VERSION,
            "rule": {
                "scope_check": self.scope_check,
                "signal": "best reranker score among in-scope chunks",
                "threshold": self.threshold,
                "confidence_scale": self.confidence_scale,
            },
            "scorer": dict(self.scorer),
            "tuning": dict(self.tuning),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> EvidencePolicy:
        """Rebuild a policy from :meth:`to_record` output.

        Raises:
            ValueError: If the record's version is not :data:`POLICY_VERSION` or a
                field is missing or invalid.
        """
        if record.get("policy_version") != POLICY_VERSION:
            raise ValueError(
                f"unsupported policy_version {record.get('policy_version')!r}; "
                f"expected {POLICY_VERSION!r}"
            )
        try:
            rule = record["rule"]
            return cls(
                threshold=rule["threshold"],
                confidence_scale=rule["confidence_scale"],
                scope_check=rule["scope_check"],
                scorer=dict(record["scorer"]),
                tuning=dict(record.get("tuning", {})),
            )
        except KeyError as error:
            raise ValueError(f"policy record is missing {error}") from error

    @classmethod
    def load(cls, path: Path | str) -> EvidencePolicy:
        """Read a policy written by :func:`src.abstention.tune.write_policy`."""
        return cls.from_record(json.loads(Path(path).read_text(encoding="utf-8")))
