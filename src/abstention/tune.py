"""Tune the insufficient-evidence threshold on development data only.

Run ``python -m src.abstention.tune --help``. The rule being tuned is described in
:mod:`src.abstention.policy`; this module chooses its one number, the threshold, and
decides whether the scope check earns its place.

**Development data only.** Questions come from :func:`src.retrieval.queries.load_dev_queries`
and the corpus is restricted to the filings those questions belong to, so no
frozen-test question and no frozen-test filing is ever read. The frozen test ID file is
never opened; ``tests/test_abstention_tune.py`` enforces that.

**Labelled data.** Every FinanceBench question is answerable, so unanswerable cases are
constructed. Each development question is run twice:

* ``full``: against the whole development corpus. The question is *supported* if the
  gold evidence page is among the top ``context_k`` reranked chunks (page level, as in
  :func:`src.evaluation.metrics.is_gold_match`). This includes real retrieval failures.
* ``withheld``: against the same corpus with the question's gold filing removed. No
  gold evidence exists, so the question is never supported and the right behaviour is
  to decline. Related filings (the company's other years) stay in the index, which is
  the hard realistic case: right topic, wrong period.

"Supported" never depends on the policy being tuned, so a question the scope check
wrongly rejects shows up as lost coverage rather than disappearing from the numbers.

**Metrics** (answering means the policy lets the pipeline proceed):

* ``unsupported_answer_rate``: answered questions that are unsupported / answered
  questions. The project target is at most 10%. Defined as 0 when nothing is answered.
* ``answer_coverage``: supported questions answered / supported questions.
* ``abstention_precision`` / ``abstention_recall``: of the questions the policy
  declined, the share that were unsupported; of the unsupported questions, the share it
  declined.

**Selection rule.** Over every threshold that sits between two consecutive observed
scores (plus one below all of them and one above), keep those whose
``unsupported_answer_rate`` is at most the target, and take the one with the highest
``answer_coverage``; ties go to the higher, more conservative threshold. Abstaining on
everything always qualifies, so the rule can degrade to never answering; the report
flags that. With the scope check on ``auto``, the variant with the higher coverage wins
and ties go to the scope check being on.

**Limits.** Development numbers are in-sample. A grouped bootstrap over filings gives a
90% interval for the chosen threshold, which on a few dozen filings is wide. The pooled
numbers weight ``full`` and ``withheld`` equally, which is a property of this
evaluation and not of production traffic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.abstention.policy import EvidencePolicy, evidence_signal
from src.abstention.reference_scorer import IdfCoverageScorer
from src.data.split import DEFAULT_DATASET, load_rows
from src.evaluation.metrics import is_gold_match
from src.retrieval.bm25 import BM25Index
from src.retrieval.queries import DEFAULT_DEV_IDS, load_dev_queries
from src.retrieval.rerank import RetrieveAndRerank

__all__ = [
    "CANDIDATE_K",
    "CONTEXT_K",
    "MIN_ANSWERS_FOR_REPORT",
    "TARGET_UNSUPPORTED_RATE",
    "EvalItem",
    "ExcludingRetriever",
    "TuningOutcome",
    "candidate_thresholds",
    "collect_items",
    "grouped_bootstrap",
    "load_corpus",
    "load_dev_rows",
    "metrics_at",
    "scope_effect",
    "select_threshold",
    "sweep",
    "tune_policy",
    "write_policy",
    "write_sweep",
]

DEFAULT_CHUNKS = Path("data/processed/chunks_w300.jsonl")
DEFAULT_OUT_DIR = Path("results/abstention")

TARGET_UNSUPPORTED_RATE = 0.10
CONTEXT_K = 5
CANDIDATE_K = 50
CONDITIONS = ("full", "withheld")
#: Fewest answered questions for a rate to be reported as the best achievable.
MIN_ANSWERS_FOR_REPORT = 10
_META_KEYS = ("doc_name", "page_num", "company", "doc_type", "doc_period", "doc_link")

SELECTION_RULE = (
    "highest answer_coverage among thresholds with unsupported_answer_rate <= target; "
    "ties go to the higher threshold"
)


# --------------------------------------------------------------------------- data


class ExcludingRetriever:
    """Wraps a retriever so every chunk of one filing is invisible to it.

    This builds the ``withheld`` condition without rebuilding an index. Hits are
    over-fetched until ``k`` survive or the base retriever runs out. Index statistics
    such as BM25's IDF still include the withheld filing, a negligible difference.
    """

    def __init__(self, base: Any, chunk_doc: Mapping[str, str], excluded_doc: str) -> None:
        self._base = base
        self._chunk_doc = chunk_doc
        self._excluded = excluded_doc

    def retrieve(self, query: str, k: int) -> list[Any]:
        """Return up to ``k`` hits, none from the excluded filing."""
        fetch = k
        while True:
            hits = self._base.retrieve(query, fetch)
            kept = [h for h in hits if self._chunk_doc[h.chunk_id] != self._excluded]
            if len(kept) >= k or len(hits) < fetch:
                return kept[:k]
            fetch *= 4


def load_dev_rows(
    dataset: Path = DEFAULT_DATASET, dev_ids_path: Path = DEFAULT_DEV_IDS
) -> list[dict[str, Any]]:
    """Load the full dataset row of every development question, sorted by ID.

    Only ``dev_ids.txt`` is read, and its count is checked against the split metadata
    by :func:`src.retrieval.queries.load_dev_queries`.
    """
    queries = load_dev_queries(dataset, dev_ids_path)
    by_id = {row["financebench_id"]: row for row in load_rows(dataset)}
    return [by_id[q.financebench_id] for q in queries]


def load_corpus(
    chunks_path: Path, dev_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the chunks of the development filings and the questions they can answer.

    Chunks of any other filing are dropped, so a chunk file that also holds
    frozen-test filings cannot leak them into tuning. Questions whose filing has no
    chunks (not downloaded) are dropped too.

    Returns:
        ``(chunks, rows)``.

    Raises:
        ValueError: If no development question has a filing in the chunk file.
    """
    dev_docs = {row["doc_name"] for row in dev_rows}
    chunks = []
    with Path(chunks_path).open(encoding="utf-8") as handle:
        for line in handle:
            chunk = json.loads(line)
            if chunk["doc_name"] in dev_docs:
                chunks.append(chunk)
    present = {chunk["doc_name"] for chunk in chunks}
    rows = [dict(row) for row in dev_rows if row["doc_name"] in present]
    if not rows:
        raise ValueError(f"no development filing has chunks in {chunks_path}")
    return chunks, rows


@dataclass(frozen=True)
class EvalItem:
    """One development question under one condition.

    Attributes:
        financebench_id: The question's ID.
        doc_name: The question's gold filing; also the bootstrap group.
        condition: ``"full"`` or ``"withheld"``.
        question: The question text.
        supported: Whether a gold evidence page is among ``results``.
        results: The top ``context_k`` reranked chunks.
    """

    financebench_id: str
    doc_name: str
    condition: str
    question: str
    supported: bool
    results: tuple[Any, ...]


def collect_items(
    rows: Sequence[Mapping[str, Any]],
    retrievers: Mapping[str, Any],
    chunk_meta: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    scorer: Any,
    *,
    candidate_k: int = CANDIDATE_K,
    context_k: int = CONTEXT_K,
) -> list[EvalItem]:
    """Run every question under both conditions and label it.

    Args:
        rows: Development rows (``financebench_id``, ``doc_name``, ``question``,
            ``evidence``).
        retrievers: First-stage retrievers for :class:`RetrieveAndRerank`.
        chunk_meta: ``chunk_id`` -> metadata with ``doc_name`` and ``page_num``.
        texts: ``chunk_id`` -> chunk text.
        scorer: The :class:`~src.retrieval.rerank.PairScorer` being tuned for.
        candidate_k: Candidates handed to the reranker.
        context_k: Reranked chunks kept and labelled.

    Returns:
        Two items per row, ``full`` then ``withheld``.
    """
    chunk_doc = {cid: meta["doc_name"] for cid, meta in chunk_meta.items()}
    full = RetrieveAndRerank(retrievers, texts, scorer, candidate_k=candidate_k)
    items = []
    for row in rows:
        withheld = RetrieveAndRerank(
            {name: ExcludingRetriever(r, chunk_doc, row["doc_name"]) for name, r in retrievers.items()},
            texts,
            scorer,
            candidate_k=candidate_k,
        )
        for condition, search in (("full", full), ("withheld", withheld)):
            results = tuple(search.search(row["question"], top_k=context_k).results)
            pages = [
                {"doc_name": chunk_meta[r.chunk_id]["doc_name"], "evidence_page_num": chunk_meta[r.chunk_id]["page_num"]}
                for r in results
            ]
            supported = any(is_gold_match(page, list(row["evidence"])) for page in pages)
            items.append(
                EvalItem(row["financebench_id"], row["doc_name"], condition, row["question"], supported, results)
            )
    return items


# --------------------------------------------------------------------------- metrics


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics_at(
    threshold: float, signals: Sequence[float | None], supported: Sequence[bool]
) -> dict[str, Any]:
    """Score the policy "answer iff signal >= threshold" on labelled items.

    A ``None`` signal (no in-scope evidence) is never answered. Undefined ratios are
    ``None``.
    """
    if len(signals) != len(supported):
        raise ValueError("signals and supported must have the same length")
    answered = [s is not None and s >= threshold for s in signals]
    n = len(signals)
    n_answered = sum(answered)
    n_supported = sum(supported)
    answered_supported = sum(a and s for a, s in zip(answered, supported))
    answered_unsupported = n_answered - answered_supported
    abstained_unsupported = sum((not a) and (not s) for a, s in zip(answered, supported))
    return {
        "threshold": float(threshold),
        "n": n,
        "n_supported": n_supported,
        "n_answered": n_answered,
        "n_abstained": n - n_answered,
        "answered_supported": answered_supported,
        "answered_unsupported": answered_unsupported,
        "unsupported_answer_rate": _ratio(answered_unsupported, n_answered),
        "answer_coverage": _ratio(answered_supported, n_supported),
        "abstention_precision": _ratio(abstained_unsupported, n - n_answered),
        "abstention_recall": _ratio(abstained_unsupported, n - n_supported),
    }


def candidate_thresholds(signals: Sequence[float | None]) -> list[float]:
    """Thresholds to try: below every score, between consecutive scores, above every score."""
    values = sorted({s for s in signals if s is not None})
    if not values:
        return [0.0]
    low, high = values[0], values[-1]
    pad = (high - low) * 0.01 if high > low else 1.0
    middles = [(a + b) / 2 for a, b in zip(values, values[1:])]
    return [low - pad, *middles, high + pad]


def sweep(signals: Sequence[float | None], supported: Sequence[bool]) -> list[dict[str, Any]]:
    """Return :func:`metrics_at` for every candidate threshold, ascending."""
    return [metrics_at(t, signals, supported) for t in candidate_thresholds(signals)]


def select_threshold(points: Sequence[Mapping[str, Any]], target: float) -> Mapping[str, Any]:
    """Pick the operating point by the module's selection rule.

    Raises:
        ValueError: If ``points`` is empty or ``target`` is outside [0, 1].
    """
    if not points:
        raise ValueError("points must be non-empty")
    if not 0 <= target <= 1:
        raise ValueError(f"target must lie in [0, 1], got {target!r}")
    feasible = [
        p for p in points if p["unsupported_answer_rate"] is None or p["unsupported_answer_rate"] <= target
    ]
    return max(feasible, key=lambda p: (p["answer_coverage"] or 0.0, p["threshold"]))


def grouped_bootstrap(
    threshold: float,
    signals: Sequence[float | None],
    supported: Sequence[bool],
    groups: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.90,
) -> dict[str, Any]:
    """Interval for the unsupported-answer rate and coverage, resampling whole filings.

    Items from one filing are correlated (both conditions of a question, and every
    question about the same filing), so filings, not items, are resampled. A resample
    that answers nothing has rate 0, matching :func:`metrics_at`.
    """
    names = sorted(set(groups))
    index = {g: i for i, g in enumerate(names)}
    counts = np.zeros((len(names), 4))
    for signal, is_supported, group in zip(signals, supported, groups):
        answered = signal is not None and signal >= threshold
        row = counts[index[group]]
        row[0] += answered and not is_supported
        row[1] += answered
        row[2] += answered and is_supported
        row[3] += is_supported
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(n_boot, len(names)))
    sums = counts[picks].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(sums[:, 1] > 0, sums[:, 0] / sums[:, 1], 0.0)
        coverage = np.where(sums[:, 3] > 0, sums[:, 2] / sums[:, 3], 0.0)
    low, high = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {
        "level": level,
        "n_boot": n_boot,
        "seed": seed,
        "n_filings": len(names),
        "unsupported_answer_rate": [float(np.percentile(rate, low)), float(np.percentile(rate, high))],
        "answer_coverage": [float(np.percentile(coverage, low)), float(np.percentile(coverage, high))],
    }


# --------------------------------------------------------------------------- tuning


@dataclass(frozen=True)
class TuningOutcome:
    """The result of tuning one variant (scope check on or off).

    Attributes:
        scope_check: Whether the scope check was applied.
        points: The full sweep, ascending by threshold.
        chosen: The selected operating point (pooled).
        by_condition: Metrics at the chosen threshold for ``pooled``, ``full``, ``withheld``.
        bootstrap: Grouped bootstrap interval at the chosen threshold.
        confidence_scale: Standard deviation of the best in-scope scores (1.0 if constant).
        degenerate: ``True`` when the chosen policy answers nothing.
        best_rate_at_min_answers: The lowest unsupported-answer rate reachable while
            answering at least :data:`MIN_ANSWERS_FOR_REPORT` questions, with the
            threshold and coverage that gives it; ``None`` if no threshold answers that
            many. It quantifies how far the target is when the policy is degenerate.
    """

    scope_check: bool
    points: list[dict[str, Any]]
    chosen: Mapping[str, Any]
    by_condition: dict[str, dict[str, Any]]
    bootstrap: dict[str, Any]
    confidence_scale: float
    degenerate: bool
    best_rate_at_min_answers: dict[str, Any] | None


def scope_effect(
    items: Sequence[EvalItem],
    chunk_meta: Mapping[str, Mapping[str, Any]],
    known_companies: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Count the items the scope check leaves with no in-scope evidence, by kind of item.

    The check earns its place when it removes withheld and unsupported items and spares
    supported ones; this is that evidence, independent of the chosen threshold.
    """
    counts = {k: {"n": 0, "out_of_scope": 0} for k in ("withheld", "full_supported", "full_unsupported")}
    for item in items:
        kind = "withheld" if item.condition == "withheld" else ("full_supported" if item.supported else "full_unsupported")
        top = evidence_signal(item.question, item.results, chunk_meta, scope_check=True, known_companies=known_companies)[0]
        counts[kind]["n"] += 1
        counts[kind]["out_of_scope"] += top is None
    return counts


def tune_policy(
    items: Sequence[EvalItem],
    chunk_meta: Mapping[str, Mapping[str, Any]],
    known_companies: Sequence[str],
    *,
    scope_check: bool,
    target: float = TARGET_UNSUPPORTED_RATE,
    seed: int = 0,
) -> TuningOutcome:
    """Choose the threshold for one variant of the rule from labelled development items."""
    signals = [
        evidence_signal(
            item.question, item.results, chunk_meta, scope_check=scope_check, known_companies=known_companies
        )[0]
        for item in items
    ]
    supported = [item.supported for item in items]
    points = sweep(signals, supported)
    chosen = select_threshold(points, target)
    threshold = chosen["threshold"]
    by_condition = {"pooled": dict(chosen)}
    for condition in CONDITIONS:
        picked = [i for i, item in enumerate(items) if item.condition == condition]
        by_condition[condition] = metrics_at(threshold, [signals[i] for i in picked], [supported[i] for i in picked])
    reachable = [p for p in points if p["n_answered"] >= MIN_ANSWERS_FOR_REPORT]
    best = None
    if reachable:
        pick = min(reachable, key=lambda p: (p["unsupported_answer_rate"], -(p["answer_coverage"] or 0.0)))
        best = {
            "min_answered": MIN_ANSWERS_FOR_REPORT,
            "threshold": pick["threshold"],
            "n_answered": pick["n_answered"],
            "unsupported_answer_rate": pick["unsupported_answer_rate"],
            "answer_coverage": pick["answer_coverage"],
        }
    observed = [s for s in signals if s is not None]
    scale = float(np.std(observed)) if len(observed) > 1 and np.std(observed) > 0 else 1.0
    return TuningOutcome(
        scope_check=scope_check,
        points=points,
        chosen=chosen,
        by_condition=by_condition,
        bootstrap=grouped_bootstrap(threshold, signals, supported, [i.doc_name for i in items], seed=seed),
        confidence_scale=scale,
        degenerate=chosen["n_answered"] == 0,
        best_rate_at_min_answers=best,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_policy(
    outcome: TuningOutcome,
    *,
    scorer_description: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    provenance: Mapping[str, Any],
    ablation: Mapping[str, Any],
    scope_counts: Mapping[str, Any],
    target: float,
) -> EvidencePolicy:
    """Wrap a tuning outcome in an :class:`EvidencePolicy` that carries its provenance."""
    return EvidencePolicy(
        threshold=float(outcome.chosen["threshold"]),
        confidence_scale=outcome.confidence_scale,
        scope_check=outcome.scope_check,
        scorer=dict(scorer_description),
        tuning={
            "split": "dev",
            "test_ids_read": False,
            **provenance,
            "pipeline": dict(pipeline),
            "selection": {"rule": SELECTION_RULE, "target_unsupported_answer_rate": target},
            "metrics": outcome.by_condition,
            "bootstrap": outcome.bootstrap,
            "degenerate_always_abstains": outcome.degenerate,
            "best_rate_at_min_answers": outcome.best_rate_at_min_answers,
            "scope_check_ablation": dict(ablation),
            "scope_check_effect": dict(scope_counts),
        },
    )


def write_policy(policy: EvidencePolicy, path: Path) -> Path:
    """Write ``policy`` as strict, sorted, timestamp-free JSON so reruns are byte-identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(policy.to_record(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def write_sweep(points: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Write the threshold sweep as CSV, one row per candidate threshold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(points)
    return path


# --------------------------------------------------------------------------- command line


def _summary(label: str, outcome: TuningOutcome) -> str:
    pooled, full, withheld = (outcome.by_condition[k] for k in ("pooled", "full", "withheld"))

    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    low, high = outcome.bootstrap["unsupported_answer_rate"]
    return (
        f"{label}: threshold={pooled['threshold']:.4f} answered={pooled['n_answered']}/{pooled['n']} "
        f"unsupported_rate={fmt(pooled['unsupported_answer_rate'])} (90% CI {low:.3f}-{high:.3f}) "
        f"coverage={fmt(pooled['answer_coverage'])} abst_prec={fmt(pooled['abstention_precision'])} "
        f"abst_rec={fmt(pooled['abstention_recall'])} | full: unsupported={fmt(full['unsupported_answer_rate'])} "
        f"coverage={fmt(full['answer_coverage'])} | withheld: answered={withheld['n_answered']}/{withheld['n']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Tune and write a policy. See the module docstring and ``docs/insufficient_evidence.md``."""
    parser = argparse.ArgumentParser(description="Tune the insufficient-evidence threshold on dev data only.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dev-ids", type=Path, default=DEFAULT_DEV_IDS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--scorer", choices=("idf-coverage", "cross-encoder"), default="idf-coverage")
    parser.add_argument("--model-id", help="cross-encoder model ID (with --scorer cross-encoder)")
    parser.add_argument("--revision", help="full 40-character commit hash of the cross-encoder")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--dense-model", help="optional dense first stage: model ID")
    parser.add_argument("--dense-revision", help="optional dense first stage: 40-character commit hash")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--context-k", type=int, default=CONTEXT_K)
    parser.add_argument("--target", type=float, default=TARGET_UNSUPPORTED_RATE)
    parser.add_argument("--scope-check", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    dev_rows = load_dev_rows(args.dataset, args.dev_ids)
    chunks, rows = load_corpus(args.chunks, dev_rows)
    chunk_meta = {c["chunk_id"]: {k: c.get(k) for k in _META_KEYS} for c in chunks}
    texts = {c["chunk_id"]: c["text"] for c in chunks}
    companies = sorted({m["company"] for m in chunk_meta.values() if m["company"]})
    print(f"{len(rows)} dev questions over {len({r['doc_name'] for r in rows})} filings; {len(chunks)} chunks")

    bm25 = BM25Index.build((c["chunk_id"], c["text"]) for c in chunks)
    retrievers: dict[str, Any] = {"bm25": bm25}
    if args.dense_model:
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.embeddings import SentenceTransformerEncoder

        encoder = SentenceTransformerEncoder.load(args.dense_model, args.dense_revision or "", device=args.device)
        retrievers["dense"] = DenseRetriever.build(((c["chunk_id"], c["text"]) for c in chunks), encoder)

    if args.scorer == "cross-encoder":
        from src.retrieval.rerank import CrossEncoderScorer

        if not args.model_id or not args.revision:
            parser.error("--scorer cross-encoder needs --model-id and --revision")
        scorer: Any = CrossEncoderScorer.load(
            args.model_id, args.revision, device=args.device, max_seq_length=args.max_seq_length
        )
        tag = f"ce-{args.model_id.split('/')[-1]}-{args.revision[:8]}"
    else:
        scorer = IdfCoverageScorer(bm25.idf)
        tag = "idf-coverage"
    if args.dense_model:
        tag += "+dense"

    items = collect_items(
        rows, retrievers, chunk_meta, texts, scorer, candidate_k=args.candidate_k, context_k=args.context_k
    )
    outcomes = {
        flag: tune_policy(items, chunk_meta, companies, scope_check=flag, target=args.target, seed=args.seed)
        for flag in (True, False)
    }
    print(_summary("scope check ON ", outcomes[True]))
    print(_summary("scope check OFF", outcomes[False]))

    if args.scope_check == "auto":
        on, off = outcomes[True].chosen["answer_coverage"] or 0.0, outcomes[False].chosen["answer_coverage"] or 0.0
        use_scope = on >= off
    else:
        use_scope = args.scope_check == "on"
    outcome = outcomes[use_scope]

    pipeline = {
        **RetrieveAndRerank(retrievers, texts, scorer, candidate_k=args.candidate_k).config_record(),
        "context_k": args.context_k,
    }
    pipeline = {k: pipeline[k] for k in ("first_stage", "fusion", "rrf_k", "per_retriever_k", "candidate_k", "context_k")}
    provenance = {
        "dataset_sha256": _sha256(args.dataset),
        "dev_ids_sha256": _sha256(args.dev_ids),
        "n_questions": len(rows),
        "n_filings": len({r["doc_name"] for r in rows}),
        "n_chunks": len(chunks),
        "chunks_file": args.chunks.name,
        "scope_check_choice": args.scope_check,
        "conditions": {
            "full": "whole development corpus; supported = gold evidence page among the top context_k reranked chunks",
            "withheld": "gold filing removed from the index; never supported",
        },
    }
    ablation = {
        ("on" if flag else "off"): {
            "threshold": o.chosen["threshold"],
            "pooled": o.by_condition["pooled"],
            "full": o.by_condition["full"],
            "withheld": o.by_condition["withheld"],
            "best_rate_at_min_answers": o.best_rate_at_min_answers,
        }
        for flag, o in outcomes.items()
    }
    policy = build_policy(
        outcome,
        scorer_description=scorer.describe(),
        pipeline=pipeline,
        provenance=provenance,
        ablation=ablation,
        scope_counts=scope_effect(items, chunk_meta, companies),
        target=args.target,
    )
    policy_path = write_policy(policy, args.out_dir / f"policy_{tag}.json")
    sweep_path = write_sweep(outcome.points, args.out_dir / f"sweep_{tag}.csv")
    print(f"wrote {policy_path} and {sweep_path}")
    if outcome.degenerate:
        best = outcome.best_rate_at_min_answers
        detail = (
            f" The lowest unsupported-answer rate reachable while answering at least {best['min_answered']} "
            f"questions is {best['unsupported_answer_rate']:.3f}."
            if best
            else ""
        )
        print(f"WARNING: no threshold meets the {args.target:.0%} target, so the policy answers nothing.{detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
