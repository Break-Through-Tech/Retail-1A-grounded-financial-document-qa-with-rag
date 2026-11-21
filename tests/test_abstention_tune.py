"""Tests for threshold tuning (``src/abstention/tune.py``) and the reference scorer.

Metrics are checked against hand-worked counts. The load-bearing tests are
:func:`test_tuning_never_opens_the_frozen_test_id_file` (the threshold is tuned on
development data only, and the test proves its own guard is live), the withheld-filing
construction, and :func:`test_scope_check_earns_its_place_when_wrong_filings_score_high`.
"""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

from src.abstention import tune
from src.abstention.policy import EvidencePolicy
from src.abstention.reference_scorer import IdfCoverageScorer
from src.abstention.tune import (
    EvalItem,
    ExcludingRetriever,
    candidate_thresholds,
    collect_items,
    grouped_bootstrap,
    load_corpus,
    load_dev_rows,
    metrics_at,
    scope_effect,
    select_threshold,
    sweep,
    tune_policy,
    write_policy,
    write_sweep,
)
from src.retrieval.bm25 import BM25Index

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "financebench_merged.jsonl"
SPLITS = ROOT / "data" / "splits"
DEV_IDS = SPLITS / "dev_ids.txt"
FROZEN_ID_FILE = (SPLITS / "test_ids.txt").resolve()


# --------------------------------------------------------------------------- metrics

# Six items: 0.9 T, 0.8 F, 0.7 T answered at 0.5; 0.4 T, None F, 0.2 F abstained.
SIGNALS: list[float | None] = [0.9, 0.8, 0.4, None, 0.2, 0.7]
SUPPORTED = [True, False, True, False, False, True]


def test_metrics_match_a_hand_count() -> None:
    m = metrics_at(0.5, SIGNALS, SUPPORTED)
    assert (m["n"], m["n_supported"], m["n_answered"], m["n_abstained"]) == (6, 3, 3, 3)
    assert (m["answered_supported"], m["answered_unsupported"]) == (2, 1)
    assert m["unsupported_answer_rate"] == pytest.approx(1 / 3)
    assert m["answer_coverage"] == pytest.approx(2 / 3)
    # abstained: 0.4 (supported), None (unsupported), 0.2 (unsupported) -> 2 of 3 unsupported
    assert m["abstention_precision"] == pytest.approx(2 / 3)
    # unsupported overall: 0.8, None, 0.2 -> 2 of those 3 were abstained
    assert m["abstention_recall"] == pytest.approx(2 / 3)


def test_a_missing_signal_is_never_answered() -> None:
    assert metrics_at(-1e9, [None, None], [False, False])["n_answered"] == 0


def test_undefined_ratios_are_none_not_zero() -> None:
    nothing_answered = metrics_at(2.0, [0.1, 0.2], [True, False])
    assert nothing_answered["unsupported_answer_rate"] is None
    assert nothing_answered["answer_coverage"] == 0.0
    nothing_supported = metrics_at(0.0, [0.1, 0.2], [False, False])
    assert nothing_supported["answer_coverage"] is None
    all_answered = metrics_at(0.0, [0.1, 0.2], [True, False])
    assert all_answered["abstention_precision"] is None


def test_metrics_reject_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        metrics_at(0.5, [0.1], [True, False])


def test_threshold_is_inclusive() -> None:
    assert metrics_at(0.5, [0.5], [True])["n_answered"] == 1
    assert metrics_at(0.5000001, [0.5], [True])["n_answered"] == 0


# --------------------------------------------------------------------------- thresholds


def test_candidate_thresholds_sit_between_observed_scores() -> None:
    values = candidate_thresholds([0.5, None, 0.2, 0.9, 0.5])
    assert values[1:-1] == pytest.approx([0.35, 0.7])
    assert values[0] < 0.2 and values[-1] > 0.9


def test_candidate_thresholds_edge_cases() -> None:
    assert candidate_thresholds([None, None]) == [0.0]
    assert candidate_thresholds([0.4]) == [-0.6, 1.4]


def test_sweep_has_one_point_per_gap_plus_both_extremes() -> None:
    points = sweep(SIGNALS, SUPPORTED)
    assert len(points) == len({s for s in SIGNALS if s is not None}) + 1
    assert points[0]["n_answered"] == 5 and points[-1]["n_answered"] == 0  # the None never answers


def test_selection_takes_the_most_coverage_within_the_target() -> None:
    separable_signals = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    separable_supported = [True, True, True, False, False, False]
    best = select_threshold(sweep(separable_signals, separable_supported), 0.10)
    assert best["threshold"] == pytest.approx(0.5)
    assert (best["answer_coverage"], best["unsupported_answer_rate"]) == (1.0, 0.0)


def test_selection_respects_the_target_when_scores_overlap() -> None:
    signals = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    supported = [True, True, False, True, False, False]
    strict = select_threshold(sweep(signals, supported), 0.0)
    assert strict["threshold"] == pytest.approx(0.75) and strict["answer_coverage"] == pytest.approx(2 / 3)
    loose = select_threshold(sweep(signals, supported), 0.5)
    assert loose["answer_coverage"] == 1.0 and loose["unsupported_answer_rate"] <= 0.5


def test_ties_go_to_the_higher_threshold() -> None:
    points = [
        {"threshold": 0.2, "unsupported_answer_rate": 0.0, "answer_coverage": 0.5},
        {"threshold": 0.8, "unsupported_answer_rate": 0.0, "answer_coverage": 0.5},
    ]
    assert select_threshold(points, 0.1)["threshold"] == 0.8


def test_selection_can_degrade_to_answering_nothing() -> None:
    # Every answerable-looking item is unsupported, so only abstaining on everything qualifies.
    best = select_threshold(sweep([0.9, 0.8, 0.7], [False, False, False]), 0.10)
    assert best["n_answered"] == 0


def test_selection_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_threshold([], 0.1)
    with pytest.raises(ValueError, match="target"):
        select_threshold(sweep([0.5], [True]), 1.5)


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_is_deterministic_and_bounded() -> None:
    groups = ["a", "a", "b", "b", "c", "c"]
    first = grouped_bootstrap(0.5, SIGNALS, SUPPORTED, groups, seed=3)
    assert first == grouped_bootstrap(0.5, SIGNALS, SUPPORTED, groups, seed=3)
    low, high = first["unsupported_answer_rate"]
    assert 0.0 <= low <= high <= 1.0 and first["n_filings"] == 3


def test_bootstrap_of_a_perfect_separation_never_finds_an_unsupported_answer() -> None:
    signals = [0.9, 0.8, 0.1, 0.2]
    out = grouped_bootstrap(0.5, signals, [True, True, False, False], ["a", "b", "c", "d"])
    assert out["unsupported_answer_rate"] == [0.0, 0.0]
    # A resample of four single-item filings can hold no supported one, so coverage
    # honestly ranges from 0 up to 1.
    assert out["answer_coverage"][1] == 1.0


def test_bootstrap_resamples_filings_not_items() -> None:
    # One filing holds the only unsupported answer; resamples without it have rate 0 and
    # resamples with it do not, so the upper end must be positive.
    signals = [0.9] * 8
    supported = [True] * 6 + [False, False]
    groups = ["a", "b", "c", "d", "e", "f", "g", "g"]
    out = grouped_bootstrap(0.5, signals, supported, groups, n_boot=500)
    assert out["unsupported_answer_rate"][1] > 0.0


# --------------------------------------------------------------------------- withheld filing


class FakeBase:
    """Ranked list of chunk IDs; ``retrieve`` returns the first ``k``."""

    def __init__(self, ranking: Sequence[str]) -> None:
        self.ranking = list(ranking)
        self.asked: list[int] = []

    def retrieve(self, query: str, k: int) -> list[Any]:
        self.asked.append(k)
        return [SimpleNamespace(chunk_id=c, score=float(len(self.ranking) - i)) for i, c in enumerate(self.ranking[:k])]


CHUNK_DOC = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B"}


def test_excluding_retriever_removes_the_filing() -> None:
    base = FakeBase(["a1", "b1", "a2", "b2"])
    hits = ExcludingRetriever(base, CHUNK_DOC, "A").retrieve("q", 2)
    assert [h.chunk_id for h in hits] == ["b1", "b2"]


def test_excluding_retriever_overfetches_until_k_survive() -> None:
    base = FakeBase(["a1", "a2", "a3", "b1", "b2"])
    hits = ExcludingRetriever(base, CHUNK_DOC, "A").retrieve("q", 2)
    assert [h.chunk_id for h in hits] == ["b1", "b2"]
    assert base.asked == [2, 8]


def test_excluding_retriever_returns_fewer_when_the_base_runs_out() -> None:
    base = FakeBase(["a1", "b1"])
    assert [h.chunk_id for h in ExcludingRetriever(base, CHUNK_DOC, "A").retrieve("q", 5)] == ["b1"]
    assert [h.chunk_id for h in ExcludingRetriever(FakeBase(["a1"]), CHUNK_DOC, "A").retrieve("q", 3)] == []


def _mini_corpus() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    chunks = [
        {"chunk_id": "A_p3_0", "doc_name": "A", "page_num": 3, "company": "AlphaCo", "doc_period": 2018, "text": "alphaco capital expenditure purchases of property plant equipment 2018"},
        {"chunk_id": "A_p4_0", "doc_name": "A", "page_num": 4, "company": "AlphaCo", "doc_period": 2018, "text": "alphaco dividends paid to shareholders"},
        {"chunk_id": "B_p1_0", "doc_name": "B", "page_num": 1, "company": "AlphaCo", "doc_period": 2022, "text": "alphaco capital expenditure 2022 purchases"},
        {"chunk_id": "C_p9_0", "doc_name": "C", "page_num": 9, "company": "BetaCo", "doc_period": 2018, "text": "betaco risk factors and supply chain"},
    ]
    meta = {c["chunk_id"]: {k: c[k] for k in ("doc_name", "page_num", "company", "doc_period")} for c in chunks}
    texts = {c["chunk_id"]: c["text"] for c in chunks}
    rows = [
        {
            "financebench_id": "q1",
            "doc_name": "A",
            "question": "What was AlphaCo capital expenditure purchases of property plant equipment in 2018?",
            "evidence": [{"doc_name": "A", "evidence_page_num": 3}],
        }
    ]
    return chunks, meta, texts, rows


def test_collect_items_builds_a_full_and_a_withheld_item_per_question() -> None:
    chunks, meta, texts, rows = _mini_corpus()
    bm25 = BM25Index.build((c["chunk_id"], c["text"]) for c in chunks)
    items = collect_items(rows, {"bm25": bm25}, meta, texts, IdfCoverageScorer(bm25.idf), candidate_k=10, context_k=3)
    assert [(i.condition, i.doc_name) for i in items] == [("full", "A"), ("withheld", "A")]
    full, withheld = items
    assert full.supported is True and full.results[0].chunk_id == "A_p3_0"
    assert withheld.supported is False
    assert all(meta[r.chunk_id]["doc_name"] != "A" for r in withheld.results)
    assert withheld.results  # related filings stay in the index: the hard realistic case


def test_supported_is_judged_on_the_page_not_the_chunk() -> None:
    chunks, meta, texts, rows = _mini_corpus()
    bm25 = BM25Index.build((c["chunk_id"], c["text"]) for c in chunks)
    rows[0]["evidence"] = [{"doc_name": "A", "evidence_page_num": 4}]  # a different page of the same filing
    full = collect_items(rows, {"bm25": bm25}, meta, texts, IdfCoverageScorer(bm25.idf), candidate_k=10, context_k=1)[0]
    assert full.results[0].chunk_id == "A_p3_0" and full.supported is False


# --------------------------------------------------------------------------- tuning end to end


def _item(fid: str, doc: str, condition: str, supported: bool, hits: list[tuple[str, float]], question: str) -> EvalItem:
    results = tuple(SimpleNamespace(chunk_id=c, score=s, retrieval_rank=i + 1, retrieval_score=s) for i, (c, s) in enumerate(hits))
    return EvalItem(fid, doc, condition, question, supported, results)


SYN_META = {
    "right": {"doc_name": "A", "page_num": 1, "company": "AlphaCo", "doc_period": 2018},
    "wrong_year": {"doc_name": "B", "page_num": 1, "company": "AlphaCo", "doc_period": 2022},
}
SYN_COMPANIES = ["AlphaCo"]
SYN_Q = "What did AlphaCo spend in 2018?"


def _synthetic_items() -> list[EvalItem]:
    """Four supported questions score 0.8-0.9 from the right filing. Four withheld ones
    retrieve the wrong year's filing, which scores just as high (0.7-0.95)."""
    items = []
    for i, score in enumerate([0.9, 0.85, 0.8, 0.82]):
        items.append(_item(f"s{i}", f"D{i}", "full", True, [("right", score)], SYN_Q))
    for i, score in enumerate([0.95, 0.9, 0.7, 0.8]):
        items.append(_item(f"w{i}", f"D{i}", "withheld", False, [("wrong_year", score)], SYN_Q))
    return items


def test_scope_check_earns_its_place_when_wrong_filings_score_high() -> None:
    items = _synthetic_items()
    off = tune_policy(items, SYN_META, SYN_COMPANIES, scope_check=False, target=0.10)
    on = tune_policy(items, SYN_META, SYN_COMPANIES, scope_check=True, target=0.10)
    # Without the scope check the wrong-year scores sit among the right ones, so meeting
    # the target costs coverage (or everything).
    assert (off.chosen["answer_coverage"] or 0.0) < 1.0
    # With it the wrong-year chunks have no in-scope evidence and cannot be answered.
    assert on.chosen["answer_coverage"] == 1.0 and on.chosen["unsupported_answer_rate"] == 0.0
    assert on.by_condition["withheld"]["n_answered"] == 0
    assert not on.degenerate


def test_outcome_reports_each_condition_at_the_chosen_threshold() -> None:
    on = tune_policy(_synthetic_items(), SYN_META, SYN_COMPANIES, scope_check=True)
    assert set(on.by_condition) == {"pooled", "full", "withheld"}
    threshold = on.chosen["threshold"]
    assert all(on.by_condition[k]["threshold"] == threshold for k in on.by_condition)
    assert on.by_condition["full"]["n"] == on.by_condition["withheld"]["n"] == 4


def test_a_degenerate_outcome_is_flagged_and_quantified() -> None:
    items = [_item(f"u{i}", f"D{i}", "full", False, [("right", 0.5 + i / 100)], SYN_Q) for i in range(12)]
    outcome = tune_policy(items, SYN_META, SYN_COMPANIES, scope_check=True, target=0.10)
    assert outcome.degenerate and outcome.chosen["n_answered"] == 0
    assert outcome.best_rate_at_min_answers["unsupported_answer_rate"] == 1.0
    assert outcome.best_rate_at_min_answers["n_answered"] >= tune.MIN_ANSWERS_FOR_REPORT


def test_scope_effect_counts_by_kind_of_item() -> None:
    counts = scope_effect(_synthetic_items(), SYN_META, SYN_COMPANIES)
    assert counts["withheld"] == {"n": 4, "out_of_scope": 4}
    assert counts["full_supported"] == {"n": 4, "out_of_scope": 0}
    assert counts["full_unsupported"] == {"n": 0, "out_of_scope": 0}


def test_confidence_scale_is_the_spread_of_the_scores() -> None:
    outcome = tune_policy(_synthetic_items(), SYN_META, SYN_COMPANIES, scope_check=True)
    assert outcome.confidence_scale == pytest.approx(float(np.std([0.9, 0.85, 0.8, 0.82])))
    constant = [_item(f"c{i}", f"D{i}", "full", True, [("right", 0.5)], SYN_Q) for i in range(3)]
    assert tune_policy(constant, SYN_META, SYN_COMPANIES, scope_check=True).confidence_scale == 1.0


# --------------------------------------------------------------------------- artifacts


def _policy() -> EvidencePolicy:
    outcome = tune_policy(_synthetic_items(), SYN_META, SYN_COMPANIES, scope_check=True)
    return tune.build_policy(
        outcome,
        scorer_description={"scorer": "stub"},
        pipeline={"first_stage": ["bm25"], "fusion": "none", "rrf_k": None, "per_retriever_k": None, "candidate_k": 50, "context_k": 5},
        provenance={"dataset_sha256": "x"},
        ablation={},
        scope_counts={},
        target=0.10,
    )


def test_written_policy_is_strict_json_that_loads_back(tmp_path: Path) -> None:
    policy = _policy()
    path = write_policy(policy, tmp_path / "nested" / "policy.json")
    json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(f"non-strict {c}"))
    assert EvidencePolicy.load(path) == policy


def test_the_policy_records_that_it_was_tuned_on_dev_only() -> None:
    tuning = _policy().tuning
    assert tuning["split"] == "dev" and tuning["test_ids_read"] is False
    assert tuning["selection"]["target_unsupported_answer_rate"] == 0.10
    assert set(tuning["metrics"]) == {"pooled", "full", "withheld"}


def test_rewriting_a_policy_is_byte_identical(tmp_path: Path) -> None:
    first = write_policy(_policy(), tmp_path / "a.json").read_bytes()
    second = write_policy(_policy(), tmp_path / "b.json").read_bytes()
    assert first == second


def test_sweep_csv_has_one_row_per_threshold(tmp_path: Path) -> None:
    points = sweep(SIGNALS, SUPPORTED)
    path = write_sweep(points, tmp_path / "sweep.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(points) + 1 and lines[0].startswith("threshold,")


# --------------------------------------------------------------------------- development data only


def test_dev_rows_are_exactly_the_development_split() -> None:
    rows = load_dev_rows(DATASET, DEV_IDS)
    dev_ids = DEV_IDS.read_text(encoding="utf-8").split()
    assert len(rows) == 111
    assert [r["financebench_id"] for r in rows] == sorted(dev_ids)
    assert not {r["financebench_id"] for r in rows} & set(FROZEN_ID_FILE.read_text(encoding="utf-8").split())


def test_tuning_never_opens_the_frozen_test_id_file(monkeypatch: pytest.MonkeyPatch) -> None:
    real_path_open, real_builtin_open = Path.open, builtins.open

    def guarded_path_open(self: Path, *args: object, **kwargs: object):
        if Path(self).resolve() == FROZEN_ID_FILE:
            raise AssertionError("opened the frozen test ID file")
        return real_path_open(self, *args, **kwargs)

    def guarded_builtin_open(file: object, *args: object, **kwargs: object):
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == FROZEN_ID_FILE:
            raise AssertionError("opened the frozen test ID file")
        return real_builtin_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)

    assert len(load_dev_rows(DATASET, DEV_IDS)) == 111
    # The guard must be live, or the pass above proves nothing.
    with pytest.raises(AssertionError, match="frozen test ID file"):
        FROZEN_ID_FILE.read_text(encoding="utf-8")


def test_no_module_in_the_package_names_the_frozen_file() -> None:
    for path in (ROOT / "src" / "abstention").glob("*.py"):
        assert "test_ids.txt" not in path.read_text(encoding="utf-8"), path.name


def test_corpus_loading_drops_filings_outside_the_development_split(tmp_path: Path) -> None:
    dev_rows = [{"doc_name": "DEV_A", "financebench_id": "q1"}, {"doc_name": "DEV_NO_CHUNKS", "financebench_id": "q2"}]
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps({"chunk_id": f"{d}_{i}", "doc_name": d, "text": "t"}) for i, d in enumerate(["DEV_A", "TEST_X", "DEV_A"])),
        encoding="utf-8",
    )
    chunks, rows = load_corpus(path, dev_rows)
    assert {c["doc_name"] for c in chunks} == {"DEV_A"}  # a frozen-test filing cannot leak in
    assert [r["financebench_id"] for r in rows] == ["q1"]  # no chunks, no question


def test_corpus_loading_fails_when_no_development_filing_is_present(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps({"chunk_id": "x", "doc_name": "OTHER", "text": "t"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no development filing"):
        load_corpus(path, [{"doc_name": "DEV_A", "financebench_id": "q1"}])


# --------------------------------------------------------------------------- command line


def _cli_chunks(tmp_path: Path) -> Path:
    """Chunks for four real development filings, built from their own gold evidence text."""
    rows = load_dev_rows(DATASET, DEV_IDS)[:12]
    lines = []
    for row in rows:
        for j, evidence in enumerate(row["evidence"]):
            lines.append(
                {
                    "chunk_id": f"{row['doc_name']}_p{evidence['evidence_page_num']}_{j}_{row['financebench_id']}",
                    "doc_name": row["doc_name"],
                    "page_num": evidence["evidence_page_num"],
                    "company": row["company"],
                    "doc_type": row["doc_type"],
                    "doc_period": row["doc_period"],
                    "doc_link": row["doc_link"],
                    "text": evidence["evidence_text"],
                }
            )
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    return path


def test_command_line_writes_a_policy_and_a_sweep_deterministically(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chunks = _cli_chunks(tmp_path)
    args = ["--dataset", str(DATASET), "--dev-ids", str(DEV_IDS), "--chunks", str(chunks)]
    assert tune.main([*args, "--out-dir", str(tmp_path / "one")]) == 0
    assert tune.main([*args, "--out-dir", str(tmp_path / "two")]) == 0
    first, second = (tmp_path / d / "policy_idf-coverage.json" for d in ("one", "two"))
    assert first.read_bytes() == second.read_bytes()
    assert (tmp_path / "one" / "sweep_idf-coverage.csv").exists()
    record = json.loads(first.read_text(encoding="utf-8"))
    assert record["tuning"]["split"] == "dev" and record["tuning"]["test_ids_read"] is False
    assert record["scorer"] == {"scorer": "idf-coverage", "version": "1.0.0"}
    assert "scope check ON" in capsys.readouterr().out


def test_command_line_requires_a_pinned_cross_encoder(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        tune.main(["--chunks", str(_cli_chunks(tmp_path)), "--scorer", "cross-encoder", "--out-dir", str(tmp_path)])


# --------------------------------------------------------------------------- reference scorer


def test_idf_coverage_matches_a_hand_calculation() -> None:
    scorer = IdfCoverageScorer({"a": 1.0, "b": 3.0})
    # Query terms a, b and an unseen term c (weight = max idf = 3.0): total weight 7.
    scores = scorer.score_pairs("a b c", ["a b", "b", "x y", "a b c"])
    assert scores.tolist() == pytest.approx([4 / 7, 3 / 7, 0.0, 1.0])
    assert scores.dtype == np.float64


def test_idf_coverage_ignores_repeats_and_case() -> None:
    scorer = IdfCoverageScorer({"a": 1.0, "b": 3.0})
    assert scorer.score_pairs("A a A", ["a a a"]).tolist() == [1.0]


def test_idf_coverage_of_a_query_with_no_terms_is_zero() -> None:
    assert IdfCoverageScorer({"a": 1.0}).score_pairs("?!", ["a"]).tolist() == [0.0]


def test_idf_coverage_scores_stay_in_the_unit_interval() -> None:
    scorer = IdfCoverageScorer({"a": 0.5, "b": 2.0, "c": 4.0})
    scores = scorer.score_pairs("a b c d", ["a", "a b c d", "zzz", "b c"])
    assert ((scores >= 0) & (scores <= 1)).all()


def test_idf_coverage_requires_an_idf_map_and_describes_itself() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        IdfCoverageScorer({})
    assert IdfCoverageScorer({"a": 1.0}).describe() == {"scorer": "idf-coverage", "version": "1.0.0"}
