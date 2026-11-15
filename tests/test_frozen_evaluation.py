"""Synthetic integration coverage for Issue #33's frozen-evaluation plumbing."""

from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from src.evaluation.evaluate_test import (
    evaluate_frozen_rankings,
    load_rankings,
    _require_clean_worktree,
    validate_corpus_manifest,
    validate_retrieval_config,
    write_outputs,
)
from src.prepare_test_corpus import prepare_test_corpus, select_test_docs
from src.retrieval.queries import load_rows, load_test_queries, validate_frozen_split
from src.data.split import sha256_of

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "financebench_merged.jsonl"
TEST_IDS = ROOT / "data" / "splits" / "test_ids.txt"
DEV_IDS = ROOT / "data" / "splits" / "dev_ids.txt"


def query(question_id: str):
    from src.retrieval.queries import DevQuery

    return DevQuery(question_id, f"question for {question_id}")


def test_test_loader_has_exactly_39_unique_disjoint_queries() -> None:
    queries = load_test_queries(DATASET, TEST_IDS)
    test_ids = {item.financebench_id for item in queries}
    dev_ids = set(DEV_IDS.read_text(encoding="utf-8").split())
    assert len(queries) == 39
    assert len(test_ids) == 39
    assert not test_ids & dev_ids


def test_synthetic_matching_and_multi_page_all_at_5() -> None:
    queries = [query("q1"), query("q2")]
    gold = {
        "q1": [{"doc_name": "filing-a", "evidence_page_num": 3}],
        "q2": [
            {"doc_name": "filing-b", "evidence_page_num": 10},
            {"doc_name": "filing-b", "evidence_page_num": 12},
        ],
    }
    rankings = {
        "q1": [
            {"doc_name": "wrong-filing", "evidence_page_num": 3},
            {"doc_name": "filing-a", "evidence_page_num": 3},
        ],
        "q2": [
            {"doc_name": "filing-b", "evidence_page_num": 10},
            {"doc_name": "filing-b", "evidence_page_num": 99},
            {"doc_name": "filing-b", "evidence_page_num": 12},
        ],
    }

    rows, metrics = evaluate_frozen_rankings(queries, rankings, gold)

    assert rows[0]["hit@1"] == 0
    assert rows[0]["hit@5"] == 1
    assert rows[1]["hit@5"] == 1
    assert rows[1]["all@5"] == 1
    assert metrics == {
        "hit@1": 0.5,
        "hit@5": 1.0,
        "mrr@10": 0.75,
        "all@5": 1.0,
        "questions": 2,
    }


def test_synthetic_multi_page_failure_is_not_hidden_by_hit_at_5() -> None:
    queries = [query("q")]
    gold = {
        "q": [
            {"doc_name": "filing", "evidence_page_num": 1},
            {"doc_name": "filing", "evidence_page_num": 2},
        ]
    }
    rankings = {"q": [{"doc_name": "filing", "evidence_page_num": 1}]}

    rows, metrics = evaluate_frozen_rankings(queries, rankings, gold)

    assert rows[0]["hit@5"] == 1
    assert rows[0]["all@5"] == 0
    assert metrics["all@5"] == 0.0


def test_evaluator_requires_complete_query_coverage() -> None:
    with pytest.raises(ValueError, match="exactly one record"):
        evaluate_frozen_rankings(
            [query("q1"), query("q2")],
            {"q1": []},
            {"q1": [], "q2": []},
        )


def test_synthetic_corpus_selection_excludes_development_filing() -> None:
    docs = {
        "dev-filing": [{"doc_name": "dev-filing", "page_num": 0, "text": "dev"}],
        "test-filing": [{"doc_name": "test-filing", "page_num": 0, "text": "test"}],
    }

    selected = select_test_docs(docs, {"test-filing"})

    assert set(selected) == {"test-filing"}
    assert "dev-filing" not in selected


def test_outputs_record_config_rankings_manifest_and_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.evaluation.evaluate_test._commit", lambda: "commit-for-test")
    rows = [{"financebench_id": "q", "rankings": [], "hit@1": 0, "hit@5": 0,
             "all@5": 0, "mrr@10": 0.0}]
    write_outputs(
        tmp_path,
        {"retriever": "synthetic", "chunking": "synthetic", "retrieval_depth": 10},
        {"filings": ["filing"], "artifacts": []},
        rows,
        {"hit@1": 0.0, "hit@5": 0.0, "mrr@10": 0.0, "all@5": 0.0, "questions": 1},
    )

    assert json.loads((tmp_path / "config.json").read_text())["git"]["commit"] == "commit-for-test"
    assert json.loads((tmp_path / "corpus_manifest.json").read_text())["filings"] == ["filing"]
    assert json.loads((tmp_path / "metrics.json").read_text())["questions"] == 1
    assert (tmp_path / "per_question_rankings.jsonl").read_text().count("\n") == 1
    with pytest.raises(FileExistsError, match="not empty"):
        write_outputs(
            tmp_path,
            {"retriever": "synthetic", "chunking": "synthetic", "retrieval_depth": 1},
            {},
            [],
            {},
        )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"retriever": "bm25", "chunking": "page"},
        {"retriever": "", "chunking": "page", "retrieval_depth": 10},
        {"retriever": "bm25", "chunking": {}, "retrieval_depth": 10},
        {"retriever": "bm25", "chunking": "page", "retrieval_depth": 0},
        {"retriever": "bm25", "chunking": "page", "retrieval_depth": True},
    ],
)
def test_retrieval_config_rejects_missing_or_invalid_required_fields(config) -> None:
    with pytest.raises(ValueError):
        validate_retrieval_config(config)


def test_retrieval_config_allows_future_values() -> None:
    config = validate_retrieval_config(
        {"retriever": "future-hybrid", "chunking": {"strategy": "custom"}, "retrieval_depth": 25,
         "future_option": {"alpha": 0.3}}
    )
    assert config["retriever"] == "future-hybrid"


def _ranking_file(tmp_path: Path, value: str) -> Path:
    path = tmp_path / "rankings.jsonl"
    path.write_text(value, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "value",
    [
        "not-json\n",
        "[]\n",
        '{"financebench_id":"q","rankings":[{"doc_name":"d","evidence_page_num":true}]}\n',
        '{"financebench_id":"q","rankings":[{"doc_name":"d","evidence_page_num":-1}]}\n',
        '{"financebench_id":"q","rankings":[{"doc_name":"d","evidence_page_num":1,"score":Infinity}]}\n',
        '{"financebench_id":"q","rankings":[{"doc_name":"d","evidence_page_num":1},{"doc_name":"d","evidence_page_num":1}]}\n',
    ],
)
def test_rankings_reject_malformed_or_duplicate_items(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError):
        load_rankings(_ranking_file(tmp_path, value), {"q"})


def test_rankings_reject_duplicate_question_and_wrong_coverage(tmp_path: Path) -> None:
    duplicate = _ranking_file(
        tmp_path,
        '{"financebench_id":"q","rankings":[]}\n{"financebench_id":"q","rankings":[]}\n',
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_rankings(duplicate)
    wrong_ids = _ranking_file(tmp_path, '{"financebench_id":"unexpected","rankings":[]}\n')
    with pytest.raises(ValueError, match="expected question IDs"):
        load_rankings(wrong_ids, {"q"})


def test_split_validator_rejects_partition_and_group_invariant_changes() -> None:
    rows = load_rows(DATASET)
    metadata = json.loads((ROOT / "data/splits/split_metadata.json").read_text(encoding="utf-8"))
    test_ids = set(TEST_IDS.read_text(encoding="utf-8").split())
    dev_ids = set(DEV_IDS.read_text(encoding="utf-8").split())
    validate_frozen_split(rows, metadata, test_ids, dev_ids)

    bad_metadata = copy.deepcopy(metadata)
    bad_metadata["invariants"]["grouping_key_overlap"] = 1
    with pytest.raises(ValueError, match="grouping_key_overlap"):
        validate_frozen_split(rows, bad_metadata, test_ids, dev_ids)

    with pytest.raises(ValueError, match="counts"):
        validate_frozen_split(rows, metadata, test_ids - {next(iter(test_ids))}, dev_ids)

    bad_rows = copy.deepcopy(rows)
    by_id = {row["financebench_id"]: row for row in bad_rows}
    by_id[next(iter(test_ids))]["doc_name"] = by_id[next(iter(dev_ids))]["doc_name"]
    with pytest.raises(ValueError, match="document groups"):
        validate_frozen_split(bad_rows, metadata, test_ids, dev_ids)


def _manifest_fixture(tmp_path: Path) -> tuple[dict, Path]:
    pdf_root = tmp_path / "pdfs"
    output_root = tmp_path / "processed"
    pdf_root.mkdir()
    output_root.mkdir()
    (pdf_root / "TEST.pdf").write_bytes(b"pdf")
    artifact_names = ["pages.jsonl", "chunks_page.jsonl", "chunks_w300.jsonl", "chunks_w600.jsonl"]
    for name in artifact_names:
        (output_root / name).write_text(name + "\n", encoding="utf-8")

    def record(path: Path, root: Path) -> dict:
        return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "sha256": sha256_of(path)}

    return {
        "purpose": "frozen-test-corpus",
        "dataset": {"sha256": "7a1c81789e0fd2f1c37057a7ec0097756d726b05e7228e68e57db8e18c54fd0b"},
        "split": {"ids_sha256": sha256_of(TEST_IDS)},
        "filings": ["TEST"],
        "roots": {"pdf_dir": str(pdf_root), "output_dir": str(output_root)},
        "pdfs": [record(pdf_root / "TEST.pdf", pdf_root)],
        "artifacts": [record(output_root / name, output_root) for name in artifact_names],
    }, tmp_path / "corpus_manifest.json"


def test_manifest_validator_checks_purpose_coverage_and_hashes(tmp_path: Path) -> None:
    manifest, manifest_path = _manifest_fixture(tmp_path)
    validate_corpus_manifest(manifest, manifest_path, {"TEST"}, DATASET, TEST_IDS)

    bad = copy.deepcopy(manifest)
    bad["purpose"] = "development-corpus"
    with pytest.raises(ValueError, match="frozen-test"):
        validate_corpus_manifest(bad, manifest_path, {"TEST"}, DATASET, TEST_IDS)

    bad = copy.deepcopy(manifest)
    bad["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        validate_corpus_manifest(bad, manifest_path, {"TEST"}, DATASET, TEST_IDS)

    bad = copy.deepcopy(manifest)
    bad["filings"] = []
    with pytest.raises(ValueError, match="filing coverage"):
        validate_corpus_manifest(bad, manifest_path, {"TEST"}, DATASET, TEST_IDS)

    bad = copy.deepcopy(manifest)
    bad["dataset"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dataset hash"):
        validate_corpus_manifest(bad, manifest_path, {"TEST"}, DATASET, TEST_IDS)

    (tmp_path / "processed" / "chunks_w600.jsonl").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_corpus_manifest(manifest, manifest_path, {"TEST"}, DATASET, TEST_IDS)


def test_corpus_preparation_rejects_development_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development artifact"):
        prepare_test_corpus(ROOT / "data" / "pdfs", tmp_path)


def test_corpus_preparation_rejects_missing_or_empty_extraction(tmp_path: Path, monkeypatch) -> None:
    import src.prepare_test_corpus as corpus

    pdf_dir = tmp_path / "pdfs"
    out_dir = tmp_path / "out"
    pdf_dir.mkdir()
    (pdf_dir / "TEST.pdf").write_bytes(b"synthetic")
    monkeypatch.setattr(corpus, "test_filings", lambda *_args: {"TEST"})
    monkeypatch.setattr(corpus, "extract_pages", lambda **kwargs: None)
    monkeypatch.setattr(corpus, "load_pages", lambda **_kwargs: {})
    with pytest.raises(ValueError, match="missing"):
        corpus.prepare_test_corpus(pdf_dir, out_dir, DATASET, TEST_IDS)
    monkeypatch.setattr(corpus, "load_pages", lambda **_kwargs: {"TEST": [{"text": ""}]})
    with pytest.raises(ValueError, match="empty"):
        corpus.prepare_test_corpus(pdf_dir, out_dir, DATASET, TEST_IDS)


def test_frozen_evaluation_requires_clean_worktree(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.evaluation.evaluate_test._git_provenance",
        lambda: {"commit": "commit", "dirty": True},
    )
    with pytest.raises(RuntimeError, match="clean working tree"):
        _require_clean_worktree()
