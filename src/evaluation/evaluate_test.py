"""Evaluate a frozen ranking file without running a retriever.

The evaluator deliberately consumes rankings produced by a separately frozen
retrieval configuration. This keeps the one-time test evaluation auditable and
prevents this metric entry point from selecting or tuning retrieval parameters.

Ranking JSONL records have the form::

    {"financebench_id": "...", "rankings": [
      {"doc_name": "...", "evidence_page_num": 12, "score": 0.5}
    ]}

Usage::

    python -m src.evaluation.evaluate_test --rankings rankings.jsonl \
      --config retrieval_config.json --corpus-manifest data/test_processed/corpus_manifest.json \
      --output-dir results/final-test
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.split import DATASET_SHA256, DEFAULT_DATASET, sha256_of
from src.evaluation.metrics import evaluate_rankings, is_gold_match
from src.retrieval.queries import (
    DEFAULT_TEST_IDS,
    DevQuery,
    load_test_queries,
    validate_frozen_split,
)

ROOT = Path(__file__).resolve().parents[2]


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_provenance() -> dict[str, object]:
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = True
    return {"commit": _commit(), "dirty": dirty}


def _require_clean_worktree() -> None:
    provenance = _git_provenance()
    if provenance["dirty"]:
        raise RuntimeError(
            "frozen evaluation requires a clean working tree; commit or remove local changes"
        )


def validate_retrieval_config(config: Any) -> dict[str, Any]:
    """Validate the minimal frozen retrieval configuration schema."""
    if not isinstance(config, dict):
        raise ValueError("retrieval configuration must be a JSON object")
    retriever = config.get("retriever")
    chunking = config.get("chunking")
    depth = config.get("retrieval_depth")
    if not isinstance(retriever, str) or not retriever.strip():
        raise ValueError("retrieval configuration requires a non-empty retriever")
    if isinstance(chunking, str):
        if not chunking.strip():
            raise ValueError("chunking strategy must be non-empty")
    elif isinstance(chunking, dict):
        if not isinstance(chunking.get("strategy"), str) or not chunking["strategy"].strip():
            raise ValueError("chunking configuration requires a non-empty strategy")
    else:
        raise ValueError("retrieval configuration requires chunking strategy")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("retrieval_depth must be a positive integer")
    return config


def load_rankings(
    path: Path, expected_ids: set[str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Load rankings, rejecting duplicate or malformed question records."""
    rankings: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"invalid ranking record at {path}:{line_number}")
            question_id = record.get("financebench_id")
            values = record.get("rankings")
            if not isinstance(question_id, str) or not isinstance(values, list):
                raise ValueError(f"invalid ranking record at {path}:{line_number}")
            if question_id in rankings:
                raise ValueError(f"duplicate ranking record for {question_id}")
            for result in values:
                if not isinstance(result, dict):
                    raise ValueError(f"invalid ranking item at {path}:{line_number}")
                page = result.get("evidence_page_num")
                if not isinstance(result.get("doc_name"), str) or not result["doc_name"].strip():
                    raise ValueError(f"ranking item lacks document/page at {path}:{line_number}")
                if isinstance(page, bool) or not isinstance(page, int) or page < 0:
                    raise ValueError(f"ranking item has invalid page at {path}:{line_number}")
                if "score" in result and (
                    isinstance(result["score"], bool)
                    or not isinstance(result["score"], (int, float))
                    or not math.isfinite(result["score"])
                ):
                    raise ValueError(f"ranking item has invalid score at {path}:{line_number}")
            page_keys = [
                (item["doc_name"], item["evidence_page_num"]) for item in values
            ]
            if len(page_keys) != len(set(page_keys)):
                raise ValueError(f"duplicate ranked document/page for {question_id}")
            rankings[question_id] = values
    if expected_ids is not None and set(rankings) != expected_ids:
        raise ValueError("rankings must contain exactly the expected question IDs")
    return rankings


def gold_for_dataset(dataset: Path, queries: Iterable[DevQuery]) -> dict[str, list[dict[str, Any]]]:
    wanted = {query.financebench_id for query in queries}
    gold: dict[str, list[dict[str, Any]]] = {}
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["financebench_id"] in wanted:
                gold[row["financebench_id"]] = [
                    {"doc_name": item["doc_name"], "evidence_page_num": item["evidence_page_num"]}
                    for item in row["evidence"]
                ]
    if set(gold) != wanted:
        raise ValueError("gold evidence does not cover every frozen-test query")
    return gold


def evaluate_frozen_rankings(
    queries: list[DevQuery],
    rankings: Mapping[str, list[dict[str, Any]]],
    gold: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Calculate headline metrics and supplementary all-pages Hit@5."""
    if (
        not queries
        or any(
            not isinstance(query.financebench_id, str) or not query.financebench_id.strip()
            for query in queries
        )
        or len({query.financebench_id for query in queries}) != len(queries)
    ):
        raise ValueError("queries must contain unique non-empty question IDs")
    expected = {query.financebench_id for query in queries}
    if set(rankings) != expected:
        raise ValueError("rankings must contain exactly one record for every frozen query")
    if set(gold) != expected:
        raise ValueError("gold must contain exactly one record for every frozen query")

    rows: list[dict[str, Any]] = []
    for query in queries:
        results = rankings[query.financebench_id]
        evidence = gold[query.financebench_id]
        top5_pages = {(item["doc_name"], item["evidence_page_num"]) for item in results[:5]}
        gold_pages = {(item["doc_name"], item["evidence_page_num"]) for item in evidence}
        first_rank = next(
            (rank for rank, result in enumerate(results[:10], start=1)
             if is_gold_match(result, evidence)),
            None,
        )
        rows.append({
            "financebench_id": query.financebench_id,
            "rankings": results,
            "gold_pages": sorted(gold_pages),
            "hit@1": int(any(is_gold_match(item, evidence) for item in results[:1])),
            "hit@5": int(any(is_gold_match(item, evidence) for item in results[:5])),
            "all@5": int(gold_pages <= top5_pages),
            "mrr@10": 1.0 / first_rank if first_rank else 0.0,
        })

    headline = evaluate_rankings(
        {row["financebench_id"]: row["rankings"] for row in rows},
        dict(gold),
    )
    metrics: dict[str, float | int] = {
        **headline,
        "all@5": sum(row["all@5"] for row in rows) / len(rows),
        "questions": len(rows),
    }
    return rows, metrics


def write_outputs(
    output_dir: Path,
    config: Mapping[str, Any],
    corpus_manifest: Mapping[str, Any],
    rows: list[dict[str, Any]],
    metrics: Mapping[str, Any],
) -> None:
    """Write immutable-style audit artifacts for one evaluation invocation."""
    config = validate_retrieval_config(config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    recorded_config = {
        "schema_version": 1,
        "git": _git_provenance(),
        "dataset_sha256": DATASET_SHA256,
        "retrieval": dict(config),
    }
    (output_dir / "config.json").write_text(
        json.dumps(recorded_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(dict(corpus_manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_question_rankings.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _manifest_path(root: Path, relative_path: str, label: str) -> Path:
    candidate = (root / relative_path).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        raise ValueError(f"{label} path escapes its manifest root: {relative_path}")
    return candidate


def validate_corpus_manifest(
    manifest: Any,
    manifest_path: Path,
    expected_filings: set[str],
    dataset: Path,
    test_ids_path: Path,
) -> dict[str, Any]:
    """Validate the generated frozen-test corpus manifest and its file hashes."""
    if not isinstance(manifest, dict) or manifest.get("purpose") != "frozen-test-corpus":
        raise ValueError("corpus manifest must identify a frozen-test corpus")
    if manifest.get("dataset", {}).get("sha256") != DATASET_SHA256:
        raise ValueError("corpus manifest has the wrong dataset hash")
    if manifest.get("split", {}).get("ids_sha256") != sha256_of(test_ids_path):
        raise ValueError("corpus manifest has the wrong test-ID hash")
    if set(manifest.get("filings", [])) != expected_filings:
        raise ValueError("corpus manifest filing coverage is inconsistent with frozen evidence")
    roots = manifest.get("roots", {})
    if not isinstance(roots, dict) or not isinstance(roots.get("pdf_dir"), str) or not isinstance(
        roots.get("output_dir"), str
    ) or not roots["pdf_dir"].strip() or not roots["output_dir"].strip():
        raise ValueError("corpus manifest must record PDF and artifact roots")
    pdf_root = Path(roots["pdf_dir"])
    artifact_root = Path(roots["output_dir"])
    if not pdf_root.is_absolute():
        pdf_root = (manifest_path.parent / pdf_root).resolve()
    if not artifact_root.is_absolute():
        artifact_root = (manifest_path.parent / artifact_root).resolve()

    for collection, root, label in (
        (manifest.get("pdfs"), pdf_root, "PDF"),
        (manifest.get("artifacts"), artifact_root, "artifact"),
    ):
        if not isinstance(collection, list) or not collection:
            raise ValueError(f"corpus manifest has no {label} records")
        seen: set[str] = set()
        for item in collection:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError(f"invalid {label} manifest record")
            path = _manifest_path(root, item["path"], label)
            if item["path"] in seen or not path.is_file():
                raise ValueError(f"missing or duplicate {label}: {item['path']}")
            seen.add(item["path"])
            if item.get("bytes") != path.stat().st_size or item.get("sha256") != sha256_of(path):
                raise ValueError(f"{label} hash or size mismatch: {item['path']}")
        if label == "PDF":
            if {Path(path).stem for path in seen} != expected_filings:
                raise ValueError("corpus manifest PDF coverage is inconsistent")
        elif {Path(path).name for path in seen} != {
            "pages.jsonl", "chunks_page.jsonl", "chunks_w300.jsonl", "chunks_w600.jsonl"
        }:
            raise ValueError("corpus manifest artifact coverage is incomplete")
    if sha256_of(dataset) != DATASET_SHA256:
        raise ValueError("dataset hash does not match the committed anchor")
    return manifest


def run_from_files(
    rankings_path: Path,
    config_path: Path,
    corpus_manifest_path: Path,
    output_dir: Path,
    dataset: Path = DEFAULT_DATASET,
    test_ids_path: Path = DEFAULT_TEST_IDS,
) -> dict[str, float | int]:
    _require_clean_worktree()
    queries = load_test_queries(dataset, test_ids_path)
    gold = gold_for_dataset(dataset, queries)
    expected_filings = {
        item["doc_name"] for evidence in gold.values() for item in evidence
    }
    rankings = load_rankings(rankings_path, {query.financebench_id for query in queries})
    config = validate_retrieval_config(json.loads(config_path.read_text(encoding="utf-8")))
    corpus_manifest = validate_corpus_manifest(
        json.loads(corpus_manifest_path.read_text(encoding="utf-8")),
        corpus_manifest_path,
        expected_filings,
        dataset,
        test_ids_path,
    )
    rows, metrics = evaluate_frozen_rankings(queries, rankings, gold)
    write_outputs(output_dir, config, corpus_manifest, rows, metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_from_files(args.rankings, args.config, args.corpus_manifest, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
