"""Validated development and frozen-test query sets.

A query is a FinanceBench row's ``question`` field, verbatim, and nothing else: never
the answer, the justification, or any evidence text.

The development loader continues to read only ``dev_ids.txt``. The test loader is
explicit and reads only ``test_ids.txt`` plus the split metadata. Both loaders use
:func:`src.data.split.load_rows`, which refuses a dataset whose SHA-256 differs from
the published value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.data.split import (
    DATASET_SHA256,
    DEFAULT_DATASET,
    DEFAULT_OUT_DIR,
    EXPECTED_ROWS,
    load_rows,
)

__all__ = [
    "DEFAULT_DEV_IDS",
    "DEFAULT_TEST_IDS",
    "EXPECTED_TEST_QUERIES",
    "DevQuery",
    "load_dev_queries",
    "load_test_queries",
    "validate_frozen_split",
]

DEFAULT_DEV_IDS = DEFAULT_OUT_DIR / "dev_ids.txt"
DEFAULT_TEST_IDS = DEFAULT_OUT_DIR / "test_ids.txt"
EXPECTED_TEST_QUERIES = 39


@dataclass(frozen=True)
class DevQuery:
    """One development-set retrieval query.

    Attributes:
        financebench_id: The row's ``financebench_id``.
        question: The row's ``question`` field, verbatim.
    """

    financebench_id: str
    question: str


def _load_split_queries(
    dataset: Path,
    ids_path: Path,
    split_name: str,
    expected_count: int,
    other_ids_path: Path | None,
) -> list[DevQuery]:
    """Load and validate one side of the committed split."""
    ids = [line for line in ids_path.read_text(encoding="utf-8").splitlines() if line]
    metadata_path = ids_path.with_name("split_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded = metadata["sizes"][split_name]
    if len(ids) != expected_count:
        if split_name == "dev":
            raise ValueError(
                f"{ids_path} lists {len(ids)} IDs but {metadata_path} records {recorded}"
            )
        raise ValueError(
            f"{ids_path} lists {len(ids)} IDs; expected exactly {expected_count}"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"{ids_path} lists a financebench_id more than once")

    if recorded != expected_count:
        raise ValueError(
            f"{metadata_path} records {recorded} {split_name} IDs; expected {expected_count}"
        )

    if other_ids_path is not None:
        other_ids = set(other_ids_path.read_text(encoding="utf-8").split())
        overlap = sorted(set(ids) & other_ids)
        if overlap:
            raise ValueError(f"{ids_path} overlaps the other split: {overlap}")

    by_id = {row["financebench_id"]: row for row in load_rows(dataset)}
    missing = sorted(set(ids) - set(by_id))
    if missing:
        raise ValueError(f"{split_name} IDs absent from {dataset}: {missing}")

    queries = []
    for financebench_id in sorted(ids):
        question = by_id[financebench_id]["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{financebench_id} has an empty question")
        queries.append(DevQuery(financebench_id=financebench_id, question=question))
    return queries


def load_dev_queries(
    dataset: Path = DEFAULT_DATASET, dev_ids_path: Path = DEFAULT_DEV_IDS
) -> list[DevQuery]:
    """Load exactly the committed development questions.

    The ID count is checked against ``split_metadata.json`` in the same directory,
    so a truncated or edited ID file fails loudly instead of shrinking the query set.

    Args:
        dataset: Path to ``financebench_merged.jsonl``.
        dev_ids_path: Path to ``dev_ids.txt``.

    Returns:
        One :class:`DevQuery` per development ID, sorted by ``financebench_id``.

    Raises:
        FileNotFoundError: If the dataset, the ID file or the split metadata is
            missing.
        ValueError: If the dataset fails :func:`src.data.split.load_rows`; if the ID
            file repeats an ID or its count differs from the metadata's dev size; if
            an ID is absent from the dataset; or if a question is empty.
    """
    expected = json.loads(
        dev_ids_path.with_name("split_metadata.json").read_text(encoding="utf-8")
    )["sizes"]["dev"]
    return _load_split_queries(
        dataset, dev_ids_path, "dev", expected, None
    )


def load_test_queries(
    dataset: Path = DEFAULT_DATASET, test_ids_path: Path = DEFAULT_TEST_IDS
) -> list[DevQuery]:
    """Load exactly the 39 committed frozen-test questions.

    The loader validates uniqueness, the split metadata count, dataset membership,
    and disjointness from the development IDs. It does not load or modify any
    document or chunk artifact.
    """
    queries = _load_split_queries(
        dataset,
        test_ids_path,
        "test",
        EXPECTED_TEST_QUERIES,
        test_ids_path.with_name("dev_ids.txt"),
    )
    metadata = json.loads(
        test_ids_path.with_name("split_metadata.json").read_text(encoding="utf-8")
    )
    validate_frozen_split(
        load_rows(dataset),
        metadata,
        set(test_ids_path.read_text(encoding="utf-8").split()),
        set(test_ids_path.with_name("dev_ids.txt").read_text(encoding="utf-8").split()),
    )
    return queries


def validate_frozen_split(
    rows: list[dict],
    metadata: dict,
    test_ids: set[str],
    dev_ids: set[str],
) -> None:
    """Validate the committed dataset partition and document-group isolation."""
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} dataset rows, found {len(rows)}")
    row_ids = {row["financebench_id"] for row in rows}
    if len(row_ids) != EXPECTED_ROWS:
        raise ValueError("dataset financebench_id values are not unique")
    if len(test_ids) != EXPECTED_TEST_QUERIES or len(dev_ids) != metadata["sizes"]["dev"]:
        raise ValueError("frozen split ID counts do not match split metadata")
    if test_ids & dev_ids:
        raise ValueError("development and frozen-test IDs overlap")
    if test_ids | dev_ids != row_ids:
        raise ValueError("development and frozen-test IDs do not partition the dataset")
    if metadata.get("dataset", {}).get("sha256") != DATASET_SHA256:
        raise ValueError("split metadata dataset hash does not match the committed anchor")
    if metadata.get("dataset", {}).get("rows") != EXPECTED_ROWS:
        raise ValueError("split metadata dataset row count is inconsistent")
    if metadata.get("grouping_key") != "doc_name":
        raise ValueError("split metadata grouping key is not doc_name")
    if metadata.get("sizes") != {"dev": len(dev_ids), "test": len(test_ids), "total": EXPECTED_ROWS}:
        raise ValueError("split metadata sizes are inconsistent")

    by_id = {row["financebench_id"]: row for row in rows}
    dev_docs = {by_id[question_id]["doc_name"] for question_id in dev_ids}
    test_docs = {by_id[question_id]["doc_name"] for question_id in test_ids}
    if dev_docs & test_docs:
        raise ValueError("development and frozen-test document groups overlap")

    invariants = metadata.get("invariants", {})
    required_invariants = {
        "ids_partition_dataset": True,
        "ids_disjoint": True,
        "grouping_key_overlap": 0,
        "test_size_at_or_above_floor": True,
        "all_stratum_values_present_both_sides": True,
    }
    for key, expected in required_invariants.items():
        if invariants.get(key) != expected:
            raise ValueError(f"split metadata invariant {key!r} is not {expected!r}")
