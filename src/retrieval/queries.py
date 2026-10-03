"""Development query set for the Issue #9 BM25 baseline.

A query is a FinanceBench row's ``question`` field, verbatim, and nothing else: never
the answer, the justification, or any evidence text.

Only ``dev_ids.txt`` and the split metadata are read. The frozen test ID file is
never opened, so no frozen-test question can enter a development query set through
this module. The dataset itself is loaded through :func:`src.data.split.load_rows`,
which refuses a file whose SHA-256 differs from the published value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.data.split import DEFAULT_DATASET, DEFAULT_OUT_DIR, load_rows

__all__ = ["DEFAULT_DEV_IDS", "DevQuery", "load_dev_queries"]

DEFAULT_DEV_IDS = DEFAULT_OUT_DIR / "dev_ids.txt"


@dataclass(frozen=True)
class DevQuery:
    """One development-set retrieval query.

    Attributes:
        financebench_id: The row's ``financebench_id``.
        question: The row's ``question`` field, verbatim.
    """

    financebench_id: str
    question: str


def load_dev_queries(
    dataset: Path = DEFAULT_DATASET, dev_ids_path: Path = DEFAULT_DEV_IDS
) -> list[DevQuery]:
    """Load the development questions named in ``dev_ids_path``.

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
    dev_ids = [line for line in dev_ids_path.read_text(encoding="utf-8").splitlines() if line]
    if len(set(dev_ids)) != len(dev_ids):
        raise ValueError(f"{dev_ids_path} lists a financebench_id more than once")

    metadata_path = dev_ids_path.with_name("split_metadata.json")
    expected = json.loads(metadata_path.read_text(encoding="utf-8"))["sizes"]["dev"]
    if len(dev_ids) != expected:
        raise ValueError(
            f"{dev_ids_path} lists {len(dev_ids)} IDs but {metadata_path} records {expected}"
        )

    by_id = {row["financebench_id"]: row for row in load_rows(dataset)}
    missing = sorted(set(dev_ids) - set(by_id))
    if missing:
        raise ValueError(f"development IDs absent from {dataset}: {missing}")

    queries = []
    for financebench_id in sorted(dev_ids):
        question = by_id[financebench_id]["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{financebench_id} has an empty question")
        queries.append(DevQuery(financebench_id=financebench_id, question=question))
    return queries
