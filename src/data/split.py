"""Deterministic, group-aware dev/test split for the FinanceBench public sample.

Issue #6 of the Retail-1A challenge project. Produces a frozen split in which no
source filing (``doc_name``) appears on both sides, stratified on ``question_type``
and ``doc_type``.

Design notes that the final report depends on:

* The grouping unit is ``doc_name``, per the Challenge Advisor's instruction that
  questions about the same filing must not appear in both development and test
  sets. Every ``doc_name`` group is atomic, so filing-level isolation holds by
  construction rather than by assertion.
* Company-level overlap is *measured and reported*, never used as a grouping
  constraint. 135 of the 150 questions belong to a company that owns more than one
  document, so a company-clean split under ``doc_name`` grouping does not exist at
  any usable test size.
* The seed only decides the order groups are visited. Quotas, not the seed, decide
  balance. No search over seeds is performed, because selecting a seed by how good
  its output looks is a form of tuning on the split itself and is not defensible in
  the final report.
* Nothing written by this module contains a timestamp, so two runs with the same
  seed produce byte-identical files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ALGORITHM",
    "ALGORITHM_VERSION",
    "DATASET_SHA256",
    "DEFAULT_DATASET",
    "DEFAULT_OUT_DIR",
    "DEFAULT_SEED",
    "DEFAULT_TARGET_TEST_SIZE",
    "EXPECTED_ROWS",
    "MIN_TEST_SIZE",
    "STRATA",
    "SplitConstraintError",
    "SplitResult",
    "build_groups",
    "company_overlap_diagnostic",
    "largest_remainder_quota",
    "load_rows",
    "make_split",
    "sha256_of",
    "write_split",
]

DEFAULT_DATASET = Path("data/financebench_merged.jsonl")
DEFAULT_OUT_DIR = Path("data/splits")

#: Published in data/README.md. The reproducibility anchor for every reported number.
DATASET_SHA256 = "7a1c81789e0fd2f1c37057a7ec0097756d726b05e7228e68e57db8e18c54fd0b"
EXPECTED_ROWS = 150

#: Hard floor from the Challenge Advisor's brief: at least 30 of 150 examples frozen.
MIN_TEST_SIZE = 30
DEFAULT_TARGET_TEST_SIZE = 40

#: Arbitrary fixed seed. It orders group visits and nothing else; see module docstring.
DEFAULT_SEED = 42

GROUPING_KEY = "doc_name"
STRATA: tuple[str, ...] = ("question_type", "doc_type")

ALGORITHM = "quota-constrained-greedy-group-split"
ALGORITHM_VERSION = "1.0.0"

#: How far a pass may exceed a stratum quota. Escalated only when the previous
#: tolerance failed to satisfy a hard invariant, and the value used is recorded.
TOLERANCE_LADDER: tuple[int, ...] = (0, 1, 2, 3)


class SplitConstraintError(RuntimeError):
    """Raised when no tolerance on the ladder satisfies every hard invariant."""


@dataclass(frozen=True)
class SplitResult:
    """The outcome of one split run.

    Attributes:
        dev_ids: Sorted ``financebench_id`` values assigned to development.
        test_ids: Sorted ``financebench_id`` values assigned to the frozen test set.
        dev_groups: Sorted ``doc_name`` values on the development side.
        test_groups: Sorted ``doc_name`` values on the test side.
        metadata: Fully reproducible description of the run, safe to serialise.
    """

    dev_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    dev_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of ``path``.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path = DEFAULT_DATASET, *, verify_sha256: bool = True) -> list[dict[str, Any]]:
    """Load the FinanceBench JSONL sample.

    Args:
        path: Path to ``financebench_merged.jsonl``.
        verify_sha256: When True, refuse to proceed unless the file hash matches
            :data:`DATASET_SHA256`.

    Returns:
        One dict per line, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the hash does not match, the row count is not
            :data:`EXPECTED_ROWS`, or ``financebench_id`` values are not unique.
    """
    if verify_sha256:
        actual = sha256_of(path)
        if actual != DATASET_SHA256:
            raise ValueError(
                f"dataset hash mismatch for {path}: expected {DATASET_SHA256}, got {actual}. "
                "The dataset is read-only and is the anchor for every reported number."
            )
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows in {path}, found {len(rows)}")
    ids = [row["financebench_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("financebench_id values are not unique; they cannot key a split")
    return rows


def build_groups(
    rows: Sequence[Mapping[str, Any]], grouping_key: str = GROUPING_KEY
) -> dict[str, list[Mapping[str, Any]]]:
    """Group rows by ``grouping_key``.

    Args:
        rows: FinanceBench rows.
        grouping_key: Row field whose value defines an atomic, unsplittable group.

    Returns:
        Mapping of group value to its rows, keys in sorted order.

    Raises:
        KeyError: If a row is missing ``grouping_key``.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[grouping_key], []).append(row)
    return {key: groups[key] for key in sorted(groups)}


def largest_remainder_quota(counts: Mapping[str, int], total: int, target: int) -> dict[str, int]:
    """Apportion ``target`` items across strata values in proportion to ``counts``.

    Uses the largest-remainder method so the quotas sum exactly to ``target``. Ties
    on the remainder are broken by sorted value name, keeping the result
    deterministic.

    Args:
        counts: Population count per stratum value.
        total: Sum of ``counts``.
        target: Number of items to apportion.

    Returns:
        Quota per stratum value, summing to ``target``.

    Raises:
        ValueError: If ``total`` is not positive or does not equal the sum of counts.
    """
    if total <= 0 or total != sum(counts.values()):
        raise ValueError(f"total must be the positive sum of counts; got total={total}")
    exact = {value: counts[value] * target / total for value in sorted(counts)}
    quota = {value: int(amount) for value, amount in exact.items()}
    shortfall = target - sum(quota.values())
    ranked = sorted(exact, key=lambda value: (-(exact[value] - quota[value]), value))
    for value in ranked[:shortfall]:
        quota[value] += 1
    return quota


def _distribution(rows: Iterable[Mapping[str, Any]], field_name: str) -> dict[str, int]:
    """Return a sorted count of ``field_name`` values across ``rows``."""
    counter = Counter(row[field_name] for row in rows)
    return {value: counter[value] for value in sorted(counter)}


def _group_fits(
    group_counts: Mapping[str, Mapping[str, int]],
    running: Mapping[str, Counter],
    quotas: Mapping[str, Mapping[str, int]],
    tolerance: int,
) -> bool:
    """Return True if adding a group keeps every stratum within quota + tolerance."""
    for stratum, per_value in group_counts.items():
        for value, count in per_value.items():
            if running[stratum][value] + count > quotas[stratum].get(value, 0) + tolerance:
                return False
    return True


def company_overlap_diagnostic(
    rows: Sequence[Mapping[str, Any]], dev_ids: Sequence[str], test_ids: Sequence[str]
) -> dict[str, Any]:
    """Measure company-level overlap between the two sides.

    Reported, never enforced. Under ``doc_name`` grouping a company that owns more
    than one filing can legitimately appear on both sides; this quantifies how much
    of that happens so the limitation is documented rather than silent.

    Args:
        rows: FinanceBench rows.
        dev_ids: Development ``financebench_id`` values.
        test_ids: Test ``financebench_id`` values.

    Returns:
        Counts of companies per side, the overlapping companies, and the structural
        reason (how many questions belong to a multi-document company).
    """
    by_id = {row["financebench_id"]: row for row in rows}
    dev_companies = {by_id[i]["company"] for i in dev_ids}
    test_companies = {by_id[i]["company"] for i in test_ids}
    overlap = sorted(dev_companies & test_companies)
    docs_per_company: dict[str, set[str]] = {}
    for row in rows:
        docs_per_company.setdefault(row["company"], set()).add(row[GROUPING_KEY])
    multi_doc = {c for c, docs in docs_per_company.items() if len(docs) > 1}
    return {
        "note": "diagnostic only; company is not a grouping constraint",
        "companies_dev": len(dev_companies),
        "companies_test": len(test_companies),
        "companies_in_both": len(overlap),
        "overlapping_companies": overlap,
        "companies_owning_multiple_documents": len(multi_doc),
        "companies_total": len(docs_per_company),
        "questions_from_multi_document_companies": sum(
            1 for row in rows if row["company"] in multi_doc
        ),
    }


def make_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    target_test_size: int = DEFAULT_TARGET_TEST_SIZE,
    min_test_size: int = MIN_TEST_SIZE,
    grouping_key: str = GROUPING_KEY,
    strata: Sequence[str] = STRATA,
    dataset_sha256: str = DATASET_SHA256,
) -> SplitResult:
    """Build a frozen dev/test split grouped by ``grouping_key``.

    Groups are visited in a seeded shuffle order and accepted into test while every
    stratum stays within its quota. Tolerance is escalated along
    :data:`TOLERANCE_LADDER` only when a hard invariant is unmet, and the tolerance
    actually used is recorded in the metadata.

    Args:
        rows: FinanceBench rows.
        seed: Seeds the group visit order only.
        target_test_size: Desired number of test questions.
        min_test_size: Hard floor on test questions.
        grouping_key: Field defining atomic groups.
        strata: Fields whose distribution is held near the population share.
        dataset_sha256: Hash recorded in the metadata.

    Returns:
        A :class:`SplitResult` whose metadata records quotas, distributions, the
        satisfied invariants, and the company-overlap diagnostic.

    Raises:
        ValueError: If ``target_test_size`` is below ``min_test_size`` or exceeds the
            number of rows.
        SplitConstraintError: If no tolerance on the ladder satisfies every hard
            invariant.
    """
    total = len(rows)
    if not min_test_size <= target_test_size <= total:
        raise ValueError(
            f"target_test_size must lie in [{min_test_size}, {total}]; got {target_test_size}"
        )

    groups = build_groups(rows, grouping_key)
    quotas = {
        stratum: largest_remainder_quota(_distribution(rows, stratum), total, target_test_size)
        for stratum in strata
    }
    group_counts = {
        name: {stratum: _distribution(members, stratum) for stratum in strata}
        for name, members in groups.items()
    }

    order = sorted(groups)
    random.Random(seed).shuffle(order)

    failures: list[str] = []
    for tolerance in TOLERANCE_LADDER:
        running = {stratum: Counter() for stratum in strata}
        test_groups: list[str] = []
        test_size = 0
        for name in order:
            if test_size >= target_test_size:
                break
            if _group_fits(group_counts[name], running, quotas, tolerance):
                test_groups.append(name)
                test_size += len(groups[name])
                for stratum in strata:
                    running[stratum].update(group_counts[name][stratum])

        test_set = set(test_groups)
        dev_groups = [name for name in sorted(groups) if name not in test_set]
        test_rows = [row for name in test_groups for row in groups[name]]
        dev_rows = [row for name in dev_groups for row in groups[name]]

        problems: list[str] = []
        if len(test_rows) < min_test_size:
            problems.append(f"test size {len(test_rows)} below floor {min_test_size}")
        for stratum in strata:
            missing_test = set(_distribution(rows, stratum)) - set(_distribution(test_rows, stratum))
            missing_dev = set(_distribution(rows, stratum)) - set(_distribution(dev_rows, stratum))
            if missing_test:
                problems.append(f"{stratum} values absent from test: {sorted(missing_test)}")
            if missing_dev:
                problems.append(f"{stratum} values absent from dev: {sorted(missing_dev)}")
        if problems:
            failures.append(f"tolerance={tolerance}: " + "; ".join(problems))
            continue

        dev_ids = tuple(sorted(row["financebench_id"] for row in dev_rows))
        test_ids = tuple(sorted(row["financebench_id"] for row in test_rows))
        overlap = set(dev_groups) & test_set
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "algorithm": ALGORITHM,
            "algorithm_version": ALGORITHM_VERSION,
            "issue": "#6 Create Training and Testing Sets",
            "split_kind": "two-way: development and frozen test (no separate train set; "
            "the September baseline is a keyword retriever and fits no parameters)",
            "dataset": {
                "path": str(DEFAULT_DATASET),
                "rows": total,
                "sha256": dataset_sha256,
            },
            "grouping_key": grouping_key,
            "strata": list(strata),
            "seed": seed,
            "target_test_size": target_test_size,
            "min_test_size": min_test_size,
            "tolerance_used": tolerance,
            "tolerance_escalations": failures,
            "quotas": quotas,
            "sizes": {"dev": len(dev_rows), "test": len(test_rows), "total": total},
            "groups": {
                "dev": len(dev_groups),
                "test": len(test_groups),
                "total": len(groups),
            },
            "distributions": {
                stratum: {
                    "population": _distribution(rows, stratum),
                    "dev": _distribution(dev_rows, stratum),
                    "test": _distribution(test_rows, stratum),
                }
                for stratum in (*strata, "gics_sector")
            },
            "invariants": {
                "ids_partition_dataset": sorted(dev_ids + test_ids)
                == sorted(row["financebench_id"] for row in rows),
                "ids_disjoint": not set(dev_ids) & set(test_ids),
                "grouping_key_overlap": len(overlap),
                "test_size_at_or_above_floor": len(test_rows) >= min_test_size,
                "all_stratum_values_present_both_sides": True,
            },
            "diagnostics": {"company_overlap": company_overlap_diagnostic(rows, dev_ids, test_ids)},
        }
        return SplitResult(
            dev_ids=dev_ids,
            test_ids=test_ids,
            dev_groups=tuple(dev_groups),
            test_groups=tuple(sorted(test_groups)),
            metadata=metadata,
        )

    raise SplitConstraintError(
        "no tolerance on the ladder satisfied every hard invariant:\n  " + "\n  ".join(failures)
    )


def write_split(result: SplitResult, out_dir: Path = DEFAULT_OUT_DIR) -> list[Path]:
    """Write the split artifacts to ``out_dir``.

    Output contains no timestamps, so repeated runs with the same seed produce
    byte-identical files.

    Args:
        result: Split to serialise.
        out_dir: Directory to create and write into.

    Returns:
        The paths written, in a stable order.

    Raises:
        OSError: If ``out_dir`` cannot be created or written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_path = out_dir / "dev_ids.txt"
    test_path = out_dir / "test_ids.txt"
    meta_path = out_dir / "split_metadata.json"
    dev_path.write_text("\n".join(result.dev_ids) + "\n", encoding="utf-8")
    test_path.write_text("\n".join(result.test_ids) + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return [dev_path, test_path, meta_path]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code, 0 on success.

    Raises:
        SplitConstraintError: Propagated when the split cannot satisfy its invariants.
    """
    parser = argparse.ArgumentParser(description="Create the frozen FinanceBench dev/test split.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-test-size", type=int, default=DEFAULT_TARGET_TEST_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    args = parser.parse_args(argv)

    rows = load_rows(args.dataset)
    result = make_split(rows, seed=args.seed, target_test_size=args.target_test_size)
    if not args.dry_run:
        for path in write_split(result, args.out_dir):
            print(f"wrote {path}")
    print(json.dumps(result.metadata, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
