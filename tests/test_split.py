"""Invariant tests for the frozen FinanceBench dev/test split (issue #6).

These tests exist to make the split's guarantees mechanical rather than reviewed by
eye. The load-bearing one is :func:`test_no_source_filing_appears_on_both_sides`:
filing-level isolation is the property the Challenge Advisor's brief requires, and
the property every acceptance number later depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.split import (
    DATASET_SHA256,
    DEFAULT_SEED,
    DEFAULT_TARGET_TEST_SIZE,
    EXPECTED_ROWS,
    MIN_TEST_SIZE,
    STRATA,
    build_groups,
    company_overlap_diagnostic,
    largest_remainder_quota,
    load_rows,
    make_split,
    sha256_of,
    write_split,
)

DATASET = Path(__file__).resolve().parents[1] / "data" / "financebench_merged.jsonl"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    """The 150 FinanceBench rows, hash-verified."""
    return load_rows(DATASET)


@pytest.fixture(scope="module")
def split(rows: list[dict]):
    """The split produced by the committed operating configuration."""
    return make_split(rows, seed=DEFAULT_SEED, target_test_size=DEFAULT_TARGET_TEST_SIZE)


# --- dataset anchor -------------------------------------------------------------


def test_dataset_hash_matches_published_value() -> None:
    assert sha256_of(DATASET) == DATASET_SHA256


def test_dataset_has_expected_row_count_and_unique_ids(rows: list[dict]) -> None:
    assert len(rows) == EXPECTED_ROWS
    ids = [row["financebench_id"] for row in rows]
    assert len(set(ids)) == EXPECTED_ROWS


def test_load_rows_rejects_a_modified_dataset(tmp_path: Path) -> None:
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text(DATASET.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_rows(tampered)


# --- hard invariants ------------------------------------------------------------


def test_ids_partition_the_dataset(rows: list[dict], split) -> None:
    assert sorted(split.dev_ids + split.test_ids) == sorted(
        row["financebench_id"] for row in rows
    )
    assert not set(split.dev_ids) & set(split.test_ids)


def test_test_set_meets_the_advisor_floor(split) -> None:
    assert len(split.test_ids) >= MIN_TEST_SIZE


def test_no_source_filing_appears_on_both_sides(split) -> None:
    assert not set(split.dev_groups) & set(split.test_groups)
    assert split.metadata["invariants"]["grouping_key_overlap"] == 0


def test_groups_are_atomic(rows: list[dict], split) -> None:
    """Every row of a doc_name group lands on the same side."""
    groups = build_groups(rows)
    test_ids = set(split.test_ids)
    for name, members in groups.items():
        sides = {row["financebench_id"] in test_ids for row in members}
        assert len(sides) == 1, f"group {name} was split across sides"
    assert len(groups) == len(split.dev_groups) + len(split.test_groups)


@pytest.mark.parametrize("stratum", STRATA)
def test_every_stratum_value_is_present_on_both_sides(rows: list[dict], split, stratum: str) -> None:
    by_id = {row["financebench_id"]: row for row in rows}
    population = {row[stratum] for row in rows}
    dev = {by_id[i][stratum] for i in split.dev_ids}
    test = {by_id[i][stratum] for i in split.test_ids}
    assert population == dev, f"{stratum} values missing from dev: {sorted(population - dev)}"
    assert population == test, f"{stratum} values missing from test: {sorted(population - test)}"


# --- reproducibility ------------------------------------------------------------


def test_same_seed_reproduces_the_same_split(rows: list[dict]) -> None:
    first = make_split(rows, seed=DEFAULT_SEED, target_test_size=DEFAULT_TARGET_TEST_SIZE)
    second = make_split(rows, seed=DEFAULT_SEED, target_test_size=DEFAULT_TARGET_TEST_SIZE)
    assert first.dev_ids == second.dev_ids
    assert first.test_ids == second.test_ids
    assert first.metadata == second.metadata


def test_seed_actually_changes_the_split(rows: list[dict]) -> None:
    """Guards against a seed parameter that is accepted and then ignored."""
    a = make_split(rows, seed=DEFAULT_SEED, target_test_size=DEFAULT_TARGET_TEST_SIZE)
    b = make_split(rows, seed=DEFAULT_SEED + 1, target_test_size=DEFAULT_TARGET_TEST_SIZE)
    assert a.test_ids != b.test_ids


def test_written_files_are_byte_identical_across_runs(rows: list[dict], tmp_path: Path) -> None:
    result = make_split(rows, seed=DEFAULT_SEED, target_test_size=DEFAULT_TARGET_TEST_SIZE)
    first = {p.name: p.read_bytes() for p in write_split(result, tmp_path / "a")}
    second = {p.name: p.read_bytes() for p in write_split(result, tmp_path / "b")}
    assert first == second
    assert set(first) == {"dev_ids.txt", "test_ids.txt", "split_metadata.json"}


def test_written_ids_are_sorted_and_newline_terminated(rows: list[dict], tmp_path: Path) -> None:
    result = make_split(rows, seed=DEFAULT_SEED, target_test_size=DEFAULT_TARGET_TEST_SIZE)
    paths = {p.name: p for p in write_split(result, tmp_path)}
    for name in ("dev_ids.txt", "test_ids.txt"):
        text = paths[name].read_text(encoding="utf-8")
        assert text.endswith("\n")
        lines = text.splitlines()
        assert lines == sorted(lines)
        assert len(lines) == len(set(lines))


def test_metadata_counts_agree_with_the_id_files(rows: list[dict], tmp_path: Path) -> None:
    result = make_split(rows, seed=DEFAULT_SEED, target_test_size=DEFAULT_TARGET_TEST_SIZE)
    paths = {p.name: p for p in write_split(result, tmp_path)}
    meta = json.loads(paths["split_metadata.json"].read_text(encoding="utf-8"))
    dev_lines = paths["dev_ids.txt"].read_text(encoding="utf-8").splitlines()
    test_lines = paths["test_ids.txt"].read_text(encoding="utf-8").splitlines()
    assert meta["sizes"]["dev"] == len(dev_lines)
    assert meta["sizes"]["test"] == len(test_lines)
    assert meta["sizes"]["total"] == len(dev_lines) + len(test_lines) == EXPECTED_ROWS
    assert meta["dataset"]["sha256"] == DATASET_SHA256


def test_metadata_contains_no_timestamp(rows: list[dict], split) -> None:
    """A timestamp would break the byte-identical determinism guarantee."""
    blob = json.dumps(split.metadata).lower()
    for forbidden in ("timestamp", "created_at", "generated_at", "datetime"):
        assert forbidden not in blob


# --- apportionment --------------------------------------------------------------


def test_quotas_sum_to_the_target(rows: list[dict], split) -> None:
    for stratum, quota in split.metadata["quotas"].items():
        assert sum(quota.values()) == DEFAULT_TARGET_TEST_SIZE, stratum


def test_largest_remainder_is_exact_and_deterministic() -> None:
    counts = {"a": 50, "b": 50, "c": 50}
    quota = largest_remainder_quota(counts, 150, 40)
    assert sum(quota.values()) == 40
    assert quota == largest_remainder_quota(counts, 150, 40)
    assert max(quota.values()) - min(quota.values()) <= 1


def test_largest_remainder_rejects_inconsistent_totals() -> None:
    with pytest.raises(ValueError):
        largest_remainder_quota({"a": 1, "b": 2}, 5, 2)


# --- configuration guards -------------------------------------------------------


def test_target_below_the_floor_is_rejected(rows: list[dict]) -> None:
    with pytest.raises(ValueError):
        make_split(rows, seed=DEFAULT_SEED, target_test_size=MIN_TEST_SIZE - 1)


def test_target_above_the_dataset_is_rejected(rows: list[dict]) -> None:
    with pytest.raises(ValueError):
        make_split(rows, seed=DEFAULT_SEED, target_test_size=EXPECTED_ROWS + 1)


# --- diagnostic, not a constraint -----------------------------------------------


def test_company_overlap_is_reported_not_enforced(rows: list[dict], split) -> None:
    diag = company_overlap_diagnostic(rows, split.dev_ids, split.test_ids)
    assert diag["companies_total"] == 32
    assert diag["companies_owning_multiple_documents"] == 26
    assert diag["questions_from_multi_document_companies"] == 135
    # Overlap is expected under doc_name grouping. Asserting it is zero would be
    # asserting a property this split deliberately does not promise.
    assert diag["companies_in_both"] >= 0
    assert diag["companies_in_both"] == split.metadata["diagnostics"]["company_overlap"][
        "companies_in_both"
    ]
