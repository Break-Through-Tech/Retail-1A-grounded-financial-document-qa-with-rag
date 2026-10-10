"""Tests for the development query loader (issue #9).

The load-bearing test is :func:`test_loader_never_opens_the_frozen_test_id_file`:
the development query set must be built without touching the frozen split, and the
test proves its own guard is live before trusting a pass.
"""

from __future__ import annotations

import builtins
import json
import os
import shutil
from dataclasses import fields
from pathlib import Path

import pytest

from src.data.split import load_rows
from src.retrieval.queries import DevQuery, load_dev_queries

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "financebench_merged.jsonl"
SPLITS = ROOT / "data" / "splits"
DEV_IDS = SPLITS / "dev_ids.txt"
METADATA = SPLITS / "split_metadata.json"

#: The frozen split's ID file. Named only so the guard below can refuse to open it.
FROZEN_ID_FILE = (SPLITS / "test_ids.txt").resolve()


@pytest.fixture(scope="module")
def dev_id_lines() -> list[str]:
    """The committed development IDs, in file order."""
    return DEV_IDS.read_text(encoding="utf-8").splitlines()


@pytest.fixture(scope="module")
def queries() -> list[DevQuery]:
    """The development query set loaded from the committed split."""
    return load_dev_queries(DATASET, DEV_IDS)


def _write_split_dir(tmp_path: Path, id_lines: list[str]) -> Path:
    """Write a dev ID file plus a copy of the real split metadata; return the ID path."""
    shutil.copy(METADATA, tmp_path / "split_metadata.json")
    path = tmp_path / "dev_ids.txt"
    path.write_text("\n".join(id_lines) + "\n", encoding="utf-8")
    return path


def test_dev_queries_match_dev_ids_exactly(queries: list[DevQuery], dev_id_lines: list[str]) -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert len(queries) == 111 == metadata["sizes"]["dev"]
    assert [q.financebench_id for q in queries] == sorted(dev_id_lines)


def test_query_text_is_question_verbatim(queries: list[DevQuery], dev_id_lines: list[str]) -> None:
    dev_ids = set(dev_id_lines)
    questions = {
        row["financebench_id"]: row["question"]
        for row in load_rows(DATASET)
        if row["financebench_id"] in dev_ids
    }
    assert {q.financebench_id: q.question for q in queries} == questions


def test_dev_query_has_only_id_and_question() -> None:
    assert [f.name for f in fields(DevQuery)] == ["financebench_id", "question"]


def test_loader_is_sorted_and_deterministic(queries: list[DevQuery]) -> None:
    ids = [q.financebench_id for q in queries]
    assert ids == sorted(ids)
    assert load_dev_queries(DATASET, DEV_IDS) == queries


def test_loader_never_opens_the_frozen_test_id_file(monkeypatch: pytest.MonkeyPatch) -> None:
    real_path_open = Path.open
    real_builtin_open = builtins.open

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

    assert len(load_dev_queries(DATASET, DEV_IDS)) == 111
    # The guard must be live, or the pass above proves nothing.
    with pytest.raises(AssertionError, match="frozen test ID file"):
        FROZEN_ID_FILE.read_text(encoding="utf-8")


def test_loader_rejects_a_repeated_id(tmp_path: Path, dev_id_lines: list[str]) -> None:
    path = _write_split_dir(tmp_path, dev_id_lines[:-1] + [dev_id_lines[0]])
    with pytest.raises(ValueError, match="more than once"):
        load_dev_queries(DATASET, path)


def test_loader_rejects_a_count_that_disagrees_with_metadata(
    tmp_path: Path, dev_id_lines: list[str]
) -> None:
    path = _write_split_dir(tmp_path, dev_id_lines[:-1])
    with pytest.raises(ValueError, match="records 111"):
        load_dev_queries(DATASET, path)


def test_loader_rejects_an_id_absent_from_the_dataset(
    tmp_path: Path, dev_id_lines: list[str]
) -> None:
    path = _write_split_dir(tmp_path, dev_id_lines[:-1] + ["not-a-real-id"])
    with pytest.raises(ValueError, match="absent"):
        load_dev_queries(DATASET, path)
