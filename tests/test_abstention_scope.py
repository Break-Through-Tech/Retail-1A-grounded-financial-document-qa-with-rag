"""Tests for the company/period scope matcher (``src/abstention/scope.py``).

The matcher is strict on purpose: a missed mention leaves a question unrestricted, but
a false mention would restrict it to the wrong filings and cause a wrong abstention.
The tests pin that asymmetry, using only hand-written strings.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.abstention.scope import chunk_in_scope, named_companies, named_years

COMPANIES = ["3M", "AMD", "American Express", "Block", "JPMorgan", "Johnson & Johnson"]


# --------------------------------------------------------------------------- companies


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What was 3M's capital expenditure in FY2018?", {"3M"}),
        ("How did AMD, Inc. do?", {"AMD"}),
        ("Compare American Express and 3M.", {"American Express", "3M"}),
        ("Did JPMorgan Chase grow?", {"JPMorgan"}),
        ("Revenue of Johnson & Johnson in 2022", {"Johnson & Johnson"}),
        ("What is the capital expenditure?", set()),
    ],
)
def test_named_companies_finds_exact_names(question: str, expected: set[str]) -> None:
    assert named_companies(question, COMPANIES) == frozenset(expected)


def test_company_match_is_case_sensitive() -> None:
    # "block" the noun must not restrict a question to Block the company.
    assert named_companies("What is the largest building block of revenue?", COMPANIES) == frozenset()
    assert named_companies("Block reported higher gross profit.", COMPANIES) == frozenset({"Block"})


@pytest.mark.parametrize("question", ["3MX reported", "The AMDX unit", "Blocks of revenue", "MyBlock inc"])
def test_company_match_is_whole_word(question: str) -> None:
    assert named_companies(question, COMPANIES) == frozenset()


def test_aliases_are_deliberately_not_recognised() -> None:
    assert named_companies("How did JPM and J&J do?", COMPANIES) == frozenset()


def test_blank_and_non_string_company_names_are_ignored() -> None:
    assert named_companies("3M grew", ["", "  ", None, 3, "3M"]) == frozenset({"3M"})  # type: ignore[list-item]


def test_named_companies_requires_a_string_question() -> None:
    with pytest.raises(TypeError, match="question"):
        named_companies(None, COMPANIES)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- years


@pytest.mark.parametrize(
    "question, expected",
    [
        ("FY2018 capital expenditure", {2018}),
        ("revenue in 2022 versus 2021", {2021, 2022}),
        ("Q2 2023 results", {2023}),
        ("fiscal years 2019-2021", {2019, 2021}),
        ("no year here", set()),
        ("the year 2000 and 2029", {2000, 2029}),
    ],
)
def test_named_years(question: str, expected: set[int]) -> None:
    assert named_years(question) == frozenset(expected)


@pytest.mark.parametrize(
    "question", ["12018 units", "$2,018 million", "a margin of 20.23%", "in 1999", "the 2030 plan", "2018.5"]
)
def test_numbers_that_are_not_years_are_ignored(question: str) -> None:
    assert named_years(question) == frozenset()


def test_named_years_requires_a_string_question() -> None:
    with pytest.raises(TypeError, match="question"):
        named_years(2018)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- chunk_in_scope

META = {"company": "3M", "doc_period": 2018}


def test_no_restriction_accepts_anything_even_without_metadata() -> None:
    assert chunk_in_scope({}, frozenset(), frozenset())
    assert chunk_in_scope(META, frozenset(), frozenset())


def test_company_restriction() -> None:
    assert chunk_in_scope(META, frozenset({"3M"}), frozenset())
    assert chunk_in_scope(META, frozenset({"3M", "AMD"}), frozenset())
    assert not chunk_in_scope(META, frozenset({"AMD"}), frozenset())


def test_year_restriction() -> None:
    assert chunk_in_scope(META, frozenset(), frozenset({2018}))
    assert chunk_in_scope(META, frozenset(), frozenset({2017, 2018}))
    assert not chunk_in_scope(META, frozenset(), frozenset({2019}))


def test_both_restrictions_must_hold() -> None:
    assert chunk_in_scope(META, frozenset({"3M"}), frozenset({2018}))
    assert not chunk_in_scope(META, frozenset({"3M"}), frozenset({2019}))
    assert not chunk_in_scope(META, frozenset({"AMD"}), frozenset({2018}))


def test_missing_metadata_fails_a_restriction_it_cannot_verify() -> None:
    assert not chunk_in_scope({"doc_period": 2018}, frozenset({"3M"}), frozenset())
    assert not chunk_in_scope({"company": "3M"}, frozenset(), frozenset({2018}))
    assert not chunk_in_scope({"company": "3M", "doc_period": None}, frozenset(), frozenset({2018}))


@pytest.mark.parametrize("period", ["2018", 2018.0, True])
def test_malformed_period_fails_a_year_restriction(period: object) -> None:
    assert not chunk_in_scope({"company": "3M", "doc_period": period}, frozenset(), frozenset({2018, 1}))


def test_numpy_integer_period_is_accepted() -> None:
    assert chunk_in_scope({"company": "3M", "doc_period": np.int64(2018)}, frozenset(), frozenset({2018}))
