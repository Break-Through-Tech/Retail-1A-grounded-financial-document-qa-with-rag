"""Which companies and fiscal years a question names, and whether a chunk's filing fits.

FinanceBench questions usually name the company and the fiscal year they are about.
A passage from a different company's filing, or from a different year's filing, cannot
support such a question however well its words match, and a relevance score alone does
not catch it: "capital expenditure" reads the same in every year's cash-flow statement.

The matcher is deliberately strict. A missed mention only means the question is not
restricted, which is the safe direction. A false mention would restrict the question
to the wrong filings and cause a wrong abstention. So company names are matched
case-sensitively as whole words (``Block`` the company, never ``block`` the noun), and
an alias such as ``J&J`` is simply not recognised.

On the 111 development questions, 82 name a known company by its exact name and all 82
name the right one and only that one; the other 29 use an alias ("JPM", "MGM") or name
no company, so they are left unrestricted. 98 name a year and 95 of them include the
gold filing's period. The three exceptions name a comparison or guidance year outside
the filing's own period; they are the measured cost of the period rule.
"""

from __future__ import annotations

import numbers
import re
from typing import Any, Iterable, Mapping

__all__ = ["chunk_in_scope", "named_companies", "named_years"]

# 2000-2029 as a standalone four-digit number: "FY2018" and "2018" match; "12018",
# "2,018" and the decimal "2018.5" do not. A sentence-final "2018." still matches.
_YEAR = re.compile(r"(?<![0-9,.])20[0-2][0-9](?![0-9]|\.[0-9])")


def named_companies(question: str, companies: Iterable[str]) -> frozenset[str]:
    """Return the known companies whose exact name appears in ``question``.

    Args:
        question: Question text.
        companies: Known company names, as stored in chunk metadata.

    Returns:
        The companies mentioned. Matching is case-sensitive and whole-word.

    Raises:
        TypeError: If ``question`` is not a ``str``.
    """
    if not isinstance(question, str):
        raise TypeError(f"question must be str, got {type(question).__name__}")
    found = set()
    for company in companies:
        if not isinstance(company, str) or not company.strip():
            continue
        pattern = rf"(?<![A-Za-z0-9]){re.escape(company)}(?![A-Za-z0-9])"
        if re.search(pattern, question):
            found.add(company)
    return frozenset(found)


def named_years(question: str) -> frozenset[int]:
    """Return the four-digit years 2000-2029 that appear in ``question``.

    Raises:
        TypeError: If ``question`` is not a ``str``.
    """
    if not isinstance(question, str):
        raise TypeError(f"question must be str, got {type(question).__name__}")
    return frozenset(int(match) for match in _YEAR.findall(question))


def chunk_in_scope(
    meta: Mapping[str, Any],
    companies: frozenset[str],
    years: frozenset[int],
) -> bool:
    """Whether a chunk's filing matches every restriction the question names.

    A restriction that is empty never excludes anything. A restriction that is set
    excludes any chunk whose metadata is missing or malformed, because the match
    cannot be verified; a corpus without company and period metadata should run with
    the scope check off instead of passing silently.

    Args:
        meta: Chunk metadata with ``company`` and ``doc_period``.
        companies: Companies the question names; empty means no restriction.
        years: Years the question names; empty means no restriction.

    Returns:
        ``True`` when the chunk's company is among ``companies`` (if any) and its
        ``doc_period`` is among ``years`` (if any).
    """
    if companies and meta.get("company") not in companies:
        return False
    if years:
        period = meta.get("doc_period")
        if isinstance(period, bool) or not isinstance(period, numbers.Integral) or period not in years:
            return False
    return True
