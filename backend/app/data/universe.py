"""Universes.

Phase 1 uses a fixed pair. Phase 2 momentum and Fama–French strategies need a
multi-stock universe. We use a hand-curated list of large caps that were in the
S&P 500 throughout 2014–2024. This minimizes but does not eliminate survivorship
bias — see the README for the full caveat.

Replacing this with point-in-time CRSP-style membership is a Phase 3 deliverable.
"""
from __future__ import annotations


def pairs_universe() -> list[tuple[str, str]]:
    """Canonical mean-reversion pairs to probe in Phase 1."""
    return [("KO", "PEP"), ("XOM", "CVX"), ("GM", "F"), ("MA", "V"), ("HD", "LOW")]


# ~40 large caps that were continuously in the S&P 500 from 2014-01 through 2024-12.
# Skews toward stable mega-caps; this is the survivorship caveat that the README documents.
LARGE_CAP_UNIVERSE: tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "WFC", "GS", "MS", "C",
    "JNJ", "PFE", "MRK", "ABT", "LLY", "UNH",
    "XOM", "CVX", "COP",
    "KO", "PEP", "PG", "WMT", "COST", "MCD", "NKE",
    "HD", "LOW",
    "V", "MA", "AXP",
    "IBM", "INTC", "CSCO", "ORCL", "ADBE", "CRM",
)


def large_cap_universe() -> list[str]:
    return list(LARGE_CAP_UNIVERSE)


def sp500_constituents(as_of: str) -> list[str]:  # noqa: ARG001
    raise NotImplementedError(
        "Point-in-time S&P 500 membership is a Phase 3 deliverable. "
        "Phase 2 uses the fixed `LARGE_CAP_UNIVERSE` instead and documents the bias."
    )
