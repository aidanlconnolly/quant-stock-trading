"""Synthetic fixtures so tests never need network access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _business_index(n: int, start: str = "2014-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


@pytest.fixture
def random_walk_pair() -> pd.DataFrame:
    """Two independent geometric random walks — should fail cointegration.

    Seed picked to give an ADF p-value of ~0.96 on the residuals, well clear of the
    0.05 threshold. Different seeds occasionally produce spurious cointegration —
    that is a known feature of finite-sample ADF, not a strategy bug.
    """
    rng = np.random.default_rng(5)
    n = 1200
    a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"A": a, "B": b}, index=_business_index(n))


@pytest.fixture
def cointegrated_pair() -> pd.DataFrame:
    """log(A) = 0.5 + 1.2 * log(B) + stationary AR(1) noise — strongly cointegrated."""
    rng = np.random.default_rng(7)
    n = 1500
    log_b = np.cumsum(rng.normal(0, 0.01, n)) + np.log(50.0)

    eps = np.zeros(n)
    phi = -0.10  # AR(1) coefficient on Δspread → half-life ≈ -ln(2)/ln(1+phi) ~ 6.6 days
    sigma = 0.02
    for i in range(1, n):
        eps[i] = (1 + phi) * eps[i - 1] + rng.normal(0, sigma)

    log_a = 0.5 + 1.2 * log_b + eps
    a = np.exp(log_a)
    b = np.exp(log_b)
    return pd.DataFrame({"A": a, "B": b}, index=_business_index(n))


@pytest.fixture
def cointegrated_pair_long() -> pd.DataFrame:
    """A longer cointegrated pair (~3500 bars) for walk-forward fold tests."""
    rng = np.random.default_rng(11)
    n = 3500
    log_b = np.cumsum(rng.normal(0, 0.01, n)) + np.log(50.0)
    eps = np.zeros(n)
    phi = -0.08
    sigma = 0.02
    for i in range(1, n):
        eps[i] = (1 + phi) * eps[i - 1] + rng.normal(0, sigma)
    log_a = 0.5 + 1.2 * log_b + eps
    return pd.DataFrame(
        {"A": np.exp(log_a), "B": np.exp(log_b)},
        index=_business_index(n),
    )
