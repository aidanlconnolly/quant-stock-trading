"""Fama-French tilt: regression freezes per fold, no look-ahead, dollar-neutral."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.strategies.factors import FamaFrenchTiltStrategy


@pytest.fixture
def synthetic_ff5() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2014-01-01", periods=2000)
    df = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.01, len(idx)),
            "SMB": rng.normal(0.0, 0.005, len(idx)),
            "HML": rng.normal(0.0, 0.005, len(idx)),
            "RMW": rng.normal(0.0, 0.005, len(idx)),
            "CMA": rng.normal(0.0, 0.005, len(idx)),
            "RF": np.full(len(idx), 0.00005),
        },
        index=idx,
    )
    df.index.name = "date"
    return df


@pytest.fixture
def synthetic_universe(synthetic_ff5) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n_tickers = 30
    idx = synthetic_ff5.index
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    # Assign each stock different factor loadings.
    loadings = rng.uniform(-1.5, 1.5, size=(n_tickers, 5))
    factor_rets = synthetic_ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values

    idio = rng.normal(0.0, 0.01, size=(len(idx), n_tickers))
    rets = (factor_rets @ loadings.T) + idio + synthetic_ff5["RF"].values[:, None]
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=tickers)


def test_loadings_estimated_per_fold(synthetic_universe, synthetic_ff5):
    strat = FamaFrenchTiltStrategy(
        tickers=list(synthetic_universe.columns),
        tilt_factor="HML",
        ff5=synthetic_ff5,
    )
    strat.fit(synthetic_universe.iloc[:600])
    assert strat.params is not None
    assert "HML" in strat.params.loadings.columns
    # All 30 stocks should have estimates with 600 bars.
    assert len(strat.params.loadings) == 30


def test_tilt_picks_extremes(synthetic_universe, synthetic_ff5):
    strat = FamaFrenchTiltStrategy(
        tickers=list(synthetic_universe.columns),
        tilt_factor="HML",
        decile=0.10,
        ff5=synthetic_ff5,
    )
    strat.fit(synthetic_universe.iloc[:600])

    signals = strat.generate_signals(synthetic_universe.iloc[600:])
    active = signals[(signals != 0).any(axis=1)].iloc[0]
    long_tickers = set(active[active > 0].index)
    short_tickers = set(active[active < 0].index)

    hml_loadings = strat.params.loadings["HML"]
    top = set(hml_loadings.sort_values().tail(3).index)
    bottom = set(hml_loadings.sort_values().head(3).index)
    assert long_tickers == top
    assert short_tickers == bottom


def test_dollar_neutral(synthetic_universe, synthetic_ff5):
    strat = FamaFrenchTiltStrategy(
        tickers=list(synthetic_universe.columns), ff5=synthetic_ff5
    )
    strat.fit(synthetic_universe.iloc[:600])
    signals = strat.generate_signals(synthetic_universe.iloc[600:])
    nonzero = signals.sum(axis=1)[(signals != 0).any(axis=1)]
    assert (nonzero.abs() < 1e-9).all()


def test_no_lookahead(synthetic_universe, synthetic_ff5):
    strat = FamaFrenchTiltStrategy(
        tickers=list(synthetic_universe.columns), ff5=synthetic_ff5
    )
    strat.fit(synthetic_universe.iloc[:600])

    full = strat.generate_signals(synthetic_universe.iloc[600:])
    truncated = strat.generate_signals(synthetic_universe.iloc[600:-1])
    common = truncated.index
    pd.testing.assert_frame_equal(
        full.loc[common, truncated.columns],
        truncated[truncated.columns],
        check_dtype=False,
    )
