"""MomentumStrategy: decile construction, look-ahead, market-neutrality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.strategies.momentum import MomentumStrategy


def _synthetic_multi_asset(n_bars: int = 1500, n_tickers: int = 30, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n_bars)
    drifts = np.linspace(-0.0005, 0.0008, n_tickers)
    rng.shuffle(drifts)
    log_rets = rng.normal(0.0, 0.012, size=(n_bars, n_tickers)) + drifts
    prices = 100 * np.exp(np.cumsum(log_rets, axis=0))
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    return pd.DataFrame(prices, index=idx, columns=cols)


@pytest.fixture
def multi_asset_prices() -> pd.DataFrame:
    return _synthetic_multi_asset()


def test_market_neutral_by_construction(multi_asset_prices):
    tickers = list(multi_asset_prices.columns)
    strat = MomentumStrategy(tickers=tickers)
    strat.fit(multi_asset_prices.iloc[:500])

    signals = strat.generate_signals(multi_asset_prices.iloc[500:])
    # On every nonzero-weight bar, longs and shorts should net to ~0 dollar exposure.
    row_sums = signals.sum(axis=1)
    nonzero = row_sums[(signals != 0).any(axis=1)]
    assert (nonzero.abs() < 1e-9).all(), (
        f"Strategy is not dollar-neutral: max |row sum| = {nonzero.abs().max()}"
    )


def test_decile_count_matches_universe(multi_asset_prices):
    tickers = list(multi_asset_prices.columns)
    strat = MomentumStrategy(tickers=tickers, decile=0.10)
    strat.fit(multi_asset_prices.iloc[:500])
    signals = strat.generate_signals(multi_asset_prices.iloc[500:])

    # On any rebalance bar, expected 3 longs + 3 shorts for 30 stocks at 10% decile.
    active_bar = signals[(signals != 0).any(axis=1)].iloc[0]
    n_long = int((active_bar > 0).sum())
    n_short = int((active_bar < 0).sum())
    assert n_long == n_short == 3, f"Expected 3/3, got {n_long} long / {n_short} short"


def test_no_lookahead(multi_asset_prices):
    strat = MomentumStrategy(tickers=list(multi_asset_prices.columns))
    strat.fit(multi_asset_prices.iloc[:500])

    full = strat.generate_signals(multi_asset_prices.iloc[500:])
    truncated = strat.generate_signals(multi_asset_prices.iloc[500:-1])

    common = truncated.index
    pd.testing.assert_frame_equal(
        full.loc[common, truncated.columns],
        truncated[truncated.columns],
        check_dtype=False,
    )
