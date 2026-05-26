"""Look-ahead guard: signal at index T must not depend on data at index >= T."""
from __future__ import annotations

import pandas as pd

from backend.app.strategies.pairs_trading import PairsTradingStrategy


def _fit(cointegrated_pair: pd.DataFrame) -> PairsTradingStrategy:
    strat = PairsTradingStrategy("A", "B", zscore_window=60)
    train = cointegrated_pair.iloc[:600]
    strat.fit(train)
    return strat


def test_signal_independence_from_future_bars(cointegrated_pair):
    strat = _fit(cointegrated_pair)
    test_full = cointegrated_pair.iloc[600:]

    signals_full = strat.generate_signals(test_full)
    signals_truncated = strat.generate_signals(test_full.iloc[:-1])

    common_idx = signals_truncated.index
    cols = ["A", "B", "state"]
    pd.testing.assert_frame_equal(
        signals_full.loc[common_idx, cols],
        signals_truncated.loc[common_idx, cols],
        check_dtype=False,
    )


def test_signal_uses_only_past_data(cointegrated_pair):
    """Setting a future bar to NaN must not change any earlier signal."""
    strat = _fit(cointegrated_pair)
    test_full = cointegrated_pair.iloc[600:].copy()

    poisoned = test_full.copy()
    poisoned.iloc[-1] = float("nan")

    s1 = strat.generate_signals(test_full).iloc[:-1]
    s2 = strat.generate_signals(poisoned).iloc[:-1]
    cols = ["A", "B", "state"]
    pd.testing.assert_frame_equal(s1[cols], s2[cols], check_dtype=False)
