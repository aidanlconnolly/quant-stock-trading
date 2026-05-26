"""Costs must reduce net returns vs gross."""
from __future__ import annotations

import pandas as pd

from backend.app.backtest.engine import run_backtest
from backend.app.strategies.pairs_trading import PairsTradingStrategy


def _build_signals(cointegrated_pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat = PairsTradingStrategy("A", "B", zscore_window=60)
    train = cointegrated_pair.iloc[:600]
    test = cointegrated_pair.iloc[600:]
    strat.fit(train)
    signals = strat.generate_signals(test)
    sized = strat.size_positions(signals[["A", "B"]], None, portfolio_value=100_000.0)
    prices = test[["A", "B"]]
    return prices, sized[["A", "B"]]


def test_net_returns_strictly_lower_than_gross(cointegrated_pair):
    prices, sized = _build_signals(cointegrated_pair)

    gross = run_backtest(prices=prices, target_dollars=sized, fees_bps=0.0, borrow_rate_annual=0.0)
    net = run_backtest(prices=prices, target_dollars=sized, fees_bps=5.0, borrow_rate_annual=0.0025)

    gross_terminal = float(gross.equity.iloc[-1])
    net_terminal = float(net.equity.iloc[-1])

    # If the strategy traded at all (it should on this synthetic), net should be strictly lower.
    assert (sized.diff().abs().sum().sum() > 0), "test fixture failed to trigger any trades"
    assert net_terminal < gross_terminal, (
        f"Costs failed to reduce terminal NAV: gross={gross_terminal}, net={net_terminal}"
    )
