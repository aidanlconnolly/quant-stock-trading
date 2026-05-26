"""Walk-forward driver: 24-month train, 6-month test, rolling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from backend.app.backtest.engine import BacktestResult, run_backtest
from backend.app.config import DEFAULT_INIT_CASH
from backend.app.strategies.base import Strategy
from backend.app.strategies.pairs_trading import (
    NotCointegratedError,
    PairsTradingStrategy,
)


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    beta: float | None = None
    half_life: float | None = None
    skipped: bool = False
    skip_reason: str = ""
    result: BacktestResult | None = None


def make_folds(
    index: pd.DatetimeIndex,
    train_months: int = 24,
    test_months: int = 6,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (train_start, train_end, test_start, test_end) tuples with no overlap."""
    if len(index) == 0:
        return []
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(index[0])
    end_of_data = pd.Timestamp(index[-1])

    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > end_of_data:
            break
        folds.append((train_start, train_end, test_start, test_end))
        cursor = cursor + pd.DateOffset(months=test_months)
    return folds


def walk_forward(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    train_months: int = 24,
    test_months: int = 6,
    zscore_window: int = 60,
    init_cash: float = DEFAULT_INIT_CASH,
    strategy_kwargs: dict | None = None,
    fees_bps: float | None = None,
) -> tuple[pd.Series, list[Fold]]:
    """Run rolling walk-forward. Returns (oos_equity_concat, folds).

    Test-period equity curves are stitched into one continuous series. Each fold's
    starting NAV is the prior fold's terminal NAV so the curve is comparable to a
    single long backtest. The β used in fold N's test slice is the β fit on fold N's
    training slice, never refit on test data.
    """
    strategy_kwargs = dict(strategy_kwargs or {})
    strategy_kwargs.setdefault("zscore_window", zscore_window)

    fold_specs = make_folds(prices.index, train_months=train_months, test_months=test_months)
    folds: list[Fold] = []
    oos_curves: list[pd.Series] = []
    running_nav = init_cash

    for train_start, train_end, test_start, test_end in fold_specs:
        # Hard non-overlap assertion (the spec's data-integrity check).
        assert train_end < test_start, (
            f"Walk-forward fold has overlapping train/test: train ends {train_end}, "
            f"test starts {test_start}"
        )

        train_df = prices.loc[train_start:train_end]
        if len(train_df) < zscore_window * 2:
            folds.append(
                Fold(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    skipped=True,
                    skip_reason="insufficient training bars",
                )
            )
            continue

        strategy = PairsTradingStrategy(
            ticker_a=ticker_a, ticker_b=ticker_b, **strategy_kwargs
        )
        try:
            strategy.fit(train_df)
        except NotCointegratedError as e:
            folds.append(
                Fold(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    skipped=True,
                    skip_reason=f"not cointegrated: {e}",
                )
            )
            continue

        # Include a warm-up window of training data so rolling z-score is well-defined
        # at test_start. The warm-up is past data so no future info leaks.
        warmup_start_idx = max(0, prices.index.get_indexer([test_start])[0] - zscore_window)
        warmup_start = prices.index[warmup_start_idx]
        signal_window = prices.loc[warmup_start:test_end]

        signals = strategy.generate_signals(signal_window)
        sized = strategy.size_positions(signals[[ticker_a, ticker_b]], None, running_nav)

        # Restrict execution to the actual test window.
        test_slice = sized.loc[test_start:test_end, [ticker_a, ticker_b]]
        price_slice = prices.loc[test_slice.index]

        fees_kwargs = {} if fees_bps is None else {"fees_bps": fees_bps}
        result = run_backtest(
            prices=price_slice,
            target_dollars=test_slice,
            init_cash=running_nav,
            **fees_kwargs,
        )
        oos_curves.append(result.equity)
        running_nav = float(result.equity.iloc[-1])

        folds.append(
            Fold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                beta=strategy.params.beta if strategy.params else None,
                half_life=strategy.params.half_life if strategy.params else None,
                result=result,
            )
        )

    if not oos_curves:
        return pd.Series(dtype=float, name="equity"), folds

    oos_equity = pd.concat(oos_curves).sort_index()
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
    oos_equity.name = "equity"
    return oos_equity, folds


def walk_forward_multi(
    prices: pd.DataFrame,
    strategy_factory: Callable[[], Strategy],
    tradable_columns: list[str] | None = None,
    train_months: int = 24,
    test_months: int = 6,
    warmup_bars: int = 252,
    init_cash: float = DEFAULT_INIT_CASH,
    fees_bps: float | None = None,
) -> tuple[pd.Series, list[Fold]]:
    """Walk-forward runner for any multi-asset Strategy (momentum, factor, ...).

    `strategy_factory` is called once per fold to produce a fresh, un-fit Strategy.
    `tradable_columns` defaults to all columns of `prices`. `warmup_bars` is the
    number of pre-test bars (drawn from the past) passed into `generate_signals`
    so rolling stats are warm; the warm-up is past data, no leakage.
    """
    cols = tradable_columns or list(prices.columns)
    fold_specs = make_folds(prices.index, train_months=train_months, test_months=test_months)
    folds: list[Fold] = []
    oos_curves: list[pd.Series] = []
    running_nav = init_cash

    for train_start, train_end, test_start, test_end in fold_specs:
        assert train_end < test_start, "walk-forward train/test overlap"

        train_df = prices.loc[train_start:train_end]
        if train_df.empty:
            folds.append(
                Fold(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    skipped=True,
                    skip_reason="empty training slice",
                )
            )
            continue

        strategy = strategy_factory()
        try:
            strategy.fit(train_df)
        except Exception as e:  # noqa: BLE001
            folds.append(
                Fold(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    skipped=True,
                    skip_reason=f"fit failed: {e}",
                )
            )
            continue

        # Warm-up: include `warmup_bars` of past data so rolling stats are valid at test_start.
        test_pos = prices.index.get_indexer([test_start])[0]
        warmup_idx = max(0, test_pos - warmup_bars)
        warmup_start = prices.index[warmup_idx]
        signal_window = prices.loc[warmup_start:test_end]

        signals = strategy.generate_signals(signal_window)
        signal_cols = [c for c in cols if c in signals.columns]
        sized = strategy.size_positions(signals[signal_cols], None, running_nav)

        test_slice = sized.loc[test_start:test_end, signal_cols]
        price_slice = prices.loc[test_slice.index, signal_cols]

        fees_kwargs = {} if fees_bps is None else {"fees_bps": fees_bps}
        result = run_backtest(
            prices=price_slice,
            target_dollars=test_slice,
            init_cash=running_nav,
            **fees_kwargs,
        )
        oos_curves.append(result.equity)
        running_nav = float(result.equity.iloc[-1])

        folds.append(
            Fold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                result=result,
            )
        )

    if not oos_curves:
        return pd.Series(dtype=float, name="equity"), folds

    oos_equity = pd.concat(oos_curves).sort_index()
    oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
    oos_equity.name = "equity"
    return oos_equity, folds
