"""Backtest engine. Thin wrapper around vectorbt that applies realistic costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from backend.app.config import (
    BORROW_RATE_ANNUAL,
    DEFAULT_INIT_CASH,
    SLIPPAGE_BPS,
    TRADING_DAYS_PER_YEAR,
)


@dataclass
class BacktestResult:
    portfolio: vbt.Portfolio
    equity: pd.Series          # NAV through time, includes borrow-cost adjustment
    gross_equity: pd.Series    # NAV before borrow cost
    returns: pd.Series         # daily simple returns of equity
    trades: pd.DataFrame       # vectorbt trades_readable
    target_dollars: pd.DataFrame  # what we asked for, for diagnostics


def _slippage_fraction(bps: float = SLIPPAGE_BPS) -> float:
    return bps / 10_000.0


def run_backtest(
    prices: pd.DataFrame,
    target_dollars: pd.DataFrame,
    init_cash: float = DEFAULT_INIT_CASH,
    fees_bps: float = SLIPPAGE_BPS,
    borrow_rate_annual: float = BORROW_RATE_ANNUAL,
    freq: str = "1D",
) -> BacktestResult:
    """Run a backtest given a price frame and per-bar target dollar allocations.

    `prices` columns must match `target_dollars` columns. Both indexed by date.
    `fees_bps` is applied per leg per fill (vectorbt applies to notional, not P&L).
    Short-leg borrow cost is applied as a post-hoc daily haircut.
    """
    if list(prices.columns) != list(target_dollars.columns):
        raise ValueError("prices and target_dollars must have identical column order.")
    if not prices.index.equals(target_dollars.index):
        raise ValueError("prices and target_dollars must share the same index.")

    fees = _slippage_fraction(fees_bps)

    pf = vbt.Portfolio.from_orders(
        close=prices,
        size=target_dollars,
        size_type="targetvalue",
        init_cash=init_cash,
        fees=fees,
        slippage=0.0,
        freq=freq,
        group_by=True,
        cash_sharing=True,
    )

    gross_equity = pf.value()
    if isinstance(gross_equity, pd.DataFrame):
        gross_equity = gross_equity.iloc[:, 0]
    gross_equity = gross_equity.copy()
    gross_equity.name = "gross_equity"

    # Borrow cost on the short leg: daily haircut on the absolute short notional.
    short_notional = target_dollars.clip(upper=0).abs().sum(axis=1)
    daily_borrow_rate = borrow_rate_annual / TRADING_DAYS_PER_YEAR
    borrow_cost = short_notional * daily_borrow_rate
    borrow_cost = borrow_cost.reindex(gross_equity.index).fillna(0.0)

    equity = (gross_equity - borrow_cost.cumsum()).rename("equity")
    returns = equity.pct_change().fillna(0.0).rename("returns")

    trades = pf.trades.records_readable.copy() if len(pf.trades.records_readable) else pd.DataFrame()

    return BacktestResult(
        portfolio=pf,
        equity=equity,
        gross_equity=gross_equity,
        returns=returns,
        trades=trades,
        target_dollars=target_dollars,
    )


def sanity_check_fees_on_notional(result: BacktestResult, tolerance: float = 1e-6) -> dict:
    """Confirm vectorbt is charging fees on notional (not on P&L).

    Returns a small dict the engine can dump to a debug file the first time it runs.
    """
    trades = result.trades
    if trades.empty:
        return {"checked": False, "reason": "no trades"}

    sample = trades.iloc[0]
    notional = float(abs(sample.get("Size", 0.0) * sample.get("Avg Entry Price", 0.0)))
    fee = float(sample.get("Entry Fees", 0.0))
    expected = notional * _slippage_fraction()
    rel_err = abs(fee - expected) / max(abs(expected), 1.0)
    return {
        "checked": True,
        "notional": notional,
        "actual_fee": fee,
        "expected_fee_on_notional": expected,
        "relative_error": rel_err,
        "ok": rel_err < tolerance + 1e-3 or np.isclose(fee, expected, rtol=0.05),
    }
