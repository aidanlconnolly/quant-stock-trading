"""Performance metrics. All risk-adjusted figures use daily simple returns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.config import TRADING_DAYS_PER_YEAR


def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = returns.dropna()
    if r.empty or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if r.empty or downside.empty or downside.std() == 0:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    e = equity.dropna().astype(float)
    if e.empty:
        return 0.0
    peak = e.cummax()
    dd = e / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    e = equity.dropna().astype(float)
    if len(e) < 2 or e.iloc[0] <= 0:
        return 0.0
    total_return = e.iloc[-1] / e.iloc[0]
    years = len(e) / periods_per_year
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1.0 / years) - 1.0)


def calmar(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(cagr(equity, periods_per_year) / abs(mdd))


def hit_rate(trades: pd.DataFrame) -> float:
    if trades.empty or "PnL" not in trades.columns:
        return 0.0
    wins = (trades["PnL"] > 0).sum()
    return float(wins / len(trades))


def avg_holding_days(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    if "Entry Timestamp" in trades.columns and "Exit Timestamp" in trades.columns:
        durations = (
            pd.to_datetime(trades["Exit Timestamp"]) - pd.to_datetime(trades["Entry Timestamp"])
        ).dt.days
        return float(durations.mean())
    if "Duration" in trades.columns:
        return float(trades["Duration"].mean())
    return 0.0


def turnover(
    target_dollars: pd.DataFrame,
    equity: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    if target_dollars.empty or equity.empty:
        return 0.0
    notional_change = target_dollars.diff().abs().sum(axis=1).fillna(0.0)
    avg_equity = float(equity.mean())
    if avg_equity == 0:
        return 0.0
    annualized = notional_change.sum() / avg_equity * (periods_per_year / max(1, len(equity)))
    return float(annualized)


def summarize(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    target_dollars: pd.DataFrame | None = None,
) -> dict[str, float]:
    """One-shot bundle of all the metrics we report."""
    trades_df = trades if trades is not None else pd.DataFrame()
    returns = equity.pct_change().dropna()
    out = {
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "cagr": cagr(equity),
        "calmar": calmar(equity),
        "hit_rate": hit_rate(trades_df),
        "avg_holding_days": avg_holding_days(trades_df),
        "n_trades": int(len(trades_df)),
    }
    if target_dollars is not None:
        out["turnover"] = turnover(target_dollars, equity)
    return out


def format_table(metrics: dict[str, float]) -> str:
    from tabulate import tabulate

    rows = []
    for k, v in metrics.items():
        if isinstance(v, float):
            if k in {"max_drawdown", "cagr", "hit_rate"}:
                rows.append([k, f"{v:.2%}"])
            else:
                rows.append([k, f"{v:.3f}"])
        else:
            rows.append([k, v])
    return tabulate(rows, headers=["metric", "value"], tablefmt="github")
