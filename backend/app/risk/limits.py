from __future__ import annotations

import pandas as pd

from backend.app.config import (
    DRAWDOWN_LOOKBACK_DAYS,
    MAX_DRAWDOWN_CIRCUIT,
    MAX_POSITION_PCT,
)


def cap_per_leg(weights: pd.DataFrame, max_pct: float = MAX_POSITION_PCT) -> pd.DataFrame:
    """Clip each leg so |weight| <= max_pct of NAV. Weights are fractions of NAV."""
    capped = weights.clip(lower=-max_pct, upper=max_pct)
    return capped


def rolling_drawdown(equity: pd.Series, lookback: int = DRAWDOWN_LOOKBACK_DAYS) -> pd.Series:
    """For each bar, return the worst peak-to-trough drawdown over the trailing `lookback` bars."""
    e = equity.astype(float)
    rolling_peak = e.rolling(lookback, min_periods=1).max()
    return (e / rolling_peak) - 1.0


def circuit_breaker_mask(
    equity: pd.Series,
    threshold: float = MAX_DRAWDOWN_CIRCUIT,
    lookback: int = DRAWDOWN_LOOKBACK_DAYS,
) -> pd.Series:
    """True where trading should be DISABLED because trailing drawdown exceeded `threshold`."""
    dd = rolling_drawdown(equity, lookback=lookback)
    return dd <= -abs(threshold)


def apply_circuit_breaker(
    sized_weights: pd.DataFrame, equity: pd.Series, threshold: float = MAX_DRAWDOWN_CIRCUIT
) -> pd.DataFrame:
    """Force weights to zero on any bar where the rolling-30d drawdown triggered the breaker."""
    mask = circuit_breaker_mask(equity, threshold=threshold)
    out = sized_weights.copy()
    out.loc[mask, :] = 0.0
    return out


__all__ = [
    "cap_per_leg",
    "rolling_drawdown",
    "circuit_breaker_mask",
    "apply_circuit_breaker",
    "MAX_POSITION_PCT",
    "MAX_DRAWDOWN_CIRCUIT",
]
