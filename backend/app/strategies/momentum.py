"""Cross-sectional momentum: top-decile long, bottom-decile short, monthly rebalance.

Signal = 12-month return ending one month ago (the canonical 12-1 momentum that
skips the most recent month to avoid short-term reversal). Market-neutral by
construction: equal dollar long and short legs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.app.config import (
    MAX_POSITION_PCT,
    TARGET_VOL_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)
from backend.app.strategies.base import Strategy


@dataclass
class MomentumFitted:
    target_scalar: float
    realized_vol_annual: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    universe_used: tuple[str, ...]


class MomentumStrategy(Strategy):
    def __init__(
        self,
        tickers: list[str],
        lookback_months: int = 12,
        skip_months: int = 1,
        decile: float = 0.10,
        target_vol: float = TARGET_VOL_ANNUAL,
        max_position_pct: float = MAX_POSITION_PCT,
        min_universe_size: int = 10,
    ) -> None:
        self.tickers = list(tickers)
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.decile = decile
        self.target_vol = target_vol
        self.max_position_pct = max_position_pct
        self.min_universe_size = min_universe_size
        self.params: MomentumFitted | None = None

    # ----------------------------------------------------- helpers
    @staticmethod
    def _month_end_mask(index: pd.DatetimeIndex) -> pd.Series:
        """True on bars that are the last trading day of their calendar month."""
        periods = pd.Series(index.to_period("M"), index=index)
        return periods.ne(periods.shift(-1)).fillna(True)

    def _unit_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute pre-shift target weights (unit long-short, sums to ~0 by row)."""
        available = [t for t in self.tickers if t in data.columns]
        prices = data[available].astype(float)
        log_p = np.log(prices.where(prices > 0))

        lookback_bars = self.lookback_months * 21
        skip_bars = self.skip_months * 21
        momentum = log_p.shift(skip_bars) - log_p.shift(skip_bars + lookback_bars)

        rebal_dates = data.index[self._month_end_mask(data.index).values]
        unit = pd.DataFrame(0.0, index=data.index, columns=available)

        last_weights: pd.Series | None = None
        for date in rebal_dates:
            m = momentum.loc[date].dropna()
            if len(m) < self.min_universe_size:
                if last_weights is not None:
                    unit.loc[date] = last_weights
                continue
            sorted_m = m.sort_values()
            n = max(1, int(round(len(m) * self.decile)))
            shorts = sorted_m.iloc[:n].index
            longs = sorted_m.iloc[-n:].index
            w = pd.Series(0.0, index=available)
            w[longs] = 0.5 / n     # 50% NAV on long leg, equally weighted
            w[shorts] = -0.5 / n   # 50% NAV on short leg
            unit.loc[date] = w
            last_weights = w

        # Replace non-rebalance days with NaN so ffill carries the most recent rebalance weights.
        is_rebal = pd.Series(False, index=data.index)
        is_rebal.loc[rebal_dates] = True
        carried = unit.copy()
        carried.loc[~is_rebal.values] = np.nan
        carried = carried.ffill().fillna(0.0)
        return carried

    # ----------------------------------------------------- fit
    def fit(self, train_data: pd.DataFrame) -> "MomentumStrategy":
        available = [t for t in self.tickers if t in train_data.columns]
        if len(available) < self.min_universe_size:
            raise ValueError(
                f"Training universe has only {len(available)} of {len(self.tickers)} tickers"
            )

        unit = self._unit_weights(train_data)
        executable = unit.shift(1).fillna(0.0)  # would-be live weights
        rets = train_data[unit.columns].astype(float).pct_change().fillna(0.0)
        port_rets = (executable * rets).sum(axis=1)
        realized_vol = float(port_rets.std()) * float(np.sqrt(TRADING_DAYS_PER_YEAR))

        scalar = (self.target_vol / realized_vol) if realized_vol > 0 else 0.0

        self.params = MomentumFitted(
            target_scalar=scalar,
            realized_vol_annual=realized_vol,
            train_start=pd.Timestamp(train_data.index[0]),
            train_end=pd.Timestamp(train_data.index[-1]),
            universe_used=tuple(available),
        )
        return self

    # ----------------------------------------------------- signal
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.params is None:
            raise RuntimeError("Strategy must be fit() before generate_signals().")
        unit = self._unit_weights(data)
        # Look-ahead guard: shift weights by one bar so signal_T uses only data through T-1.
        executable = unit.shift(1).fillna(0.0)
        return executable

    # ----------------------------------------------------- sizing
    def size_positions(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None,
        portfolio_value: float,
    ) -> pd.DataFrame:
        if self.params is None:
            raise RuntimeError("Strategy must be fit() before size_positions().")
        del returns

        scalar = self.params.target_scalar
        sized = signals * scalar * portfolio_value

        # Per-leg cap.
        cap = self.max_position_pct * portfolio_value
        sized = sized.clip(lower=-cap, upper=cap)
        return sized
