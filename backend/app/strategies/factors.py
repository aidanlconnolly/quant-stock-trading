"""Fama–French 5-factor tilt strategy.

For each stock in the universe, regress training-period excess returns on
[Mkt-RF, SMB, HML, RMW, CMA]. Rank stocks by their loading on the chosen tilt
factor (default HML, the value factor). Go long the top decile, short the bottom.
Monthly rebalance, market-neutral.

The factor betas are estimated ONCE on the training window and frozen for the
test period — same look-ahead guard as the other strategies. Loadings are
re-estimated each walk-forward fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backend.app.config import (
    MAX_POSITION_PCT,
    TARGET_VOL_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)
from backend.app.data.factors import load_ff5_daily
from backend.app.strategies.base import Strategy
from backend.app.strategies.momentum import MomentumStrategy


FACTOR_NAMES = ("Mkt-RF", "SMB", "HML", "RMW", "CMA")


@dataclass
class FactorFitted:
    loadings: pd.DataFrame      # rows = ticker, columns = factor names
    target_scalar: float
    realized_vol_annual: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    universe_used: tuple[str, ...]


class FamaFrenchTiltStrategy(Strategy):
    def __init__(
        self,
        tickers: list[str],
        tilt_factor: str = "HML",
        decile: float = 0.10,
        target_vol: float = TARGET_VOL_ANNUAL,
        max_position_pct: float = MAX_POSITION_PCT,
        min_universe_size: int = 10,
        ff5: pd.DataFrame | None = None,
    ) -> None:
        if tilt_factor not in FACTOR_NAMES:
            raise ValueError(f"tilt_factor must be one of {FACTOR_NAMES}, got {tilt_factor}")
        self.tickers = list(tickers)
        self.tilt_factor = tilt_factor
        self.decile = decile
        self.target_vol = target_vol
        self.max_position_pct = max_position_pct
        self.min_universe_size = min_universe_size
        self._ff5 = ff5 if ff5 is not None else load_ff5_daily()
        self.params: FactorFitted | None = None

    # ----------------------------------------------------- fit
    def fit(self, train_data: pd.DataFrame) -> "FamaFrenchTiltStrategy":
        available = [t for t in self.tickers if t in train_data.columns]
        if len(available) < self.min_universe_size:
            raise ValueError(
                f"Training universe has only {len(available)} of {len(self.tickers)} tickers"
            )

        rets = train_data[available].astype(float).pct_change().dropna(how="all")
        ff = self._ff5.reindex(rets.index).dropna()
        if ff.empty:
            raise RuntimeError("FF5 factor frame is empty after aligning to training dates.")
        rets = rets.loc[ff.index]

        rf = ff["RF"]
        factors = ff[list(FACTOR_NAMES)]
        loadings = {}
        for ticker in available:
            excess = rets[ticker].dropna() - rf.loc[rets[ticker].dropna().index]
            common = excess.index.intersection(factors.index)
            if len(common) < 60:
                continue
            X = sm.add_constant(factors.loc[common].values)
            y = excess.loc[common].values
            try:
                ols = sm.OLS(y, X, missing="drop").fit()
                loadings[ticker] = dict(zip(FACTOR_NAMES, ols.params[1:]))
            except Exception:  # noqa: BLE001
                continue

        if len(loadings) < self.min_universe_size:
            raise ValueError(
                f"Only {len(loadings)} usable factor regressions; need {self.min_universe_size}."
            )
        loadings_df = pd.DataFrame(loadings).T  # rows = ticker, cols = factor

        # Estimate the in-sample portfolio vol the tilt would have produced.
        unit = self._unit_weights(train_data, loadings_df)
        executable = unit.shift(1).fillna(0.0)
        port_rets = (executable * rets).sum(axis=1)
        realized_vol = float(port_rets.std()) * float(np.sqrt(TRADING_DAYS_PER_YEAR))
        scalar = (self.target_vol / realized_vol) if realized_vol > 0 else 0.0

        self.params = FactorFitted(
            loadings=loadings_df,
            target_scalar=scalar,
            realized_vol_annual=realized_vol,
            train_start=pd.Timestamp(train_data.index[0]),
            train_end=pd.Timestamp(train_data.index[-1]),
            universe_used=tuple(loadings_df.index.tolist()),
        )
        return self

    # ----------------------------------------------------- weights
    def _unit_weights(self, data: pd.DataFrame, loadings: pd.DataFrame) -> pd.DataFrame:
        available = [t for t in self.tickers if t in data.columns and t in loadings.index]
        rebal_dates = data.index[MomentumStrategy._month_end_mask(data.index).values]
        unit = pd.DataFrame(0.0, index=data.index, columns=available)

        ranks = loadings[self.tilt_factor].reindex(available).dropna()
        if len(ranks) < self.min_universe_size:
            return unit
        sorted_ranks = ranks.sort_values()
        n = max(1, int(round(len(sorted_ranks) * self.decile)))
        shorts = sorted_ranks.iloc[:n].index
        longs = sorted_ranks.iloc[-n:].index
        w = pd.Series(0.0, index=available)
        w[longs] = 0.5 / n
        w[shorts] = -0.5 / n

        # Tilt is static given fitted loadings; we just apply `w` on every rebalance.
        is_rebal = pd.Series(False, index=data.index)
        is_rebal.loc[rebal_dates] = True
        unit.loc[is_rebal.values] = w.values
        unit.loc[~is_rebal.values] = np.nan
        carried = unit.ffill().fillna(0.0)
        return carried

    # ----------------------------------------------------- signal
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.params is None:
            raise RuntimeError("Strategy must be fit() before generate_signals().")
        unit = self._unit_weights(data, self.params.loadings)
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
        cap = self.max_position_pct * portfolio_value
        sized = sized.clip(lower=-cap, upper=cap)
        return sized
