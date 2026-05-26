"""Engle–Granger pairs trading.

Look-ahead bias is the entire game here. Two guards:

  1. `fit()` runs ONLY on training data. The estimated hedge ratio β is frozen
     and re-used unchanged in `generate_signals()` on test data.

  2. In `generate_signals()`, the raw position derived from z-score is shifted
     forward by one bar before being returned. Signal at time T is therefore
     position computed from prices through T-1, which is then executed at T.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from backend.app.config import MAX_POSITION_PCT, TARGET_VOL_ANNUAL, TRADING_DAYS_PER_YEAR
from backend.app.strategies.base import Strategy


class NotCointegratedError(ValueError):
    pass


@dataclass
class FittedParams:
    alpha: float
    beta: float
    half_life: float
    adf_pvalue: float
    spread_vol_annual: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp


class PairsTradingStrategy(Strategy):
    def __init__(
        self,
        ticker_a: str,
        ticker_b: str,
        zscore_window: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
        target_vol: float = TARGET_VOL_ANNUAL,
        max_position_pct: float = MAX_POSITION_PCT,
        adf_pvalue_threshold: float = 0.05,
    ) -> None:
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.target_vol = target_vol
        self.max_position_pct = max_position_pct
        self.adf_pvalue_threshold = adf_pvalue_threshold
        self.params: FittedParams | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, train_data: pd.DataFrame) -> "PairsTradingStrategy":
        """Calibrate hedge ratio on training data. Raises if pair is not cointegrated."""
        self._validate_columns(train_data)
        log_a = np.log(train_data[self.ticker_a].astype(float))
        log_b = np.log(train_data[self.ticker_b].astype(float))

        if adfuller(log_a, autolag="AIC")[1] < 0.05:
            raise NotCointegratedError(
                f"{self.ticker_a} log price is stationary (I(0)); pairs framework does not apply."
            )
        if adfuller(log_b, autolag="AIC")[1] < 0.05:
            raise NotCointegratedError(
                f"{self.ticker_b} log price is stationary (I(0)); pairs framework does not apply."
            )

        X = sm.add_constant(log_b.values)
        ols = sm.OLS(log_a.values, X).fit()
        alpha, beta = float(ols.params[0]), float(ols.params[1])

        residuals = log_a.values - (alpha + beta * log_b.values)
        adf_p = float(adfuller(residuals, autolag="AIC")[1])
        if adf_p >= self.adf_pvalue_threshold:
            raise NotCointegratedError(
                f"Residual ADF p-value {adf_p:.4f} >= {self.adf_pvalue_threshold}; "
                f"{self.ticker_a}/{self.ticker_b} is not cointegrated on the training window."
            )

        half_life = self._half_life_from_ar1(pd.Series(residuals, index=train_data.index))
        if not np.isfinite(half_life) or half_life <= 0 or half_life >= self.zscore_window:
            raise NotCointegratedError(
                f"Estimated half-life {half_life:.2f} is incompatible with z-score window "
                f"{self.zscore_window}; pick a different pair or widen the window."
            )

        spread_diff = pd.Series(residuals, index=train_data.index).diff().dropna()
        spread_vol_annual = float(spread_diff.std()) * float(np.sqrt(TRADING_DAYS_PER_YEAR))

        self.params = FittedParams(
            alpha=alpha,
            beta=beta,
            half_life=half_life,
            adf_pvalue=adf_p,
            spread_vol_annual=spread_vol_annual,
            train_start=pd.Timestamp(train_data.index[0]),
            train_end=pd.Timestamp(train_data.index[-1]),
        )
        return self

    @staticmethod
    def _half_life_from_ar1(spread: pd.Series) -> float:
        delta = spread.diff().dropna()
        lagged = spread.shift(1).dropna().loc[delta.index]
        X = sm.add_constant(lagged.values)
        res = sm.OLS(delta.values, X).fit()
        phi = float(res.params[1])
        if phi >= 0:
            return float("inf")
        return float(-np.log(2) / np.log(1 + phi))

    # -------------------------------------------------------- signal generation
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return per-bar target weights for each leg, shifted so signal_T uses data through T-1.

        Output columns: [`<ticker_a>`, `<ticker_b>`, `z`, `state`].
        Weights are in *units of leg-A notional*: long_spread → (+1, -β), short_spread → (-1, +β).
        """
        if self.params is None:
            raise RuntimeError("Strategy must be fit() before generate_signals().")
        self._validate_columns(data)

        beta = self.params.beta
        log_a = np.log(data[self.ticker_a].astype(float))
        log_b = np.log(data[self.ticker_b].astype(float))
        spread = log_a - beta * log_b

        prior = spread.shift(1)
        rolling_mean = prior.rolling(self.zscore_window).mean()
        rolling_std = prior.rolling(self.zscore_window).std()
        z = (spread - rolling_mean) / rolling_std

        state = self._run_state_machine(z.values, max_hold=2 * self.params.half_life)
        state_series = pd.Series(state, index=data.index, name="state")

        weight_a = state_series.map({0: 0.0, 1: 1.0, -1: -1.0})
        weight_b = state_series.map({0: 0.0, 1: -beta, -1: beta})

        signals = pd.DataFrame(
            {
                self.ticker_a: weight_a,
                self.ticker_b: weight_b,
                "z": z,
                "state": state_series,
            },
            index=data.index,
        )

        # Look-ahead guard: shift positions forward by one bar.
        # signal_T now uses only data through T-1.
        signals[[self.ticker_a, self.ticker_b, "state"]] = signals[
            [self.ticker_a, self.ticker_b, "state"]
        ].shift(1)
        signals[[self.ticker_a, self.ticker_b]] = signals[
            [self.ticker_a, self.ticker_b]
        ].fillna(0.0)
        signals["state"] = signals["state"].fillna(0).astype(int)
        return signals

    def _run_state_machine(self, z: np.ndarray, max_hold: float) -> np.ndarray:
        """0 = flat, +1 = long spread (long A, short B), -1 = short spread."""
        state = np.zeros(len(z), dtype=int)
        current = 0
        entry_idx = -1
        max_hold_bars = int(np.ceil(max_hold))
        for i, zi in enumerate(z):
            if not np.isfinite(zi):
                state[i] = current
                continue

            if current == 0:
                if zi < -self.entry_z:
                    current, entry_idx = 1, i
                elif zi > self.entry_z:
                    current, entry_idx = -1, i
            elif current == 1:
                held = i - entry_idx
                if zi >= -self.exit_z or zi < -self.stop_z or held > max_hold_bars:
                    current, entry_idx = 0, -1
            elif current == -1:
                held = i - entry_idx
                if zi <= self.exit_z or zi > self.stop_z or held > max_hold_bars:
                    current, entry_idx = 0, -1
            state[i] = current
        return state

    # ----------------------------------------------------------- sizing
    def size_positions(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame | None,
        portfolio_value: float,
    ) -> pd.DataFrame:
        """Scale unit weights by the vol-target scalar fixed at fit time, then cap per leg.

        `returns` is accepted for API symmetry but ignored: the volatility estimate is
        frozen on the training window inside `fit()` so it never leaks future data.
        """
        if self.params is None:
            raise RuntimeError("Strategy must be fit() before size_positions().")
        del returns  # explicitly unused; preserved for Strategy contract symmetry.

        beta = self.params.beta
        spread_vol = self.params.spread_vol_annual
        scalar = (self.target_vol / spread_vol) if spread_vol > 0 else 0.0

        # Cap the leg-A notional at max_position_pct of NAV; leg B scales with |β|.
        scalar = min(scalar, self.max_position_pct)
        if abs(beta) > 1.0:
            scalar = min(scalar, self.max_position_pct / abs(beta))

        sized = signals.copy()
        sized[self.ticker_a] = sized[self.ticker_a] * scalar * portfolio_value
        sized[self.ticker_b] = sized[self.ticker_b] * scalar * portfolio_value
        sized["target_scalar"] = scalar
        return sized

    # ---------------------------------------------------------- helpers
    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [t for t in (self.ticker_a, self.ticker_b) if t not in df.columns]
        if missing:
            raise KeyError(f"Input frame is missing required price columns: {missing}")
