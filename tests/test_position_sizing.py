from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.config import TRADING_DAYS_PER_YEAR
from backend.app.risk.position_sizing import vol_target_scalar


def test_vol_target_scalar_hits_target_within_tolerance():
    rng = np.random.default_rng(0)
    daily_sigma = 0.02  # 2% daily → ~31.7% annualized
    n = 5000
    spread_returns = pd.Series(rng.normal(0, daily_sigma, n))

    target = 0.10
    scalar = vol_target_scalar(spread_returns, target_vol=target)

    realized_vol = (spread_returns * scalar).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert abs(realized_vol - target) / target < 0.05  # within 5%


def test_vol_target_scalar_handles_zero_vol():
    flat = pd.Series([0.0] * 100)
    assert vol_target_scalar(flat) == 0.0


def test_vol_target_scalar_handles_empty():
    assert vol_target_scalar(pd.Series(dtype=float)) == 0.0
