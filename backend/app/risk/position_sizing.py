from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.config import TARGET_VOL_ANNUAL, TRADING_DAYS_PER_YEAR


def vol_target_scalar(
    spread_returns: pd.Series,
    target_vol: float = TARGET_VOL_ANNUAL,
) -> float:
    """Return the scalar that, multiplied by spread returns, gives `target_vol` annualized.

    Uses the *sample* standard deviation of the supplied spread returns.
    """
    s = spread_returns.dropna()
    if s.empty:
        return 0.0
    realized_vol = float(s.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if realized_vol <= 0:
        return 0.0
    return float(target_vol / realized_vol)
