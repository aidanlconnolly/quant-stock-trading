from __future__ import annotations

import pandas as pd

from backend.app.backtest.walk_forward import make_folds, walk_forward


def test_no_overlapping_folds(cointegrated_pair_long):
    folds = make_folds(cointegrated_pair_long.index, train_months=24, test_months=6)
    assert len(folds) > 1
    for train_start, train_end, test_start, test_end in folds:
        assert train_start < train_end < test_start < test_end


def test_beta_frozen_per_fold(cointegrated_pair_long):
    oos_equity, folds = walk_forward(
        prices=cointegrated_pair_long,
        ticker_a="A",
        ticker_b="B",
        train_months=24,
        test_months=6,
        zscore_window=60,
        fees_bps=5.0,
    )
    used = [f for f in folds if not f.skipped]
    assert len(used) >= 2, "expected multiple usable folds in long fixture"
    # Each fold's β must be finite and not identical across all folds (refit fresh on each train).
    betas = [f.beta for f in used]
    assert all(b is not None for b in betas)
    assert any(abs(b1 - b2) > 1e-9 for b1, b2 in zip(betas, betas[1:])), (
        "β looks identical across all folds — refit may not be working"
    )
    assert isinstance(oos_equity, pd.Series)
    assert not oos_equity.empty
