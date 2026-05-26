from __future__ import annotations

import pytest

from backend.app.strategies.pairs_trading import (
    NotCointegratedError,
    PairsTradingStrategy,
)


def test_cointegrated_pair_fits(cointegrated_pair):
    strat = PairsTradingStrategy("A", "B", zscore_window=60)
    strat.fit(cointegrated_pair)
    assert strat.params is not None
    assert strat.params.adf_pvalue < 0.05
    # True β was 1.2; OLS should land close.
    assert abs(strat.params.beta - 1.2) < 0.1
    assert 0 < strat.params.half_life < 60


def test_two_random_walks_rejected(random_walk_pair):
    strat = PairsTradingStrategy("A", "B", zscore_window=60)
    with pytest.raises(NotCointegratedError):
        strat.fit(random_walk_pair)
