from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Abstract base for all strategies.

    Lifecycle:
      1. `fit(train_df)` calibrates parameters on training data only and freezes them.
      2. `generate_signals(test_df)` produces trade signals using ONLY frozen params
         and data lagged to t-1.
      3. `size_positions(signals, ...)` converts signals to dollar weights.
    """

    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> "Strategy":
        ...

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def size_positions(
        self, signals: pd.DataFrame, returns: pd.DataFrame, portfolio_value: float
    ) -> pd.DataFrame:
        ...
