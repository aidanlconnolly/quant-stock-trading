from __future__ import annotations

import pandas as pd
import yfinance as yf

from backend.app.data import cache


REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def get_ohlcv(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch daily OHLCV for a ticker, with Parquet caching keyed by (ticker, start, end).

    Returns a DataFrame indexed by date with columns: Open, High, Low, Close, Adj Close, Volume.
    Adj Close is what return math should use. Close is preserved for diagnostics.
    """
    if use_cache:
        cached = cache.read(ticker, start, end)
        if cached is not None and not cached.empty:
            return cached

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned empty data for {ticker} {start}..{end}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"yfinance response missing columns {missing} for {ticker}")

    df = raw[REQUIRED_COLS].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    cache.write(ticker, start, end, df)
    return df


def get_pair(
    ticker_a: str, ticker_b: str, start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = get_ohlcv(ticker_a, start, end)
    b = get_ohlcv(ticker_b, start, end)
    common_idx = a.index.intersection(b.index)
    return a.loc[common_idx], b.loc[common_idx]
