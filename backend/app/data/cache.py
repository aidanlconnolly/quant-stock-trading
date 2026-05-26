from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.config import CACHE_DIR


def _cache_path(ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("/", "_").upper()
    return CACHE_DIR / f"{safe}_{start}_{end}.parquet"


def read(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    path = _cache_path(ticker, start, end)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def write(ticker: str, start: str, end: str, df: pd.DataFrame) -> Path:
    path = _cache_path(ticker, start, end)
    df.to_parquet(path)
    return path
