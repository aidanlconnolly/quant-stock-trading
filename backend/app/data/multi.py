"""Multi-asset price loader. Pulls each ticker via the single-asset fetcher and aligns."""
from __future__ import annotations

import pandas as pd

from backend.app.data.fetcher import get_ohlcv


def load_adj_close(
    tickers: list[str],
    start: str,
    end: str,
    drop_threshold: float = 0.9,
) -> pd.DataFrame:
    """Load adjusted close for many tickers and align them on the union of dates.

    Tickers missing more than `(1 - drop_threshold)` of bars are dropped to keep
    the cross-section comparable. The dropped tickers are surfaced via the logger.
    """
    series = {}
    failed = []
    for t in tickers:
        try:
            df = get_ohlcv(t, start, end)
            series[t] = df["Adj Close"].astype(float)
        except Exception as e:  # noqa: BLE001
            failed.append((t, str(e)))

    if not series:
        raise RuntimeError(f"No tickers could be fetched. Failures: {failed}")

    frame = pd.concat(series, axis=1).sort_index()
    coverage = frame.notna().mean()
    keep = coverage[coverage >= drop_threshold].index
    dropped = sorted(set(frame.columns) - set(keep))
    if dropped:
        print(f"[universe] dropped low-coverage tickers: {dropped}")
    if failed:
        print(f"[universe] failed to fetch: {[t for t, _ in failed]}")

    return frame[keep]
