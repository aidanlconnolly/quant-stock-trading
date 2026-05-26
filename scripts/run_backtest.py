"""CLI: end-to-end pairs-trading backtest with walk-forward validation.

Usage:
    uv run python -m scripts.run_backtest --pair KO PEP --start 2014-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from backend.app.backtest.metrics import format_table, summarize
from backend.app.backtest.walk_forward import walk_forward
from backend.app.config import DEFAULT_INIT_CASH, RESULTS_DIR
from backend.app.data.fetcher import get_pair


def _adj_close_frame(ohlcv_a: pd.DataFrame, ohlcv_b: pd.DataFrame, ticker_a: str, ticker_b: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ticker_a: ohlcv_a["Adj Close"].astype(float),
            ticker_b: ohlcv_b["Adj Close"].astype(float),
        }
    ).dropna()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, metavar=("TICKER_A", "TICKER_B"), required=True)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--zscore-window", type=int, default=60)
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--init-cash", type=float, default=DEFAULT_INIT_CASH)
    parser.add_argument(
        "--adf-pvalue",
        type=float,
        default=0.05,
        help="Cointegration ADF p-value threshold. Looser values trade more folds but with weaker evidence.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ticker_a, ticker_b = args.pair
    print(f"Fetching {ticker_a} / {ticker_b} from {args.start} to {args.end}...")
    ohlcv_a, ohlcv_b = get_pair(ticker_a, ticker_b, args.start, args.end)
    prices = _adj_close_frame(ohlcv_a, ohlcv_b, ticker_a, ticker_b)
    print(f"  {len(prices)} aligned bars.")

    print("Running walk-forward (train={tm}m, test={sm}m)...".format(
        tm=args.train_months, sm=args.test_months
    ))
    oos_equity, folds = walk_forward(
        prices=prices,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        train_months=args.train_months,
        test_months=args.test_months,
        zscore_window=args.zscore_window,
        init_cash=args.init_cash,
        strategy_kwargs={"adf_pvalue_threshold": args.adf_pvalue},
    )

    used_folds = [f for f in folds if not f.skipped]
    skipped = [f for f in folds if f.skipped]
    print(f"  folds: {len(folds)} total, {len(used_folds)} traded, {len(skipped)} skipped.")
    for f in skipped:
        print(f"    SKIP {f.test_start.date()}..{f.test_end.date()}: {f.skip_reason}")

    if oos_equity.empty:
        print("No tradable folds. Nothing to write.")
        return 1

    all_trades = pd.concat(
        [f.result.trades for f in used_folds if f.result is not None and not f.result.trades.empty],
        ignore_index=True,
    ) if used_folds else pd.DataFrame()
    all_targets = pd.concat(
        [f.result.target_dollars for f in used_folds if f.result is not None],
    ) if used_folds else pd.DataFrame()

    metrics = summarize(oos_equity, all_trades, all_targets)
    print()
    print(format_table(metrics))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or RESULTS_DIR / f"{ticker_a}{ticker_b}_{timestamp}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)

    oos_equity.to_frame("equity").to_parquet(output)
    trades_path = output.with_name(output.stem + "_trades.parquet")
    if not all_trades.empty:
        all_trades.to_parquet(trades_path)
    folds_df = pd.DataFrame(
        [
            {
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "beta": f.beta,
                "half_life": f.half_life,
                "skipped": f.skipped,
                "skip_reason": f.skip_reason,
            }
            for f in folds
        ]
    )
    folds_path = output.with_name(output.stem + "_folds.parquet")
    folds_df.to_parquet(folds_path)

    print()
    print(f"Wrote equity curve   → {output}")
    if not all_trades.empty:
        print(f"Wrote trade log      → {trades_path}")
    print(f"Wrote fold summary   → {folds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
