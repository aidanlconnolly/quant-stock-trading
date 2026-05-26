"""CLI: Fama-French factor-tilt backtest with walk-forward validation."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from backend.app.backtest.metrics import format_table, summarize
from backend.app.backtest.walk_forward import walk_forward_multi
from backend.app.config import DEFAULT_INIT_CASH, RESULTS_DIR
from backend.app.data.factors import load_ff5_daily
from backend.app.data.multi import load_adj_close
from backend.app.data.universe import large_cap_universe
from backend.app.strategies.factors import FACTOR_NAMES, FamaFrenchTiltStrategy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--tilt-factor", choices=list(FACTOR_NAMES), default="HML")
    parser.add_argument("--decile", type=float, default=0.10)
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--init-cash", type=float, default=DEFAULT_INIT_CASH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    tickers = large_cap_universe()
    print(f"Loading {len(tickers)} large caps from {args.start} to {args.end}...")
    prices = load_adj_close(tickers, args.start, args.end).dropna(how="all")
    available = list(prices.columns)
    print(f"  {len(prices)} bars, {len(available)} tickers retained.")

    print("Loading Ken French FF5 daily factors...")
    ff5 = load_ff5_daily()
    print(f"  FF5 spans {ff5.index.min().date()} → {ff5.index.max().date()}.")

    def factory() -> FamaFrenchTiltStrategy:
        return FamaFrenchTiltStrategy(
            tickers=available,
            tilt_factor=args.tilt_factor,
            decile=args.decile,
            ff5=ff5,
        )

    print(
        f"Running walk-forward (train={args.train_months}m, test={args.test_months}m, "
        f"tilt={args.tilt_factor})..."
    )
    oos_equity, folds = walk_forward_multi(
        prices=prices,
        strategy_factory=factory,
        tradable_columns=available,
        train_months=args.train_months,
        test_months=args.test_months,
        warmup_bars=30,
        init_cash=args.init_cash,
    )

    used = [f for f in folds if not f.skipped]
    print(f"  folds: {len(folds)} total, {len(used)} traded.")
    if oos_equity.empty:
        print("No tradable folds.")
        return 1

    all_trades = pd.concat(
        [f.result.trades for f in used if f.result is not None and not f.result.trades.empty],
        ignore_index=True,
    ) if used else pd.DataFrame()
    all_targets = pd.concat(
        [f.result.target_dollars for f in used if f.result is not None]
    ) if used else pd.DataFrame()

    metrics = summarize(oos_equity, all_trades, all_targets)
    print()
    print(format_table(metrics))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or RESULTS_DIR / f"factor_{args.tilt_factor}_{timestamp}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    oos_equity.to_frame("equity").to_parquet(output)
    if not all_trades.empty:
        all_trades.to_parquet(output.with_name(output.stem + "_trades.parquet"))

    print()
    print(f"Wrote equity curve → {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
