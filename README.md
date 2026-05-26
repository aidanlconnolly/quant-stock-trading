# Quant Stock Trading

Personal quant research and paper-trading platform. **Phase 1 MVP**: rigorous pairs-trading backtest on KO/PEP with walk-forward validation, no look-ahead bias, and realistic costs.

## Setup

```bash
brew install python@3.12 uv     # one-time
make setup                      # uv sync
make test                       # run all 17 test cases
make backtest                   # Phase 1: KO/PEP pairs trade, 2014-2024
make momentum                   # Phase 2a: cross-sectional momentum
make factor                     # Phase 2b: Fama-French HML tilt
make notebook                   # open Jupyter Lab on results
```

## Phase 1 scope

- Engle–Granger cointegration on a single equity pair (KO/PEP).
- Walk-forward validation: 2-year train, 6-month test, rolling.
- Costs: 5 bps slippage per leg, $0 commission, 25 bps/yr borrow on short leg.
- Position sizing: volatility-targeted at 10% annualized, capped at 5% NAV per leg.
- Risk: 15% rolling-30d drawdown circuit breaker.
- **No live trading.** `config.LIVE_TRADING_ENABLED` is `False` and not runtime-mutable.

## Phase 2 scope (current)

- **Momentum** ([momentum.py](backend/app/strategies/momentum.py)) — cross-sectional 12-1 month return, top/bottom decile, monthly rebalance, dollar-neutral.
- **Fama-French tilt** ([factors.py](backend/app/strategies/factors.py)) — per-stock OLS on Ken French FF5 daily factors over training window; long top decile of chosen factor loading, short bottom decile.
- **Multi-asset walk-forward** ([walk_forward.py](backend/app/backtest/walk_forward.py)) — `walk_forward_multi` runs any `Strategy` over the large-cap universe with frozen-per-fold params.
- **Universe** ([universe.py](backend/app/data/universe.py)) — 40 mega-caps continuously in the S&P 500 from 2014-2024. See survivorship-bias caveat below.

## What's deferred to Phase 3

- React frontend with strategy comparison view.
- Alpaca paper-trading executor.
- Combined portfolio across strategies with risk parity.
- Regime detection (HMM or vol-based).
- Point-in-time S&P 500 membership (CRSP-style).

## Phase 2 results snapshot (2014-2024, net of costs)

| Strategy            | Net Sharpe | Max DD  | CAGR    | Notes                                    |
|---------------------|-----------:|--------:|--------:|------------------------------------------|
| Pairs (KO/PEP)      |     −1.69  |  −0.6%  |  −0.5%  | Most folds skipped (not cointegrated)    |
| Pairs (XOM/CVX,p=0.10) | −0.08  |  −1.2%  |   0.0%  | More tradable, but costs ~= alpha        |
| Momentum            |      0.35  | −14.3%  |   2.0%  | Consistent w/ post-2009 momentum drag    |
| FF5 HML tilt        |     −0.42  | −25.3%  |  −2.7%  | Value factor has been brutal 2014-2024   |

These results are realistic, not impressive — that is the point. Treat any net Sharpe > 2 as a bug.

## Survivorship-bias caveat

Phase 1 (KO/PEP) is immune. Phase 2 uses `LARGE_CAP_UNIVERSE` — 40 stocks that *survived* in the S&P 500 from 2014 through 2024. This biases backtested returns upward because the universe excludes companies that were delisted, acquired, or fell out of the index. The bias is modest for mega-caps over a 10-year window but it is non-zero. Replacing the fixed universe with point-in-time membership (CRSP or similar) is in Phase 3.

## Expected Phase 1 results (sanity ranges)

| Metric            | Range            |
|-------------------|------------------|
| Gross Sharpe      | 1.0 – 1.5        |
| Net Sharpe        | 0.4 – 0.8        |
| Max drawdown      | 8 – 12%          |
| Trades / year     | 8 – 15           |
| Avg holding (days)| 15 – 25          |

If net Sharpe > 2, treat as a bug — most likely cause is a look-ahead leak in the z-score calculation or `β` getting refit on test data.
