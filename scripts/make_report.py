"""Generate a single self-contained HTML report from all backtest results in data/results/.

Just double-click the produced file to view in a browser. No server, no notebook.
"""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backend.app.backtest.metrics import summarize
from backend.app.config import RESULTS_DIR


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _equity_chart(equity: pd.Series, title: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    equity.plot(ax=ax, linewidth=2, color="#2563eb")
    ax.fill_between(equity.index, equity.values, equity.iloc[0], alpha=0.08, color="#2563eb")
    ax.axhline(equity.iloc[0], color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("NAV ($)")
    ax.grid(alpha=0.25)
    return _fig_to_data_uri(fig)


def _drawdown_chart(equity: pd.Series) -> str:
    peak = equity.cummax()
    dd = (equity / peak - 1.0) * 100
    fig, ax = plt.subplots(figsize=(10, 2.6))
    dd.plot(ax=ax, color="#dc2626")
    ax.fill_between(dd.index, dd.values, 0, color="#dc2626", alpha=0.2)
    ax.set_title(f"Drawdown (max = {dd.min():.1f}%)")
    ax.set_ylabel("DD (%)")
    ax.grid(alpha=0.25)
    return _fig_to_data_uri(fig)


def _metrics_html(metrics: dict) -> str:
    rows = []
    fmt_pct = {"max_drawdown", "cagr", "hit_rate"}
    for k, v in metrics.items():
        if isinstance(v, float):
            cell = f"{v:.2%}" if k in fmt_pct else f"{v:.3f}"
        else:
            cell = str(v)
        rows.append(f"<tr><td>{k}</td><td style='text-align:right'><code>{cell}</code></td></tr>")
    return (
        "<table style='border-collapse:collapse;min-width:300px'>"
        "<thead><tr><th style='text-align:left;border-bottom:1px solid #ccc;padding:6px 12px'>metric</th>"
        "<th style='text-align:right;border-bottom:1px solid #ccc;padding:6px 12px'>value</th></tr></thead>"
        f"<tbody style='font-family:ui-monospace,monospace;font-size:14px'>{''.join(rows)}</tbody>"
        "</table>"
    )


def _load_latest(prefix: str) -> tuple[pd.Series, pd.DataFrame] | None:
    files = sorted(
        p for p in RESULTS_DIR.glob(f"{prefix}*.parquet")
        if "trades" not in p.name and "folds" not in p.name
    )
    if not files:
        return None
    latest = files[-1]
    equity = pd.read_parquet(latest)["equity"]
    trades_path = latest.with_name(latest.stem + "_trades.parquet")
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    return equity, trades, latest.name


def _section(label: str, prefix: str, color: str) -> str:
    loaded = _load_latest(prefix)
    if loaded is None:
        return (
            f"<section><h2 style='color:{color}'>{label}</h2>"
            f"<p><em>No results yet — run the relevant <code>make</code> target.</em></p></section>"
        )
    equity, trades, fname = loaded
    metrics = summarize(equity, trades)
    return (
        f"<section style='margin:32px 0;padding:24px;border-left:4px solid {color};background:#fafafa'>"
        f"<h2 style='margin-top:0;color:{color}'>{label}</h2>"
        f"<p style='color:#666;font-size:13px'>source: <code>{fname}</code></p>"
        f"<img src='{_equity_chart(equity, label + ' — out-of-sample equity')}' style='max-width:100%' />"
        f"<img src='{_drawdown_chart(equity)}' style='max-width:100%' />"
        f"<div style='margin-top:16px'>{_metrics_html(metrics)}</div>"
        "</section>"
    )


def main() -> int:
    sections = [
        _section("Pairs trade (KO/PEP)", "KOPEP_", "#0f766e"),
        _section("Pairs trade (XOM/CVX)", "XOMCVX_", "#0f766e"),
        _section("Cross-sectional momentum (12-1m)", "momentum_", "#2563eb"),
        _section("Fama-French HML tilt", "factor_HML_", "#9333ea"),
    ]

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>Quant Stock Trading — Results</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #111; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ color: #666; margin-top: 0; }}
  table td {{ padding: 4px 12px; }}
  code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 3px; }}
</style></head><body>
  <h1>Quant Stock Trading — Backtest Results</h1>
  <p class='sub'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · all results are net of 5 bps slippage per leg and 25 bps/yr borrow on the short leg.</p>
  <p class='sub'><strong>Honest take:</strong> momentum was the only strategy with positive Sharpe (~0.35) over 2014-2024. Pairs trading on KO/PEP correctly refused to trade most periods (rolling cointegration broke down). FF5 value tilt lost money — value got crushed this decade.</p>
  {''.join(sections)}
  <footer style='color:#999;font-size:12px;margin-top:48px'>
    <a href='https://github.com/aidanlconnolly/quant-stock-trading'>github.com/aidanlconnolly/quant-stock-trading</a>
  </footer>
</body></html>"""

    out = RESULTS_DIR / "report.html"
    out.write_text(html)
    print(f"Wrote → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
