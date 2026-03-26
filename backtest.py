"""
backtest.py
-----------
Simple regime-aware backtest for the Market Regime Detection System.

Strategy
--------
  - LONG  during Bull regime  (regime == 0)
  - CASH  during Bear and Sideways regimes (regime 1, 2)

Benchmark
---------
  - Buy-and-hold the same ticker over the same period.

Usage
-----
  python backtest.py
  python backtest.py --ticker AAPL --method hmm --start 2010-01-01
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from config import (
    DEFAULT_TICKER, DEFAULT_START, DEFAULT_END,
    INITIAL_CAPITAL, REGIME_LABELS, FIGURE_SIZE, PLOT_DPI,
)
from regime_model import build_features, classify_regimes
from utils.stats import portfolio_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regime-aware backtest: Long on Bull, Cash on Bear/Sideways",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker", default=DEFAULT_TICKER)
    p.add_argument("--start",  default=DEFAULT_START)
    p.add_argument("--end",    default=DEFAULT_END)
    p.add_argument("--method", default="kmeans", choices=["kmeans", "hmm"])
    p.add_argument("--export", default=None, metavar="FILE")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, regimes: np.ndarray) -> dict:
    """
    Simulate regime-aware strategy and buy-and-hold.

    Returns
    -------
    dict with DataFrames and summary metrics.
    """
    close = df["Close"].squeeze()
    daily_returns = close.pct_change().dropna()

    # Align to features window
    feat_len   = len(regimes)
    aligned_ret = daily_returns.iloc[-feat_len:]
    aligned_reg = pd.Series(regimes, index=aligned_ret.index)

    # Strategy: invest only during Bull (regime 0)
    strat_ret = aligned_ret.where(aligned_reg == 0, other=0.0)

    # Equity curves (starting from INITIAL_CAPITAL)
    strat_equity = INITIAL_CAPITAL * (1 + strat_ret).cumprod()
    bnh_equity   = INITIAL_CAPITAL * (1 + aligned_ret).cumprod()

    strat_metrics = portfolio_metrics(strat_ret)
    bnh_metrics   = portfolio_metrics(aligned_ret)

    return {
        "strat_equity":   strat_equity,
        "bnh_equity":     bnh_equity,
        "strat_ret":      strat_ret,
        "bnh_ret":        aligned_ret,
        "strat_metrics":  strat_metrics,
        "bnh_metrics":    bnh_metrics,
        "regimes":        aligned_reg,
    }


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def _print_comparison(strat: dict, bnh: dict, ticker: str, method: str) -> None:
    print(f"\n{'─'*62}")
    print(f"  Backtest Results  |  {ticker}  |  method={method}")
    print(f"{'─'*62}")
    print(f"  {'Metric':<24}  {'Strategy':>12}  {'Buy & Hold':>12}")
    print(f"  {'─'*24}  {'─'*12}  {'─'*12}")
    for key in strat:
        sv = strat[key]
        bv = bnh.get(key, "—")
        sv_str = f"{sv}%" if "%" in key else str(sv)
        bv_str = f"{bv}%" if "%" in key else str(bv)
        print(f"  {key:<24}  {sv_str:>12}  {bv_str:>12}")
    print(f"{'─'*62}\n")


def plot_backtest(result: dict, ticker: str, export_path: str | None = None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, facecolor="#0d1117",
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    # ── Panel 0: Equity curves ───────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor("#161b22")
    ax0.plot(result["strat_equity"].index, result["strat_equity"].values,
             color="#3fb950", linewidth=1.2, label="Regime Strategy (Long Bull)")
    ax0.plot(result["bnh_equity"].index,   result["bnh_equity"].values,
             color="#58a6ff", linewidth=1.0, linestyle="--", label="Buy & Hold")

    ax0.set_title(f"{ticker}  ·  Regime Backtest: Strategy vs Buy & Hold",
                  color="#e6edf3", fontsize=12, pad=10)
    ax0.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax0.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)
    ax0.tick_params(colors="#8b949e", labelbottom=False)
    for spine in ax0.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 1: Regime bar ──────────────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor("#161b22")
    regimes = result["regimes"]
    for rid, meta in REGIME_LABELS.items():
        mask = regimes == rid
        ax1.fill_between(regimes.index, 0, 1, where=mask,
                         color=meta["color"], alpha=0.6, label=meta["label"])

    ax1.set_yticks([])
    ax1.set_ylabel("Regime", color="#8b949e", fontsize=8)
    ax1.legend(loc="upper left", facecolor="#21262d", edgecolor="#30363d",
               labelcolor="#e6edf3", fontsize=8)
    ax1.tick_params(colors="#8b949e")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    plt.setp(ax1.get_xticklabels(), color="#8b949e", fontsize=7)

    if export_path:
        plt.savefig(export_path, dpi=PLOT_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✅  Backtest chart exported → {export_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"\n📥  Downloading {args.ticker} …")
    df = yf.download(args.ticker, start=args.start, end=args.end,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        print(f"❌  Download failed for {args.ticker}.")
        sys.exit(1)
    df = df.dropna()

    print("⚙️   Engineering features …")
    features = build_features(df)
    if features.empty:
        print("❌  Not enough data.")
        sys.exit(1)

    print(f"🔍  Classifying regimes (method={args.method}) …")
    regimes = classify_regimes(features, method=args.method)

    print("📊  Running backtest …")
    result = run_backtest(df, regimes)

    _print_comparison(result["strat_metrics"], result["bnh_metrics"],
                      ticker=args.ticker, method=args.method)

    if not args.no_plot:
        plot_backtest(result, ticker=args.ticker, export_path=args.export)


if __name__ == "__main__":
    main()
