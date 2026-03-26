"""
main.py
-------
Entry point for the Market Regime Detection System.

Usage
-----
  python main.py                              # defaults: SPY, 2000-01-01, KMeans
  python main.py --ticker AAPL --method hmm
  python main.py --ticker QQQ --start 2010-01-01 --end 2023-12-31 --export chart.png
  python main.py --no-plot                    # print stats only
"""

import argparse
import sys

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from config import (
    DEFAULT_TICKER, DEFAULT_START, DEFAULT_END,
    REGIME_LABELS, FIGURE_SIZE, PLOT_DPI,
)
from regime_model import build_features, classify_regimes
from utils.stats import regime_statistics
from utils.indicators import rsi as compute_rsi, macd as compute_macd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Market Regime Detection System — detect Bull / Bear / Sideways regimes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker", default=DEFAULT_TICKER, help="Yahoo Finance ticker symbol")
    p.add_argument("--start",  default=DEFAULT_START,  help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",    default=DEFAULT_END,    help="End date (YYYY-MM-DD), default = today")
    p.add_argument(
        "--method", default="kmeans", choices=["kmeans", "hmm"],
        help="Classification method: kmeans | hmm",
    )
    p.add_argument("--export",  default=None, metavar="FILE",
                   help="Export chart to file (e.g. chart.png, report.pdf)")
    p.add_argument("--no-plot", action="store_true", help="Skip chart; print stats only")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_regimes(df: pd.DataFrame, regimes: pd.Series, ticker: str,
                 export_path: str | None = None) -> None:
    """
    3-panel chart:
      [0] Price with regime shading
      [1] RSI
      [2] MACD histogram
    """
    close  = df["Close"].squeeze()
    feat_idx = df.index[-len(regimes):]

    _rsi  = compute_rsi(close)
    _macd = compute_macd(close)

    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="#0d1117")
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.05)

    # ── Panel 0: Price ──────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor("#161b22")
    ax0.plot(close.index, close.values, color="#e6edf3", linewidth=0.9, label=ticker)

    price_min = float(close.min())
    price_max = float(close.max())

    for rid, meta in REGIME_LABELS.items():
        mask = regimes == rid
        ax0.fill_between(feat_idx, price_min, price_max,
                         where=mask, alpha=0.18, color=meta["color"])

    legend_handles = [
        mpatches.Patch(color=REGIME_LABELS[r]["color"], label=REGIME_LABELS[r]["label"])
        for r in REGIME_LABELS
    ]
    legend_handles.insert(0, plt.Line2D([0], [0], color="#e6edf3", lw=1.5, label=ticker))
    ax0.legend(handles=legend_handles, loc="upper left",
               facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9)

    ax0.set_title(f"{ticker}  ·  Market Regime Detection",
                  color="#e6edf3", fontsize=13, pad=10)
    ax0.tick_params(colors="#8b949e", labelbottom=False)
    ax0.set_xlim(close.index[0], close.index[-1])
    for spine in ax0.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 1: RSI ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.set_facecolor("#161b22")
    ax1.plot(_rsi.index, _rsi.values, color="#58a6ff", linewidth=0.8)
    ax1.axhline(70, color="#f85149", linewidth=0.6, linestyle="--", alpha=0.7)
    ax1.axhline(30, color="#3fb950", linewidth=0.6, linestyle="--", alpha=0.7)
    ax1.set_ylabel("RSI", color="#8b949e", fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.tick_params(colors="#8b949e", labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 2: MACD histogram ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.set_facecolor("#161b22")
    hist = _macd["histogram"]
    colors_hist = ["#3fb950" if v >= 0 else "#f85149" for v in hist.values]
    ax2.bar(hist.index, hist.values, color=colors_hist, width=1.5, alpha=0.8)
    ax2.axhline(0, color="#8b949e", linewidth=0.5)
    ax2.set_ylabel("MACD", color="#8b949e", fontsize=8)
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    plt.setp(ax2.get_xticklabels(), color="#8b949e", fontsize=7)

    if export_path:
        plt.savefig(export_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\n✅  Chart exported → {export_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Statistics table
# ---------------------------------------------------------------------------

def print_stats(df: pd.DataFrame, regimes: pd.Series, method: str, ticker: str) -> None:
    stats = regime_statistics(df, regimes)
    print(f"\n{'─'*60}")
    print(f"  Regime Statistics  |  {ticker}  |  method={method}")
    print(f"{'─'*60}")
    print(stats.to_string())
    print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Download ──────────────────────────────────────────────────────────
    print(f"\n📥  Downloading {args.ticker} from {args.start} …")
    df = yf.download(args.ticker, start=args.start, end=args.end, auto_adjust=True, progress=False)

    if df is None or df.empty:
        print(f"❌  Failed to download data for {args.ticker}. Check ticker or internet.")
        sys.exit(1)

    df = df.dropna()
    print(f"✅  {len(df)} trading days loaded.")

    # ── Features ──────────────────────────────────────────────────────────
    print("⚙️   Engineering features …")
    features = build_features(df)
    if features.empty:
        print("❌  Not enough data to compute features. Use a longer date range.")
        sys.exit(1)

    # ── Classification ────────────────────────────────────────────────────
    print(f"🔍  Classifying regimes (method={args.method}) …")
    regimes = classify_regimes(features, method=args.method)

    # ── Stats ─────────────────────────────────────────────────────────────
    print_stats(df, regimes, method=args.method, ticker=args.ticker)

    # ── Plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_regimes(df, regimes, ticker=args.ticker, export_path=args.export)


if __name__ == "__main__":
    main()
