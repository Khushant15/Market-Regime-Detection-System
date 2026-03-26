"""
dashboard.py
------------
Interactive Plotly dashboard for the Market Regime Detection System.

Launches a browser tab showing:
  · Candlestick chart with regime background shading
  · RSI indicator subplot
  · MACD histogram subplot
  · Regime distribution pie chart
  · Regime statistics table

Usage
-----
  python dashboard.py
  python dashboard.py --ticker AAPL --method hmm --start 2015-01-01
"""

import argparse
import sys

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    DEFAULT_TICKER, DEFAULT_START, DEFAULT_END,
    REGIME_LABELS,
)
from regime_model import build_features, classify_regimes
from utils.stats import regime_statistics
from utils.indicators import rsi as compute_rsi, macd as compute_macd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive Plotly dashboard for market regime analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker", default=DEFAULT_TICKER)
    p.add_argument("--start",  default=DEFAULT_START)
    p.add_argument("--end",    default=DEFAULT_END)
    p.add_argument("--method", default="kmeans", choices=["kmeans", "hmm"])
    p.add_argument("--port",   default=8050, type=int, help="Local port for the dashboard")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

DARK_BG     = "#0d1117"
DARK_PANEL  = "#161b22"
TEXT_COLOR  = "#e6edf3"
GRID_COLOR  = "#21262d"


def build_dashboard(df: pd.DataFrame, regimes: np.ndarray, ticker: str) -> go.Figure:
    close = df["Close"].squeeze()
    feat_idx = df.index[-len(regimes):]

    _rsi  = compute_rsi(close)
    _macd = compute_macd(close)

    # ── Layout: 4 rows ────────────────────────────────────────────────────
    fig = make_subplots(
        rows=4, cols=2,
        column_widths=[0.72, 0.28],
        row_heights=[0.45, 0.18, 0.18, 0.19],
        specs=[
            [{"rowspan": 1}, {"rowspan": 2, "type": "pie"}],
            [{"rowspan": 1}, None],
            [{"rowspan": 1}, {"rowspan": 2, "type": "table"}],
            [{"rowspan": 1}, None],
        ],
        subplot_titles=[
            f"{ticker} — Price & Regimes", "",
            "RSI (14)", "",
            "MACD Histogram", "Regime Statistics",
            "", "",
        ],
        shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.04,
    )

    # ── Candlestick ────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(),
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=close,
        name=ticker,
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
        showlegend=False,
    ), row=1, col=1)

    # Regime shading via shapes
    price_min = float(close.min()) * 0.95
    price_max = float(close.max()) * 1.05
    _add_regime_shapes(fig, feat_idx, regimes, y0=price_min, y1=price_max, yref="y")

    # ── RSI ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=_rsi.index, y=_rsi.values,
                              line=dict(color="#58a6ff", width=1),
                              name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_width=0.8, line_dash="dot", line_color="#f85149", row=2, col=1)
    fig.add_hline(y=30, line_width=0.8, line_dash="dot", line_color="#3fb950", row=2, col=1)

    # ── MACD ──────────────────────────────────────────────────────────────
    hist = _macd["histogram"]
    bar_colors = ["#3fb950" if v >= 0 else "#f85149" for v in hist.values]
    fig.add_trace(go.Bar(x=hist.index, y=hist.values,
                          marker_color=bar_colors, name="MACD Hist"), row=3, col=1)

    # ── Regime duration placeholders (row 4 = spacer) ─────────────────────
    # Regime label strip
    reg_vals = pd.Series(regimes, index=feat_idx)
    for rid, meta in REGIME_LABELS.items():
        mask = reg_vals == rid
        fig.add_trace(go.Scatter(
            x=feat_idx[mask], y=[0.5] * mask.sum(),
            mode="markers",
            marker=dict(color=meta["color"], size=4, symbol="square"),
            name=meta["label"],
            legendgroup=meta["label"],
            showlegend=True,
        ), row=4, col=1)

    fig.update_yaxes(title_text="Regime", tickvals=[], row=4, col=1)

    # ── Pie chart ─────────────────────────────────────────────────────────
    labels = [REGIME_LABELS[r]["label"] for r in REGIME_LABELS]
    counts = [int((regimes == r).sum()) for r in REGIME_LABELS]
    colors = [REGIME_LABELS[r]["color"] for r in REGIME_LABELS]
    fig.add_trace(go.Pie(
        labels=labels, values=counts,
        marker_colors=colors,
        textfont_color=TEXT_COLOR,
        hole=0.4,
        name="",
    ), row=1, col=2)

    # ── Stats table ───────────────────────────────────────────────────────
    stats = regime_statistics(df, regimes)
    fig.add_trace(go.Table(
        header=dict(
            values=["Regime"] + list(stats.columns),
            fill_color="#21262d",
            font=dict(color=TEXT_COLOR, size=11),
            align="left",
        ),
        cells=dict(
            values=[stats.index.tolist()] + [stats[c].tolist() for c in stats.columns],
            fill_color=DARK_PANEL,
            font=dict(color="#8b949e", size=10),
            align="left",
        ),
    ), row=3, col=2)

    # ── Global styling ────────────────────────────────────────────────────
    fig.update_layout(
        height=900,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor="#21262d", bordercolor="#30363d",
            font=dict(color=TEXT_COLOR),
        ),
        title=dict(
            text=f"<b>Market Regime Detection Dashboard</b>   ·   {ticker}",
            font=dict(size=16, color=TEXT_COLOR),
            x=0.02,
        ),
        margin=dict(t=60, l=10, r=10, b=10),
    )

    _style_axes(fig)
    return fig


def _add_regime_shapes(fig, feat_idx, regimes, y0, y1, yref="y"):
    """Add filled rectangles for each consecutive regime run."""
    reg_series = pd.Series(regimes, index=feat_idx)
    prev_regime = None
    start_date  = None

    for date, regime in reg_series.items():
        if regime != prev_regime:
            if prev_regime is not None:
                _rect(fig, start_date, date, y0, y1, REGIME_LABELS[prev_regime]["color"], yref)
            start_date  = date
            prev_regime = regime

    if prev_regime is not None:
        _rect(fig, start_date, feat_idx[-1], y0, y1, REGIME_LABELS[prev_regime]["color"], yref)


def _rect(fig, x0, x1, y0, y1, color, yref):
    fig.add_shape(
        type="rect",
        x0=x0, x1=x1,
        y0=y0, y1=y1,
        yref=yref,
        fillcolor=color,
        opacity=0.12,
        line_width=0,
        layer="below",
    )


def _style_axes(fig):
    axis_style = dict(
        gridcolor=GRID_COLOR,
        linecolor="#30363d",
        zerolinecolor=GRID_COLOR,
        tickfont=dict(color="#8b949e", size=9),
    )
    for k in fig.layout:
        if k.startswith("xaxis") or k.startswith("yaxis"):
            fig.layout[k].update(axis_style)


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

    print("🚀  Building dashboard …")
    fig = build_dashboard(df, regimes, ticker=args.ticker)

    print(f"🌐  Opening dashboard in browser …")
    fig.show()


if __name__ == "__main__":
    main()
