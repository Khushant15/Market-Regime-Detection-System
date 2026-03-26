"""
utils/stats.py
--------------
Regime statistics: per-regime return, volatility, Sharpe ratio,
average duration, and trade count.
"""

import numpy as np
import pandas as pd
from config import REGIME_LABELS, TRADING_DAYS


def regime_statistics(df: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
    """
    Compute per-regime performance metrics.

    Parameters
    ----------
    df      : Full price DataFrame (must contain 'Close').
    regimes : 1-D array of regime labels aligned to the *tail* of df
              (i.e., after feature-engineering NaN rows are dropped).

    Returns
    -------
    DataFrame indexed by regime label with columns:
        count, avg_daily_return, avg_volatility, annualised_return,
        sharpe_ratio, avg_duration_days
    """
    close = df["Close"].copy()
    returns_full = close.pct_change()

    # Align returns to the features window
    aligned_returns = returns_full.iloc[-len(regimes):].values
    aligned_close   = close.iloc[-len(regimes):].values

    rows = []
    for regime_id, meta in REGIME_LABELS.items():
        mask = regimes == regime_id
        if mask.sum() == 0:
            continue

        r = aligned_returns[mask]
        avg_ret  = float(np.nanmean(r))
        avg_vol  = float(np.nanstd(r))
        ann_ret  = avg_ret * TRADING_DAYS
        sharpe   = (avg_ret / avg_vol * np.sqrt(TRADING_DAYS)) if avg_vol > 0 else np.nan

        # Average consecutive-run length (duration in days)
        durations = _run_lengths(mask)
        avg_dur  = float(np.mean(durations)) if durations else 0.0

        rows.append({
            "Regime":            meta["label"],
            "Days":              int(mask.sum()),
            "Avg Daily Ret (%)": round(avg_ret * 100, 4),
            "Avg Volatility":    round(avg_vol, 6),
            "Ann. Return (%)":   round(ann_ret * 100, 2),
            "Sharpe Ratio":      round(sharpe, 3) if not np.isnan(sharpe) else "N/A",
            "Avg Duration (days)": round(avg_dur, 1),
        })

    return pd.DataFrame(rows).set_index("Regime")


def _run_lengths(mask: np.ndarray) -> list:
    """Return a list of lengths of consecutive True runs."""
    lengths = []
    count = 0
    for val in mask:
        if val:
            count += 1
        else:
            if count > 0:
                lengths.append(count)
                count = 0
    if count > 0:
        lengths.append(count)
    return lengths


def portfolio_metrics(returns: pd.Series, risk_free: float = 0.0) -> dict:
    """
    Compute overall portfolio metrics.

    Parameters
    ----------
    returns    : daily portfolio returns (Series).
    risk_free  : annual risk-free rate (default 0).

    Returns
    -------
    dict with keys: total_return, cagr, sharpe, max_drawdown, calmar
    """
    cum   = (1 + returns).cumprod()
    total = float(cum.iloc[-1] - 1)
    n     = len(returns) / TRADING_DAYS
    cagr  = float((cum.iloc[-1]) ** (1 / n) - 1) if n > 0 else 0.0

    excess = returns - risk_free / TRADING_DAYS
    sharpe = float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)) if excess.std() > 0 else np.nan

    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd   = float(drawdown.min())

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Total Return (%)": round(total * 100, 2),
        "CAGR (%)":         round(cagr * 100, 2),
        "Sharpe Ratio":     round(sharpe, 3),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Calmar Ratio":     round(calmar, 3) if not np.isnan(calmar) else "N/A",
    }
