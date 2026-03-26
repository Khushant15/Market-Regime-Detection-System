"""
utils/indicators.py
-------------------
Standalone, reusable technical indicator functions.
All functions accept a pandas Series / DataFrame and return a Series.
"""

import pandas as pd
import numpy as np


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = close.diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up   = up.rolling(window=window, min_periods=window).mean()
    avg_down = down.rolling(window=window, min_periods=window).mean()
    rs = avg_up / avg_down
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.

    Returns
    -------
    DataFrame with columns: ['macd', 'signal', 'histogram']
    """
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": histogram,
    })


def bollinger_bandwidth(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger Band Width = (Upper - Lower) / Middle.
    Measures relative volatility; spikes signal regime transitions.
    """
    mid   = close.rolling(window=window).mean()
    std   = close.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return (upper - lower) / mid


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range — a pure volatility measure not affected by gaps.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def rolling_sharpe(returns: pd.Series, window: int = 252, risk_free: float = 0.0) -> pd.Series:
    """Annualised rolling Sharpe ratio."""
    excess = returns - risk_free / 252
    mean   = excess.rolling(window=window).mean()
    std    = excess.rolling(window=window).std()
    return (mean / std) * np.sqrt(252)
