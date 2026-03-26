"""
config.py
---------
Centralized configuration for the Market Regime Detection System.
All tunable parameters live here — no magic numbers scattered in the code.
"""

# ---------------------------------------------------------------------------
# Data defaults
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "SPY"
DEFAULT_START  = "2000-01-01"
DEFAULT_END    = None          # None  → today

# ---------------------------------------------------------------------------
# Feature-engineering windows
# ---------------------------------------------------------------------------
WINDOW_VOLATILITY   = 21   # trading-days (~1 month)
WINDOW_MA_SHORT     = 50
WINDOW_MA_LONG      = 200
WINDOW_RSI          = 14
WINDOW_MACD_FAST    = 12
WINDOW_MACD_SLOW    = 26
WINDOW_MACD_SIGNAL  = 9
WINDOW_BBAND        = 20
WINDOW_ATR          = 14

# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
N_CLUSTERS     = 3
RANDOM_STATE   = 42
HMM_ITERATIONS = 1000

# ---------------------------------------------------------------------------
# Regime labels & colours
# ---------------------------------------------------------------------------
REGIME_LABELS = {
    0: {"label": "Bull",     "color": "#2ecc71"},
    1: {"label": "Bear",     "color": "#e74c3c"},
    2: {"label": "Sideways", "color": "#3498db"},
}

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 10_000.0    # USD
TRADING_DAYS    = 252

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
PLOT_DPI    = 150
FIGURE_SIZE = (16, 8)
