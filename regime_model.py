"""
regime_model.py
---------------
Feature engineering and regime classification for the Market Regime Detection System.

Supported methods
-----------------
- 'kmeans'  : KMeans clustering (default, no extra dependencies)
- 'hmm'     : Gaussian Hidden Markov Model via hmmlearn
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import (
    WINDOW_VOLATILITY, WINDOW_MA_SHORT, WINDOW_MA_LONG,
    WINDOW_RSI, WINDOW_MACD_FAST, WINDOW_MACD_SLOW, WINDOW_MACD_SIGNAL,
    WINDOW_BBAND, WINDOW_ATR,
    N_CLUSTERS, RANDOM_STATE, HMM_ITERATIONS,
    REGIME_LABELS,
)
from utils.indicators import rsi, macd, bollinger_bandwidth, atr


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rich feature matrix from OHLCV price data.

    Features
    --------
    returns, volatility, ma_ratio (50/200), rsi, macd_hist,
    bb_width, atr_pct

    Parameters
    ----------
    df : DataFrame with at least columns ['Open', 'High', 'Low', 'Close', 'Volume'].

    Returns
    -------
    DataFrame of features (NaN rows dropped).
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    feat = pd.DataFrame(index=df.index)

    # 1. Daily returns
    feat["returns"] = close.pct_change()

    # 2. Rolling volatility (std of returns)
    feat["volatility"] = feat["returns"].rolling(window=WINDOW_VOLATILITY).std()

    # 3. MA ratio (trend signal: > 1 → bullish)
    ma_short = close.rolling(window=WINDOW_MA_SHORT).mean()
    ma_long  = close.rolling(window=WINDOW_MA_LONG).mean()
    feat["ma_ratio"] = ma_short / ma_long

    # 4. RSI
    feat["rsi"] = rsi(close, window=WINDOW_RSI)

    # 5. MACD histogram (momentum)
    _macd = macd(close, fast=WINDOW_MACD_FAST, slow=WINDOW_MACD_SLOW, signal=WINDOW_MACD_SIGNAL)
    feat["macd_hist"] = _macd["histogram"]

    # 6. Bollinger Band Width (volatility regime)
    feat["bb_width"] = bollinger_bandwidth(close, window=WINDOW_BBAND)

    # 7. ATR % (normalise ATR by price for cross-ticker comparison)
    _atr = atr(high, low, close, window=WINDOW_ATR)
    feat["atr_pct"] = _atr / close

    return feat.dropna()


# ---------------------------------------------------------------------------
# Regime Classification
# ---------------------------------------------------------------------------

FEATURE_COLS = ["returns", "volatility", "rsi", "macd_hist", "bb_width", "atr_pct"]


def classify_regimes(features: pd.DataFrame, method: str = "kmeans") -> np.ndarray:
    """
    Classify market regimes using KMeans or HMM.

    Parameters
    ----------
    features : Feature DataFrame from build_features().
    method   : 'kmeans' (default) or 'hmm'.

    Returns
    -------
    1-D NumPy array of regime integers (0 = Bull, 1 = Bear, 2 = Sideways),
    aligned to features.index.
    """
    X = features[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "hmm":
        raw_labels = _hmm_classify(X_scaled)
    else:
        raw_labels = _kmeans_classify(X_scaled)

    return _map_to_regimes(raw_labels, features)


def _kmeans_classify(X: np.ndarray) -> np.ndarray:
    """Fit KMeans and return raw cluster labels."""
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    return km.fit_predict(X)


def _hmm_classify(X: np.ndarray) -> np.ndarray:
    """Fit a Gaussian HMM and return raw state labels."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise ImportError(
            "hmmlearn is required for the HMM method.\n"
            "Install it with:  pip install hmmlearn"
        ) from exc

    model = GaussianHMM(
        n_components=N_CLUSTERS,
        covariance_type="full",
        n_iter=HMM_ITERATIONS,
        random_state=RANDOM_STATE,
    )
    model.fit(X)
    return model.predict(X)


def _map_to_regimes(raw_labels: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """
    Map arbitrary cluster IDs → (0=Bull, 1=Bear, 2=Sideways) by ranking
    mean daily returns of each cluster.
    Tie-breaking: highest volatility cluster among equal-return clusters → Bear.
    """
    df_tmp = pd.DataFrame({
        "label":      raw_labels,
        "returns":    features["returns"].values,
        "volatility": features["volatility"].values,
    })
    cluster_stats = df_tmp.groupby("label").agg(
        mean_ret=("returns",    "mean"),
        mean_vol=("volatility", "mean"),
    )

    sorted_by_ret = cluster_stats.sort_values("mean_ret", ascending=False)
    cluster_ids   = sorted_by_ret.index.tolist()   # [best_ret, ..., worst_ret]

    bull     = cluster_ids[0]
    bear     = cluster_ids[-1]
    sideways = cluster_ids[1]  # middle

    mapping = {bull: 0, bear: 1, sideways: 2}
    return np.array([mapping[c] for c in raw_labels])
