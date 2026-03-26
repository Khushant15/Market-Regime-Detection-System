# Market Regime Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/ML-KMeans%20%7C%20HMM-orange" />
  <img src="https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Plotly-purple" />
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" />
</p>

A professional-grade Python toolkit for **detecting and analyzing market regimes** — Bull 🟢, Bear 🔴, and Sideways 🔵 — using machine learning on historical stock data.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Rich Feature Engineering** | Returns, Volatility, MA Ratio, RSI, MACD, Bollinger Width, ATR |
| **Two Classification Methods** | KMeans (default) · Hidden Markov Model (HMM) |
| **Per-Regime Statistics** | Count, Avg Return, Annualised Return, Sharpe Ratio, Avg Duration |
| **Regime-Aware Backtest** | Long-Bull / Cash-otherwise vs. Buy & Hold benchmark |
| **Interactive Dashboard** | Plotly dashboard: Candlestick + RSI + MACD + Pie chart + Stats table |
| **CLI Interface** | Custom ticker, date range, method, PNG/PDF export |
| **Dark-Themed Charts** | Professional matplotlib output with 3-panel layout |

---

## 🗂️ Project Structure

```
Market-Regime-Detection-System/
│
├── main.py              # Entry point — regime chart + stats (CLI)
├── backtest.py          # Regime-aware backtest vs buy & hold (CLI)
├── dashboard.py         # Interactive Plotly dashboard (CLI)
├── regime_model.py      # Feature engineering + KMeans / HMM classification
├── config.py            # Centralized configuration (windows, colors, defaults)
│
├── utils/
│   ├── __init__.py
│   ├── indicators.py    # RSI, MACD, Bollinger Width, ATR, Rolling Sharpe
│   └── stats.py         # Regime statistics, portfolio metrics
│
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Market-Regime-Detection-System.git
cd Market-Regime-Detection-System
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the regime chart (default: SPY, KMeans)
```bash
python main.py
```

---

## 🖥️ CLI Reference

### `main.py` — Regime Detection Chart

```
python main.py [OPTIONS]

Options:
  --ticker TEXT      Yahoo Finance ticker  [default: SPY]
  --start  DATE      Start date YYYY-MM-DD [default: 2000-01-01]
  --end    DATE      End date YYYY-MM-DD   [default: today]
  --method TEXT      kmeans | hmm          [default: kmeans]
  --export FILE      Export chart to PNG/PDF
  --no-plot          Print stats only, skip chart
```

**Examples:**
```bash
# Default SPY with KMeans
python main.py

# Apple stock, HMM, custom range
python main.py --ticker AAPL --method hmm --start 2015-01-01 --end 2023-12-31

# Export chart to file without displaying
python main.py --ticker QQQ --export qqq_regimes.png --no-plot
```

### `backtest.py` — Regime-Aware Backtest

```bash
python backtest.py
python backtest.py --ticker MSFT --method hmm --export backtest.png
```

Strategy: **Long during Bull regime, hold Cash during Bear / Sideways**.

Sample output:
```
──────────────────────────────────────────────────────
  Backtest Results  |  SPY  |  method=kmeans
──────────────────────────────────────────────────────
  Metric                    Strategy    Buy & Hold
  ────────────────────────  ────────────  ────────────
  Total Return (%)            +312.4%      +498.2%
  CAGR (%)                     +6.4%        +8.1%
  Sharpe Ratio                  1.21         0.54
  Max Drawdown (%)             -17.3%       -56.8%
  Calmar Ratio                  0.37         0.14
──────────────────────────────────────────────────────
```

### `dashboard.py` — Interactive Plotly Dashboard

```bash
python dashboard.py
python dashboard.py --ticker TSLA --method hmm
```

Opens a browser tab with a fully interactive 4-panel dashboard.

---

## ⚙️ Configuration

All tunable parameters are in [`config.py`](config.py):

```python
DEFAULT_TICKER    = "SPY"
DEFAULT_START     = "2000-01-01"
WINDOW_VOLATILITY = 21     # rolling window in trading days
WINDOW_RSI        = 14
N_CLUSTERS        = 3
HMM_ITERATIONS    = 1000
INITIAL_CAPITAL   = 10_000.0
```

---

## 🔬 Methodology

```
Raw OHLCV Data (yfinance)
        │
        ▼
Feature Engineering (regime_model.py + utils/indicators.py)
  · Daily returns         · RSI (14)
  · Rolling volatility    · MACD histogram
  · MA ratio (50/200)     · Bollinger Band Width
  · ATR %
        │
        ▼
StandardScaler normalisation
        │
        ├─── KMeans (n=3)  ──┐
        └─── Gaussian HMM ──►  Cluster → Regime mapping
                               (sorted by mean return)
                               0 = Bull  /  1 = Bear  /  2 = Sideways
        │
        ▼
Stats + Visualisation + Backtest
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Commit changes**: `git commit -m "feat: add your feature"`
4. **Push**: `git push origin feature/your-feature-name`
5. **Open a Pull Request** — describe what you changed and why

### Ideas for future contributions
- [ ] Add more tickers (multi-asset regime comparison)
- [ ] Add Kalman filter as a third classification method
- [ ] Add regime transition probability matrix (HMM)
- [ ] Add walk-forward validation / expanding window backtest
- [ ] Add risk-parity position sizing in backtest
- [ ] Export backtest results to CSV / Excel

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

**Original project by:** [tubakhxn](https://github.com/tubakhxn)  
**Enhanced & contributed by:** [Khushant15](https://github.com/Khushant15)
