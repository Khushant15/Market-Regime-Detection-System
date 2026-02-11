
# Market Regime Detection System

**What is this project about?**

This project is a Python-based tool for detecting and visualizing market regimes—Bull, Bear, and Sideways—using historical stock data (e.g., SPY). It leverages machine learning and technical indicators to classify different market environments, helping traders and analysts better understand market dynamics.

## Methodology

1. **Data Acquisition**: Downloads historical price data using yfinance.
2. **Feature Engineering**: Calculates rolling volatility, moving averages, and momentum indicators (e.g., RSI).
3. **Regime Classification**: Uses KMeans clustering (or optionally Hidden Markov Model) to classify regimes based on returns and volatility patterns.
4. **Visualization**: Plots the stock price with background shading for each detected regime and a clear legend.

## Structure
- `main.py`: Entry point, visualization
- `regime_model.py`: Feature engineering, regime classification
- `requirements.txt`: Dependencies
- `README.md`: Methodology, usage, and project info

## Usage
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the project:
    ```bash
    python main.py
    ```

## Notes
- Modular, clean, production-style code
- Professional visualization

---
**Developer/Creator:** tubakhxn
