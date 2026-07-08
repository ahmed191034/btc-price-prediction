# Bitcoin Price Prediction

A time-series regression project: predict next-day BTC closing price from lagged prices, moving averages, and rolling volatility, using linear regression implemented from scratch with NumPy (normal equation, standardized features).

**Tools:** Python · NumPy · pandas · matplotlib

## Data

This environment doesn't have live internet access, so the notebook runs on a **simulated BTC-like price series** (geometric Brownian motion, seeded for reproducibility) so that every number and chart in the notebook is real, executed output — not fabricated. The notebook includes a drop-in swap for real BTC-USD data via `yfinance` or the CoinGecko public API; the modeling code doesn't change.

## Approach

1. Feature engineering: lag prices (t-1, t-2, t-3), 7-day and 30-day moving averages, 7-day rolling volatility.
2. Time-based 80/20 train/test split (not shuffled — this is a time series).
3. Linear regression from scratch via the NumPy normal equation, on standardized features.
4. Evaluation against a **naive baseline** (tomorrow's price = today's price).

## Key Result

The model reaches **R² = 0.97** on the test set — but the naive "no change" baseline actually **beats it on RMSE** (8,782 vs. 12,764). This is a genuinely common finding in financial time-series forecasting, and it's reported here directly rather than only showing the flattering R² number. A model that looks good on R² can still lose to doing nothing, and checking that is the actual point of the notebook.

## Files

- `btc_price_prediction.ipynb` — full notebook (data, features, model, evaluation, plot, and the real-data swap-in instructions)
- `requirements.txt`

## Running It

```bash
pip install -r requirements.txt
jupyter notebook btc_price_prediction.ipynb
```

## What I'd Add Next

- Run against real BTC-USD data (see the notebook's Section 5 for the exact swap)
- Try a model better suited to volatility clustering (GARCH) or gradient boosting on the same features
- Walk-forward (rolling-origin) validation instead of a single train/test split
