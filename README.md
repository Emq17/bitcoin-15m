# Candle Close Predictor

Research-focused machine learning pipeline for short-horizon crypto candle classification.

This project explores whether intraperiod market information can be used to estimate directional outcomes at candle close under strict out-of-sample validation.

## Why This Project

Most trading experiments fail because of leakage, weak evaluation design, or overfitting to noise.  
This repository is structured to prioritize:

- Temporal integrity (features only use information available at decision time)
- Walk-forward validation (rolling train/test windows)
- Honest out-of-sample reporting
- Repeatable experimentation and risk-aware analysis

## Scope

- Instruments: BTC-USD (1-minute OHLCV)
- Target horizons: 5-minute and 15-minute candles
- Task type: binary classification (up-close vs down-close)
- Modeling: baseline linear and tree-based classifiers

## Project Structure

```text
src/
  dataset.py       # data loading, normalization, bucket/label prep
  features.py      # feature frame generation at configurable entry minute
  models.py        # model factory + probability inference
  walkforward.py   # rolling OOS evaluation and reporting
  montecarlo.py    # bootstrap risk simulation for trade-return samples
  download_data.py # market data downloader
  utils.py         # shared helpers
```

## Methodology (High Level)

1. Load and normalize 1-minute OHLCV data.
2. Re-bucket into target horizon candles.
3. Snapshot each candle at a chosen entry minute.
4. Build feature vectors from information available up to that entry point.
5. Train/test via rolling walk-forward windows.
6. Evaluate OOS quality metrics and probability behavior.
7. Run bootstrap Monte Carlo on sampled trade outcomes for risk context.

## Current State

- End-to-end pipeline implemented and modularized
- Walk-forward experimentation working for both 5m and 15m horizons
- Confidence-filtered trade simulation and Monte Carlo analysis integrated
- Baseline model comparisons implemented for iterative research

## Roadmap

- Expand historical coverage and regime diversity in datasets
- Add stronger model diagnostics and probability calibration checks
- Introduce experiment tracking and run metadata logging
- Extend feature families and regime-aware filtering
- Add robustness tests across alternate assets/time windows

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download data:

```bash
python -m src.download_data --days 90 --source coinbase --product BTC-USD --out data/btcusd_90d.csv
```

Run walk-forward (example):

```bash
python -m src.walkforward --data data/btcusd_90d.csv --horizon 15 --entry_minute 13 --model logreg --calibrate --lookback_days 60 --test_days 14 --mc_sims 5000
python -m src.walkforward --data data/btcusd_90d.csv --horizon 5 --entry_minute 3 --model rf --lookback_days 60 --test_days 14 --mc_sims 5000
```

## Portfolio Notes

This repository is intentionally public-safe.  
It demonstrates research process, engineering rigor, and evaluation discipline without publishing private strategy details, proprietary tuning decisions, or deployment logic.

## Disclaimer

This is a research project, not financial advice or a production trading system.
