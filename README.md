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

Run reproducible experiment grid + report:

```bash
python -m src.run_experiments --data data/btcusd_90d.csv --out_dir results/runs --horizons 15,5 --entry_minutes 13,3 --models logreg,rf --calibrate_logreg --lookback_days 60 --test_days 14 --mc_sims 5000
python -m src.report --runs_dir results/runs --out_dir results/report --mc_sims 2000
python -m src.report --runs_dir results/runs --out_dir results/report --mc_sims 2000 --inject_readme
```

Run with confidence-threshold sweep:

```bash
python -m src.run_experiments --data data/btcusd_90d.csv --out_dir results/runs --horizons 15,5 --entry_minutes 13,3 --models logreg,rf --calibrate_logreg --min_conf_values 0.0,0.02,0.05,0.1 --lookback_days 60 --test_days 14 --mc_sims 5000
python -m src.report --runs_dir results/runs --out_dir results/report --mc_sims 2000
```

## Results Snapshot

Representative visuals (lightweight view) are generated into `results/report/snapshots/`.

15m summary metrics are listed in the **Key Findings** table below.
![Best run Monte Carlo paths](results/report/snapshots/best_run_monte_carlo_paths.png)

Detailed visuals (calibration, fold metrics, Monte Carlo distribution, equity/drawdown) stay in:
- `results/report/per_run/<run_id>/`

## Key Findings

<!-- AUTO_KEY_FINDINGS_START -->
Updated automatically from `results/report/summary_table.csv`.

### Performance Report (15m Focus)

| Setup | Net Quality | Hit Rate | Prob Error | Trade Participation |
|---|---:|---:|---:|---:|
| e13, LogReg Cal, conf 0.00 | 0.9731 | 0.9181 | 0.0655 | 1.0000 |
| e13, LogReg Cal, conf 0.05 | 0.9731 | 0.9181 | 0.0655 | 0.9959 |
| e13, RF NoCal, conf 0.00 | 0.9574 | 0.9203 | 0.0720 | 1.0000 |

### Quick Read

- Current top setup: 15-minute horizon, entry minute 13, logreg (calibrated), confidence threshold 0.0.
- AUC: higher is better at separating up candles from down candles.
- Brier: lower is better for probability quality.
<!-- AUTO_KEY_FINDINGS_END -->

## Portfolio Notes

This repository is intentionally public-safe.  
It demonstrates research process, engineering rigor, and evaluation discipline without publishing private strategy details, proprietary tuning decisions, or deployment logic.

## Disclaimer

This is a research project, not financial advice or a production trading system.
