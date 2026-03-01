from __future__ import annotations
import numpy as np
import pandas as pd

def floor_to_interval(dt: pd.Series, minutes: int) -> pd.Series:
    return dt.dt.floor(f"{minutes}min")

def safe_div(a, b, eps: float = 1e-12):
    return a / (b + eps)

def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())

def bootstrap_equity(
    returns: np.ndarray,
    n_sims: int = 10000,
    seed: int = 42,
    start_equity: float = 1.0
):
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n == 0:
        raise ValueError("No returns provided.")

    end_equity = np.empty(n_sims)
    mdd = np.empty(n_sims)

    for i in range(n_sims):
        sample = rng.choice(returns, size=n, replace=True)
        equity = start_equity * np.cumprod(1.0 + sample)
        end_equity[i] = equity[-1]
        mdd[i] = max_drawdown(equity)

    return end_equity, mdd