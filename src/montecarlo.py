from __future__ import annotations
import numpy as np
from src.utils import bootstrap_equity

def run_bootstrap(oos_returns: np.ndarray, sims: int = 5000) -> dict:
    """
    Bootstrap resample the OOS returns to estimate distribution of outcomes.
    Returns summary stats for ending equity and max drawdown.
    """
    end_eq, mdd = bootstrap_equity(oos_returns, n_sims=sims)

    return {
        "sims": int(sims),
        "median_end": float(np.median(end_eq)),
        "p5_end": float(np.percentile(end_eq, 5)),
        "p95_end": float(np.percentile(end_eq, 95)),
        "median_mdd": float(np.median(mdd)),
        "p5_mdd": float(np.percentile(mdd, 5)),
        "p95_mdd": float(np.percentile(mdd, 95)),
    }