from __future__ import annotations
import numpy as np
import pandas as pd

from src.dataset import add_bucket
from src.utils import safe_div

def build_feature_frame(df_1m: pd.DataFrame, horizon_min: int, entry_minute: int) -> pd.DataFrame:
    """
    One row per horizon candle.
    Features are computed using information available up to `entry_minute` inside each candle.

    Output columns:
      bucket, y_green, <features...>
    """
    df = add_bucket(df_1m, horizon_min)

    # Final OHLCV + label per bucket
    g = df.groupby("bucket", sort=True)
    final = g.agg(
        o=("open", "first"),
        h=("high", "max"),
        l=("low", "min"),
        c=("close", "last"),
        v=("volume", "sum"),
    ).reset_index()
    final["y_green"] = (final["c"] > final["o"]).astype(int)

    # Snapshot at entry_minute
    entry = df[df["minute_in_bucket"] == entry_minute][
        ["bucket", "open", "high", "low", "close", "volume"]
    ].copy()
    entry = entry.rename(columns={
        "open": "e_open",
        "high": "e_high",
        "low": "e_low",
        "close": "e_close",
        "volume": "e_vol",
    })

    m = final.merge(entry, on="bucket", how="inner")

    # Pre-entry mean volume (adaptive)
    lookback = 3 if horizon_min == 5 else 5
    pre_lo = max(entry_minute - lookback, 0)

    if entry_minute - 1 >= pre_lo:
        pre = df[df["minute_in_bucket"].between(pre_lo, entry_minute - 1)]
        pre_mean = pre.groupby("bucket")["volume"].mean().rename("pre_mean_vol").reset_index()
        m = m.merge(pre_mean, on="bucket", how="left")
    else:
        m["pre_mean_vol"] = np.nan

    # ---- Intrabucket feature set (entry snapshot only) ----
    m["ret_from_open"] = (m["e_close"] - m["o"]) / m["o"]
    m["range_so_far"] = (m["e_high"] - m["e_low"]) / m["o"]
    m["dist_to_ehigh"] = (m["e_high"] - m["e_close"]) / m["o"]
    m["dist_to_elow"] = (m["e_close"] - m["e_low"]) / m["o"]
    m["vol_burst"] = safe_div(m["e_vol"], m["pre_mean_vol"].fillna(0.0))

    # Chop score: sign flips in 1m returns up to entry
    df = df.copy()
    df["ret_1m"] = df["close"].pct_change().fillna(0.0)
    df_entry = df[df["minute_in_bucket"] <= entry_minute].copy()

    def chop_score(x: pd.Series) -> float:
        s = np.sign(x.values)
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i - 1]
        return float(np.sum(s[1:] * s[:-1] < 0))

    chop = df_entry.groupby("bucket")["ret_1m"].apply(chop_score).rename("chop").reset_index()
    m = m.merge(chop, on="bucket", how="left")
    m["chop"] = m["chop"].fillna(0.0)

    # Time remaining fraction
    total_minutes = horizon_min - 1
    m["t_remaining"] = (total_minutes - entry_minute) / max(total_minutes, 1)

    # ---- Regime + momentum features from prior completed buckets only ----
    m = m.sort_values("bucket").reset_index(drop=True)

    # Prior completed bucket return/range context
    m["bucket_ret"] = (m["c"] - m["o"]) / m["o"]
    m["bucket_range"] = (m["h"] - m["l"]) / m["o"]
    m["prev_green"] = (m["bucket_ret"].shift(1) > 0).astype(float)
    m["bucket_ret_lag1"] = m["bucket_ret"].shift(1)
    m["bucket_ret_lag2"] = m["bucket_ret"].shift(2)
    m["bucket_range_lag1"] = m["bucket_range"].shift(1)

    # Prior bucket momentum aggregates
    m["mom3"] = m["bucket_ret"].shift(1).rolling(3, min_periods=3).mean()
    m["mom6"] = m["bucket_ret"].shift(1).rolling(6, min_periods=6).mean()

    # Higher timeframe trend slope via linear-regression slope of prior closes
    def _slope(vals: np.ndarray) -> float:
        x = np.arange(len(vals), dtype=float)
        x = x - x.mean()
        y = vals.astype(float)
        y = y - y.mean()
        den = np.dot(x, x)
        if den <= 0:
            return 0.0
        return float(np.dot(x, y) / den)

    known_close = m["c"].shift(1)
    m["trend_slope_4"] = known_close.rolling(4, min_periods=4).apply(_slope, raw=True)
    m["trend_slope_8"] = known_close.rolling(8, min_periods=8).apply(_slope, raw=True)
    m["dist_from_trend_8"] = safe_div(
        known_close - known_close.rolling(8, min_periods=8).mean(),
        known_close.rolling(8, min_periods=8).mean().fillna(0.0),
    )

    # RSI-style momentum from prior completed bucket closes
    diff = known_close.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = safe_div(avg_gain, avg_loss)
    m["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # Volatility regime from prior bucket returns
    known_ret = m["bucket_ret"].shift(1)
    m["vol_std_6"] = known_ret.rolling(6, min_periods=6).std()
    m["vol_std_24"] = known_ret.rolling(24, min_periods=24).std()
    m["vol_regime"] = safe_div(m["vol_std_6"], m["vol_std_24"])

    # Range compression/expansion regime
    prior_range = m["bucket_range"].shift(1)
    range_base = prior_range.rolling(12, min_periods=12).mean()
    m["range_regime"] = safe_div(prior_range, range_base)
    m["range_expanding"] = (m["range_regime"] > 1.2).astype(float)
    m["range_compressing"] = (m["range_regime"] < 0.8).astype(float)

    keep = [
        "bucket", "y_green",
        "ret_from_open", "range_so_far", "dist_to_ehigh", "dist_to_elow",
        "vol_burst", "chop", "t_remaining",
        "prev_green", "bucket_ret_lag1", "bucket_ret_lag2", "bucket_range_lag1",
        "mom3", "mom6",
        "trend_slope_4", "trend_slope_8", "dist_from_trend_8",
        "rsi_14",
        "vol_std_6", "vol_std_24", "vol_regime",
        "range_regime", "range_expanding", "range_compressing",
    ]
    return m[keep].copy()
