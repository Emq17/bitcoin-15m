from __future__ import annotations
import pandas as pd

def _detect_time_col(df: pd.DataFrame) -> str:
    for c in ["open_time", "timestamp", "time", "date", "datetime"]:
        if c in df.columns:
            return c
    raise ValueError(f"No recognized time column found. Columns: {list(df.columns)}")

def load_1m_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    time_col = _detect_time_col(df)
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # Normalize to a standard name we use everywhere
    df = df.rename(columns={time_col: "open_time"})

    # Ensure required columns exist
    required = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.sort_values("open_time").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    return df

def add_bucket(df_1m: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """
    Adds:
      - bucket (floored timestamp to horizon)
      - minute_in_bucket (0..horizon-1)
    """
    df = df_1m.copy()
    df["bucket"] = df["open_time"].dt.floor(f"{horizon_min}min")
    df["minute_in_bucket"] = ((df["open_time"] - df["bucket"]) / pd.Timedelta(minutes=1)).astype(int)
    return df

def build_labels(df_1m: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """
    One row per horizon candle with final OHLCV + label:
      y_green = 1 if close > open else 0
    """
    df = add_bucket(df_1m, horizon_min)
    g = df.groupby("bucket", sort=True)

    final = g.agg(
        o=("open", "first"),
        h=("high", "max"),
        l=("low", "min"),
        c=("close", "last"),
        v=("volume", "sum"),
    ).reset_index()

    final["y_green"] = (final["c"] > final["o"]).astype(int)
    return final