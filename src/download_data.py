from __future__ import annotations
import argparse
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from tqdm import tqdm


# ----------------------------
# Helpers
# ----------------------------
def to_iso(dt: datetime) -> str:
    # Coinbase expects RFC3339-ish timestamps
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------
# COINBASE (recommended)
# ----------------------------
COINBASE_BASE = "https://api.exchange.coinbase.com"

def fetch_coinbase_candles(
    product: str,
    granularity_sec: int,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Coinbase candles:
      GET /products/<product>/candles?granularity=60&start=...&end=...
    Returns rows: [time, low, high, open, close, volume]
    """
    url = f"{COINBASE_BASE}/products/{product}/candles"
    params = {
        "granularity": int(granularity_sec),
        "start": to_iso(start),
        "end": to_iso(end),
    }
    r = requests.get(url, params=params, timeout=20, headers={"User-Agent": "candle-close-predictor"})
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop(columns=["time"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    return df.sort_values("open_time").reset_index(drop=True)

def download_coinbase_1m(product: str, days: int) -> pd.DataFrame:
    """
    Coinbase limits:
    - Candles endpoint returns up to ~300 rows per request depending on granularity.
    We chunk time windows to accumulate full history.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    # For 1m granularity, 300 candles = 300 minutes = 5 hours.
    step = timedelta(hours=5)

    cur = start
    out = []
    pbar = tqdm(total=int(((end - start) / step) + 1), desc=f"Downloading {product} 1m candles (Coinbase)")

    while cur < end:
        nxt = min(cur + step, end)
        df = fetch_coinbase_candles(product, 60, cur, nxt)
        if not df.empty:
            out.append(df)
        cur = nxt
        pbar.update(1)
        time.sleep(0.12)  # be polite to the API

    pbar.close()

    if not out:
        return pd.DataFrame()

    res = (
        pd.concat(out, ignore_index=True)
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    return res


# ----------------------------
# BINANCE (optional, may fail)
# ----------------------------
BINANCE_BASE = "https://api.binance.com"

def fetch_binance_klines(symbol: str, interval: str, start_ms: int, limit: int = 1000) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": int(start_ms)}
    r = requests.get(f"{BINANCE_BASE}/api/v3/klines", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()

    cols = ["open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","tbbav","tbqav","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df = df[["open_time","open","high","low","close","volume","close_time"]]
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[["open_time","open","high","low","close","volume"]]


def download_binance_1m(symbol: str, days: int) -> pd.DataFrame:
    end = int(time.time() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    cur = start
    out = []
    pbar = tqdm(desc=f"Downloading {symbol} 1m candles (Binance)")
    while cur < end:
        df = fetch_binance_klines(symbol, "1m", cur, 1000)
        if df.empty:
            break
        out.append(df)
        cur = int(df["open_time"].iloc[-1].value / 1e6) + 60_000
        pbar.update(1)
        time.sleep(0.1)
    pbar.close()
    if not out:
        return pd.DataFrame()
    return (
        pd.concat(out)
        .drop_duplicates(subset=["open_time"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/btc_1m.csv")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--source", choices=["coinbase", "binance"], default="coinbase")

    # Coinbase uses product IDs like BTC-USD
    ap.add_argument("--product", default="BTC-USD", help="Coinbase product id, e.g., BTC-USD")

    # Binance uses symbols like BTCUSDT
    ap.add_argument("--symbol", default="BTCUSDT", help="Binance symbol, e.g., BTCUSDT")

    args = ap.parse_args()

    if args.source == "coinbase":
        df = download_coinbase_1m(args.product, args.days)
    else:
        df = download_binance_1m(args.symbol, args.days)

    if df.empty:
        raise SystemExit("No data downloaded. Try a different source or reduce days.")

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()