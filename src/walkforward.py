from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

from src.dataset import load_1m_csv
from src.features import build_feature_frame
from src.models import get_model, fit_predict_p_green
from src.montecarlo import run_bootstrap


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def walk_forward(
    data_path: str,
    horizon: int,
    entry_minute: int,
    model_name: str,
    calibrate: bool,
    lookback_days: int,
    test_days: int,
    min_conf: float,
    mc_sims: int,
):
    df_1m = load_1m_csv(data_path)
    feat = build_feature_frame(df_1m, horizon, entry_minute)

    feat = feat.sort_values("bucket").reset_index(drop=True)
    feat = feat.dropna().reset_index(drop=True)

    feat["date"] = feat["bucket"].dt.date
    unique_days = sorted(feat["date"].unique())

    lookback = lookback_days
    test_len = test_days

    oos_probs = []
    oos_y = []
    oos_taken_returns = []

    start = lookback
    fold = 0

    if len(unique_days) < lookback + test_len:
        raise ValueError(
            f"Not enough days for walk-forward: have {len(unique_days)}, "
            f"need at least {lookback + test_len}."
        )

    while start + test_len <= len(unique_days):
        train_days = unique_days[start - lookback:start]
        test_days_list = unique_days[start:start + test_len]

        train = feat[feat["date"].isin(train_days)]
        test = feat[feat["date"].isin(test_days_list)]

        X_train = train.drop(columns=["bucket", "y_green", "date"]).values
        y_train = train["y_green"].values

        X_test = test.drop(columns=["bucket", "y_green", "date"]).values
        y_test = test["y_green"].values

        if len(train) == 0 or len(test) == 0:
            start += test_len
            continue
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            start += test_len
            continue

        model = get_model(model_name, calibrate=calibrate)
        try:
            probs = fit_predict_p_green(model, X_train, y_train, X_test)
        except ValueError:
            # Calibration can fail in small/imbalanced folds; fallback keeps run alive.
            fallback = get_model(model_name, calibrate=False)
            probs = fit_predict_p_green(fallback, X_train, y_train, X_test)

        oos_probs.extend(probs)
        oos_y.extend(y_test)

        pred = (probs >= 0.5).astype(int)
        auc = _safe_auc(y_test, probs)
        acc = accuracy_score(y_test, pred)
        brier = brier_score_loss(y_test, probs)

        # Even-money placeholder return model
        take_mask = np.abs(probs - 0.5) >= min_conf
        taken = np.where(pred == y_test, 0.01, -0.01)
        taken = taken * take_mask

        oos_taken_returns.extend(taken[take_mask])

        print(
            f"Fold {fold:02d} | "
            f"{test_days_list[0]}->{test_days_list[-1]} | "
            f"AUC {auc:.3f} | ACC {acc:.3f} | "
            f"Brier {brier:.4f} | "
            f"Trades {take_mask.sum()}"
        )

        fold += 1
        start += test_len

    oos_probs = np.array(oos_probs)
    oos_y = np.array(oos_y)
    oos_taken_returns = np.array(oos_taken_returns)

    if len(oos_y) == 0:
        raise ValueError("No OOS predictions generated. Check split settings and data coverage.")

    print("\n=== OVERALL OOS METRICS ===")
    print("AUC:", _safe_auc(oos_y, oos_probs))
    print("ACC:", accuracy_score(oos_y, (oos_probs >= 0.5)))
    print("Brier:", brier_score_loss(oos_y, oos_probs))
    print("Predictions:", len(oos_y))

    if len(oos_taken_returns) > 0:
        equity = np.cumprod(1.0 + oos_taken_returns)
        print("\nTotal Return (sim):", equity[-1] - 1)
        print("Trades taken:", len(oos_taken_returns))
        print("Take rate:", len(oos_taken_returns) / len(oos_y))

        mc = run_bootstrap(oos_taken_returns, sims=mc_sims)
        print("\n=== MONTE CARLO ===")
        for k, v in mc.items():
            print(k, ":", v)
    else:
        print("\nNo trades taken. Lower min_conf or adjust model.")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--entry_minute", type=int, default=13)
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--lookback_days", type=int, default=3)
    ap.add_argument("--test_days", type=int, default=1)
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--mc_sims", type=int, default=3000)

    args = ap.parse_args()

    walk_forward(
        data_path=args.data,
        horizon=args.horizon,
        entry_minute=args.entry_minute,
        model_name=args.model,
        calibrate=args.calibrate,
        lookback_days=args.lookback_days,
        test_days=args.test_days,
        min_conf=args.min_conf,
        mc_sims=args.mc_sims,
    )


if __name__ == "__main__":
    main()
