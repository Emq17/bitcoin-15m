from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
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
    out_dir: Optional[str] = None,
    run_name: Optional[str] = None,
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
    fold_rows = []
    pred_frames = []

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
        fold_rows.append({
            "fold": fold,
            "test_start": str(test_days_list[0]),
            "test_end": str(test_days_list[-1]),
            "rows": int(len(y_test)),
            "auc": float(auc),
            "acc": float(acc),
            "brier": float(brier),
            "trades_taken": int(take_mask.sum()),
            "take_rate": float(take_mask.mean()),
        })

        pred_frames.append(pd.DataFrame({
            "bucket": test["bucket"].values,
            "date": test["date"].astype(str).values,
            "y_true": y_test,
            "p_green": probs,
            "y_pred": pred,
            "take_trade": take_mask.astype(int),
            "trade_return": taken,
            "fold": fold,
        }))

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

    overall_auc = _safe_auc(oos_y, oos_probs)
    overall_acc = float(accuracy_score(oos_y, (oos_probs >= 0.5)))
    overall_brier = float(brier_score_loss(oos_y, oos_probs))

    print("\n=== OVERALL OOS METRICS ===")
    print("AUC:", overall_auc)
    print("ACC:", overall_acc)
    print("Brier:", overall_brier)
    print("Predictions:", len(oos_y))

    total_return = float("nan")
    trades_taken = int(len(oos_taken_returns))
    take_rate = float(trades_taken / len(oos_y))
    mc = {}
    if len(oos_taken_returns) > 0:
        equity = np.cumprod(1.0 + oos_taken_returns)
        total_return = float(equity[-1] - 1)
        print("\nTotal Return (sim):", total_return)
        print("Trades taken:", trades_taken)
        print("Take rate:", take_rate)

        mc = run_bootstrap(oos_taken_returns, sims=mc_sims)
        print("\n=== MONTE CARLO ===")
        for k, v in mc.items():
            print(k, ":", v)
    else:
        print("\nNo trades taken. Lower min_conf or adjust model.")


    summary = {
        "auc": float(overall_auc),
        "acc": float(overall_acc),
        "brier": float(overall_brier),
        "predictions": int(len(oos_y)),
        "total_return_sim": float(total_return),
        "trades_taken": trades_taken,
        "take_rate": float(take_rate),
        "mc": mc,
    }
    config = {
        "data_path": data_path,
        "horizon": int(horizon),
        "entry_minute": int(entry_minute),
        "model_name": model_name,
        "calibrate": bool(calibrate),
        "lookback_days": int(lookback_days),
        "test_days": int(test_days),
        "min_conf": float(min_conf),
        "mc_sims": int(mc_sims),
    }

    if out_dir:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        default_name = f"h{horizon}_e{entry_minute}_{model_name}_{stamp}"
        run_id = run_name or default_name
        run_dir = Path(out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(fold_rows).to_csv(run_dir / "fold_metrics.csv", index=False)
        pd.concat(pred_frames, ignore_index=True).to_csv(run_dir / "predictions.csv", index=False)
        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"\nSaved run artifacts to: {run_dir}")

    return {
        "summary": summary,
        "config": config,
        "fold_metrics": pd.DataFrame(fold_rows),
    }


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
    ap.add_argument("--out_dir", default=None, help="Optional output directory for run artifacts")
    ap.add_argument("--run_name", default=None, help="Optional run folder name")

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
        out_dir=args.out_dir,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
