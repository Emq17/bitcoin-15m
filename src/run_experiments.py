from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.walkforward import walk_forward


def _parse_csv_list(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", default="results/runs")
    ap.add_argument("--horizons", default="15,5", help="Comma-separated, e.g. 15,5")
    ap.add_argument("--entry_minutes", default="13,3", help="Comma-separated, aligns with horizons")
    ap.add_argument("--models", default="logreg,rf", help="Comma-separated model names")
    ap.add_argument("--calibrate_logreg", action="store_true", help="Use calibration for logistic runs")
    ap.add_argument(
        "--min_conf_values",
        default=None,
        help="Optional comma-separated confidence thresholds, e.g. 0.0,0.02,0.05",
    )
    ap.add_argument("--lookback_days", type=int, default=60)
    ap.add_argument("--test_days", type=int, default=14)
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--mc_sims", type=int, default=5000)
    args = ap.parse_args()

    horizons = [int(x) for x in _parse_csv_list(args.horizons)]
    entries = [int(x) for x in _parse_csv_list(args.entry_minutes)]
    models = _parse_csv_list(args.models)
    min_conf_values = (
        [float(x) for x in _parse_csv_list(args.min_conf_values)]
        if args.min_conf_values
        else [float(args.min_conf)]
    )
    if len(horizons) != len(entries):
        raise ValueError("--horizons and --entry_minutes must have equal length.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for horizon, entry in zip(horizons, entries):
        for model in models:
            for min_conf in min_conf_values:
                calibrate = args.calibrate_logreg if model == "logreg" else False
                run_name = (
                    f"{stamp}_h{horizon}_e{entry}_{model}_{'cal' if calibrate else 'noc'}"
                    f"_mc{str(min_conf).replace('.', 'p')}"
                )
                print(f"\n=== RUN {run_name} ===")
                walk_forward(
                    data_path=args.data,
                    horizon=horizon,
                    entry_minute=entry,
                    model_name=model,
                    calibrate=calibrate,
                    lookback_days=args.lookback_days,
                    test_days=args.test_days,
                    min_conf=min_conf,
                    mc_sims=args.mc_sims,
                    out_dir=args.out_dir,
                    run_name=run_name,
                )


if __name__ == "__main__":
    main()
