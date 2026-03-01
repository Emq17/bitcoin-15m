from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/.cache").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import bootstrap_equity


def _load_run(run_dir: Path) -> Optional[dict[str, Any]]:
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    preds_path = run_dir / "predictions.csv"
    folds_path = run_dir / "fold_metrics.csv"

    if not (config_path.exists() and summary_path.exists() and preds_path.exists() and folds_path.exists()):
        return None

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    preds = pd.read_csv(preds_path, parse_dates=["bucket"])
    folds = pd.read_csv(folds_path)
    return {
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "config": config,
        "summary": summary,
        "preds": preds,
        "folds": folds,
    }


def _make_summary_table(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        cfg = r["config"]
        s = r["summary"]
        rows.append({
            "run_id": r["run_id"],
            "horizon": cfg["horizon"],
            "entry_minute": cfg["entry_minute"],
            "model": cfg["model_name"],
            "calibrate": cfg["calibrate"],
            "lookback_days": cfg["lookback_days"],
            "test_days": cfg["test_days"],
            "min_conf": cfg["min_conf"],
            "auc": s["auc"],
            "acc": s["acc"],
            "brier": s["brier"],
            "predictions": s["predictions"],
            "trades_taken": s["trades_taken"],
            "take_rate": s["take_rate"],
            "total_return_sim": s["total_return_sim"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["horizon", "model", "min_conf", "run_id"]).reset_index(drop=True)


def _run_readable_label(row: pd.Series) -> str:
    model = "LogReg" if str(row["model"]).lower() == "logreg" else "RF"
    cal = "Cal" if bool(row["calibrate"]) else "NoCal"
    return (
        f"{int(row['horizon'])}m | e{int(row['entry_minute'])} | "
        f"{model} {cal} | conf {float(row['min_conf']):.2f}"
    )


def _plot_metric_panels(summary: pd.DataFrame, title: str, out_path: Path) -> None:
    labels = [f"R{i+1}" for i in range(len(summary))]
    readable = [_run_readable_label(r) for _, r in summary.iterrows()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axes[0].bar(x, summary["auc"], color="#1f77b4")
    axes[0].set_ylabel("AUC")
    axes[0].grid(alpha=0.2)

    axes[1].bar(x, summary["acc"], color="#2ca02c")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.2)

    axes[2].bar(x, summary["brier"], color="#d62728")
    axes[2].set_ylabel("Brier")
    axes[2].grid(alpha=0.2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")

    fig.suptitle(title)
    legend_lines = [f"{labels[i]}: {readable[i]}" for i in range(len(labels))]
    fig.text(0.02, 0.02, "\n".join(legend_lines), fontsize=8, va="bottom", family="monospace")
    fig.tight_layout(rect=[0, 0.18, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_overall_metrics(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    _plot_metric_panels(summary, "Run Comparison Metrics (OOS)", out_dir / "overall_metrics.png")


def _plot_overall_metrics_by_horizon(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    for h in sorted(summary["horizon"].unique()):
        sh = summary[summary["horizon"] == h].reset_index(drop=True)
        if sh.empty:
            continue
        _plot_metric_panels(
            sh,
            f"Run Comparison Metrics (OOS) - {int(h)}m Horizon",
            out_dir / f"overall_metrics_h{int(h)}.png",
        )


def _plot_confidence_tradeoff(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty or summary["min_conf"].nunique() <= 1:
        return

    key_cols = ["horizon", "entry_minute", "model", "calibrate"]
    grouped = summary.groupby(key_cols, dropna=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for key, g in grouped:
        g = g.sort_values("min_conf")
        label = f"h={key[0]} e={key[1]} {key[2]} {'cal' if key[3] else 'noc'}"
        axes[0].plot(g["min_conf"], g["auc"], marker="o", label=label)
        axes[1].plot(g["min_conf"], g["take_rate"], marker="o", label=label)

    axes[0].set_ylabel("AUC")
    axes[0].set_title("Confidence Threshold Tradeoff")
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("Take Rate")
    axes[1].set_xlabel("min_conf")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "confidence_tradeoff.png", dpi=160)
    plt.close(fig)


def _plot_fold_metrics(folds: pd.DataFrame, title: str, out_path: Path) -> None:
    x = np.arange(len(folds))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(x, folds["auc"], marker="o")
    axes[0].set_ylabel("AUC")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, folds["acc"], marker="o", color="#2ca02c")
    axes[1].set_ylabel("ACC")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, folds["brier"], marker="o", color="#d62728")
    axes[2].set_ylabel("Brier")
    axes[2].set_xlabel("Fold")
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"{title}: Fold Metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_calibration(preds: pd.DataFrame, title: str, out_path: Path, n_bins: int = 10) -> None:
    df = preds.copy()
    df["bin"] = pd.cut(df["p_green"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    g = df.groupby("bin", observed=False).agg(
        p_mean=("p_green", "mean"),
        y_rate=("y_true", "mean"),
        n=("y_true", "size"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(g["p_mean"], g["y_rate"], marker="o", label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted p(green)")
    ax.set_ylabel("Observed green frequency")
    ax.set_title(f"{title}: Calibration")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_probability_hist(preds: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(preds.loc[preds["y_true"] == 1, "p_green"], bins=30, alpha=0.6, label="true green")
    ax.hist(preds.loc[preds["y_true"] == 0, "p_green"], bins=30, alpha=0.6, label="true red")
    ax.set_xlabel("Predicted p(green)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}: Probability Distribution by Class")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_equity_drawdown(preds: pd.DataFrame, title: str, out_path: Path) -> None:
    r = preds[preds["take_trade"] == 1]["trade_return"].to_numpy()
    if len(r) == 0:
        return

    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    x = np.arange(len(r))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, eq, color="#1f77b4")
    axes[0].set_ylabel("Equity")
    axes[0].set_title(f"{title}: Equity Curve (Sim)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, dd, color="#d62728")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Trade Index")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_monte_carlo(preds: pd.DataFrame, title: str, out_path: Path, sims: int = 2000) -> None:
    r = preds[preds["take_trade"] == 1]["trade_return"].to_numpy()
    if len(r) == 0:
        return
    end_eq, _ = bootstrap_equity(r, n_sims=sims)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(end_eq, bins=50, color="#9467bd", alpha=0.8)
    ax.set_xlabel("Terminal Equity")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title}: Monte Carlo Terminal Equity ({sims} sims)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_monte_carlo_paths(preds: pd.DataFrame, title: str, out_path: Path, sims: int = 300) -> None:
    r = preds[preds["take_trade"] == 1]["trade_return"].to_numpy()
    if len(r) == 0:
        return

    rng = np.random.default_rng(42)
    n = len(r)
    # Keep the chart readable by limiting plotted paths.
    n_paths = int(min(max(sims, 50), 400))
    samples = rng.choice(r, size=(n_paths, n), replace=True)
    eq_paths = np.cumprod(1.0 + samples, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_paths):
        ax.plot(eq_paths[i], alpha=0.08, color="#1f77b4", linewidth=1.0)

    median_path = np.median(eq_paths, axis=0)
    p05 = np.percentile(eq_paths, 5, axis=0)
    p95 = np.percentile(eq_paths, 95, axis=0)
    ax.plot(median_path, color="#d62728", linewidth=2.0, label="median path")
    ax.plot(p05, color="#2ca02c", linewidth=1.5, linestyle="--", label="5th/95th percentile")
    ax.plot(p95, color="#2ca02c", linewidth=1.5, linestyle="--")

    ax.set_title(f"{title}: Monte Carlo Equity Paths")
    ax.set_xlabel("Trade Index")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _make_run_title(run: dict[str, Any]) -> str:
    cfg = run["config"]
    cal = "cal" if cfg["calibrate"] else "noc"
    return f"{run['run_id']} | h={cfg['horizon']} e={cfg['entry_minute']} {cfg['model_name']} {cal}"


def _write_snapshot_images(summary: pd.DataFrame, out_root: Path) -> None:
    if summary.empty:
        return
    best_auc_idx = summary["auc"].astype(float).idxmax()
    best_run_id = str(summary.loc[best_auc_idx, "run_id"])
    best_dir = out_root / "per_run" / best_run_id
    if not best_dir.exists():
        return

    snapshot_dir = out_root / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        out_root / "overall_metrics.png": snapshot_dir / "overall_metrics.png",
        out_root / "overall_metrics_h5.png": snapshot_dir / "overall_metrics_h5.png",
        out_root / "overall_metrics_h15.png": snapshot_dir / "overall_metrics_h15.png",
        out_root / "confidence_tradeoff.png": snapshot_dir / "confidence_tradeoff.png",
        best_dir / "fold_metrics.png": snapshot_dir / "best_run_fold_metrics.png",
        best_dir / "calibration.png": snapshot_dir / "best_run_calibration.png",
        best_dir / "equity_drawdown.png": snapshot_dir / "best_run_equity_drawdown.png",
        best_dir / "monte_carlo_terminal_equity.png": snapshot_dir / "best_run_monte_carlo.png",
        best_dir / "monte_carlo_paths.png": snapshot_dir / "best_run_monte_carlo_paths.png",
    }
    for src, dst in mapping.items():
        if src.exists():
            shutil.copyfile(src, dst)


def _fmt_float(v: Any, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "nan"


def _model_label(row: pd.Series) -> str:
    cal = "calibrated" if bool(row["calibrate"]) else "uncalibrated"
    return (
        f"{int(row['horizon'])}-minute horizon, entry minute {int(row['entry_minute'])}, "
        f"{str(row['model'])} ({cal}), confidence threshold {row['min_conf']}"
    )


def _write_key_findings(summary: pd.DataFrame, out_root: Path) -> None:
    if summary.empty:
        return

    s = summary.copy()
    s["auc"] = s["auc"].astype(float)
    s["acc"] = s["acc"].astype(float)
    s["brier"] = s["brier"].astype(float)
    s["take_rate"] = s["take_rate"].astype(float)
    s["min_conf"] = s["min_conf"].astype(float)

    best_idx = s["auc"].idxmax()
    best = s.loc[best_idx]

    lines = []
    lines.append("Updated automatically from `results/report/summary_table.csv`.")
    lines.append("")
    lines.append("### Performance Report (15m Focus)")
    lines.append("")
    lines.append("| Setup | Net Quality | Hit Rate | Prob Error | Trade Participation |")
    lines.append("|---|---:|---:|---:|---:|")
    s15 = s[s["horizon"] == 15].copy()
    if s15.empty:
        lines.append("| No 15-minute runs found | - | - | - | - |")
    else:
        s15 = s15.sort_values(["auc", "acc"], ascending=[False, False])
        s15 = s15.drop_duplicates(
            subset=["horizon", "entry_minute", "model", "calibrate", "min_conf"],
            keep="first",
        )
        for _, row in s15.iterrows():
            model = "LogReg" if str(row["model"]).lower() == "logreg" else "RF"
            cal = "Cal" if bool(row["calibrate"]) else "NoCal"
            setup = f"e{int(row['entry_minute'])}, {model} {cal}, conf {float(row['min_conf']):.2f}"
            lines.append(
                f"| {setup} | {_fmt_float(row['auc'])} | {_fmt_float(row['acc'])} | {_fmt_float(row['brier'])} | {_fmt_float(row['take_rate'])} |"
            )
    lines.append("")
    lines.append("### Quick Read")
    lines.append("")
    lines.append(
        f"- Current top setup: {_model_label(best)}."
    )
    lines.append(
        "- AUC: higher is better at separating up candles from down candles."
    )
    lines.append(
        "- Brier: lower is better for probability quality."
    )
    lines.append("")

    (out_root / "key_findings.md").write_text("\n".join(lines), encoding="utf-8")


def _inject_findings_into_readme(readme_path: str, findings_path: Path) -> None:
    readme = Path(readme_path)
    if not readme.exists():
        raise ValueError(f"README file not found: {readme}")
    if not findings_path.exists():
        raise ValueError(f"Findings file not found: {findings_path}")

    start_marker = "<!-- AUTO_KEY_FINDINGS_START -->"
    end_marker = "<!-- AUTO_KEY_FINDINGS_END -->"
    text = readme.read_text(encoding="utf-8")
    if start_marker not in text or end_marker not in text:
        raise ValueError(
            "README markers not found. Add AUTO_KEY_FINDINGS_START/END markers first."
        )

    findings = findings_path.read_text(encoding="utf-8").strip()
    replacement = f"{start_marker}\n{findings}\n{end_marker}"
    pre = text.split(start_marker, 1)[0]
    post = text.split(end_marker, 1)[1]
    updated = pre + replacement + post
    readme.write_text(updated, encoding="utf-8")


def build_report(
    runs_dir: str,
    out_dir: str,
    mc_sims: int = 2000,
    inject_readme: bool = False,
    readme_path: str = "README.md",
) -> None:
    runs_root = Path(runs_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for sub in sorted(runs_root.iterdir()):
        if not sub.is_dir():
            continue
        run = _load_run(sub)
        if run is not None:
            runs.append(run)

    if not runs:
        raise ValueError(f"No valid run artifacts found in {runs_root}")

    summary = _make_summary_table(runs)
    summary.to_csv(out_root / "summary_table.csv", index=False)
    _plot_overall_metrics(summary, out_root)
    _plot_overall_metrics_by_horizon(summary, out_root)
    _plot_confidence_tradeoff(summary, out_root)

    per_run_dir = out_root / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run_out = per_run_dir / run["run_id"]
        run_out.mkdir(parents=True, exist_ok=True)
        title = _make_run_title(run)

        _plot_fold_metrics(run["folds"], title, run_out / "fold_metrics.png")
        _plot_calibration(run["preds"], title, run_out / "calibration.png")
        _plot_probability_hist(run["preds"], title, run_out / "probability_hist.png")
        _plot_equity_drawdown(run["preds"], title, run_out / "equity_drawdown.png")
        _plot_monte_carlo(run["preds"], title, run_out / "monte_carlo_terminal_equity.png", sims=mc_sims)
        _plot_monte_carlo_paths(run["preds"], title, run_out / "monte_carlo_paths.png", sims=mc_sims)

    _write_snapshot_images(summary, out_root)
    _write_key_findings(summary, out_root)
    if inject_readme:
        _inject_findings_into_readme(readme_path, out_root / "key_findings.md")
        print(f"Updated README findings block: {readme_path}")

    print(f"Report written to: {out_root}")
    print(f"Runs processed: {len(runs)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results/runs")
    ap.add_argument("--out_dir", default="results/report")
    ap.add_argument("--mc_sims", type=int, default=2000)
    ap.add_argument("--inject_readme", action="store_true")
    ap.add_argument("--readme_path", default="README.md")
    args = ap.parse_args()
    build_report(
        args.runs_dir,
        args.out_dir,
        mc_sims=args.mc_sims,
        inject_readme=args.inject_readme,
        readme_path=args.readme_path,
    )


if __name__ == "__main__":
    main()
