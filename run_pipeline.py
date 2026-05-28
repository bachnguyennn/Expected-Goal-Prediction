"""
run_pipeline.py
----------------
One-command end-to-end runner for the xG Betting Engine portfolio project.

Steps:
  1. Download StatsBomb shot data (World Cup 2018, La Liga 2017/18 & 2018/19, Euro 2020)
  2. Build shot-level xG features (distance, angle, freeze frame, etc.)
  3. Train + evaluate XGBoost xG model with SHAP interpretation
  4. Demo: match simulation + EV / Kelly bet sizing

Usage:
    python run_pipeline.py                  # full run
    python run_pipeline.py --skip-download  # use existing data
    python run_pipeline.py --no-shap        # skip SHAP (faster)
    python run_pipeline.py --tune           # run Optuna tuning first
    python run_pipeline.py --demo-only      # skip training, run EV demo
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON      = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def run_step(name: str, script: str, extra_args: list[str] | None = None) -> float:
    cmd = [PYTHON, str(ROOT / script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"  cmd : {' '.join(cmd)}")
    print("="*60)
    start  = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  ERROR: {name} failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"  ✓ {name}  ({elapsed:.1f}s)")
    return elapsed


def demo_ev(python: str) -> None:
    """Run the EV engine demo directly (no subprocess overhead)."""
    print(f"\n{'='*60}")
    print("  STEP: EV + Kelly Demonstration")
    print("="*60)
    result = subprocess.run([python, "-m", "betting.ev_calculator"], cwd=ROOT)
    if result.returncode != 0:
        print("  (EV demo failed — non-fatal)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="xG Betting Engine — end-to-end pipeline"
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (use existing shots_raw.parquet)")
    parser.add_argument("--no-shap",       action="store_true",
                        help="Skip SHAP plots (much faster run)")
    parser.add_argument("--tune",          action="store_true",
                        help="Run Optuna hyperparameter tuning before training")
    parser.add_argument("--demo-only",     action="store_true",
                        help="Skip training; only run the EV/Kelly demo")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  xG BETTING ENGINE — Portfolio Pipeline")
    print("  Real-Time Expected Goals + Optimal Bet Sizing")
    print("="*60)

    timings: dict[str, float] = {}
    start_total = time.time()

    if not args.demo_only:
        # 1. Download
        if not args.skip_download:
            timings["Download"] = run_step(
                "StatsBomb Data Download", "data/download_data.py"
            )
        else:
            print("\n  ⏭  Skipping download (--skip-download)")

        # 2. Feature engineering
        timings["Features"] = run_step(
            "Shot Feature Engineering", "features/shot_features.py"
        )

        # 3. Train xG model
        train_args: list[str] = []
        if args.no_shap:
            train_args.append("--no-shap")
        if args.tune:
            train_args.append("--tune")
        timings["xG Model"] = run_step(
            "xG Model Training + SHAP + Calibration",
            "models/xg_model.py",
            train_args,
        )

    # 4. EV + Kelly demo
    demo_ev(PYTHON)

    total = time.time() - start_total

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    if timings:
        print(f"\nTimings:")
        for step, secs in timings.items():
            print(f"  {step:<35} {secs:>6.1f}s")
    print(f"\n  Total wall time: {total:.1f}s")

    print("\nKey outputs:")
    print("  models/artifacts/xg_model.joblib          — calibrated xG model")
    print("  models/artifacts/xg_metrics.json          — Brier / AUPRC / AUROC")
    print("  models/artifacts/test_predictions.parquet — per-shot xG predictions")
    print("  reports/figures/shap_bar_xg.png            — SHAP feature importance")
    print("  reports/figures/shap_beeswarm_xg.png       — SHAP direction + magnitude")
    print("  reports/figures/shap_dep_distance.png      — SHAP dependence: distance")
    print("  reports/figures/shap_dep_angle.png         — SHAP dependence: angle")
    print("  reports/figures/calibration_curve.png      — xG calibration vs StatsBomb")
    print("  reports/figures/shot_map.png               — pitch map coloured by xG")

    print("\nNext steps:")
    print("  • Open notebook/xG_Betting_Analysis.ipynb for full analysis")
    print("  • Run python run_pipeline.py --tune for optimised model")
    print("  • See inplay/engine.py for live match replay")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
