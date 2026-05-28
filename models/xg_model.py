"""
models/xg_model.py
-------------------
Trains, evaluates, and interprets an XGBoost Expected Goals (xG) model.

xG  = probability that a shot results in a goal.

Design decisions
  - Train / test split is by competition (temporal): train on World Cup 2018
    and Champions League 2018/19, test on UEFA Euro 2020.  This mirrors a
    real betting use-case where the model is trained on historical seasons
    and deployed on upcoming tournaments.
  - Primary metric: Brier score (proper scoring rule for probabilities) +
    AUPRC + calibration curve.
  - Model is calibrated with Platt scaling (LogisticRegression on raw
    XGBoost probabilities) so outputs can be trusted as true probabilities.
  - Comparison against StatsBomb's own xG (shot_statsbomb_xg) as an
    independent benchmark.
  - Full SHAP suite: bar, beeswarm, dependence (distance & angle).

Usage:
    python models/xg_model.py
    python models/xg_model.py --no-shap   # skip SHAP plots (faster)
    python models/xg_model.py --tune      # run Optuna hyperparameter search first
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.shot_features import FEATURE_COLS

ROOT_DIR     = Path(__file__).resolve().parents[1]
PROC_DIR     = ROOT_DIR / "data" / "processed"
ARTIFACTS    = ROOT_DIR / "models" / "artifacts"
FIGURES      = ROOT_DIR / "reports" / "figures"
DATA_PATH    = PROC_DIR / "shots_featured.parquet"

TARGET       = "goal"
TRAIN_SPLIT  = "train"
TEST_SPLIT   = "test"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Featured data not found: {DATA_PATH}\n"
            "Run: python features/shot_features.py"
        )

    df = pd.read_parquet(DATA_PATH)

    # Use competitions as the train/test boundary
    train_mask = df["split"] == TRAIN_SPLIT
    test_mask  = df["split"] == TEST_SPLIT

    if test_mask.sum() == 0:
        # Fallback: random 80/20 by match
        log.warning("No 'test' split found — using random 80% train / 20% test by match.")
        match_ids   = df["match_id"].unique()
        rng         = np.random.default_rng(42)
        test_matches= set(rng.choice(match_ids, size=int(0.2 * len(match_ids)), replace=False))
        test_mask   = df["match_id"].isin(test_matches)
        train_mask  = ~test_mask

    X_train = df.loc[train_mask, FEATURE_COLS]
    X_test  = df.loc[test_mask,  FEATURE_COLS]
    y_train = df.loc[train_mask, TARGET]
    y_test  = df.loc[test_mask,  TARGET]

    log.info(f"Train: {len(X_train):,} shots  goal rate {y_train.mean():.2%}")
    log.info(f"Test : {len(X_test):,}  shots  goal rate {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, df.loc[test_mask]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """Train XGBoost with early stopping on the supplied validation set."""
    scale_pw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    log.info(f"scale_pos_weight = {scale_pw:.2f}")

    defaults = {
        "objective":         "binary:logistic",
        "eval_metric":       "aucpr",
        "n_estimators":      800,
        "max_depth":         5,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  3,
        "scale_pos_weight":  scale_pw,
        "random_state":      42,
        "n_jobs":            -1,
        "early_stopping_rounds": 40,
    }
    if params:
        defaults.update(params)

    model = xgb.XGBClassifier(**defaults)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    log.info(f"Best iteration: {model.best_iteration}  best val AUCPR: {model.best_score:.4f}")
    return model


def calibrate_model(
    model: xgb.XGBClassifier,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> CalibratedClassifierCV:
    """
    Platt-scale calibration on a held-out calibration set.
    Raw XGBoost probabilities can be systematically miscalibrated;
    calibration is required before using outputs as true probabilities in EV/Kelly.
    """
    log.info("Calibrating with Platt scaling ...")
    cal = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
    cal.fit(X_cal, y_cal)
    return cal


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    y_true: pd.Series,
    y_proba: np.ndarray,
    label: str,
    sb_xg: pd.Series | None = None,
) -> dict:
    """Compute Brier score, AUPRC, AUROC. Optionally compare to StatsBomb xG."""
    metrics = {
        "model":       label,
        "brier":       float(brier_score_loss(y_true, y_proba)),
        "auprc":       float(average_precision_score(y_true, y_proba)),
        "auroc":       float(roc_auc_score(y_true, y_proba)),
        "n_shots":     int(len(y_true)),
        "goal_rate":   float(y_true.mean()),
    }

    log.info(
        f"{label:<28}  Brier={metrics['brier']:.4f}  "
        f"AUPRC={metrics['auprc']:.4f}  AUROC={metrics['auroc']:.4f}"
    )

    if sb_xg is not None and not sb_xg.isna().all():
        sb_proba = sb_xg.fillna(sb_xg.mean()).values
        sb_metrics = {
            "statsbomb_brier": float(brier_score_loss(y_true, sb_proba)),
            "statsbomb_auprc": float(average_precision_score(y_true, sb_proba)),
            "statsbomb_auroc": float(roc_auc_score(y_true, sb_proba)),
        }
        metrics.update(sb_metrics)
        log.info(
            f"  StatsBomb baseline xG:       Brier={sb_metrics['statsbomb_brier']:.4f}  "
            f"AUPRC={sb_metrics['statsbomb_auprc']:.4f}  AUROC={sb_metrics['statsbomb_auroc']:.4f}"
        )

    return metrics


# ---------------------------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------------------------

def save_calibration_plot(
    y_true: pd.Series,
    y_our: np.ndarray,
    y_sb: pd.Series | None,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))

    frac_pos, mean_pred = calibration_curve(y_true, y_our, n_bins=10)
    plt.plot(mean_pred, frac_pos, "s-", label="Our xG model", linewidth=2)

    if y_sb is not None and not y_sb.isna().all():
        fp2, mp2 = calibration_curve(y_true, y_sb.fillna(y_sb.mean()), n_bins=10)
        plt.plot(mp2, fp2, "o--", label="StatsBomb xG", linewidth=1.5, alpha=0.8)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives (actual goal rate)")
    plt.title("xG Model Calibration")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = FIGURES / "calibration_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def save_shap_plots(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)

    log.info("Computing SHAP values ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 1. Bar — global importance
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.title("xG Model: Feature Importance (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_bar_xg.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved shap_bar_xg.png")

    # 2. Beeswarm
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("xG Model: Feature Effect Direction & Magnitude (SHAP Beeswarm)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_beeswarm_xg.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved shap_beeswarm_xg.png")

    # 3. Dependence: distance
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot("distance", shap_values, X_sample, ax=ax, show=False)
    ax.set_title("SHAP Dependence: Shot Distance (yards from goal)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_dep_distance.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved shap_dep_distance.png")

    # 4. Dependence: angle
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot("angle", shap_values, X_sample, ax=ax, show=False)
    ax.set_title("SHAP Dependence: Shot Angle (degrees)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_dep_angle.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved shap_dep_angle.png")

    # 5. Scatter: predicted xG vs actual outcomes
    y_proba = model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(8, 5))
    plt.scatter(y_proba[y_test == 0], np.zeros(int((y_test == 0).sum())),
                alpha=0.3, s=8, label="No goal", color="#5B9BD5")
    plt.scatter(y_proba[y_test == 1], np.ones(int((y_test == 1).sum())),
                alpha=0.5, s=12, label="Goal", color="#ED7D31")
    plt.xlabel("Predicted xG")
    plt.title("xG Distribution: Goals vs Non-Goals")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "xg_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved xg_distribution.png")


# ---------------------------------------------------------------------------
# Shot-location heatmap
# ---------------------------------------------------------------------------

def save_shot_map(df_test: pd.DataFrame) -> None:
    """Pitch map: shot locations coloured by predicted xG."""
    if "x" not in df_test.columns or "xg_pred" not in df_test.columns:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(60, 122)
    ax.set_ylim(-2, 82)
    ax.set_facecolor("#2d6a4f")

    # Goal rectangle
    ax.add_patch(plt.Rectangle((120, 36), 2, 8, fc="white", ec="white", lw=2, zorder=3))
    ax.axvline(120, color="white", lw=1.5)

    scatter = ax.scatter(
        df_test["x"], df_test["y"],
        c=df_test["xg_pred"], cmap="YlOrRd",
        s=50, alpha=0.8, vmin=0, vmax=0.5, zorder=4,
    )
    plt.colorbar(scatter, ax=ax, label="Predicted xG")

    goals = df_test[df_test["goal"] == 1]
    ax.scatter(goals["x"], goals["y"],
               s=120, marker="*", color="gold", zorder=5, label="Goal scored")
    ax.legend(loc="upper left")
    ax.set_title("Test Set: Shot Locations Coloured by Predicted xG\n(attacking direction → right)")
    ax.set_xlabel("Pitch x (yards)")
    ax.set_ylabel("Pitch y (yards)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shot_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved shot_map.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--tune",    action="store_true",
                        help="Run a short Optuna search before training")
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, df_test = load_data()

    # Carve out calibration set from train (10%)
    cal_size   = max(200, int(0.10 * len(X_train)))
    cal_idx    = X_train.sample(cal_size, random_state=42).index
    X_cal      = X_train.loc[cal_idx]
    y_cal      = y_train.loc[cal_idx]
    X_tr       = X_train.drop(index=cal_idx)
    y_tr       = y_train.drop(index=cal_idx)

    # Optional Optuna tuning
    best_params = None
    if args.tune:
        best_params = _run_optuna(X_tr, y_tr, n_trials=30)

    # Train and calibrate
    val_size  = int(0.15 * len(X_tr))
    val_idx   = X_tr.sample(val_size, random_state=42).index
    X_val     = X_tr.loc[val_idx]
    y_val     = y_tr.loc[val_idx]
    X_tr2     = X_tr.drop(index=val_idx)
    y_tr2     = y_tr.drop(index=val_idx)

    model_raw = train_xgboost(X_tr2, y_tr2, X_val, y_val, best_params)
    model     = calibrate_model(model_raw, X_cal, y_cal)

    # Evaluate
    y_proba   = model.predict_proba(X_test)[:, 1]
    sb_xg     = df_test.get("shot_statsbomb_xg")
    metrics   = evaluate(y_test, y_proba, "Our XGBoost xG (calibrated)", sb_xg)

    # Attach predictions to test set for visualisation
    df_test = df_test.copy()
    df_test["xg_pred"] = y_proba
    if "x" not in df_test.columns and "location" in df_test.columns:
        df_test["x"] = df_test["location"].apply(
            lambda l: float(l[0]) if isinstance(l, list) else np.nan
        )
        df_test["y"] = df_test["location"].apply(
            lambda l: float(l[1]) if isinstance(l, list) else np.nan
        )

    # Save
    joblib.dump(model, ARTIFACTS / "xg_model.joblib")
    df_test[["id", "match_id", "player", "competition", "xg_pred", "goal",
             "shot_statsbomb_xg"] + FEATURE_COLS
            if "id" in df_test.columns else
            ["match_id", "player", "competition", "xg_pred", "goal"] + FEATURE_COLS
           ].to_parquet(ARTIFACTS / "test_predictions.parquet", index=False)

    with open(ARTIFACTS / "xg_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Saved xg_model.joblib, xg_metrics.json, test_predictions.parquet")

    # Plots
    save_calibration_plot(y_test, y_proba, sb_xg)
    save_shot_map(df_test)

    if not args.no_shap:
        try:
            save_shap_plots(model_raw, X_test, y_test)
        except Exception as exc:
            log.warning(f"SHAP failed (non-fatal): {exc}")

    # Console summary
    print("\n" + "=" * 56)
    print("  xG MODEL RESULTS")
    print("=" * 56)
    print(f"{'Metric':<20} {'Our model':>12} {'StatsBomb xG':>14}")
    print("-" * 56)
    for key, label in [("brier", "Brier ↓"), ("auprc", "AUPRC ↑"), ("auroc", "AUROC ↑")]:
        ours = metrics[key]
        sb   = metrics.get(f"statsbomb_{key}", float("nan"))
        print(f"{label:<20} {ours:>12.4f} {sb:>14.4f}")
    print("=" * 56)
    print("\nNote: lower Brier is better; higher AUPRC/AUROC is better.")
    print(f"Artifacts → {ARTIFACTS}")


def _run_optuna(X: pd.DataFrame, y: pd.Series, n_trials: int = 30) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    val_size = int(0.2 * len(X))
    val_idx  = X.sample(val_size, random_state=0).index
    X_val, y_val = X.loc[val_idx],  y.loc[val_idx]
    X_tr,  y_tr  = X.drop(index=val_idx), y.drop(index=val_idx)
    scale_pw     = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "n_estimators": 600,
            "scale_pos_weight": scale_pw,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 30,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return average_precision_score(y_val, m.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"Optuna best AUPRC: {study.best_value:.4f}  params: {study.best_params}")
    return study.best_params


if __name__ == "__main__":
    main()
