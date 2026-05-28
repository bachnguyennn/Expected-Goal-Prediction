"""
features/shot_features.py
--------------------------
Transforms raw StatsBomb shot rows into a clean, model-ready feature matrix.

StatsBomb pitch coordinate system
  x : 0 → 120  (goal-to-goal length, attacking direction = increasing x)
  y : 0 → 80   (side-to-side width)
  Attacking goal posts : x=120, y=36 (left post) and x=120, y=44 (right post)
  Goal centre          : (120, 40)

Features produced
  Geometry    : distance, angle
  Body part   : is_header
  Shot type   : is_penalty, is_freekick, is_open_play
  Context     : under_pressure, is_first_time, is_one_on_one, is_second_half
  Play origin : is_corner, is_counter, is_freekick_buildup
  Technique   : technique_Normal, technique_Lob, technique_Volley, technique_Half Volley
  Freeze frame: n_defenders_in_cone, goalkeeper_dist_from_goal, n_opponents_visible
  Time        : match_time (fractional minutes)
  Target      : goal (1 = goal scored)

Usage:
    python features/shot_features.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT_DIR / "data" / "raw"
PROC_DIR = ROOT_DIR / "data" / "processed"

RAW_PARQUET  = RAW_DIR  / "shots_raw.parquet"
OUT_PARQUET  = PROC_DIR / "shots_featured.parquet"

GOAL_LEFT_POST  = (120.0, 36.0)
GOAL_RIGHT_POST = (120.0, 44.0)
GOAL_CENTRE     = (120.0, 40.0)

FEATURE_COLS = [
    "distance", "angle",
    "is_header", "is_penalty", "is_freekick", "is_open_play",
    "under_pressure", "is_first_time", "is_one_on_one",
    "is_corner", "is_counter",
    "is_second_half", "match_time",
    "n_defenders_in_cone", "goalkeeper_dist_from_goal", "n_opponents_visible",
    "technique_Normal", "technique_Lob", "technique_Volley", "technique_Half Volley",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def shot_distance(x: float, y: float) -> float:
    """Euclidean distance from (x, y) to goal centre."""
    return float(np.sqrt((GOAL_CENTRE[0] - x) ** 2 + (GOAL_CENTRE[1] - y) ** 2))


def shot_angle(x: float, y: float) -> float:
    """
    Angle (degrees) subtended by the goal opening from position (x, y).
    Uses arctangent of the two goal-post vectors.
    """
    a1 = np.arctan2(GOAL_LEFT_POST[1]  - y, GOAL_LEFT_POST[0]  - x)
    a2 = np.arctan2(GOAL_RIGHT_POST[1] - y, GOAL_RIGHT_POST[0] - x)
    angle = abs(a1 - a2)
    if angle > np.pi:
        angle = 2 * np.pi - angle
    return float(np.degrees(angle))


def _in_shooting_cone(sx: float, sy: float, dx: float, dy: float) -> bool:
    """
    True if defender at (dx, dy) lies inside the cone from (sx, sy) to both posts.
    Uses linear interpolation at the defender's x position.
    """
    if dx <= sx:
        return False
    t = (dx - sx) / max(GOAL_CENTRE[0] - sx, 1e-6)
    y_lo = sy + t * (GOAL_LEFT_POST[1]  - sy)
    y_hi = sy + t * (GOAL_RIGHT_POST[1] - sy)
    if y_lo > y_hi:
        y_lo, y_hi = y_hi, y_lo
    return y_lo <= dy <= y_hi


# ---------------------------------------------------------------------------
# Freeze-frame features
# ---------------------------------------------------------------------------

def freeze_frame_features(freeze_frame, shot_x: float, shot_y: float) -> dict:
    """
    Extract defensive context from StatsBomb freeze-frame data.

    Returns:
        n_defenders_in_cone       — opponents blocking the direct path to goal
        goalkeeper_dist_from_goal — how far off his line the keeper is (yards)
        n_opponents_visible       — total non-teammate players in frame
    """
    result = {
        "n_defenders_in_cone":       0,
        "goalkeeper_dist_from_goal": 5.0,  # reasonable default (keeper on line)
        "n_opponents_visible":       0,
    }

    if not hasattr(freeze_frame, '__iter__') or isinstance(freeze_frame, (str, bytes)):
        return result

    for p in freeze_frame:
        if not isinstance(p, dict):
            continue
        loc = p.get("location", [])
        try:
            px, py = float(loc[0]), float(loc[1])
        except (TypeError, IndexError, ValueError):
            continue

        if p.get("teammate", False):
            continue

        result["n_opponents_visible"] += 1

        pos = p.get("position", {})
        pos_name = pos.get("name", "") if isinstance(pos, dict) else str(pos)

        if "Goalkeeper" in pos_name:
            result["goalkeeper_dist_from_goal"] = shot_distance(px, py)
        elif _in_shooting_cone(shot_x, shot_y, px, py):
            result["n_defenders_in_cone"] += 1

    return result


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_shot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw StatsBomb shot rows into a model-ready feature DataFrame.

    Input  : raw shots_raw.parquet columns
    Output : DataFrame with FEATURE_COLS + metadata + target (goal)
    """
    # If a 'type' column is present (raw events DataFrame), filter to shots only.
    # The pre-saved shots_raw.parquet has already been filtered, so 'type' is absent.
    if "type" in df.columns:
        shots = df[df["type"] == "Shot"].copy()
    else:
        shots = df.copy()

    # Coordinates — location can be a Python list or numpy array depending on source
    def _parse_coord(l, idx):
        try:
            return float(l[idx])
        except (TypeError, IndexError, ValueError):
            return np.nan

    shots["x"] = shots["location"].apply(lambda l: _parse_coord(l, 0))
    shots["y"] = shots["location"].apply(lambda l: _parse_coord(l, 1))
    shots = shots.dropna(subset=["x", "y"])

    # Geometry
    shots["distance"] = shots.apply(lambda r: shot_distance(r["x"], r["y"]), axis=1)
    shots["angle"]    = shots.apply(lambda r: shot_angle(r["x"], r["y"]),    axis=1)

    # Body part
    bp = shots.get("shot_body_part", pd.Series("", index=shots.index)).fillna("")
    shots["is_header"] = bp.str.contains("Head", case=False).astype(int)

    # Shot type
    st = shots.get("shot_type", pd.Series("", index=shots.index)).fillna("")
    shots["is_penalty"]  = st.str.contains("Penalty",   case=False).astype(int)
    shots["is_freekick"] = st.str.contains("Free Kick", case=False).astype(int)
    shots["is_open_play"]= st.str.contains("Open Play", case=False).astype(int)

    # Play pattern
    pp = shots.get("play_pattern", pd.Series("", index=shots.index)).fillna("")
    shots["is_corner"]  = pp.str.contains("Corner",  case=False).astype(int)
    shots["is_counter"] = pp.str.contains("Counter", case=False).astype(int)

    # Situational flags
    shots["under_pressure"] = shots.get("under_pressure", pd.Series(False, index=shots.index)).fillna(False).astype(int)
    shots["is_first_time"]  = shots.get("shot_first_time",   pd.Series(False, index=shots.index)).fillna(False).astype(int)
    shots["is_one_on_one"]  = shots.get("shot_one_on_one",   pd.Series(False, index=shots.index)).fillna(False).astype(int)

    # Time
    shots["match_time"]    = shots["minute"].fillna(0) + shots["second"].fillna(0) / 60.0
    shots["is_second_half"]= (shots["period"].fillna(1) >= 2).astype(int)

    # Technique dummies
    tech = shots.get("shot_technique", pd.Series("Normal", index=shots.index)).fillna("Normal")
    for t in ["Normal", "Lob", "Volley", "Half Volley"]:
        shots[f"technique_{t}"] = (tech == t).astype(int)

    # Freeze-frame
    ff_col = "shot_freeze_frame"
    if ff_col in shots.columns:
        ff_features = shots.apply(
            lambda r: freeze_frame_features(r[ff_col], r["x"], r["y"]), axis=1
        )
        ff_df = pd.DataFrame(ff_features.tolist(), index=shots.index)
        shots = pd.concat([shots, ff_df], axis=1)
    else:
        shots["n_defenders_in_cone"]       = 0
        shots["goalkeeper_dist_from_goal"] = 5.0
        shots["n_opponents_visible"]       = 0

    # Target variable
    outcome = shots.get("shot_outcome", pd.Series("", index=shots.index)).fillna("")
    shots["goal"] = (outcome == "Goal").astype(int)

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in shots.columns:
            shots[col] = 0

    # Fill any remaining NaNs in feature columns
    shots[FEATURE_COLS] = shots[FEATURE_COLS].fillna(0)

    log.info(
        f"Built features: {len(shots):,} shots  "
        f"|  goal rate: {shots['goal'].mean():.2%}  "
        f"|  {len(FEATURE_COLS)} features"
    )
    return shots


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not RAW_PARQUET.exists():
        raise FileNotFoundError(
            f"Raw shots not found: {RAW_PARQUET}\nRun: python data/download_data.py"
        )

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading {RAW_PARQUET}")
    df = pd.read_parquet(RAW_PARQUET)

    featured = build_shot_features(df)

    # Preserve metadata columns for analysis
    meta = ["id", "match_id", "competition", "split", "team", "player",
            "minute", "second", "period", "shot_statsbomb_xg",
            "home_team", "away_team"]
    meta_existing = [c for c in meta if c in featured.columns]

    out = featured[meta_existing + FEATURE_COLS + ["goal"]].copy()
    out.to_parquet(OUT_PARQUET, index=False)
    log.info(f"✓ Saved {len(out):,} rows → {OUT_PARQUET}")

    # Split stats
    if "split" in out.columns:
        for split, grp in out.groupby("split"):
            log.info(f"  {split:>6}: {len(grp):>5,} shots  "
                     f"goal rate {grp['goal'].mean():.2%}")


if __name__ == "__main__":
    main()
