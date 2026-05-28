"""
data/download_data.py
----------------------
Downloads StatsBomb open-data shots for multiple competitions and
assembles a single shot-level Parquet dataset.

StatsBomb open data is free for public / academic use:
  https://github.com/statsbomb/open-data

Default competitions downloaded
  Training  : FIFA World Cup 2018  +  La Liga 2017/18 & 2018/19
  Test split: UEFA Euro 2020

The script is idempotent — if shots_raw.parquet already exists it
verifies it and skips downloading, unless --force is passed.

Usage:
    python data/download_data.py
    python data/download_data.py --force
    python data/download_data.py --competitions 43,3 11,1 11,4 55,43
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT = RAW_DIR / "shots_raw.parquet"

# (competition_id, season_id, label, split)
# Train on three diverse competitions from 2017-2019 for temporal realism.
# Test on Euro 2020 (held out — never seen during training).
DEFAULT_COMPETITIONS = [
    (43, 3,  "FIFA World Cup 2018",  "train"),
    (11, 1,  "La Liga 2017/18",      "train"),
    (11, 4,  "La Liga 2018/19",      "train"),
    (55, 43, "UEFA Euro 2020",       "test"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def download_competition(
    competition_id: int,
    season_id: int,
    label: str,
    split: str,
) -> pd.DataFrame:
    """Download all shot events for one competition/season."""
    from statsbombpy import sb

    log.info(f"Fetching matches for {label} ...")
    try:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
    except Exception as exc:
        log.error(f"  Could not load matches for {label}: {exc}")
        return pd.DataFrame()

    log.info(f"  {len(matches)} matches found — downloading events ...")

    all_shots: list[pd.DataFrame] = []
    for _, match in tqdm(matches.iterrows(), total=len(matches), desc=label, leave=False):
        try:
            events = sb.events(match_id=match["match_id"])
            shots = events[events["type"] == "Shot"].copy()
            if shots.empty:
                continue
            shots["competition"]     = label
            shots["split"]           = split
            shots["match_id"]        = match["match_id"]
            shots["home_team"]       = match.get("home_team", "")
            shots["away_team"]       = match.get("away_team", "")
            shots["home_score"]      = match.get("home_score", None)
            shots["away_score"]      = match.get("away_score", None)
            all_shots.append(shots)
        except Exception as exc:
            log.warning(f"  Skipped match {match['match_id']}: {exc}")

    if not all_shots:
        log.warning(f"  No shots retrieved for {label}")
        return pd.DataFrame()

    df = pd.concat(all_shots, ignore_index=True)
    log.info(f"  {len(df):,} shots from {label}")
    return df


def main(competitions: list[tuple] | None = None, force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists() and not force:
        df = pd.read_parquet(OUTPUT)
        log.info(f"shots_raw.parquet already exists ({len(df):,} rows).  Use --force to re-download.")
        _print_summary(df)
        return

    if competitions is None:
        competitions = DEFAULT_COMPETITIONS

    frames: list[pd.DataFrame] = []
    for comp_id, season_id, label, split in competitions:
        df = download_competition(comp_id, season_id, label, split)
        if not df.empty:
            frames.append(df)

    if not frames:
        log.error("No data downloaded — aborting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Keep only columns that exist and are useful
    keep = [
        "id", "match_id", "competition", "split",
        "home_team", "away_team", "home_score", "away_score",
        "period", "minute", "second", "timestamp",
        "team", "player",
        "location", "shot_end_location",
        "shot_outcome", "shot_technique", "shot_body_part", "shot_type",
        "shot_statsbomb_xg",
        "shot_freeze_frame",
        "shot_first_time", "shot_one_on_one", "shot_aerial_won",
        "shot_key_pass_id",
        "under_pressure",
        "play_pattern",
        "possession_team",
    ]
    keep_existing = [c for c in keep if c in combined.columns]
    combined = combined[keep_existing].copy()

    combined.to_parquet(OUTPUT, index=False)
    log.info(f"\n✓ Saved {len(combined):,} shots → {OUTPUT}")
    _print_summary(combined)


def _print_summary(df: pd.DataFrame) -> None:
    log.info("\nDataset summary:")
    if "competition" in df.columns:
        for comp, grp in df.groupby("competition"):
            goals = (grp["shot_outcome"] == "Goal").sum() if "shot_outcome" in grp else 0
            log.info(f"  {comp:<35} {len(grp):>5,} shots  {goals:>4,} goals  "
                     f"({goals/len(grp):.1%} conversion)")
    log.info(f"  {'TOTAL':<35} {len(df):>5,} shots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download StatsBomb shot data")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument(
        "--competitions", nargs="*", metavar="COMP_ID,SEASON_ID",
        help="Override competitions: e.g. 43,3 11,1 (defaults to World Cup + La Liga + Euro)"
    )
    args = parser.parse_args()

    comps = None
    if args.competitions:
        comps = []
        for entry in args.competitions:
            parts = entry.split(",")
            if len(parts) >= 2:
                comps.append((int(parts[0]), int(parts[1]), f"comp_{parts[0]}_{parts[1]}", "train"))

    main(competitions=comps, force=args.force)
