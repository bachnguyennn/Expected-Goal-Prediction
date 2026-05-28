"""
inplay/engine.py
-----------------
Real-time in-play betting engine.

Processes a stream of StatsBomb match events in chronological order and
continuously updates:
  - Running xG totals (home and away)
  - Match outcome probabilities (from Poisson simulator)
  - EV + Kelly recommendations versus live bookmaker odds

In a production system this engine would receive events from a live data
feed (e.g. Opta, StatsBomb Live) and emit bet signals via a message queue.
Here it replays a recorded match event-by-event to demonstrate the logic.

Usage:
    from inplay.engine import InPlayEngine
    engine = InPlayEngine(xg_model, ev_engine, sim)
    for event in match_events:
        signal = engine.process_event(event, live_odds)
        if signal:
            print(signal)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class MatchState:
    """Running state of an in-play match."""
    minute:          int   = 0
    period:          int   = 1
    home_goals:      int   = 0
    away_goals:      int   = 0
    xg_home:         float = 0.0
    xg_away:         float = 0.0
    shots_home:      int   = 0
    shots_away:      int   = 0
    n_events:        int   = 0
    home_team:       str   = "Home"
    away_team:       str   = "Away"

    @property
    def score_str(self) -> str:
        return f"{self.home_goals}–{self.away_goals}"

    @property
    def time_str(self) -> str:
        return f"{self.minute}'"


@dataclass
class InPlaySignal:
    """Emitted after processing each event; contains updated probs + bets."""
    state:         MatchState
    p_home_win:    float
    p_draw:        float
    p_away_win:    float
    value_bets:    list
    momentum:      str    = ""    # "home_pressure" / "away_pressure" / "balanced"


class InPlayEngine:
    """
    Stateful in-play engine that updates predictions on each shot event.

    Args:
        xg_model   : calibrated scikit-learn compatible classifier
                     (predict_proba returns [P(no-goal), P(goal)])
        ev_engine  : EVEngine instance for bet sizing
        simulator  : MatchSimulator instance
        emit_every : only emit signals every N shots (reduces noise)
    """

    def __init__(
        self,
        xg_model,
        ev_engine,
        simulator,
        emit_every: int = 1,
    ):
        from features.shot_features import FEATURE_COLS
        self.xg_model    = xg_model
        self.ev_engine   = ev_engine
        self.simulator   = simulator
        self.emit_every  = emit_every
        self.feature_cols= FEATURE_COLS
        self._state      = MatchState()
        self._shot_count = 0

    def reset(self, home_team: str = "Home", away_team: str = "Away") -> None:
        self._state      = MatchState(home_team=home_team, away_team=away_team)
        self._shot_count = 0

    def process_event(
        self,
        event: dict | pd.Series,
        live_odds: dict[str, float] | None = None,
    ) -> InPlaySignal | None:
        """
        Process one event from a match timeline.

        Args:
            event     : a single row from StatsBomb events (dict or Series)
            live_odds : current bookmaker decimal odds for the main markets;
                        if None, no EV calculation is performed

        Returns:
            InPlaySignal if the event triggers an update, else None
        """
        if isinstance(event, pd.Series):
            event = event.to_dict()

        event_type = event.get("type", "")
        self._state.minute   = int(event.get("minute", self._state.minute))
        self._state.period   = int(event.get("period", self._state.period))
        self._state.n_events += 1

        # Track goals from shot outcomes
        if event_type == "Shot":
            return self._process_shot(event, live_odds)

        return None

    def _process_shot(
        self,
        event: dict,
        live_odds: dict[str, float] | None,
    ) -> InPlaySignal:
        from features.shot_features import (
            build_shot_features,
            FEATURE_COLS,
        )

        self._shot_count += 1
        team            = event.get("team", "")
        is_home         = (team == self._state.home_team)

        # Compute xG for this shot
        xg = self._predict_xg(event)

        if is_home:
            self._state.xg_home   += xg
            self._state.shots_home += 1
        else:
            self._state.xg_away   += xg
            self._state.shots_away += 1

        # Update score
        outcome = event.get("shot_outcome", "")
        if outcome == "Goal":
            if is_home:
                self._state.home_goals += 1
            else:
                self._state.away_goals += 1
            log.info(
                f"  GOAL  {self._state.time_str}  "
                f"{team}  xG={xg:.3f}  score {self._state.score_str}"
            )

        # Only emit every N shots to reduce noise
        if self._shot_count % self.emit_every != 0:
            return None

        # Remaining xG estimate: use average xG rate × remaining time
        elapsed_frac    = min(self._state.minute / 90.0, 1.0)
        remaining_frac  = max(1.0 - elapsed_frac, 0.05)
        xg_rate_home    = self._state.xg_home / max(elapsed_frac, 0.1)
        xg_rate_away    = self._state.xg_away / max(elapsed_frac, 0.1)

        probs = self.simulator.simulate_inplay(
            xg_home_remaining = xg_rate_home * remaining_frac,
            xg_away_remaining = xg_rate_away * remaining_frac,
            current_home      = self._state.home_goals,
            current_away      = self._state.away_goals,
        )

        value_bets = []
        if live_odds:
            report     = self.ev_engine.evaluate(
                probs, live_odds,
                match_label=f"{self._state.time_str} {self._state.score_str}"
            )
            value_bets = report.value_bets

        # Simple momentum signal (5-shot rolling xG delta)
        momentum = self._momentum_label()

        return InPlaySignal(
            state      = MatchState(**self._state.__dict__),
            p_home_win = probs.p_home_win,
            p_draw     = probs.p_draw,
            p_away_win = probs.p_away_win,
            value_bets = value_bets,
            momentum   = momentum,
        )

    def _predict_xg(self, event: dict) -> float:
        """Run the xG model on a single shot event."""
        try:
            row_df = pd.DataFrame([event])
            featured = _featurise_single(row_df)
            if featured.empty:
                return 0.05
            X = featured[self.feature_cols].fillna(0)
            return float(self.xg_model.predict_proba(X)[0, 1])
        except Exception:
            # Fallback: geometric estimate from distance/angle
            loc = event.get("location", [])
            if isinstance(loc, list) and len(loc) == 2:
                from features.shot_features import shot_distance, shot_angle
                d = shot_distance(loc[0], loc[1])
                a = shot_angle(loc[0], loc[1])
                return float(np.clip(0.4 * np.exp(-d / 15) * (a / 30), 0.01, 0.95))
            return 0.05

    def _momentum_label(self) -> str:
        if self._state.xg_home > self._state.xg_away * 1.5:
            return "home_pressure"
        if self._state.xg_away > self._state.xg_home * 1.5:
            return "away_pressure"
        return "balanced"


def _featurise_single(df: pd.DataFrame) -> pd.DataFrame:
    """Thin wrapper — featurises a one-row shot event DataFrame."""
    from features.shot_features import build_shot_features
    try:
        return build_shot_features(df)
    except Exception:
        return pd.DataFrame()


def replay_match(
    match_id: int,
    xg_model,
    ev_engine,
    simulator,
    live_odds: dict[str, float] | None = None,
) -> list[InPlaySignal]:
    """
    Replay a complete recorded match and return all InPlaySignals.

    Args:
        match_id  : StatsBomb match ID
        xg_model  : calibrated xG model
        ev_engine : EVEngine
        simulator : MatchSimulator
        live_odds : static odds used throughout (for demo purposes)
    """
    from statsbombpy import sb

    events  = sb.events(match_id=match_id)
    matches = sb.matches(competition_id=events["possession_team_id"].iloc[0])

    # Identify home/away from the events
    teams      = events["team"].dropna().unique()
    home_team  = teams[0] if len(teams) > 0 else "Home"
    away_team  = teams[1] if len(teams) > 1 else "Away"

    engine  = InPlayEngine(xg_model, ev_engine, simulator)
    engine.reset(home_team=home_team, away_team=away_team)

    signals = []
    for _, event in events.sort_values(["period", "minute", "second"]).iterrows():
        signal = engine.process_event(event, live_odds)
        if signal:
            signals.append(signal)

    return signals
