"""
betting/ev_calculator.py
-------------------------
Expected Value (EV) engine for all standard soccer betting markets.

Given a MatchProbabilities object (from models/match_simulator.py) and
a set of bookmaker odds, this module:
  1. Calculates EV for every offered market
  2. Identifies value bets (EV > threshold)
  3. Sizes bets using fractional Kelly

Typical workflow:
    from models.match_simulator import MatchSimulator
    from betting.ev_calculator import EVEngine

    sim    = MatchSimulator()
    probs  = sim.simulate(xg_home=1.4, xg_away=0.9)
    engine = EVEngine(bankroll=1000, kelly_fraction=0.25)
    bets   = engine.evaluate(probs, market_odds, match_label="Man City vs Arsenal")
    engine.print_report(bets)

Usage (standalone):
    python betting/ev_calculator.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.match_simulator import MatchProbabilities

from betting.kelly import BetRecommendation, multi_market_kelly

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market definitions
# ---------------------------------------------------------------------------

STANDARD_MARKETS = [
    "home_win", "draw", "away_win",           # 1X2
    "over_0.5", "under_0.5",                  # total goals
    "over_1.5", "under_1.5",
    "over_2.5", "under_2.5",
    "over_3.5", "under_3.5",
    "btts_yes", "btts_no",
]


@dataclass
class MatchEVReport:
    match_label:  str
    xg_home:      float
    xg_away:      float
    value_bets:   list[tuple[str, float, BetRecommendation]] = field(default_factory=list)
    all_markets:  list[BetRecommendation]                    = field(default_factory=list)

    @property
    def n_value_bets(self) -> int:
        return len(self.value_bets)

    @property
    def total_stake(self) -> float:
        return sum(s for _, s, _ in self.value_bets)


class EVEngine:
    """
    Converts model probabilities → EV → Kelly-sized bet recommendations.

    Args:
        bankroll       : current bankroll in any currency unit
        kelly_fraction : Kelly multiplier (0.25 = quarter Kelly)
        min_edge       : minimum EV to flag a value bet (default 0.02 = 2%)
        max_bet_pct    : hard cap as fraction of bankroll per bet (default 5%)
        vig_threshold  : skip a market if bookmaker margin > this (default 10%)
    """

    def __init__(
        self,
        bankroll:       float = 1000.0,
        kelly_fraction: float = 0.25,
        min_edge:       float = 0.02,
        max_bet_pct:    float = 0.05,
        vig_threshold:  float = 0.10,
    ):
        self.bankroll       = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge       = min_edge
        self.max_bet_pct    = max_bet_pct
        self.vig_threshold  = vig_threshold

    def _model_probs(self, match_probs: "MatchProbabilities") -> dict[str, float]:
        """Map MatchProbabilities fields to market-name keys."""
        p = {
            "home_win": match_probs.p_home_win,
            "draw":     match_probs.p_draw,
            "away_win": match_probs.p_away_win,
            "btts_yes": match_probs.btts,
            "btts_no":  1 - match_probs.btts,
        }
        for line, (p_over, p_under) in match_probs.over_under.items():
            p[f"over_{line}"]  = p_over
            p[f"under_{line}"] = p_under
        return p

    def evaluate(
        self,
        match_probs: "MatchProbabilities",
        market_odds: dict[str, float],
        match_label: str = "",
    ) -> MatchEVReport:
        """
        Evaluate all markets and return a MatchEVReport.

        Args:
            match_probs : output from MatchSimulator.simulate()
            market_odds : {market_name: decimal_odds}  e.g. {"home_win": 2.10, ...}
            match_label : human-readable match identifier
        """
        model_p = self._model_probs(match_probs)

        # Filter to markets where we have both a model probability and bookmaker odds
        available = {k: v for k, v in market_odds.items() if k in model_p}

        # Screen for excessive vig
        screened = {}
        for market, odds in available.items():
            vig = self._estimate_vig(market, odds, market_odds)
            if vig > self.vig_threshold:
                log.debug(f"  Skipping {market}: estimated vig {vig:.1%} > {self.vig_threshold:.1%}")
            else:
                screened[market] = odds

        value_bets = multi_market_kelly(
            market_probs={k: model_p[k] for k in screened},
            market_odds=screened,
            bankroll=self.bankroll,
            fraction=self.kelly_fraction,
        )
        for _, _, rec in value_bets:
            rec.event = match_label

        # Build full market list (for reporting, including non-value)
        all_markets = []
        from betting.kelly import BetRecommendation, expected_value, kelly_fraction
        for market in screened:
            p    = model_p[market]
            odds = screened[market]
            ev   = expected_value(p, odds)
            kf   = kelly_fraction(p, odds)
            rec  = BetRecommendation(
                event=match_label, market=market,
                model_prob=p, decimal_odds=odds,
                ev=ev, kelly_full=kf,
                kelly_quarter=kf * 0.25, kelly_half=kf * 0.50,
                value_bet=(ev > self.min_edge),
            )
            all_markets.append(rec)

        all_markets.sort(key=lambda r: r.ev, reverse=True)

        report = MatchEVReport(
            match_label=match_label,
            xg_home=match_probs.xg_home,
            xg_away=match_probs.xg_away,
            value_bets=value_bets,
            all_markets=all_markets,
        )
        return report

    def print_report(self, report: MatchEVReport) -> None:
        print(f"\n{'='*66}")
        print(f"  EV REPORT: {report.match_label}")
        print(f"  xG: Home {report.xg_home:.2f}  Away {report.xg_away:.2f}")
        print(f"{'='*66}")
        print(f"{'Market':<18} {'Model %':>8} {'Odds':>7} {'EV':>8} {'Q-Kelly':>9} {'Stake':>9}")
        print("-"*66)

        for rec in report.all_markets:
            stake_str = (
                f"£{report.value_bets[[m for m,_,_ in report.value_bets].index(rec.market)][1]:.2f}"
                if rec.value_bet and any(m == rec.market for m, _, _ in report.value_bets)
                else "—"
            )
            flag = " ← VALUE" if rec.value_bet else ""
            print(
                f"{rec.market:<18} {rec.model_prob:>7.1%} {rec.decimal_odds:>7.2f} "
                f"{rec.ev:>+8.3f} {rec.kelly_quarter:>8.2%}  "
                f"{stake_str:>8}{flag}"
            )

        print(f"{'='*66}")
        print(f"  Value bets: {report.n_value_bets}  |  Total stake: £{report.total_stake:.2f}")

    @staticmethod
    def _estimate_vig(market: str, odds: float, all_odds: dict[str, float]) -> float:
        """Rough vig estimate using paired markets (1X2 or over/under)."""
        pairs = {
            "home_win": ["draw", "away_win"],
            "over_2.5": ["under_2.5"],
            "btts_yes": ["btts_no"],
        }
        for anchor, companions in pairs.items():
            if market in ([anchor] + companions):
                involved = [anchor] + companions
                if all(m in all_odds for m in involved):
                    implied_sum = sum(1 / all_odds[m] for m in involved)
                    return float(implied_sum - 1.0)
        return 0.0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate the EV engine on a synthetic in-play scenario."""
    import sys
    ROOT_DIR = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT_DIR))

    from models.match_simulator import MatchSimulator

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Pre-match scenario
    sim    = MatchSimulator()
    probs  = sim.simulate(xg_home=1.4, xg_away=0.9)

    # Simulated bookmaker odds (realistic for a 1.4 vs 0.9 xG match)
    pre_match_odds = {
        "home_win":  2.00,
        "draw":      3.40,
        "away_win":  3.80,
        "over_2.5":  1.85,
        "under_2.5": 1.95,
        "over_1.5":  1.35,
        "under_1.5": 3.10,
        "btts_yes":  1.80,
        "btts_no":   2.00,
    }

    engine = EVEngine(bankroll=1000, kelly_fraction=0.25, min_edge=0.02)
    report = engine.evaluate(probs, pre_match_odds, match_label="Home vs Away (pre-match)")
    engine.print_report(report)

    print("\n--- In-play: 60', Home 1–0 Away, 5 mins of xG remain each ---")
    inplay_probs = sim.simulate_inplay(
        xg_home_remaining=0.6,
        xg_away_remaining=0.5,
        current_home=1,
        current_away=0,
    )
    live_odds = {
        "home_win":  1.40,
        "draw":      3.90,
        "away_win":  8.00,
        "over_2.5":  2.50,
        "under_2.5": 1.52,
    }
    report2 = engine.evaluate(inplay_probs, live_odds, match_label="60' — 1:0")
    engine.print_report(report2)


if __name__ == "__main__":
    main()
