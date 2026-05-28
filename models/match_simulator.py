"""
models/match_simulator.py
--------------------------
Poisson-based Monte Carlo match simulator.

Given team-level expected goals (xG_home, xG_away), simulates N matches
and returns a probability distribution over:
  - Match result  : home win / draw / away win
  - Total goals   : over/under markets (0.5, 1.5, 2.5, 3.5, 4.5)
  - Both teams score (BTTS)
  - Exact score   : top-20 most likely scorelines

Mathematical basis
  Each team's goals modelled as independent Poisson(λ) where λ = team xG.
  Dixon-Coles correction applied for low-scoring outcomes (0-0, 1-0, 0-1, 1-1)
  to account for the positive correlation between home and away goals.

Usage:
    from models.match_simulator import MatchSimulator
    sim = MatchSimulator()
    probs = sim.simulate(xg_home=1.3, xg_away=0.9)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.stats import poisson


@dataclass
class MatchProbabilities:
    """Container for all derived match probabilities."""
    xg_home: float
    xg_away: float

    # Match result
    p_home_win: float = 0.0
    p_draw:     float = 0.0
    p_away_win: float = 0.0

    # Goals markets
    over_under: dict = field(default_factory=dict)   # {0.5: (p_over, p_under), ...}
    btts:       float = 0.0                           # both teams to score

    # Scoreline distribution (sorted by probability)
    scorelines: list[tuple[int, int, float]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"xG Home={self.xg_home:.2f}  Away={self.xg_away:.2f}",
            f"Result:  Home {self.p_home_win:.1%}  Draw {self.p_draw:.1%}  Away {self.p_away_win:.1%}",
            f"BTTS:    {self.btts:.1%}",
            "Over/Under:",
        ]
        for line, (p_over, p_under) in sorted(self.over_under.items()):
            lines.append(f"  {line}:  Over {p_over:.1%}  Under {p_under:.1%}")
        lines.append("Top 5 scorelines:")
        for h, a, p in self.scorelines[:5]:
            lines.append(f"  {h}–{a}  {p:.2%}")
        return "\n".join(lines)


class MatchSimulator:
    """
    Analytical Poisson match simulator with Dixon-Coles correction.

    The Dixon-Coles (1997) paper showed that the standard bivariate Poisson
    under-predicts 0-0 and 1-1 draws and over-predicts 1-0 and 0-1 results.
    The rho parameter corrects this dependency at low scores.
    """

    def __init__(self, max_goals: int = 10, rho: float = -0.13):
        """
        Args:
            max_goals : upper bound for Poisson truncation (goals per side)
            rho       : Dixon-Coles correlation parameter (typically -0.1 to -0.2)
        """
        self.max_goals = max_goals
        self.rho       = rho

    def _tau(self, x: int, y: int, lam: float, mu: float) -> float:
        """Dixon-Coles correction factor for scores (0,0), (1,0), (0,1), (1,1)."""
        if x == 0 and y == 0:
            return 1 - lam * mu * self.rho
        if x == 1 and y == 0:
            return 1 + mu * self.rho
        if x == 0 and y == 1:
            return 1 + lam * self.rho
        if x == 1 and y == 1:
            return 1 - self.rho
        return 1.0

    def simulate(self, xg_home: float, xg_away: float) -> MatchProbabilities:
        """
        Return full probability distribution for a match.

        Args:
            xg_home : expected goals for the home team
            xg_away : expected goals for the away team
        """
        lam = max(xg_home, 1e-6)
        mu  = max(xg_away, 1e-6)

        # Build score probability matrix [home_goals x away_goals]
        p_score = np.zeros((self.max_goals + 1, self.max_goals + 1))
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                tau = self._tau(h, a, lam, mu)
                p_score[h, a] = (
                    poisson.pmf(h, lam) *
                    poisson.pmf(a, mu)  *
                    tau
                )

        # Normalise (tau can shift the sum slightly)
        p_score /= p_score.sum()

        result = MatchProbabilities(xg_home=lam, xg_away=mu)

        # Match result
        result.p_home_win = float(np.tril(p_score, k=-1).sum())
        result.p_draw     = float(np.trace(p_score))
        result.p_away_win = float(np.triu(p_score, k=1).sum())

        # BTTS
        result.btts = float(1 - p_score[0, :].sum() - p_score[:, 0].sum() + p_score[0, 0])

        # Over/Under
        total = np.array(
            [[h + a for a in range(self.max_goals + 1)]
             for h in range(self.max_goals + 1)]
        )
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            p_over  = float(p_score[total >  line].sum())
            p_under = float(p_score[total <= line].sum())
            result.over_under[line] = (p_over, p_under)

        # Scoreline distribution
        scorelines = []
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                scorelines.append((h, a, float(p_score[h, a])))
        result.scorelines = sorted(scorelines, key=lambda s: s[2], reverse=True)

        return result

    def simulate_inplay(
        self,
        xg_home_remaining: float,
        xg_away_remaining: float,
        current_home: int,
        current_away: int,
    ) -> MatchProbabilities:
        """
        In-play version: given remaining xG and current score,
        return updated match probabilities for the final result.
        """
        # Simulate remaining goals distribution
        remaining = self.simulate(xg_home_remaining, xg_away_remaining)

        # Shift result by current score
        shifted_scorelines = [
            (h + current_home, a + current_away, p)
            for h, a, p in remaining.scorelines
        ]

        p_home = sum(p for h, a, p in shifted_scorelines if h > a)
        p_draw = sum(p for h, a, p in shifted_scorelines if h == a)
        p_away = sum(p for h, a, p in shifted_scorelines if h < a)

        result              = MatchProbabilities(xg_home=xg_home_remaining, xg_away=xg_away_remaining)
        result.p_home_win   = p_home
        result.p_draw       = p_draw
        result.p_away_win   = p_away
        result.scorelines   = sorted(shifted_scorelines, key=lambda s: s[2], reverse=True)

        # Goals markets on TOTAL goals (existing + remaining)
        total_goals         = current_home + current_away
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            if total_goals > line:
                result.over_under[line] = (1.0, 0.0)   # already gone over
            else:
                needed = line - total_goals
                p_over  = sum(
                    p for h, a, p in remaining.scorelines
                    if (h + a) > needed     # remaining goals exceed remaining gap
                )
                result.over_under[line] = (p_over, 1 - p_over)

        return result
