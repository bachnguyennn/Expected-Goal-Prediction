"""
betting/kelly.py
-----------------
Kelly Criterion and fractional Kelly bet-sizing utilities.

The Kelly Criterion (1956) determines the fraction of bankroll to bet
on a wager to maximise the long-run expected logarithm of wealth.

Full Kelly:
    f* = (b·p - q) / b
         where b = decimal_odds - 1
               p = model probability of winning
               q = 1 - p

Fractional Kelly (recommended):
    f = fraction * f*   (fraction = 0.25 or 0.5 is common in practice)

Why fractional Kelly?
  Full Kelly assumes perfectly calibrated probabilities.  In reality,
  even well-calibrated models have estimation error.  Fractional Kelly
  reduces variance and drawdowns at the cost of slightly lower growth rate.
  Professional bettors typically use quarter-Kelly (0.25).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BetRecommendation:
    event:        str
    market:       str
    model_prob:   float
    decimal_odds: float
    ev:           float        # expected value per unit staked
    kelly_full:   float        # full Kelly fraction
    kelly_quarter:float        # quarter Kelly (recommended)
    kelly_half:   float        # half Kelly
    value_bet:    bool         # True if ev > 0

    def __str__(self) -> str:
        tag = "✓ VALUE" if self.value_bet else "✗ no value"
        return (
            f"{self.event} | {self.market} @ {self.decimal_odds:.2f}  "
            f"model={self.model_prob:.3f}  EV={self.ev:+.3f}  "
            f"Kelly={self.kelly_quarter:.2%} ({tag})"
        )


def kelly_fraction(
    p: float,
    decimal_odds: float,
    min_edge: float = 0.0,
) -> float:
    """
    Return full Kelly fraction.  Negative values (no edge) are clipped to 0.

    Args:
        p             : model probability of the event occurring
        decimal_odds  : bookmaker decimal odds (e.g. 2.10 = 1.10 profit per unit)
        min_edge      : minimum EV threshold to return non-zero Kelly (default 0)
    """
    b = decimal_odds - 1.0
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (b * p - q) / b
    ev = b * p - q
    if ev <= min_edge:
        return 0.0
    return max(0.0, f)


def expected_value(p: float, decimal_odds: float) -> float:
    """EV per unit staked.  Positive means profitable long-term."""
    return p * (decimal_odds - 1.0) - (1.0 - p)


def size_bet(
    p: float,
    decimal_odds: float,
    bankroll: float,
    fraction: float = 0.25,
    max_fraction: float = 0.05,
    min_edge: float = 0.01,
) -> tuple[float, BetRecommendation]:
    """
    Return (stake, BetRecommendation) for a single market.

    Args:
        p             : model probability
        decimal_odds  : bookmaker decimal odds
        bankroll      : current bankroll
        fraction      : Kelly multiplier (0.25 = quarter Kelly)
        max_fraction  : hard cap on fraction of bankroll staked
        min_edge      : minimum EV to recommend a bet
    """
    ev  = expected_value(p, decimal_odds)
    kf  = kelly_fraction(p, decimal_odds, min_edge=min_edge)
    rec = BetRecommendation(
        event="",
        market="",
        model_prob=p,
        decimal_odds=decimal_odds,
        ev=ev,
        kelly_full=kf,
        kelly_quarter=kf * 0.25,
        kelly_half=kf * 0.50,
        value_bet=(ev > min_edge),
    )
    stake_fraction = min(kf * fraction, max_fraction)
    stake          = bankroll * stake_fraction if rec.value_bet else 0.0
    return stake, rec


def multi_market_kelly(
    market_probs: dict[str, float],
    market_odds:  dict[str, float],
    bankroll:     float = 1000.0,
    fraction:     float = 0.25,
) -> list[tuple[str, float, BetRecommendation]]:
    """
    Evaluate multiple markets and return value bets sorted by EV.

    Args:
        market_probs : {market_name: model_probability}
        market_odds  : {market_name: decimal_odds}
        bankroll     : current total bankroll
        fraction     : Kelly multiplier

    Returns:
        List of (market_name, stake, BetRecommendation) for value bets only,
        sorted by EV descending.
    """
    results = []
    for market, p in market_probs.items():
        odds = market_odds.get(market)
        if odds is None:
            continue
        stake, rec = size_bet(p, odds, bankroll, fraction=fraction)
        rec.market = market
        if rec.value_bet:
            results.append((market, stake, rec))

    results.sort(key=lambda x: x[2].ev, reverse=True)
    return results
