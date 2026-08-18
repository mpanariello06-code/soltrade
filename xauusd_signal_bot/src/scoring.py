"""Weighted aggregation of the nine analysis components into 0-100 scores.

Each engine reports on its own natural scale.  Here every component is
normalised to ``[0, 1]`` and multiplied by its configured weight, so weights in
:class:`config.Weights` can be re-tuned without touching engine internals and
always sum to exactly 100.

AVOIDING DOUBLE-COUNTING
------------------------
The engines are kept deliberately disjoint in their inputs:

* trend      - EMAs and ADX/DI only
* momentum   - RSI / MACD / stochastic / ROC only (never EMA direction)
* structure  - confirmed swing pivots only
* liquidity  - reference levels and sweeps
* S/R        - clustered zones and distance to them
* volume     - tick volume only
* volatility - ATR/Bollinger width only, direction-neutral
* price action - the shape of the signal candle only

The HTF component re-runs the trend/momentum/structure engines on M15 and H1;
that is intentional confirmation across *timeframes*, not a second count of the
same bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .utils import ComponentScore, clamp

#: config attribute name for each component's weight
COMPONENT_WEIGHT_KEYS: Dict[str, str] = {
    "trend": "trend",
    "htf": "htf",
    "momentum": "momentum",
    "structure": "structure",
    "liquidity": "liquidity",
    "support_resistance": "support_resistance",
    "volume": "volume",
    "volatility": "volatility",
    "price_action": "price_action",
}

#: a component counts as "confirming" once it reaches this fraction of its max
CONFIRMATION_THRESHOLD = 0.45


@dataclass
class ScoreCard:
    """Aggregate result of one evaluation."""

    bullish_score: float = 0.0
    bearish_score: float = 0.0
    components: Dict[str, ComponentScore] = field(default_factory=dict)
    weighted_bull: Dict[str, float] = field(default_factory=dict)
    weighted_bear: Dict[str, float] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        """Provisional direction before any filtering."""
        if self.bullish_score > self.bearish_score:
            return "BUY"
        if self.bearish_score > self.bullish_score:
            return "SELL"
        return "NONE"

    @property
    def separation(self) -> float:
        """Absolute gap between the two sides."""
        return abs(self.bullish_score - self.bearish_score)

    def dominant_score(self) -> float:
        """Score of the stronger side."""
        return max(self.bullish_score, self.bearish_score)

    def component_score(self, name: str, direction: str) -> float:
        """Weighted contribution of one component to one direction."""
        table = self.weighted_bull if direction == "BUY" else self.weighted_bear
        return float(table.get(name, 0.0))

    def confirmations(self, direction: str) -> Dict[str, bool]:
        """Per-component confirmation flags used in the Telegram message."""
        flags: Dict[str, bool] = {}
        for name, component in self.components.items():
            bull, bear = component.normalised()
            value = bull if direction == "BUY" else bear
            flags[name] = value >= CONFIRMATION_THRESHOLD
        return flags

    def confirmation_count(self, direction: str) -> int:
        """How many components confirm ``direction``."""
        return sum(1 for ok in self.confirmations(direction).values() if ok)


def compute_scorecard(components: Dict[str, ComponentScore], config) -> ScoreCard:
    """Combine component scores into bullish/bearish totals in ``[0, 100]``."""
    weights = config.weights
    card = ScoreCard(components=dict(components))

    total_bull = total_bear = 0.0
    for name, component in components.items():
        weight_key = COMPONENT_WEIGHT_KEYS.get(name)
        if weight_key is None:
            continue
        weight = float(getattr(weights, weight_key, 0.0))
        bull_fraction, bear_fraction = component.normalised()
        bull_points = bull_fraction * weight
        bear_points = bear_fraction * weight
        card.weighted_bull[name] = round(bull_points, 3)
        card.weighted_bear[name] = round(bear_points, 3)
        total_bull += bull_points
        total_bear += bear_points

    card.bullish_score = round(clamp(total_bull, 0.0, 100.0), 2)
    card.bearish_score = round(clamp(total_bear, 0.0, 100.0), 2)
    return card


def confidence_band(score: float, config) -> str:
    """Map a score onto a confidence label."""
    for minimum, label in config.confidence_bands:
        if score >= minimum:
            return label
    return "NO_SIGNAL"


def reason_summary(card: ScoreCard, direction: str, limit: int = 4) -> str:
    """Human-readable summary of the components that drove the signal."""
    table = card.weighted_bull if direction == "BUY" else card.weighted_bear
    ranked: List[Tuple[str, float]] = sorted(table.items(), key=lambda kv: kv[1], reverse=True)
    parts = [f"{name}={value:.1f}" for name, value in ranked[:limit] if value > 0.1]
    if not parts:
        return "no dominant component"
    return "; ".join(parts)
