"""Signal filtering: adaptive thresholds, conflict, fakeout, session, spread,
cooldown and risk-reward gates.

Every filter returns a short, stable ``reason`` string.  Those strings are
written to ``evaluations.csv`` for every evaluated candle, which is what makes
it possible to audit *why* the system stayed flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Tuple

import pandas as pd

from .scoring import ScoreCard
from .targets import Targets
from .utils import fnum, safe_div, session_allowed

@dataclass
class GateState:
    """What the tracker knows about recent signal activity."""

    last_signal_time: Optional[datetime] = None
    last_signal_direction: str = ""
    last_same_direction_time: Optional[datetime] = None
    signals_today: int = 0
    active_count: int = 0


@dataclass
class FilterInput:
    """Everything the filter chain needs for one candidate direction."""

    df: pd.DataFrame
    card: ScoreCard
    direction: str
    regime: str
    volatility_band: str
    session: str
    spread_points: float
    candle_time: datetime
    htf_alignment: str
    timeframe_minutes: int
    gate: GateState = field(default_factory=GateState)
    targets: Optional[Targets] = None


@dataclass
class FilterOutcome:
    """Result of running the chain."""

    passed: bool
    reason: str = ""
    threshold: float = 0.0


# --------------------------------------------------------------------------- #
# adaptive threshold
# --------------------------------------------------------------------------- #
def adaptive_threshold(
    regime: str, volatility_band: str, htf_alignment: str, config
) -> Optional[float]:
    """Minimum score this candidate must reach.

    Returns ``None`` when conditions forbid signalling outright (extreme
    volatility with ``block_on_extreme_volatility`` enabled).
    """
    if volatility_band == "EXTREME":
        if config.block_on_extreme_volatility:
            return None
        return min(100.0, config.base_threshold + config.extreme_volatility_extra_score)

    threshold = float(config.regime_thresholds.get(regime, config.base_threshold))
    if htf_alignment == "COUNTER":
        threshold += config.counter_trend_extra_score
    return min(threshold, 100.0)


# --------------------------------------------------------------------------- #
# individual filters
# --------------------------------------------------------------------------- #
def check_session(data: FilterInput, config) -> Optional[str]:
    """Reject candles outside the configured trading sessions."""
    if not session_allowed(data.session, config.allowed_sessions):
        return f"session not allowed ({data.session})"
    return None


def check_spread(data: FilterInput, config) -> Optional[str]:
    """Reject when the quoted spread is abnormal, absolutely or versus ATR."""
    spread = data.spread_points
    if spread is None or not pd.notna(spread):
        return None  # spread unavailable (e.g. backtest) - not a rejection
    if spread > config.max_spread_points:
        return f"excessive spread ({spread:.0f} > {config.max_spread_points:.0f} points)"
    atr_value = fnum(data.df["atr"].iloc[-1])
    spread_price = spread * config.point_value
    if atr_value > 0 and safe_div(spread_price, atr_value) > config.max_spread_atr_ratio:
        return f"excessive spread vs ATR ({spread_price:.2f} vs ATR {atr_value:.2f})"
    return None


def check_volatility(data: FilterInput, config) -> Optional[str]:
    """Reject extreme volatility when configured to do so."""
    if data.volatility_band == "EXTREME" and config.block_on_extreme_volatility:
        return "extreme volatility"
    return None


def check_threshold(data: FilterInput, config) -> Optional[str]:
    """Reject when the dominant score is below the adaptive threshold."""
    threshold = adaptive_threshold(data.regime, data.volatility_band, data.htf_alignment, config)
    if threshold is None:
        return "extreme volatility"
    score = data.card.bullish_score if data.direction == "BUY" else data.card.bearish_score
    if score < threshold:
        return f"score {score:.1f} below threshold {threshold:.1f}"
    return None


def check_separation(data: FilterInput, config) -> Optional[str]:
    """Reject conflicted markets where both sides score highly (spec 22)."""
    bull, bear = data.card.bullish_score, data.card.bearish_score
    own, other = (bull, bear) if data.direction == "BUY" else (bear, bull)
    if own - other < config.min_score_separation:
        return (
            f"insufficient separation ({own:.1f} vs {other:.1f}, "
            f"need {config.min_score_separation:.0f})"
        )
    return None


def check_fakeout(data: FilterInput, config) -> Optional[str]:
    """Reject unconfirmed breakouts and setups the higher timeframe rejects."""
    if not config.enable_fakeout_filter:
        return None
    if data.df is None or len(data.df) < 2:
        return "insufficient data for fakeout check"

    row, prev = data.df.iloc[-1], data.df.iloc[-2]
    card = data.card
    direction = data.direction

    sr_details = card.components["support_resistance"].details if "support_resistance" in card.components else {}
    pa_details = card.components["price_action"].details if "price_action" in card.components else {}
    structure_details = card.components["structure"].details if "structure" in card.components else {}
    htf_component = card.components.get("htf")

    relative_volume = fnum(row.get("rel_volume"), 1.0)
    strong_body = bool(pa_details.get("strong_body"))
    state = str(sr_details.get("state", "NEUTRAL"))

    # 1. Breakout with neither volume nor a decisive body behind it.
    breaking = (direction == "BUY" and state == "BREAKOUT") or (
        direction == "SELL" and state == "BREAKDOWN"
    )
    if breaking and relative_volume < 1.0 and not strong_body:
        return "breakout without volume or body confirmation"

    # 2. Price closed back inside the range it broke on the previous candle.
    last_high = structure_details.get("last_swing_high")
    last_low = structure_details.get("last_swing_low")
    close, prev_close = fnum(row["close"]), fnum(prev["close"])
    if direction == "BUY" and last_high is not None and prev_close > last_high >= close:
        return "failed breakout (closed back inside range)"
    if direction == "SELL" and last_low is not None and prev_close < last_low <= close:
        return "failed breakdown (closed back inside range)"

    # 3. Higher timeframe actively disagrees.
    if htf_component is not None:
        own = htf_component.bull if direction == "BUY" else htf_component.bear
        other = htf_component.bear if direction == "BUY" else htf_component.bull
        if other - own >= 5.0:
            return "higher timeframe strongly disagrees"

    # 4. Thin participation.
    if relative_volume < 0.5:
        return f"poor liquidity (relative volume {relative_volume:.2f})"

    # 5. The signal candle itself was rejected in our direction.
    if direction == "BUY" and fnum(pa_details.get("upper_wick_ratio")) > 0.60:
        return "signal candle rejected from above"
    if direction == "SELL" and fnum(pa_details.get("lower_wick_ratio")) > 0.60:
        return "signal candle rejected from below"

    return None


def check_cooldown(data: FilterInput, config) -> Optional[str]:
    """Prevent signal spam: any-direction and same-direction cooldowns."""
    gate = data.gate
    minutes = max(data.timeframe_minutes, 1)

    if gate.last_signal_time is not None:
        elapsed = (data.candle_time - gate.last_signal_time).total_seconds() / 60.0
        candles = elapsed / minutes
        if candles < config.cooldown_candles:
            return f"cooldown active ({candles:.0f}/{config.cooldown_candles} candles)"

    if gate.last_same_direction_time is not None and gate.last_signal_direction == data.direction:
        elapsed = (data.candle_time - gate.last_same_direction_time).total_seconds() / 60.0
        candles = elapsed / minutes
        if candles < config.same_direction_cooldown_candles:
            return (
                f"same-direction cooldown "
                f"({candles:.0f}/{config.same_direction_cooldown_candles} candles)"
            )
    return None


def check_limits(data: FilterInput, config) -> Optional[str]:
    """Cap signals per day and concurrently open signals."""
    if data.gate.signals_today >= config.max_signals_per_day:
        return f"daily signal limit reached ({config.max_signals_per_day})"
    if data.gate.active_count >= config.max_concurrent_active_signals:
        return f"too many active signals ({data.gate.active_count})"
    return None


def check_risk_reward(data: FilterInput, config) -> Optional[str]:
    """Reject setups whose TP2 reward does not justify the risk."""
    if data.targets is None:
        return "no valid targets"
    if data.targets.rr2 < config.min_tp2_rr:
        return f"insufficient R:R (TP2 {data.targets.rr2:.2f} < {config.min_tp2_rr:.2f})"
    return None


#: Executed in order; the first non-``None`` reason rejects the candidate.
#: Cheap/structural checks run before the ones that need targets.
PRE_TARGET_FILTERS: Tuple[Callable[[FilterInput, object], Optional[str]], ...] = (
    check_session,
    check_spread,
    check_volatility,
    check_threshold,
    check_separation,
    check_fakeout,
    check_cooldown,
    check_limits,
)

POST_TARGET_FILTERS: Tuple[Callable[[FilterInput, object], Optional[str]], ...] = (
    check_risk_reward,
)


def run_pre_target_filters(data: FilterInput, config) -> FilterOutcome:
    """Run every filter that does not need TP/SL geometry."""
    threshold = adaptive_threshold(
        data.regime, data.volatility_band, data.htf_alignment, config
    )
    for check in PRE_TARGET_FILTERS:
        reason = check(data, config)
        if reason:
            return FilterOutcome(False, reason, threshold or 0.0)
    return FilterOutcome(True, "", threshold or 0.0)


def run_post_target_filters(data: FilterInput, config, threshold: float) -> FilterOutcome:
    """Run the filters that need the finished target set."""
    for check in POST_TARGET_FILTERS:
        reason = check(data, config)
        if reason:
            return FilterOutcome(False, reason, threshold)
    return FilterOutcome(True, "", threshold)
