"""Signal filtering: adaptive thresholds, conflict, fakeout, session, spread,
cooldown and risk-reward gates.

Every filter returns a short, stable ``reason`` string.  Those strings are
written to ``evaluations.csv`` for every evaluated candle, which is what makes
it possible to audit *why* the system stayed flat.
"""

from __future__ import annotations

import math
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

    The effective threshold is built from three parts::

        base (mode + signal timeframe)  +  regime offset  +  counter-trend extra

    ``config.base_threshold`` is the mode/timeframe value folded in by
    :func:`src.runtime_state.effective_config`, so switching mode or timeframe
    from Telegram moves the whole curve without touching the regime logic.

    The result is clamped to ``[min_threshold, max_threshold]``.  Returns
    ``None`` when conditions forbid signalling outright (extreme volatility
    with ``block_on_extreme_volatility`` enabled).
    """
    base = float(config.base_threshold)
    if volatility_band == "EXTREME":
        if config.block_on_extreme_volatility:
            return None
        return config.clamp_threshold(base + config.extreme_volatility_extra_score)

    threshold = base + float(config.regime_threshold_offsets.get(regime, 0.0))
    if htf_alignment == "COUNTER":
        threshold += config.counter_trend_extra_score
    return config.clamp_threshold(threshold)


def is_near_signal(best_score: float, threshold: Optional[float], config) -> bool:
    """True when a rejected candidate came within ``near_signal_margin`` of the bar.

    Purely diagnostic: it never produces a trading signal.  The point is to
    reveal whether the active threshold is slightly too strict for the current
    market, which the score distribution alone does not show.
    """
    if threshold is None:
        return False
    margin = float(getattr(config, "near_signal_margin", 10.0))
    return bool(threshold - margin <= float(best_score) < threshold)


# --------------------------------------------------------------------------- #
# individual filters
# --------------------------------------------------------------------------- #
def check_session(data: FilterInput, config) -> Optional[str]:
    """Reject candles outside the configured trading sessions."""
    if not session_allowed(data.session, config.allowed_sessions):
        return f"session not allowed ({data.session})"
    return None


def check_spread(data: FilterInput, config) -> Optional[str]:
    """Reject when the spread is abnormal in absolute terms or vs the likely move.

    The second test is the one that matters for scalping: what counts is not the
    spread against a single M1 candle (they are nearly the same size) but the
    spread against how far price can plausibly travel inside the holding window.
    """
    spread = config.effective_spread_points(data.spread_points)
    if spread > config.max_spread_points:
        return f"excessive spread ({spread:.0f} > {config.max_spread_points:.0f} points)"

    atr_value = fnum(data.df["atr"].iloc[-1])
    if atr_value > 0:
        expected_move = atr_value * math.sqrt(max(config.max_holding_candles, 1))
        ratio = safe_div(spread * config.point_value, expected_move)
        if ratio > config.max_spread_to_expected_move:
            return (
                f"spread too large vs expected move "
                f"({config.pips(spread * config.point_value):.1f}p vs "
                f"{config.pips(expected_move):.1f}p over {config.max_holding_candles}m)"
            )
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
    """Reject setups whose TP2 reward does not justify the risk - before *and*
    after trading costs.

    The net test is not redundant: a 1.3R raw setup carrying a 1.1R cost is a
    loser dressed as a winner, and at scalping distances that is the normal
    case rather than the exception.
    """
    targets = data.targets
    if targets is None:
        return "no valid targets"
    if targets.rr2 < config.min_tp2_rr:
        return f"insufficient R:R (TP2 {targets.rr2:.2f} < {config.min_tp2_rr:.2f})"
    if targets.net_rr2 < config.min_net_tp2_rr:
        return (
            f"insufficient NET R:R after costs "
            f"(TP2 {targets.net_rr2:.2f} < {config.min_net_tp2_rr:.2f}, "
            f"cost {targets.cost_r:.2f}R)"
        )
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
