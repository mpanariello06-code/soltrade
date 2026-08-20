"""Trend engine - EMA structure plus directional movement (ADX/DI).

Scores on a 0-20 scale.  Deliberately built from several partially independent
conditions rather than one rule, so that a single indicator cannot on its own
produce a maximum trend score.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .utils import BoundedCache, ComponentScore, clamp, fnum, frame_fingerprint, safe_div, scale

MAX_SCORE = 20.0

# per-condition point budget (sums to MAX_SCORE)
_W_ALIGNMENT = 4.0
_W_VS_TREND_EMA = 3.0
_W_VS_SLOW_EMA = 2.0
_W_MID_SLOPE = 3.0
_W_SLOW_SLOPE = 2.0
_W_ADX_DIRECTION = 4.0
_W_DI_SPREAD = 2.0


def analyze_trend(df: pd.DataFrame, config) -> ComponentScore:
    """Score the trend of the signal timeframe.

    ``df`` must already carry the indicator columns from
    :func:`src.indicators.compute_indicators`, closed candles only.
    """
    if df is None or len(df) < 3:
        return ComponentScore("trend", 0.0, 0.0, MAX_SCORE, {"reason": "insufficient data"})

    row = df.iloc[-1]
    close = fnum(row["close"])
    ema_fast, ema_mid = fnum(row["ema_fast"]), fnum(row["ema_mid"])
    ema_slow, ema_trend = fnum(row["ema_slow"]), fnum(row["ema_trend"])
    atr_value = max(fnum(row["atr"]), 1e-9)
    adx_value = fnum(row["adx"])
    plus_di, minus_di = fnum(row["plus_di"]), fnum(row["minus_di"])

    bull = bear = 0.0

    # 1. EMA alignment ------------------------------------------------------ #
    if ema_fast > ema_mid > ema_slow:
        bull += _W_ALIGNMENT
    elif ema_fast > ema_mid:
        bull += _W_ALIGNMENT * 0.45
    if ema_fast < ema_mid < ema_slow:
        bear += _W_ALIGNMENT
    elif ema_fast < ema_mid:
        bear += _W_ALIGNMENT * 0.45

    # 2/3. price relative to the slower EMAs -------------------------------- #
    if ema_trend > 0:
        distance = safe_div(close - ema_trend, atr_value)
        if distance > 0:
            bull += _W_VS_TREND_EMA * scale(distance, 0.0, 1.0)
        else:
            bear += _W_VS_TREND_EMA * scale(-distance, 0.0, 1.0)
    if ema_slow > 0:
        if close > ema_slow:
            bull += _W_VS_SLOW_EMA
        elif close < ema_slow:
            bear += _W_VS_SLOW_EMA

    # 4/5. EMA slopes, normalised by ATR so the scale is instrument-agnostic - #
    mid_slope = safe_div(fnum(row["ema_mid_slope"]), atr_value)
    slow_slope = safe_div(fnum(row["ema_slow_slope"]), atr_value)
    if mid_slope > 0:
        bull += _W_MID_SLOPE * scale(mid_slope, 0.0, 0.10)
    else:
        bear += _W_MID_SLOPE * scale(-mid_slope, 0.0, 0.10)
    if slow_slope > 0:
        bull += _W_SLOW_SLOPE * scale(slow_slope, 0.0, 0.055)
    else:
        bear += _W_SLOW_SLOPE * scale(-slow_slope, 0.0, 0.055)

    # 6. ADX-weighted direction --------------------------------------------- #
    adx_strength = scale(adx_value, config.adx_range_threshold, config.adx_strong_threshold)
    if plus_di > minus_di:
        bull += _W_ADX_DIRECTION * adx_strength
    elif minus_di > plus_di:
        bear += _W_ADX_DIRECTION * adx_strength

    # 7. DI separation -------------------------------------------------------- #
    di_spread = abs(plus_di - minus_di)
    di_points = _W_DI_SPREAD * scale(di_spread, 3.0, 14.0)
    if plus_di > minus_di:
        bull += di_points
    elif minus_di > plus_di:
        bear += di_points

    details = {
        "adx": round(adx_value, 2),
        "plus_di": round(plus_di, 2),
        "minus_di": round(minus_di, 2),
        "ema_aligned_bull": bool(ema_fast > ema_mid > ema_slow),
        "ema_aligned_bear": bool(ema_fast < ema_mid < ema_slow),
        "above_ema200": bool(close > ema_trend),
        "trending": bool(adx_value >= config.adx_trend_threshold),
    }
    return ComponentScore(
        "trend", clamp(bull, 0.0, MAX_SCORE), clamp(bear, 0.0, MAX_SCORE), MAX_SCORE, details
    )


def trend_direction(df: pd.DataFrame, config) -> str:
    """Coarse label for one timeframe: ``BULL``, ``BEAR`` or ``NEUTRAL``."""
    score = analyze_trend(df, config)
    if score.bull >= score.bear + 4.0:
        return "BULL"
    if score.bear >= score.bull + 4.0:
        return "BEAR"
    return "NEUTRAL"


# --------------------------------------------------------------------------- #
# short-term context timeframe
# --------------------------------------------------------------------------- #
_CONTEXT_CACHE = BoundedCache(maxsize=8)

CONTEXT_MAX_SCORE = 15.0
#: Backwards-compatible alias - the component key in the scorecard is still
#: ``htf`` so every stored CSV column keeps working.
HTF_MAX_SCORE = CONTEXT_MAX_SCORE

#: How the context timeframe's three sub-analyses combine.  Trend dominates: at
#: this horizon the context exists to say "which way is the last half hour
#: leaning", not to add a second opinion on entry timing.
CONTEXT_MIX = {
    "trend": 0.45,
    "structure": 0.30,
    "momentum": 0.25,
}


def analyze_context(df_context: Optional[pd.DataFrame], config) -> ComponentScore:
    """Score the short-term context timeframe (M5 by default) as one component.

    Returns a **not-applicable** component when context is switched off or the
    frame is too short.  :func:`src.scoring.compute_scorecard` then redistributes
    its weight rather than scoring a flat zero, which would silently cap the
    maximum achievable score.

    Memoised on the frame's contents: one M5 candle spans five M1 candles, so
    the context view is unchanged for most evaluations.  The cache key is a
    content hash, so a changed frame can never return a stale result.
    """
    from .momentum import MAX_SCORE as MOMENTUM_MAX, analyze_momentum
    from .structure import MAX_SCORE as STRUCTURE_MAX, analyze_structure

    if df_context is None or len(df_context) < 30:
        return ComponentScore(
            "htf", 0.0, 0.0, CONTEXT_MAX_SCORE,
            {"reason": "no context timeframe", "context_direction": "NONE",
             "m15_direction": "NEUTRAL", "h1_direction": "NEUTRAL"},
            applicable=False,
        )

    cache_key = frame_fingerprint(df_context)
    cached = _CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    parts = {
        "trend": (analyze_trend(df_context, config), MAX_SCORE),
        "structure": (analyze_structure(df_context, config), STRUCTURE_MAX),
        "momentum": (analyze_momentum(df_context, config), MOMENTUM_MAX),
    }
    bull_fraction = bear_fraction = 0.0
    details: Dict[str, Any] = {}
    for key, (component, maximum) in parts.items():
        share = CONTEXT_MIX[key]
        bull_fraction += share * safe_div(component.bull, maximum)
        bear_fraction += share * safe_div(component.bear, maximum)
        details[f"context_{key}"] = (round(component.bull, 2), round(component.bear, 2))

    direction = trend_direction(df_context, config)
    details.update(
        {
            "context_direction": direction,
            # legacy key names, still read by htf_alignment
            "m15_direction": direction,
            "h1_direction": direction,
        }
    )

    return _CONTEXT_CACHE.put(
        cache_key,
        ComponentScore(
            "htf",
            clamp(bull_fraction * CONTEXT_MAX_SCORE, 0.0, CONTEXT_MAX_SCORE),
            clamp(bear_fraction * CONTEXT_MAX_SCORE, 0.0, CONTEXT_MAX_SCORE),
            CONTEXT_MAX_SCORE,
            details,
        ),
    )


def htf_alignment(context_component: ComponentScore, direction: str) -> str:
    """Is ``direction`` with, against or neutral to the context timeframe?

    Returns ``"ALIGNED"``, ``"COUNTER"`` or ``"NEUTRAL"``.  With context switched
    off every candidate is NEUTRAL, so no counter-trend penalty applies.
    """
    if not context_component.applicable:
        return "NEUTRAL"
    context_direction = str(context_component.details.get("context_direction", "NEUTRAL"))
    wanted = "BULL" if direction == "BUY" else "BEAR"
    opposite = "BEAR" if direction == "BUY" else "BULL"

    if context_direction == wanted:
        return "ALIGNED"
    if context_direction == opposite:
        return "COUNTER"
    return "NEUTRAL"
