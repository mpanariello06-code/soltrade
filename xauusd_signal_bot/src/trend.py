"""Trend engine - EMA structure plus directional movement (ADX/DI).

Scores on a 0-20 scale.  Deliberately built from several partially independent
conditions rather than one rule, so that a single indicator cannot on its own
produce a maximum trend score.
"""

from __future__ import annotations

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
# higher-timeframe confirmation
# --------------------------------------------------------------------------- #
HTF_MAX_SCORE = 15.0

#: how the M15/H1 sub-analyses combine into the single HTF component.
#: The H1 (slowest) view carries the most weight; the fractions sum to 1.0.
HTF_MIX = {
    "h1_trend": 0.30,
    "m15_trend": 0.25,
    "h1_structure": 0.15,
    "m15_structure": 0.10,
    "h1_momentum": 0.10,
    "m15_momentum": 0.10,
}


_HTF_CACHE = BoundedCache(maxsize=8)


def analyze_htf(df_m15: pd.DataFrame, df_h1: pd.DataFrame, config) -> ComponentScore:
    """Blend M15 and H1 trend, structure and momentum into one 0-15 component.

    Both frames must already carry indicator columns and contain closed candles
    only.  Imports are local to keep module import order simple.

    Memoised on the contents of both HTF frames: an M15 candle spans three M5
    candles and an H1 candle spans twelve, so the higher-timeframe view is
    unchanged for most evaluations.  The cache key is a content hash, so a
    changed HTF frame can never return a stale result.
    """
    from .momentum import MAX_SCORE as MOMENTUM_MAX, analyze_momentum
    from .structure import MAX_SCORE as STRUCTURE_MAX, analyze_structure

    if df_m15 is None or df_h1 is None or len(df_m15) < 30 or len(df_h1) < 30:
        return ComponentScore("htf", 0.0, 0.0, HTF_MAX_SCORE, {"reason": "insufficient HTF data"})

    cache_key = (frame_fingerprint(df_m15), frame_fingerprint(df_h1))
    cached = _HTF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    parts = {
        "m15_trend": (analyze_trend(df_m15, config), MAX_SCORE),
        "h1_trend": (analyze_trend(df_h1, config), MAX_SCORE),
        "m15_structure": (analyze_structure(df_m15, config), STRUCTURE_MAX),
        "h1_structure": (analyze_structure(df_h1, config), STRUCTURE_MAX),
        "m15_momentum": (analyze_momentum(df_m15, config), MOMENTUM_MAX),
        "h1_momentum": (analyze_momentum(df_h1, config), MOMENTUM_MAX),
    }

    bull_fraction = bear_fraction = 0.0
    details = {}
    for key, (component, maximum) in parts.items():
        share = HTF_MIX[key]
        bull_fraction += share * safe_div(component.bull, maximum)
        bear_fraction += share * safe_div(component.bear, maximum)
        details[key] = (round(component.bull, 2), round(component.bear, 2))

    m15_direction = trend_direction(df_m15, config)
    h1_direction = trend_direction(df_h1, config)
    details.update({"m15_direction": m15_direction, "h1_direction": h1_direction})

    return _HTF_CACHE.put(
        cache_key,
        ComponentScore(
            "htf",
            clamp(bull_fraction * HTF_MAX_SCORE, 0.0, HTF_MAX_SCORE),
            clamp(bear_fraction * HTF_MAX_SCORE, 0.0, HTF_MAX_SCORE),
            HTF_MAX_SCORE,
            details,
        ),
    )


def htf_alignment(htf_component: ComponentScore, direction: str) -> str:
    """Is ``direction`` with, against or neutral to the higher timeframe?

    Returns ``"ALIGNED"``, ``"COUNTER"`` or ``"NEUTRAL"``.
    """
    h1_direction = str(htf_component.details.get("h1_direction", "NEUTRAL"))
    m15_direction = str(htf_component.details.get("m15_direction", "NEUTRAL"))
    wanted = "BULL" if direction == "BUY" else "BEAR"
    opposite = "BEAR" if direction == "BUY" else "BULL"

    if h1_direction == wanted:
        return "ALIGNED"
    if h1_direction == opposite:
        return "COUNTER"
    if m15_direction == opposite:
        return "COUNTER"
    return "NEUTRAL"
