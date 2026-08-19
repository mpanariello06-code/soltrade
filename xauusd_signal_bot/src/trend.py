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
# higher-timeframe confirmation
# --------------------------------------------------------------------------- #
_HTF_CACHE = BoundedCache(maxsize=8)

HTF_MAX_SCORE = 15.0

#: How the two confirmation timeframes combine into the single HTF component
#: when both exist (M1..M30 signal timeframes).  The slower timeframe carries
#: the most weight; the fractions sum to 1.0.
HTF_MIX_PAIR = {
    "higher_trend": 0.30,
    "intermediate_trend": 0.25,
    "higher_structure": 0.15,
    "intermediate_structure": 0.10,
    "higher_momentum": 0.10,
    "intermediate_momentum": 0.10,
}

#: Mix used when only ONE confirmation timeframe exists (signal timeframe H1,
#: confirmed by H4 alone).  Same ordering of importance, renormalised.
HTF_MIX_SINGLE = {
    "trend": 0.45,
    "structure": 0.30,
    "momentum": 0.25,
}

#: Kept for backwards compatibility with the M5-only version.
HTF_MIX = HTF_MIX_PAIR


def _sub_scores(frame: pd.DataFrame, config):
    """Run trend / structure / momentum on one confirmation timeframe."""
    from .momentum import MAX_SCORE as MOMENTUM_MAX, analyze_momentum
    from .structure import MAX_SCORE as STRUCTURE_MAX, analyze_structure

    return {
        "trend": (analyze_trend(frame, config), MAX_SCORE),
        "structure": (analyze_structure(frame, config), STRUCTURE_MAX),
        "momentum": (analyze_momentum(frame, config), MOMENTUM_MAX),
    }


def _usable(frame: Optional[pd.DataFrame]) -> bool:
    """A confirmation frame is usable once it has enough closed candles."""
    return frame is not None and len(frame) >= 30


def analyze_htf(
    df_intermediate: Optional[pd.DataFrame],
    df_higher: Optional[pd.DataFrame],
    config,
) -> ComponentScore:
    """Blend the confirmation timeframes into one 0-15 component.

    Both frames must already carry indicator columns and contain closed candles
    only.  Either may be ``None``:

    * **both present** - the usual case (signal timeframes M1..M30)
    * **one present**  - signal timeframe H1, confirmed by H4 alone
    * **neither**      - signal timeframe H4; the component is marked *not
      applicable* and :func:`src.scoring.compute_scorecard` redistributes its
      weight rather than scoring a flat zero

    Memoised on the contents of the frames: an intermediate candle spans several
    signal candles, so the higher-timeframe view is unchanged for most
    evaluations.  The cache key is a content hash, so a changed frame can never
    return a stale result.
    """
    have_intermediate = _usable(df_intermediate)
    have_higher = _usable(df_higher)

    if not have_intermediate and not have_higher:
        return ComponentScore(
            "htf", 0.0, 0.0, HTF_MAX_SCORE,
            {"reason": "no confirmation timeframe", "intermediate_direction": "NONE",
             "higher_direction": "NONE", "m15_direction": "NEUTRAL", "h1_direction": "NEUTRAL"},
            applicable=False,
        )

    cache_key = (
        frame_fingerprint(df_intermediate) if have_intermediate else 0,
        frame_fingerprint(df_higher) if have_higher else 0,
    )
    cached = _HTF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bull_fraction = bear_fraction = 0.0
    details: Dict[str, Any] = {}

    if have_intermediate and have_higher:
        parts = {
            "intermediate": _sub_scores(df_intermediate, config),
            "higher": _sub_scores(df_higher, config),
        }
        for role, sub in parts.items():
            for key, (component, maximum) in sub.items():
                share = HTF_MIX_PAIR[f"{role}_{key}"]
                bull_fraction += share * safe_div(component.bull, maximum)
                bear_fraction += share * safe_div(component.bear, maximum)
                details[f"{role}_{key}"] = (round(component.bull, 2), round(component.bear, 2))
        intermediate_direction = trend_direction(df_intermediate, config)
        higher_direction = trend_direction(df_higher, config)
    else:
        frame = df_higher if have_higher else df_intermediate
        for key, (component, maximum) in _sub_scores(frame, config).items():
            share = HTF_MIX_SINGLE[key]
            bull_fraction += share * safe_div(component.bull, maximum)
            bear_fraction += share * safe_div(component.bear, maximum)
            details[f"single_{key}"] = (round(component.bull, 2), round(component.bear, 2))
        higher_direction = intermediate_direction = trend_direction(frame, config)

    details.update(
        {
            "intermediate_direction": intermediate_direction,
            "higher_direction": higher_direction,
            # legacy key names, still read by htf_alignment and the tests
            "m15_direction": intermediate_direction,
            "h1_direction": higher_direction,
        }
    )

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
    if not htf_component.applicable:
        return "NEUTRAL"
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
