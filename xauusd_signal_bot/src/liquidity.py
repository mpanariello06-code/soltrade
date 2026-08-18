"""Liquidity engine - reference levels, equal highs/lows and sweeps.

Scores on a 0-10 scale.

The archetype it looks for is: price runs *through* a level where stops rest,
fails to hold there, and closes back on the original side.  That is a
*potential* setup - it only becomes a signal when the other eight components
agree, which is enforced by the scoring/threshold layer, not here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import confirmed_swings
from .utils import (
    UNKNOWN_SESSION,
    BoundedCache,
    ComponentScore,
    clamp,
    frame_fingerprint,
    fnum,
    safe_div,
    scale,
)

MAX_SCORE = 10.0

_W_MAJOR_SWEEP = 5.0
_W_MINOR_SWEEP = 3.5
_W_REJECTION = 2.0
_W_EQUAL_LEVELS = 1.5
_W_CLOSE_STRENGTH = 1.5

SWEEP_LOOKBACK = 5
EQUAL_LEVEL_ATR_TOLERANCE = 0.15
PENETRATION_ATR = 0.05


_DAY_CACHE = BoundedCache(maxsize=32)


def previous_day_levels(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """High/low of the last **completed** UTC day, or ``(None, None)``.

    Memoised: both the level builder and the liquidity engine ask for this.
    """
    if df is None or df.empty or "time" not in df.columns:
        return None, None
    cache_key = ("prev_day", frame_fingerprint(df, ("high", "low", "time")))
    cached = _DAY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    times = pd.to_datetime(df["time"], utc=True)
    days = times.dt.floor("D")
    current_day = days.iloc[-1]
    prior = df[days < current_day]
    if prior.empty:
        return _DAY_CACHE.put(cache_key, (None, None))
    prior_days = days[days < current_day]
    last_completed = prior_days.iloc[-1]
    block = prior[prior_days == last_completed]
    if block.empty:
        return _DAY_CACHE.put(cache_key, (None, None))
    return _DAY_CACHE.put(
        cache_key, (float(block["high"].max()), float(block["low"].min()))
    )


def session_labels(times: pd.Series, config) -> List[str]:
    """Vectorised session labelling for a whole timestamp column.

    Equivalent to calling :func:`src.utils.detect_session` per row, but without
    the per-row Python loop - this runs on every evaluation and once per bar in
    the backtester.
    """
    times = pd.to_datetime(times, utc=True)
    hours = times.dt.hour.to_numpy() + times.dt.minute.to_numpy() / 60.0
    windows = {
        "ASIAN": config.sessions.asian,
        "LONDON": config.sessions.london,
        "NEW_YORK": config.sessions.new_york,
        "LONDON_NEW_YORK_OVERLAP": config.sessions.overlap,
    }
    masks = {}
    for name, (start, end) in windows.items():
        if start <= end:
            masks[name] = (hours >= start) & (hours < end)
        else:  # window wraps past midnight
            masks[name] = (hours >= start) | (hours < end)

    labels = np.full(len(hours), UNKNOWN_SESSION, dtype=object)
    # lowest priority first so higher-priority windows overwrite them
    ordered = list(config.session_priority)
    for name in reversed(ordered):
        if name in masks:
            labels[masks[name]] = name
    for name, mask in masks.items():
        if name not in ordered:
            labels[mask & (labels == UNKNOWN_SESSION)] = name
    return labels.tolist()


def previous_session_levels(df: pd.DataFrame, config) -> Tuple[Optional[float], Optional[float]]:
    """High/low of the session block immediately before the current one."""
    if df is None or len(df) < 10 or "time" not in df.columns:
        return None, None
    labels = session_labels(df["time"], config)
    current = labels[-1]

    # walk back over the trailing run of the current session
    index = len(labels) - 1
    while index >= 0 and labels[index] == current:
        index -= 1
    if index < 0:
        return None, None

    previous_label = labels[index]
    end = index
    while index >= 0 and labels[index] == previous_label:
        index -= 1
    block = df.iloc[index + 1 : end + 1]
    if block.empty:
        return None, None
    return float(block["high"].max()), float(block["low"].min())


def _cluster_equal_levels(prices: List[float], tolerance: float) -> List[Tuple[float, int]]:
    """Group nearby pivot prices; returns ``(mean_price, touch_count)`` pairs."""
    clusters: List[List[float]] = []
    for price in sorted(prices):
        if clusters and abs(price - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    return [(sum(group) / len(group), len(group)) for group in clusters]


_LEVEL_CACHE = BoundedCache(maxsize=32)


def collect_reference_levels(df: pd.DataFrame, config) -> Dict[str, List[Dict]]:
    """Gather meaningful liquidity levels above and below the current price.

    Each entry is ``{"price", "kind", "weight"}``.  Deliberately kept to a
    handful of *meaningful* references rather than every pivot on the chart.

    Memoised on the frame's contents: the liquidity engine, the S/R engine and
    the target builder all ask for the same levels within one evaluation.
    """
    cache_key = (
        frame_fingerprint(df),
        config.indicators.level_swing_left,
        config.indicators.level_swing_right,
        config.indicators.level_lookback_bars,
        config.sessions.asian,
        config.sessions.london,
        config.sessions.new_york,
        config.sessions.overlap,
    )
    cached = _LEVEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    params = config.indicators
    row = df.iloc[-1]
    close = fnum(row["close"])
    atr_value = max(fnum(row["atr"]), 1e-9)
    tolerance = EQUAL_LEVEL_ATR_TOLERANCE * atr_value

    above: List[Dict] = []
    below: List[Dict] = []

    day_high, day_low = previous_day_levels(df)
    session_high, session_low = previous_session_levels(df, config)
    for price, kind, weight in (
        (day_high, "PREV_DAY_HIGH", 3.0),
        (day_low, "PREV_DAY_LOW", 3.0),
        (session_high, "PREV_SESSION_HIGH", 2.0),
        (session_low, "PREV_SESSION_LOW", 2.0),
    ):
        if price is None:
            continue
        bucket = above if price > close else below
        bucket.append({"price": float(price), "kind": kind, "weight": weight})

    # NOTE: level building uses the *wide* fractal, not the structure fractal.
    swings = confirmed_swings(df, params.level_swing_left, params.level_swing_right)
    if not swings.empty:
        recent = swings[swings["index"] >= len(df) - params.level_lookback_bars]
        high_clusters = _cluster_equal_levels(
            [float(p) for p in recent[recent["kind"] == "high"]["price"]], tolerance
        )
        low_clusters = _cluster_equal_levels(
            [float(p) for p in recent[recent["kind"] == "low"]["price"]], tolerance
        )
        for price, touches in high_clusters:
            entry = {
                "price": price,
                "kind": "EQUAL_HIGHS" if touches >= 2 else "SWING_HIGH",
                "weight": 1.0 + 0.5 * min(touches, 4),
            }
            (above if price > close else below).append(entry)
        for price, touches in low_clusters:
            entry = {
                "price": price,
                "kind": "EQUAL_LOWS" if touches >= 2 else "SWING_LOW",
                "weight": 1.0 + 0.5 * min(touches, 4),
            }
            (above if price > close else below).append(entry)

    above.sort(key=lambda item: item["price"])
    below.sort(key=lambda item: item["price"], reverse=True)
    return _LEVEL_CACHE.put(cache_key, {"above": above, "below": below})


def _sweep(
    window_low: float, window_high: float, level: float, close: float, atr_value: float, bullish: bool
) -> bool:
    """True when the window penetrated ``level`` but ``close`` finished back inside.

    Takes the window's extremes rather than the frame so the caller can compute
    them once instead of once per level.
    """
    penetration = PENETRATION_ATR * atr_value
    if bullish:
        return bool(window_low < level - penetration and close > level)
    return bool(window_high > level + penetration and close < level)


def analyze_liquidity(df: pd.DataFrame, config) -> ComponentScore:
    """Score liquidity conditions on the signal timeframe (0-10 per direction)."""
    if df is None or len(df) < 40:
        return ComponentScore("liquidity", 0.0, 0.0, MAX_SCORE, {"reason": "insufficient data"})

    row = df.iloc[-1]
    close, open_price = fnum(row["close"]), fnum(row["open"])
    high, low = fnum(row["high"]), fnum(row["low"])
    atr_value = max(fnum(row["atr"]), 1e-9)
    candle_range = max(high - low, 1e-9)
    window = df.iloc[-SWEEP_LOOKBACK:]
    window_low = float(window["low"].min())
    window_high = float(window["high"].max())

    levels = collect_reference_levels(df, config)
    major_kinds = {"PREV_DAY_HIGH", "PREV_DAY_LOW", "PREV_SESSION_HIGH", "PREV_SESSION_LOW"}

    bull = bear = 0.0
    swept_bull: List[str] = []
    swept_bear: List[str] = []

    for entry in levels["below"] + levels["above"]:
        price, kind = entry["price"], entry["kind"]
        if _sweep(window_low, window_high, price, close, atr_value, bullish=True):
            swept_bull.append(kind)
        if _sweep(window_low, window_high, price, close, atr_value, bullish=False):
            swept_bear.append(kind)

    if swept_bull:
        bull += _W_MAJOR_SWEEP if any(k in major_kinds for k in swept_bull) else _W_MINOR_SWEEP
        if any(k == "EQUAL_LOWS" for k in swept_bull):
            bull += _W_EQUAL_LEVELS
        # rejection quality: long lower wick + close in the upper half
        bull += _W_REJECTION * scale(safe_div(fnum(row["lower_wick"]), candle_range), 0.20, 0.55)
        bull += _W_CLOSE_STRENGTH * scale(safe_div(close - low, candle_range), 0.45, 0.80)

    if swept_bear:
        bear += _W_MAJOR_SWEEP if any(k in major_kinds for k in swept_bear) else _W_MINOR_SWEEP
        if any(k == "EQUAL_HIGHS" for k in swept_bear):
            bear += _W_EQUAL_LEVELS
        bear += _W_REJECTION * scale(safe_div(fnum(row["upper_wick"]), candle_range), 0.20, 0.55)
        bear += _W_CLOSE_STRENGTH * scale(safe_div(high - close, candle_range), 0.45, 0.80)

    # A sweep that closes the wrong way is not a reclaim.
    if close < open_price:
        bull *= 0.6
    if close > open_price:
        bear *= 0.6

    day_high, day_low = previous_day_levels(df)
    details = {
        "swept_bull": sorted(set(swept_bull)),
        "swept_bear": sorted(set(swept_bear)),
        "prev_day_high": round(day_high, 2) if day_high is not None else None,
        "prev_day_low": round(day_low, 2) if day_low is not None else None,
        "levels_above": len(levels["above"]),
        "levels_below": len(levels["below"]),
    }
    return ComponentScore(
        "liquidity", clamp(bull, 0.0, MAX_SCORE), clamp(bear, 0.0, MAX_SCORE), MAX_SCORE, details
    )
