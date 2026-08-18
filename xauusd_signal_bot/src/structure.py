"""Market-structure engine - swings, BOS, CHoCH, continuation, consolidation.

Scores on a 0-15 scale.

NO-REPAINT GUARANTEE
--------------------
Every pivot used here comes from :func:`src.indicators.confirmed_swings`, which
discards any pivot whose confirmation bar has not closed yet.  A swing high at
bar ``i`` needs ``swing_right`` bars after it, so it is invisible to the engine
until bar ``i + swing_right`` closes.  Historical structure therefore never
changes retroactively.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .indicators import confirmed_swings
from .utils import ComponentScore, clamp, fnum, safe_div, scale

MAX_SCORE = 15.0

_W_SEQUENCE = 5.0
_W_BOS = 4.0
_W_CHOCH = 3.0
_W_CONTINUATION = 3.0

CONSOLIDATION_LOOKBACK = 20
CONSOLIDATION_ATR_RATIO = 2.0


def _pivot_list(swings: pd.DataFrame, kind: str) -> List[Dict]:
    """Confirmed pivots of one kind, oldest first.

    Built from numpy arrays rather than ``DataFrame.to_dict`` - this runs on
    every evaluation and the frame conversion dominated the profile.
    """
    if swings.empty:
        return []
    mask = (swings["kind"] == kind).to_numpy()
    if not mask.any():
        return []
    prices = swings["price"].to_numpy()[mask]
    indices = swings["index"].to_numpy()[mask]
    return [{"index": int(i), "price": float(pr)} for i, pr in zip(indices, prices)]


def _sequence_state(highs: List[Dict], lows: List[Dict]) -> str:
    """Classify the last two highs and lows as an HH/HL or LH/LL sequence."""
    if len(highs) < 2 or len(lows) < 2:
        return "UNDEFINED"
    higher_high = highs[-1]["price"] > highs[-2]["price"]
    higher_low = lows[-1]["price"] > lows[-2]["price"]
    lower_high = highs[-1]["price"] < highs[-2]["price"]
    lower_low = lows[-1]["price"] < lows[-2]["price"]

    if higher_high and higher_low:
        return "BULLISH"
    if lower_high and lower_low:
        return "BEARISH"
    if higher_high or higher_low:
        return "BULLISH_WEAK"
    if lower_high or lower_low:
        return "BEARISH_WEAK"
    return "MIXED"


def _is_consolidating(df: pd.DataFrame) -> bool:
    """True when the recent range is small relative to ATR."""
    if len(df) < CONSOLIDATION_LOOKBACK:
        return False
    window = df.iloc[-CONSOLIDATION_LOOKBACK:]
    atr_value = max(fnum(df["atr"].iloc[-1]), 1e-9)
    span = fnum(window["high"].max()) - fnum(window["low"].min())
    return safe_div(span, atr_value) < CONSOLIDATION_ATR_RATIO


def analyze_structure(df: pd.DataFrame, config) -> ComponentScore:
    """Score market structure on the signal timeframe (0-15 per direction)."""
    params = config.indicators
    if df is None or len(df) < 30:
        return ComponentScore("structure", 0.0, 0.0, MAX_SCORE, {"reason": "insufficient data"})

    swings = confirmed_swings(df, params.swing_left, params.swing_right)
    highs = _pivot_list(swings, "high")
    lows = _pivot_list(swings, "low")

    row = df.iloc[-1]
    close = fnum(row["close"])
    atr_value = max(fnum(row["atr"]), 1e-9)
    sequence = _sequence_state(highs, lows)
    consolidating = _is_consolidating(df)

    last_high = highs[-1]["price"] if highs else None
    last_low = lows[-1]["price"] if lows else None

    bull = bear = 0.0

    # 1. Swing sequence ------------------------------------------------------ #
    sequence_points = {
        "BULLISH": (_W_SEQUENCE, 0.0),
        "BULLISH_WEAK": (_W_SEQUENCE * 0.5, 0.0),
        "BEARISH": (0.0, _W_SEQUENCE),
        "BEARISH_WEAK": (0.0, _W_SEQUENCE * 0.5),
    }.get(sequence, (0.0, 0.0))
    bull += sequence_points[0]
    bear += sequence_points[1]

    # 2. Break of structure -------------------------------------------------- #
    bos_bull = last_high is not None and close > last_high
    bos_bear = last_low is not None and close < last_low
    if bos_bull:
        bull += _W_BOS
    if bos_bear:
        bear += _W_BOS

    # 3. Change of character (break against the prevailing sequence) --------- #
    choch_bull = bos_bull and sequence in ("BEARISH", "BEARISH_WEAK")
    choch_bear = bos_bear and sequence in ("BULLISH", "BULLISH_WEAK")
    if choch_bull:
        bull += _W_CHOCH
    if choch_bear:
        bear += _W_CHOCH

    # 4. Continuation: trend intact and price working above/below the last
    #    protected swing without having broken it.
    if last_low is not None and sequence in ("BULLISH", "BULLISH_WEAK") and close > last_low:
        headroom = safe_div(close - last_low, atr_value)
        bull += _W_CONTINUATION * scale(headroom, 0.15, 1.3)
    if last_high is not None and sequence in ("BEARISH", "BEARISH_WEAK") and close < last_high:
        headroom = safe_div(last_high - close, atr_value)
        bear += _W_CONTINUATION * scale(headroom, 0.15, 1.8)

    # Consolidation makes directional structure unreliable -> damp both sides.
    if consolidating:
        bull *= 0.6
        bear *= 0.6

    details = {
        "sequence": sequence,
        "bos_bull": bool(bos_bull),
        "bos_bear": bool(bos_bear),
        "choch_bull": bool(choch_bull),
        "choch_bear": bool(choch_bear),
        "consolidating": bool(consolidating),
        "last_swing_high": round(last_high, 2) if last_high is not None else None,
        "last_swing_low": round(last_low, 2) if last_low is not None else None,
        "confirmed_swings": int(len(swings)),
    }
    return ComponentScore(
        "structure", clamp(bull, 0.0, MAX_SCORE), clamp(bear, 0.0, MAX_SCORE), MAX_SCORE, details
    )


def last_protected_swing(df: pd.DataFrame, config, direction: str) -> Optional[float]:
    """Swing level a stop should sit behind: last confirmed low for a BUY."""
    params = config.indicators
    swings = confirmed_swings(df, params.swing_left, params.swing_right)
    kind = "low" if direction == "BUY" else "high"
    pivots = _pivot_list(swings, kind)
    if not pivots:
        return None
    return float(pivots[-1]["price"])
