"""Volatility engine - ATR regime classification.

Scores on a 0-5 scale and is **direction-neutral**: it grades how tradeable
conditions are, not which way price is going, so the same value is added to
both the bullish and the bearish side.  Adding equally to both leaves the
bull/bear *separation* used by the conflict filter untouched.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .utils import ComponentScore, clamp, fnum, safe_div

MAX_SCORE = 5.0

#: volatility band -> score contribution
BAND_SCORES = {"NORMAL": 5.0, "LOW": 3.0, "HIGH": 2.0, "EXTREME": 0.0}


def classify_volatility(df: pd.DataFrame, config) -> Tuple[str, float]:
    """Return ``(band, atr_ratio)`` where band is LOW/NORMAL/HIGH/EXTREME.

    ``atr_ratio`` compares the current ATR with its own long-run average, so the
    classification adapts to whatever volatility regime gold happens to be in.
    """
    if df is None or df.empty:
        return "NORMAL", 1.0
    row = df.iloc[-1]
    atr_value = fnum(row["atr"])
    atr_history = fnum(row.get("atr_history"), atr_value)
    if atr_value <= 0 or atr_history <= 0:
        return "NORMAL", 1.0

    ratio = safe_div(atr_value, atr_history, 1.0)
    if ratio >= config.vol_extreme_ratio:
        return "EXTREME", ratio
    if ratio >= config.vol_high_ratio:
        return "HIGH", ratio
    if ratio <= config.vol_low_ratio:
        return "LOW", ratio
    return "NORMAL", ratio


def analyze_volatility(df: pd.DataFrame, config) -> ComponentScore:
    """Score how favourable current volatility is (0-5, identical both sides)."""
    if df is None or len(df) < 20:
        return ComponentScore("volatility", 0.0, 0.0, MAX_SCORE, {"reason": "insufficient data"})

    band, ratio = classify_volatility(df, config)
    score = BAND_SCORES.get(band, 3.0)

    row = df.iloc[-1]
    bb_width = fnum(row.get("bb_width"))
    bb_width_ma = fnum(row.get("bb_width_ma"))
    squeeze = bool(bb_width > 0 and bb_width_ma > 0 and bb_width < 0.7 * bb_width_ma)
    expansion = bool(bb_width > 0 and bb_width_ma > 0 and bb_width > 1.3 * bb_width_ma)

    details = {
        "band": band,
        "atr": round(fnum(row["atr"]), 4),
        "atr_pct": round(fnum(row.get("atr_pct")), 4),
        "atr_ratio": round(ratio, 3),
        "bb_width": round(bb_width, 5),
        "bb_squeeze": squeeze,
        "bb_expansion": expansion,
    }
    value = clamp(score, 0.0, MAX_SCORE)
    return ComponentScore("volatility", value, value, MAX_SCORE, details)
