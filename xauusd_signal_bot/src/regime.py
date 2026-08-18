"""Market-regime classifier.

The regime does not add score directly - it selects which *threshold* the
signal must clear (see :mod:`src.filters`).  A range regime demands a much
higher score than a strong trend, which is how the system stays selective in
the conditions where multi-confirmation setups fail most often.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .utils import ComponentScore, fnum

REGIMES = (
    "STRONG_BULL_TREND",
    "STRONG_BEAR_TREND",
    "WEAK_TREND",
    "RANGE",
    "BREAKOUT",
    "HIGH_VOLATILITY",
    "LOW_VOLATILITY",
)


def classify_regime(
    df: pd.DataFrame, components: Dict[str, ComponentScore], config
) -> Tuple[str, Dict[str, object]]:
    """Return ``(regime, details)`` for the current signal candle."""
    if df is None or df.empty:
        return "WEAK_TREND", {"reason": "no data"}

    row = df.iloc[-1]
    adx_value = fnum(row["adx"])
    close = fnum(row["close"])
    ema_fast, ema_mid = fnum(row["ema_fast"]), fnum(row["ema_mid"])
    ema_slow, ema_trend = fnum(row["ema_slow"]), fnum(row["ema_trend"])

    volatility = components.get("volatility")
    volatility_band = str(volatility.details.get("band", "NORMAL")) if volatility else "NORMAL"
    bb_expansion = bool(volatility.details.get("bb_expansion")) if volatility else False

    structure = components.get("structure")
    structure_details = structure.details if structure else {}
    consolidating = bool(structure_details.get("consolidating"))
    broke_structure = bool(structure_details.get("bos_bull")) or bool(structure_details.get("bos_bear"))

    sr = components.get("support_resistance")
    sr_state = str(sr.details.get("state", "NEUTRAL")) if sr else "NEUTRAL"

    aligned_bull = ema_fast > ema_mid > ema_slow and close > ema_trend
    aligned_bear = ema_fast < ema_mid < ema_slow and close < ema_trend

    details = {
        "adx": round(adx_value, 2),
        "volatility_band": volatility_band,
        "consolidating": consolidating,
        "sr_state": sr_state,
        "aligned_bull": bool(aligned_bull),
        "aligned_bear": bool(aligned_bear),
    }

    # Order matters: unusable conditions are recognised before direction.
    if volatility_band in ("EXTREME", "HIGH"):
        return "HIGH_VOLATILITY", details
    if adx_value >= config.adx_strong_threshold and aligned_bull:
        return "STRONG_BULL_TREND", details
    if adx_value >= config.adx_strong_threshold and aligned_bear:
        return "STRONG_BEAR_TREND", details
    if (sr_state in ("BREAKOUT", "BREAKDOWN") or broke_structure) and bb_expansion:
        return "BREAKOUT", details
    if adx_value < config.adx_range_threshold or consolidating:
        return "RANGE", details
    if volatility_band == "LOW":
        return "LOW_VOLATILITY", details
    return "WEAK_TREND", details


def regime_bias(regime: str) -> str:
    """Directional bias implied by a regime: ``BUY``, ``SELL`` or ``NEUTRAL``."""
    if regime == "STRONG_BULL_TREND":
        return "BUY"
    if regime == "STRONG_BEAR_TREND":
        return "SELL"
    return "NEUTRAL"
