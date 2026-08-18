"""Support / resistance engine - meaningful zones only.

Scores on a 0-10 scale.

Levels are clustered into a small number of *zones* (repeatedly touched swing
areas plus previous day/session extremes) rather than emitting every pivot on
the chart.  Only the nearest zone on each side is used for scoring.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .liquidity import collect_reference_levels
from .utils import ComponentScore, clamp, fnum, safe_div, scale

MAX_SCORE = 10.0

# Point budget (sums to MAX_SCORE).  "Interaction" is what the price is doing
# at a zone, "room" is how far the next opposing zone is, and "clearance"
# rewards a setup with nothing meaningful standing immediately in the way.
_W_BREAK = 6.0
_W_ZONE_REACTION = 4.5
_W_ROOM = 2.0
_W_CLEARANCE = 2.0

ZONE_ATR_TOLERANCE = 0.30
NEAR_ZONE_ATR = 0.80
BREAK_CONFIRM_ATR = 0.15
CLEARANCE_ATR = 1.00
MAX_ZONES_PER_SIDE = 4


def _merge_into_zones(levels: List[Dict], tolerance: float) -> List[Dict]:
    """Merge nearby levels into zones, summing their weights."""
    if not levels:
        return []
    ordered = sorted(levels, key=lambda item: item["price"])
    zones: List[Dict] = []
    for entry in ordered:
        if zones and abs(entry["price"] - zones[-1]["price"]) <= tolerance:
            zone = zones[-1]
            total = zone["weight"] + entry["weight"]
            zone["price"] = (zone["price"] * zone["weight"] + entry["price"] * entry["weight"]) / total
            zone["weight"] = total
            zone["touches"] += 1
            zone["kinds"].append(entry["kind"])
        else:
            zones.append(
                {"price": entry["price"], "weight": entry["weight"], "touches": 1, "kinds": [entry["kind"]]}
            )
    return zones


def build_zones(df: pd.DataFrame, config) -> Dict[str, List[Dict]]:
    """Return the nearest meaningful ``support``/``resistance`` zones."""
    row = df.iloc[-1]
    close = fnum(row["close"])
    atr_value = max(fnum(row["atr"]), 1e-9)
    tolerance = ZONE_ATR_TOLERANCE * atr_value

    levels = collect_reference_levels(df, config)
    resistance = _merge_into_zones(levels["above"], tolerance)
    support = _merge_into_zones(levels["below"], tolerance)

    minimum_weight = float(getattr(config, "min_zone_weight", 2.0))
    resistance = [z for z in resistance if z["price"] > close and z["weight"] >= minimum_weight]
    support = [z for z in support if z["price"] < close and z["weight"] >= minimum_weight]
    resistance.sort(key=lambda z: z["price"])
    support.sort(key=lambda z: z["price"], reverse=True)
    return {
        "resistance": resistance[:MAX_ZONES_PER_SIDE],
        "support": support[:MAX_ZONES_PER_SIDE],
    }


def nearest_opposing_level(df: pd.DataFrame, config, direction: str) -> Optional[float]:
    """Nearest zone standing in the way of a trade (resistance for a BUY)."""
    zones = build_zones(df, config)
    side = zones["resistance"] if direction == "BUY" else zones["support"]
    if not side:
        return None
    return float(side[0]["price"])


def analyze_support_resistance(df: pd.DataFrame, config) -> ComponentScore:
    """Score the price's position relative to S/R zones (0-10 per direction)."""
    if df is None or len(df) < 40:
        return ComponentScore(
            "support_resistance", 0.0, 0.0, MAX_SCORE, {"reason": "insufficient data"}
        )

    row, prev = df.iloc[-1], df.iloc[-2]
    close, prev_close = fnum(row["close"]), fnum(prev["close"])
    open_price = fnum(row["open"])
    atr_value = max(fnum(row["atr"]), 1e-9)

    zones = build_zones(df, config)
    nearest_resistance = zones["resistance"][0]["price"] if zones["resistance"] else None
    nearest_support = zones["support"][0]["price"] if zones["support"] else None

    bull = bear = 0.0
    state = "NEUTRAL"

    # Breakout / breakdown are measured against where the *previous* close sat,
    # so a level only counts as broken once a candle actually closes through it.
    broke_up = (
        nearest_support is not None
        and prev_close <= nearest_support + BREAK_CONFIRM_ATR * atr_value
        and close > nearest_support + BREAK_CONFIRM_ATR * atr_value
    )
    broke_down = (
        nearest_resistance is not None
        and prev_close >= nearest_resistance - BREAK_CONFIRM_ATR * atr_value
        and close < nearest_resistance - BREAK_CONFIRM_ATR * atr_value
    )

    # A cleaner breakout definition: the candle closed above a zone that was
    # resistance on the previous bar.
    prior_zone_above = [z for z in zones["resistance"] if z["price"] < prev_close]
    prior_zone_below = [z for z in zones["support"] if z["price"] > prev_close]
    breakout = any(close > z["price"] + BREAK_CONFIRM_ATR * atr_value for z in prior_zone_above)
    breakdown = any(close < z["price"] - BREAK_CONFIRM_ATR * atr_value for z in prior_zone_below)

    # 1. Interaction with a zone: a confirmed break scores highest, a hold at a
    #    zone in our favour next.  The two are mutually exclusive by design.
    if breakout or broke_up:
        bull += _W_BREAK
        state = "BREAKOUT"
    if breakdown or broke_down:
        bear += _W_BREAK
        state = "BREAKDOWN"

    near_support = (
        nearest_support is not None
        and safe_div(close - nearest_support, atr_value) <= NEAR_ZONE_ATR
    )
    near_resistance = (
        nearest_resistance is not None
        and safe_div(nearest_resistance - close, atr_value) <= NEAR_ZONE_ATR
    )
    if near_support and close >= open_price and state == "NEUTRAL":
        bull += _W_ZONE_REACTION
        state = "NEAR_SUPPORT"
    if near_resistance and close <= open_price and state == "NEUTRAL":
        bear += _W_ZONE_REACTION
        state = "NEAR_RESISTANCE"

    # 2. Room to run: how far away the next opposing zone sits.
    resistance_distance = (
        safe_div(nearest_resistance - close, atr_value) if nearest_resistance is not None else 99.0
    )
    support_distance = (
        safe_div(close - nearest_support, atr_value) if nearest_support is not None else 99.0
    )
    bull += _W_ROOM * scale(resistance_distance, 0.25, 1.6)
    bear += _W_ROOM * scale(support_distance, 0.25, 1.6)

    # 3. Clearance: nothing meaningful standing immediately in the way.
    bull += _W_CLEARANCE * scale(resistance_distance, 0.15, CLEARANCE_ATR)
    bear += _W_CLEARANCE * scale(support_distance, 0.15, CLEARANCE_ATR)

    details = {
        "state": state,
        "nearest_support": round(nearest_support, 2) if nearest_support is not None else None,
        "nearest_resistance": round(nearest_resistance, 2) if nearest_resistance is not None else None,
        "support_zones": len(zones["support"]),
        "resistance_zones": len(zones["resistance"]),
        "near_support": bool(near_support),
        "near_resistance": bool(near_resistance),
    }
    return ComponentScore(
        "support_resistance",
        clamp(bull, 0.0, MAX_SCORE),
        clamp(bear, 0.0, MAX_SCORE),
        MAX_SCORE,
        details,
    )
