"""Entry / stop-loss / take-profit construction and risk-reward evaluation.

Stop placement modes
--------------------
``ATR``       - a pure volatility distance (``atr * sl_atr_multiplier``).
``STRUCTURE`` - behind the last confirmed swing plus an ATR buffer.
``HYBRID``    - the **wider** of the two, then clamped into
                ``[sl_min_atr_multiplier, sl_max_atr_multiplier] * ATR``.

HYBRID is the default because taking the wider distance keeps the stop from
sitting exactly on an obvious swing (where liquidity rests), while the clamp
stops a far-away swing from producing an absurdly large risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .structure import last_protected_swing
from .support_resistance import build_zones
from .utils import fnum, round_price, safe_div

DIRECTIONS = ("BUY", "SELL")


@dataclass
class Targets:
    """Entry, stop and the three take-profits, with their R multiples."""

    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    risk: float
    rr1: float
    rr2: float
    rr3: float
    sl_mode: str
    obstructed: Dict[str, bool]

    def as_dict(self) -> Dict[str, float]:
        return {
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "risk": self.risk,
            "rr1": self.rr1,
            "rr2": self.rr2,
            "rr3": self.rr3,
            "sl_mode": self.sl_mode,
        }


# --------------------------------------------------------------------------- #
# stop loss
# --------------------------------------------------------------------------- #
def _structure_stop_distance(
    df: pd.DataFrame, config, direction: str, entry: float, atr_value: float
) -> Optional[float]:
    """Distance from ``entry`` to just beyond the protecting swing."""
    swing = last_protected_swing(df, config, direction)
    lookback = df.iloc[-config.sl_structure_lookback :]
    if direction == "BUY":
        recent_extreme = fnum(lookback["low"].min())
        level = min(swing, recent_extreme) if swing is not None else recent_extreme
        distance = entry - (level - config.sl_structure_buffer_atr * atr_value)
    else:
        recent_extreme = fnum(lookback["high"].max())
        level = max(swing, recent_extreme) if swing is not None else recent_extreme
        distance = (level + config.sl_structure_buffer_atr * atr_value) - entry
    return distance if distance > 0 else None


def compute_stop_loss(
    df: pd.DataFrame, config, direction: str, entry: float
) -> Tuple[float, float, str]:
    """Return ``(stop_loss, risk_distance, mode_used)``."""
    atr_value = max(fnum(df["atr"].iloc[-1]), 1e-9)
    atr_distance = config.sl_atr_multiplier * atr_value
    structure_distance = _structure_stop_distance(df, config, direction, entry, atr_value)
    mode = config.sl_mode

    if mode == "ATR" or structure_distance is None:
        distance = atr_distance
        used = "ATR"
    elif mode == "STRUCTURE":
        distance = structure_distance
        used = "STRUCTURE"
    else:  # HYBRID
        distance = max(atr_distance, structure_distance)
        used = "HYBRID"

    # Never tighter than min, never wider than max - both expressed in ATR.
    distance = max(config.sl_min_atr_multiplier * atr_value, distance)
    distance = min(config.sl_max_atr_multiplier * atr_value, distance)

    stop = entry - distance if direction == "BUY" else entry + distance
    return round_price(stop, config.digits), distance, used


# --------------------------------------------------------------------------- #
# take profits
# --------------------------------------------------------------------------- #
def _opposing_zones(df: pd.DataFrame, config, direction: str) -> List[float]:
    """Prices of *major* zones standing in the way, nearest first.

    Only heavyweight zones (previous day extremes, repeatedly-rejected areas)
    are allowed to truncate a target - otherwise every minor pivot on a noisy
    chart would clip TP2/TP3 and the R:R filter would reject everything.
    """
    zones = build_zones(df, config)
    side = zones["resistance"] if direction == "BUY" else zones["support"]
    minimum_weight = float(getattr(config, "min_tp_block_zone_weight", 3.0))
    prices = [float(zone["price"]) for zone in side if zone["weight"] >= minimum_weight]
    return sorted(prices) if direction == "BUY" else sorted(prices, reverse=True)


def compute_take_profits(
    df: pd.DataFrame, config, direction: str, entry: float, risk: float
) -> Tuple[List[float], Dict[str, bool]]:
    """Project TP1/TP2/TP3 and pull them back in front of opposing structure.

    A target that would sit beyond a meaningful opposing zone is moved to just
    in front of that zone.  If the zone is so close that the pull-back would
    leave less than ``tp_min_r_after_adjustment`` of reward, the target is left
    at that floor and flagged ``obstructed`` - the R:R filter then decides
    whether the setup is still worth taking.
    """
    atr_value = max(fnum(df["atr"].iloc[-1]), 1e-9)
    buffer_distance = config.tp_sr_buffer_atr * atr_value
    zones = _opposing_zones(df, config, direction)

    targets: List[float] = []
    obstructed: Dict[str, bool] = {}

    for index, multiple in enumerate(config.tp_r_multiples):
        floor_multiple = config.tp_min_r_after_adjustment[index]
        if direction == "BUY":
            raw = entry + multiple * risk
            floor = entry + floor_multiple * risk
            blocking = [price for price in zones if entry < price < raw]
            if blocking:
                candidate = min(blocking) - buffer_distance
                if candidate < floor:
                    obstructed[f"tp{index + 1}"] = True
                    raw = floor
                else:
                    obstructed[f"tp{index + 1}"] = True
                    raw = candidate
            else:
                obstructed[f"tp{index + 1}"] = False
        else:
            raw = entry - multiple * risk
            floor = entry - floor_multiple * risk
            blocking = [price for price in zones if raw < price < entry]
            if blocking:
                candidate = max(blocking) + buffer_distance
                if candidate > floor:
                    obstructed[f"tp{index + 1}"] = True
                    raw = floor
                else:
                    obstructed[f"tp{index + 1}"] = True
                    raw = candidate
            else:
                obstructed[f"tp{index + 1}"] = False
        targets.append(raw)

    # keep the ladder strictly ordered after any pull-back
    if direction == "BUY":
        for index in range(1, len(targets)):
            targets[index] = max(targets[index], targets[index - 1] + 0.1 * atr_value)
    else:
        for index in range(1, len(targets)):
            targets[index] = min(targets[index], targets[index - 1] - 0.1 * atr_value)

    return [round_price(price, config.digits) for price in targets], obstructed


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def build_targets(
    df: pd.DataFrame, config, direction: str, entry: Optional[float] = None
) -> Tuple[Optional[Targets], str]:
    """Build the full target set for ``direction``.

    Returns ``(targets, reason)``; ``targets`` is ``None`` when the geometry is
    unusable and ``reason`` says why.
    """
    if direction not in DIRECTIONS:
        return None, f"invalid direction '{direction}'"
    if df is None or len(df) < max(config.sl_structure_lookback, 5):
        return None, "insufficient data for targets"

    entry = round_price(fnum(df["close"].iloc[-1]) if entry is None else entry, config.digits)
    if entry <= 0:
        return None, "invalid entry price"

    stop_loss, risk, mode_used = compute_stop_loss(df, config, direction, entry)
    if risk <= 0:
        return None, "non-positive risk distance"
    if direction == "BUY" and stop_loss >= entry:
        return None, "stop loss is not below entry for a BUY"
    if direction == "SELL" and stop_loss <= entry:
        return None, "stop loss is not above entry for a SELL"

    # recompute risk from the rounded stop so R multiples match the published prices
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return None, "zero risk after rounding"

    targets, obstructed = compute_take_profits(df, config, direction, entry, risk)
    tp1, tp2, tp3 = targets

    if direction == "BUY" and not (entry < tp1 < tp2 < tp3):
        return None, "take-profit ladder is not ordered above entry"
    if direction == "SELL" and not (entry > tp1 > tp2 > tp3):
        return None, "take-profit ladder is not ordered below entry"

    sign = 1.0 if direction == "BUY" else -1.0
    rr1 = safe_div(sign * (tp1 - entry), risk)
    rr2 = safe_div(sign * (tp2 - entry), risk)
    rr3 = safe_div(sign * (tp3 - entry), risk)

    return (
        Targets(
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            risk=round(risk, 6),
            rr1=round(rr1, 2),
            rr2=round(rr2, 2),
            rr3=round(rr3, 2),
            sl_mode=mode_used,
            obstructed=obstructed,
        ),
        "",
    )
