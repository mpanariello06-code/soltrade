"""Micro-scalping entry / stop / take-profit construction and the cost model.

TARGETS COME FROM MARKET CONDITIONS, NOT FIXED PIP DISTANCES
------------------------------------------------------------
Every distance is derived from the current M1 ATR and the nearest meaningful
structure, then floored so it cannot be smaller than the tick grid or - more
importantly - smaller than the cost of trading it.  A quiet minute produces a
tighter target than a volatile one, which is the whole point of sizing off ATR.

THE COST MODEL IS NOT OPTIONAL
------------------------------
At a 1-3 pip target the spread is not a rounding error, it is often the entire
trade.  Round-trip cost is

    spread + entry slippage + exit slippage + commission (both sides)

Every reward figure is therefore reported twice:

``rr1/rr2/rr3``          RAW R - price movement only
``net_rr1/net_rr2/net_rr3``  NET R - after the round-trip cost above

A setup whose TP1 does not clear ``min_tp1_cost_multiple`` times that cost is
rejected outright.  **Nothing here claims a setup is profitable**; the cost
model exists precisely so that a small favourable move is not mistaken for one.

STOP PLACEMENT
--------------
``ATR``       - a pure volatility distance
``STRUCTURE`` - behind the last confirmed micro swing plus a buffer
``HYBRID``    - the wider of the two (default), then clamped into
                ``[sl_min_atr_multiplier, sl_max_atr_multiplier] * ATR`` and
                finally floored so the stop sits beyond ordinary spread noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .structure import last_protected_swing
from .support_resistance import build_zones
from .utils import fnum, round_price, safe_div

DIRECTIONS = ("BUY", "SELL")

#: Minimum spacing between consecutive targets, as a fraction of one pip.
#: Without it, cost-flooring TP1 can collapse the ladder onto a single price.
MIN_TARGET_SPACING_PIPS = 0.3


def target_floor(config, index: int, price: float) -> float:
    """Smallest allowed distance for target ``index``, in price terms.

    Two floors are combined and the larger wins:

    * an absolute floor in the market's pip unit (``min_tp_pips``)
    * a floor as a fraction of price (``min_tp_pct``)

    Gold uses the first with the second at zero, so its behaviour is unchanged.
    Bitcoin uses the second, because a fixed dollar floor would be far too tight
    at $90,000 and far too loose at $20,000.
    """
    pip_floor = config.min_tp_pips[index] * config.pip_value
    pct_floor = config.min_tp_pct[index] * float(price)
    return max(pip_floor, pct_floor)


def max_tp3_distance(config, price: float) -> float:
    """Largest TP3 distance that still counts as a scalp, in price terms."""
    pip_ceiling = config.max_tp3_pips * config.pip_value
    pct_ceiling = config.max_tp3_pct * float(price)
    candidates = [value for value in (pip_ceiling, pct_ceiling) if value > 0]
    return max(candidates) if candidates else float("inf")


@dataclass
class Targets:
    """Entry, stop and the three take-profits, with raw and net reward."""

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

    # -- scalping additions ------------------------------------------------ #
    spread_points: float = float("nan")
    cost_price: float = 0.0      #: round-trip cost in price terms
    cost_r: float = 0.0          #: that cost expressed in R
    net_rr1: float = 0.0
    net_rr2: float = 0.0
    net_rr3: float = 0.0
    sl_pips: float = 0.0
    tp_pips: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cost_pips: float = 0.0
    atr: float = 0.0

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
            "net_rr1": self.net_rr1,
            "net_rr2": self.net_rr2,
            "net_rr3": self.net_rr3,
            "cost_r": self.cost_r,
            "cost_pips": self.cost_pips,
            "sl_pips": self.sl_pips,
            "tp_pips": self.tp_pips,
            "sl_mode": self.sl_mode,
        }


# --------------------------------------------------------------------------- #
# stop loss
# --------------------------------------------------------------------------- #
def _structure_stop_distance(
    df: pd.DataFrame, config, direction: str, entry: float, atr_value: float
) -> Optional[float]:
    """Distance from ``entry`` to just beyond the protecting micro swing."""
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
    df: pd.DataFrame,
    config,
    direction: str,
    entry: float,
    spread_points: Optional[float] = None,
) -> Tuple[float, float, str]:
    """Return ``(stop_loss, risk_distance, mode_used)``.

    The final floor is the important one for scalping: a stop closer than a
    couple of spreads is taken out by ordinary quote noise rather than by the
    market actually going the wrong way.
    """
    atr_value = max(fnum(df["atr"].iloc[-1]), 1e-9)
    atr_distance = config.sl_atr_multiplier * atr_value
    structure_distance = _structure_stop_distance(df, config, direction, entry, atr_value)
    mode = config.sl_mode

    if mode == "ATR" or structure_distance is None:
        distance, used = atr_distance, "ATR"
    elif mode == "STRUCTURE":
        distance, used = structure_distance, "STRUCTURE"
    else:  # HYBRID
        distance, used = max(atr_distance, structure_distance), "HYBRID"

    distance = max(config.sl_min_atr_multiplier * atr_value, distance)
    distance = min(config.sl_max_atr_multiplier * atr_value, distance)

    # Noise floor: keep the stop clear of ordinary spread movement.
    spread_price = config.effective_spread_points(spread_points) * config.point_value
    distance = max(distance, config.min_sl_cost_multiple * spread_price)

    stop = entry - distance if direction == "BUY" else entry + distance
    return round_price(stop, config.digits), distance, used


# --------------------------------------------------------------------------- #
# take profits
# --------------------------------------------------------------------------- #
def _opposing_zones(df: pd.DataFrame, config, direction: str) -> List[float]:
    """Prices of *major* zones standing in the way, nearest first."""
    zones = build_zones(df, config)
    side = zones["resistance"] if direction == "BUY" else zones["support"]
    minimum_weight = float(getattr(config, "min_tp_block_zone_weight", 3.0))
    prices = [float(zone["price"]) for zone in side if zone["weight"] >= minimum_weight]
    return sorted(prices) if direction == "BUY" else sorted(prices, reverse=True)


def compute_target_distances(
    config, atr_value: float, cost_price: float, price: float = 0.0
) -> List[float]:
    """Distances for TP1/TP2/TP3, in price, before any structure truncation.

    Three floors are applied in order:

    1. ``tp_atr_multiples * ATR``      - the market-conditions baseline
    2. ``min_tp_pips``                 - the tick-grid floor
    3. ``min_tp1_cost_multiple * cost``- TP1 only: it must be worth taking

    Raising TP1 for costs can push it past TP2, so the ladder is re-spaced
    afterwards rather than being allowed to collapse.
    """
    pip = config.pip_value
    baseline = [
        max(multiple * atr_value, target_floor(config, index, price))
        for index, multiple in enumerate(config.tp_atr_multiples)
    ]

    distances = list(baseline)
    distances[0] = max(distances[0], config.min_tp1_cost_multiple * cost_price)

    # When the cost floor lifts TP1, lift the whole ladder by the same ratio.
    # Merely re-spacing TP2/TP3 by a minimum gap would collapse them onto TP1
    # and quietly turn a 1:2 setup into a 1:1 one - the geometry has to say
    # honestly that a wider spread demands a bigger move, not a closer target.
    lift = safe_div(distances[0], baseline[0], 1.0)
    if lift > 1.0:
        distances = [max(distance, base * lift) for distance, base in zip(distances, baseline)]

    spacing = MIN_TARGET_SPACING_PIPS * pip
    for index in range(1, len(distances)):
        distances[index] = max(distances[index], distances[index - 1] + spacing)
    return distances


def compute_take_profits(
    df: pd.DataFrame,
    config,
    direction: str,
    entry: float,
    cost_price: float,
) -> Tuple[List[float], Dict[str, bool]]:
    """Project the target ladder and pull it back in front of major structure.

    A target that would sit beyond a meaningful opposing zone is moved to just
    in front of it.  A pull-back is never allowed below the cost floor for TP1 -
    if a wall sits that close, the setup is not worth taking and the caller
    rejects it on cost rather than quietly shipping a target that loses money.
    """
    atr_value = max(fnum(df["atr"].iloc[-1]), 1e-9)
    pip = config.pip_value
    buffer_distance = config.tp_sr_buffer_atr * atr_value
    zones = _opposing_zones(df, config, direction)
    distances = compute_target_distances(config, atr_value, cost_price, entry)

    sign = 1.0 if direction == "BUY" else -1.0
    floors = [config.min_tp1_cost_multiple * cost_price] + [
        target_floor(config, index, entry) for index in (1, 2)
    ]

    prices: List[float] = []
    obstructed: Dict[str, bool] = {}
    for index, distance in enumerate(distances):
        raw = entry + sign * distance
        if direction == "BUY":
            blocking = [price for price in zones if entry < price < raw]
        else:
            blocking = [price for price in zones if raw < price < entry]
        if blocking:
            nearest = min(blocking) if direction == "BUY" else max(blocking)
            candidate = nearest - sign * buffer_distance
            floor_price = entry + sign * floors[index]
            keep = (candidate >= floor_price) if direction == "BUY" else (candidate <= floor_price)
            raw = candidate if keep else floor_price
            obstructed[f"tp{index + 1}"] = True
        else:
            obstructed[f"tp{index + 1}"] = False
        prices.append(raw)

    spacing = MIN_TARGET_SPACING_PIPS * pip
    for index in range(1, len(prices)):
        if direction == "BUY":
            prices[index] = max(prices[index], prices[index - 1] + spacing)
        else:
            prices[index] = min(prices[index], prices[index - 1] - spacing)

    return [round_price(price, config.digits) for price in prices], obstructed


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def build_targets(
    df: pd.DataFrame,
    config,
    direction: str,
    entry: Optional[float] = None,
    spread_points: Optional[float] = None,
) -> Tuple[Optional[Targets], str]:
    """Build the full scalping target set.

    Returns ``(targets, reason)``; ``targets`` is ``None`` when the geometry is
    unusable or the move is too small to be worth its costs, and ``reason``
    says which.
    """
    if direction not in DIRECTIONS:
        return None, f"invalid direction '{direction}'"
    if df is None or len(df) < max(config.sl_structure_lookback, 5):
        return None, "insufficient data for targets"

    entry = round_price(fnum(df["close"].iloc[-1]) if entry is None else entry, config.digits)
    if entry <= 0:
        return None, "invalid entry price"

    atr_value = max(fnum(df["atr"].iloc[-1]), 1e-9)
    cost_price = config.round_trip_cost(spread_points)

    stop_loss, _distance, mode_used = compute_stop_loss(
        df, config, direction, entry, spread_points
    )
    if direction == "BUY" and stop_loss >= entry:
        return None, "stop loss is not below entry for a BUY"
    if direction == "SELL" and stop_loss <= entry:
        return None, "stop loss is not above entry for a SELL"

    # Risk is recomputed from the ROUNDED stop so every R below matches the
    # prices actually published.
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return None, "zero risk after rounding"

    prices, obstructed = compute_take_profits(df, config, direction, entry, cost_price)
    tp1, tp2, tp3 = prices

    if direction == "BUY" and not (entry < tp1 < tp2 < tp3):
        return None, "take-profit ladder is not ordered above entry"
    if direction == "SELL" and not (entry > tp1 > tp2 > tp3):
        return None, "take-profit ladder is not ordered below entry"

    sign = 1.0 if direction == "BUY" else -1.0
    tp_distances = [sign * (price - entry) for price in prices]

    # The decisive scalping check: is the first target actually worth taking
    # once the spread and slippage are paid?
    required = config.min_tp1_cost_multiple * cost_price
    if tp_distances[0] < required - 1e-9:
        return None, (
            f"target too small vs costs "
            f"(TP1 {config.pips(tp_distances[0]):.1f}{config.pip_name} < "
            f"{config.pips(required):.1f}{config.pip_name} needed)"
        )

    # A scalp that needs a large move is not a scalp.
    ceiling = max_tp3_distance(config, entry)
    if tp_distances[2] > ceiling:
        return None, (
            f"TP3 {config.pips(tp_distances[2]):.1f}{config.pip_name} exceeds the "
            f"{config.pips(ceiling):.1f}{config.pip_name} scalp range"
        )

    rr = [safe_div(distance, risk) for distance in tp_distances]
    cost_r = safe_div(cost_price, risk)

    return (
        Targets(
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            risk=round(risk, 6),
            rr1=round(rr[0], 2),
            rr2=round(rr[1], 2),
            rr3=round(rr[2], 2),
            sl_mode=mode_used,
            obstructed=obstructed,
            spread_points=config.effective_spread_points(spread_points),
            cost_price=round(cost_price, 6),
            cost_r=round(cost_r, 4),
            net_rr1=round(rr[0] - cost_r, 2),
            net_rr2=round(rr[1] - cost_r, 2),
            net_rr3=round(rr[2] - cost_r, 2),
            sl_pips=round(config.pips(risk), 2),
            tp_pips=tuple(round(config.pips(d), 2) for d in tp_distances),
            cost_pips=round(config.pips(cost_price), 2),
            atr=round(atr_value, 5),
        ),
        "",
    )
