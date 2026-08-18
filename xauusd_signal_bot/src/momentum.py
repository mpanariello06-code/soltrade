"""Momentum engine - RSI, MACD, stochastic, ROC and simple divergence.

Scores on a 0-15 scale.

Momentum is treated as *one component among nine*.  There is deliberately no
"RSI below 30 therefore buy" rule: oversold readings contribute through the
RSI-slope and divergence terms only, which require the reading to actually be
turning.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .indicators import confirmed_swings
from .utils import ComponentScore, clamp, fnum, safe_div, scale

MAX_SCORE = 15.0

_W_RSI_POSITION = 3.0
_W_RSI_SLOPE = 2.0
_W_MACD_STATE = 3.0
_W_MACD_ACCELERATION = 2.0
_W_STOCH = 2.0
_W_ROC = 2.0
_W_DIVERGENCE = 1.0

DIVERGENCE_LOOKBACK = 80


def _detect_divergence(df: pd.DataFrame, config) -> Optional[str]:
    """Classic two-pivot RSI divergence, using confirmed swings only.

    Returns ``"BULL"``, ``"BEAR"`` or ``None``.  Pivots are taken from
    :func:`src.indicators.confirmed_swings`, so no future candle is consulted.
    """
    params = config.indicators
    swings = confirmed_swings(df, params.swing_left, params.swing_right)
    if swings.empty:
        return None

    cutoff = len(df) - 1 - DIVERGENCE_LOOKBACK
    recent = swings[swings["index"] >= cutoff]
    rsi_series = df["rsi"]

    lows = recent[recent["kind"] == "low"]
    if len(lows) >= 2:
        first, second = lows.iloc[-2], lows.iloc[-1]
        rsi_first = fnum(rsi_series.iloc[int(first["index"])], 50.0)
        rsi_second = fnum(rsi_series.iloc[int(second["index"])], 50.0)
        if second["price"] < first["price"] and rsi_second > rsi_first + 2.0:
            return "BULL"

    highs = recent[recent["kind"] == "high"]
    if len(highs) >= 2:
        first, second = highs.iloc[-2], highs.iloc[-1]
        rsi_first = fnum(rsi_series.iloc[int(first["index"])], 50.0)
        rsi_second = fnum(rsi_series.iloc[int(second["index"])], 50.0)
        if second["price"] > first["price"] and rsi_second < rsi_first - 2.0:
            return "BEAR"
    return None


def analyze_momentum(df: pd.DataFrame, config) -> ComponentScore:
    """Score momentum on the signal timeframe (0-15 per direction)."""
    if df is None or len(df) < 5:
        return ComponentScore("momentum", 0.0, 0.0, MAX_SCORE, {"reason": "insufficient data"})

    row, prev = df.iloc[-1], df.iloc[-2]
    bull = bear = 0.0

    # 1. RSI position relative to the 50 midline ---------------------------- #
    rsi_value = fnum(row["rsi"], 50.0)
    if rsi_value > 50.0:
        bull += _W_RSI_POSITION * scale(rsi_value, 50.0, 61.0)
    else:
        bear += _W_RSI_POSITION * scale(50.0 - rsi_value, 0.0, 11.0)

    # 2. RSI slope ----------------------------------------------------------- #
    rsi_slope = fnum(row["rsi_slope"])
    if rsi_slope > 0:
        bull += _W_RSI_SLOPE * scale(rsi_slope, 0.0, 1.9)
    else:
        bear += _W_RSI_SLOPE * scale(-rsi_slope, 0.0, 1.9)

    # 3. MACD state (line vs signal, and histogram sign) --------------------- #
    macd_line, macd_signal = fnum(row["macd"]), fnum(row["macd_signal"])
    hist, prev_hist = fnum(row["macd_hist"]), fnum(prev["macd_hist"])
    if macd_line > macd_signal:
        bull += _W_MACD_STATE * (1.0 if hist > 0 else 0.6)
    elif macd_line < macd_signal:
        bear += _W_MACD_STATE * (1.0 if hist < 0 else 0.6)

    # 4. Histogram acceleration (momentum of momentum) ----------------------- #
    atr_value = max(fnum(row["atr"]), 1e-9)
    hist_change = safe_div(hist - prev_hist, atr_value)
    if hist_change > 0:
        bull += _W_MACD_ACCELERATION * scale(hist_change, 0.0, 0.05)
    else:
        bear += _W_MACD_ACCELERATION * scale(-hist_change, 0.0, 0.05)

    # 5. Stochastic ---------------------------------------------------------- #
    stoch_k, stoch_d = fnum(row["stoch_k"], 50.0), fnum(row["stoch_d"], 50.0)
    if stoch_k > stoch_d:
        # rising, but heavily overbought readings earn less
        bull += _W_STOCH * (0.45 if stoch_k > 85.0 else 1.0)
    elif stoch_k < stoch_d:
        bear += _W_STOCH * (0.45 if stoch_k < 15.0 else 1.0)

    # 6. Rate of change ------------------------------------------------------ #
    roc_value = fnum(row["roc"])
    if roc_value > 0:
        bull += _W_ROC * scale(roc_value, 0.0, 0.16)
    else:
        bear += _W_ROC * scale(-roc_value, 0.0, 0.16)

    # 7. Divergence ---------------------------------------------------------- #
    divergence = _detect_divergence(df, config)
    if divergence == "BULL":
        bull += _W_DIVERGENCE
    elif divergence == "BEAR":
        bear += _W_DIVERGENCE

    details = {
        "rsi": round(rsi_value, 2),
        "rsi_slope": round(rsi_slope, 3),
        "macd_hist": round(hist, 4),
        "macd_above_signal": bool(macd_line > macd_signal),
        "stoch_k": round(stoch_k, 2),
        "roc": round(roc_value, 4),
        "divergence": divergence or "NONE",
    }
    return ComponentScore(
        "momentum", clamp(bull, 0.0, MAX_SCORE), clamp(bear, 0.0, MAX_SCORE), MAX_SCORE, details
    )
