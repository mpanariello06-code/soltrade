"""Volume engine - tick-volume confirmation.

Scores on a 0-5 scale.

Volume never *creates* a setup here: it only rewards the direction the candle
already moved.  A high-volume down candle adds to the bearish side only.
"""

from __future__ import annotations

import pandas as pd

from .utils import ComponentScore, clamp, fnum, safe_div, scale

MAX_SCORE = 5.0

_W_EXPANSION = 3.0
_W_TREND = 2.0

UNUSUAL_VOLUME_RATIO = 2.0
CONTRACTION_RATIO = 0.7
#: relative-volume range mapped onto the expansion score
EXPANSION_LOW = 0.85
EXPANSION_HIGH = 1.40


def analyze_volume(df: pd.DataFrame, config) -> ComponentScore:
    """Score tick-volume confirmation (0-5 per direction)."""
    params = config.indicators
    if df is None or len(df) < params.volume_ma_period + 3:
        return ComponentScore("volume", 0.0, 0.0, MAX_SCORE, {"reason": "insufficient data"})

    row = df.iloc[-1]
    close, open_price = fnum(row["close"]), fnum(row["open"])
    relative_volume = fnum(row["rel_volume"], 1.0)
    if relative_volume <= 0:
        relative_volume = 1.0

    bullish_candle = close > open_price
    bearish_candle = close < open_price

    bull = bear = 0.0

    # 1. Expansion on the directional candle -------------------------------- #
    expansion_points = _W_EXPANSION * scale(relative_volume, EXPANSION_LOW, EXPANSION_HIGH)
    if bullish_candle:
        bull += expansion_points
    elif bearish_candle:
        bear += expansion_points

    # 2. Volume flowing with direction over the last few candles ------------- #
    window = df.iloc[-3:]
    up_volume = float(window.loc[window["close"] > window["open"], "tick_volume"].sum())
    down_volume = float(window.loc[window["close"] < window["open"], "tick_volume"].sum())
    total = up_volume + down_volume
    if total > 0:
        bias = safe_div(up_volume - down_volume, total)
        if bias > 0:
            bull += _W_TREND * scale(bias, 0.03, 0.45)
        else:
            bear += _W_TREND * scale(-bias, 0.03, 0.45)

    details = {
        "rel_volume": round(relative_volume, 3),
        "expansion": bool(relative_volume >= 1.2),
        "contraction": bool(relative_volume <= CONTRACTION_RATIO),
        "unusual": bool(relative_volume >= UNUSUAL_VOLUME_RATIO),
    }
    return ComponentScore(
        "volume", clamp(bull, 0.0, MAX_SCORE), clamp(bear, 0.0, MAX_SCORE), MAX_SCORE, details
    )
