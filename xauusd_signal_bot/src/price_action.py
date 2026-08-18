"""Price-action engine - candle-level confirmation only.

Scores on a 0-5 scale.  Price action is used strictly as *confirmation*: it can
top up a setup that the other engines already like, but 5/100 points can never
push a signal over the threshold on its own.
"""

from __future__ import annotations

import pandas as pd

from .utils import ComponentScore, clamp, fnum, safe_div, scale

MAX_SCORE = 5.0

_W_DISPLACEMENT = 2.0
_W_ENGULFING = 1.0
_W_REJECTION = 1.0
_W_CLOSE_LOCATION = 1.0

# Calibration note: these ranges come from the distribution of M5 candle
# geometry, not from backtest P&L.  The four terms are deliberately
# complementary rather than competing: a decisive body and a long rejection
# wick are mutually exclusive by construction, so scoring both on the same
# axis (as the first version did) left this component permanently starved.
DISPLACEMENT_ATR_LOW = 0.25
DISPLACEMENT_ATR_HIGH = 0.80
STRONG_BODY_RATIO = 0.65
BODY_QUALITY_LOW = 0.30
BODY_QUALITY_HIGH = 0.65
REJECTION_WICK_RATIO = 0.35
CLOSE_LOCATION_LOW = 0.55
CLOSE_LOCATION_HIGH = 0.90


def analyze_price_action(df: pd.DataFrame, config) -> ComponentScore:
    """Score the confirmation quality of the signal candle (0-5 per direction)."""
    if df is None or len(df) < 3:
        return ComponentScore("price_action", 0.0, 0.0, MAX_SCORE, {"reason": "insufficient data"})

    row, prev = df.iloc[-1], df.iloc[-2]
    open_price, close = fnum(row["open"]), fnum(row["close"])
    high, low = fnum(row["high"]), fnum(row["low"])
    prev_open, prev_close = fnum(prev["open"]), fnum(prev["close"])
    atr_value = max(fnum(row["atr"]), 1e-9)
    candle_range = max(high - low, 1e-9)
    body = abs(close - open_price)
    body_ratio = safe_div(body, candle_range)

    bullish_candle = close > open_price
    bearish_candle = close < open_price

    bull = bear = 0.0

    # 1. Displacement: body size versus ATR, weighted by how clean the body is.
    #    Scaled rather than gated - a hard "body_ratio >= 0.65" cut-off made this
    #    component score near zero on almost every candle.
    displacement = safe_div(body, atr_value)
    displacement_points = (
        _W_DISPLACEMENT
        * scale(displacement, DISPLACEMENT_ATR_LOW, DISPLACEMENT_ATR_HIGH)
        * scale(body_ratio, BODY_QUALITY_LOW, BODY_QUALITY_HIGH)
    )
    strong_body = body_ratio >= STRONG_BODY_RATIO
    if bullish_candle:
        bull += displacement_points
    elif bearish_candle:
        bear += displacement_points

    # 2. Engulfing-type confirmation ----------------------------------------- #
    bullish_engulf = (
        bullish_candle
        and prev_close < prev_open
        and close >= prev_open
        and open_price <= prev_close
    )
    bearish_engulf = (
        bearish_candle
        and prev_close > prev_open
        and close <= prev_open
        and open_price >= prev_close
    )
    if bullish_engulf:
        bull += _W_ENGULFING
    if bearish_engulf:
        bear += _W_ENGULFING

    # 3. Wick rejection ------------------------------------------------------- #
    lower_wick_ratio = safe_div(fnum(row["lower_wick"]), candle_range)
    upper_wick_ratio = safe_div(fnum(row["upper_wick"]), candle_range)
    if lower_wick_ratio >= REJECTION_WICK_RATIO and close > low + 0.5 * candle_range:
        bull += _W_REJECTION * scale(lower_wick_ratio, REJECTION_WICK_RATIO, 0.65)
    if upper_wick_ratio >= REJECTION_WICK_RATIO and close < high - 0.5 * candle_range:
        bear += _W_REJECTION * scale(upper_wick_ratio, REJECTION_WICK_RATIO, 0.65)

    # 4. Where the candle closed inside its own range - the one confirmation
    #    that a decisive body and a rejection wick can both earn.
    close_location = safe_div(close - low, candle_range)
    bull += _W_CLOSE_LOCATION * scale(close_location, CLOSE_LOCATION_LOW, CLOSE_LOCATION_HIGH)
    bear += _W_CLOSE_LOCATION * scale(1.0 - close_location, CLOSE_LOCATION_LOW, CLOSE_LOCATION_HIGH)

    details = {
        "body_ratio": round(body_ratio, 3),
        "displacement_atr": round(displacement, 3),
        "bullish_engulfing": bool(bullish_engulf),
        "bearish_engulfing": bool(bearish_engulf),
        "lower_wick_ratio": round(lower_wick_ratio, 3),
        "upper_wick_ratio": round(upper_wick_ratio, 3),
        "strong_body": bool(strong_body),
        "close_location": round(safe_div(close - low, candle_range), 3),
    }
    return ComponentScore(
        "price_action", clamp(bull, 0.0, MAX_SCORE), clamp(bear, 0.0, MAX_SCORE), MAX_SCORE, details
    )
