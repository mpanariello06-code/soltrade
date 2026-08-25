"""Causal higher-timeframe context (M5, M15, optionally H1) for the M1 agent.

WHERE MULTI-TIMEFRAME FEATURES LEAK
-----------------------------------
The classic mistake is to resample M1 into M5, join on the bucket a candle
belongs to, and call it context.  At 12:01 that hands the agent the M5 bucket
labelled 12:00 - which does not close until 12:04.  Three of its five minutes are
in the future, and the model quietly learns to read them.

This module refuses to make that mistake in two ways:

1. buckets are built only when **complete** (``resample_candles`` already does
   this, and it is reused rather than reimplemented);
2. each bucket is then stamped with its **close time**, and merged onto M1 with
   ``direction="backward"`` on that close time - so a bucket becomes visible on
   the first M1 candle at or after it closed, and not one candle sooner.

The result: at 12:01 the newest visible M5 bucket is 11:55-12:00.  That is
context; the alternative is a time machine.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.market_data import resample_candles, timeframe_minutes

from .m1_features import atr, rsi

#: Higher timeframes attached by default.  H1 is available but off: on a scalp
#: held for minutes, an hourly trend mostly adds parameters, not information.
DEFAULT_CONTEXT_TIMEFRAMES: tuple = ("M5", "M15")

MTF_VERSION = "mtf-v1"


def build_context_frame(
    m1: pd.DataFrame, timeframe: str, atr_period: int = 14, rsi_period: int = 14
) -> pd.DataFrame:
    """Resample M1 into ``timeframe`` and attach that frame's own features.

    Returns a frame keyed by ``available_at`` - the moment the bucket CLOSED and
    therefore became knowable - not by the bucket's opening label.
    """
    minutes = timeframe_minutes(timeframe)
    buckets = resample_candles(m1, "M1", timeframe)
    if buckets.empty:
        return pd.DataFrame(columns=["available_at"])

    frame = buckets.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    close = frame["close"].astype(float)
    high, low = frame["high"].astype(float), frame["low"].astype(float)

    frame["atr"] = atr(frame, atr_period)
    scale = frame["atr"].replace(0.0, np.nan)

    prefix = f"f_{timeframe.lower()}_"
    context = pd.DataFrame(index=frame.index)
    # The label is the bucket's OPEN; it is knowable only once it has closed.
    context["available_at"] = frame["time"] + pd.Timedelta(minutes=minutes)

    context[f"{prefix}range_atr"] = (high - low) / scale
    context[f"{prefix}body_atr"] = (close - frame["open"].astype(float)) / scale
    context[f"{prefix}return_1_atr"] = (close - close.shift(1)) / scale
    context[f"{prefix}return_3_atr"] = (close - close.shift(3)) / scale
    context[f"{prefix}rsi"] = rsi(close, rsi_period)

    ema_fast = close.ewm(span=9, min_periods=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, min_periods=21, adjust=False).mean()
    context[f"{prefix}ema_spread_atr"] = (ema_fast - ema_slow) / scale
    context[f"{prefix}price_vs_ema_atr"] = (close - ema_slow) / scale
    context[f"{prefix}ema_slope_atr"] = (ema_slow - ema_slow.shift(3)) / scale

    window_high = high.rolling(20, min_periods=20).max()
    window_low = low.rolling(20, min_periods=20).min()
    span = (window_high - window_low).replace(0.0, np.nan)
    context[f"{prefix}dist_high_atr"] = (window_high - close) / scale
    context[f"{prefix}dist_low_atr"] = (close - window_low) / scale
    context[f"{prefix}position"] = (close - window_low) / span
    # The higher timeframe's ATR relative to price: a scale-free volatility read
    # that is comparable between gold and Bitcoin.
    context[f"{prefix}atr_pct"] = frame["atr"] / close.replace(0.0, np.nan)

    return context.dropna(subset=["available_at"]).reset_index(drop=True)


def attach_context(
    m1: pd.DataFrame,
    timeframes: Iterable[str] = DEFAULT_CONTEXT_TIMEFRAMES,
    atr_period: int = 14,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """Merge higher-timeframe context onto an M1 frame, causally.

    ``merge_asof(direction="backward")`` on the bucket's CLOSE time is what does
    the work: every M1 row receives the most recent bucket that had already
    finished, and never the one still forming.
    """
    out = m1.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.sort_values("time", kind="mergesort").reset_index(drop=True)

    for timeframe in timeframes:
        context = build_context_frame(out, timeframe, atr_period, rsi_period)
        if context.empty:
            continue
        context = context.sort_values("available_at", kind="mergesort")
        out = pd.merge_asof(
            out,
            context,
            left_on="time",
            right_on="available_at",
            direction="backward",     # only buckets already closed
            allow_exact_matches=True,  # a bucket closing at exactly t IS knowable at t
        )
        out = out.drop(columns=["available_at"])
    return out


def context_feature_columns(frame: pd.DataFrame, timeframes: Iterable[str]) -> List[str]:
    """Context columns for the given timeframes, in a stable order."""
    prefixes = tuple(f"f_{tf.lower()}_" for tf in timeframes)
    return sorted(c for c in frame.columns if c.startswith(prefixes))


def describe_context() -> Dict[str, str]:
    return {
        "f_<tf>_range_atr": "higher-timeframe candle range / its own ATR",
        "f_<tf>_body_atr": "higher-timeframe body / its own ATR",
        "f_<tf>_return_1/3_atr": "1- and 3-bucket returns / ATR",
        "f_<tf>_rsi": "higher-timeframe RSI, 0..1",
        "f_<tf>_ema_spread_atr": "EMA9 - EMA21 on that timeframe / ATR",
        "f_<tf>_price_vs_ema_atr": "close - EMA21 / ATR",
        "f_<tf>_ema_slope_atr": "EMA21 slope over 3 buckets / ATR",
        "f_<tf>_dist_high/low_atr": "distance from the 20-bucket extremes / ATR",
        "f_<tf>_position": "price within the 20-bucket range, 0..1",
        "f_<tf>_atr_pct": "that timeframe's ATR as a fraction of price",
    }
