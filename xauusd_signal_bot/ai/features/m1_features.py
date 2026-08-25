"""Causal M1 features for the PPO observation.

THE CAUSALITY CONTRACT
----------------------
At candle ``t`` the agent may see only information derivable from candles
``<= t``.  Every function here honours that, and the rule is enforced by an
automated future-append test (``tests/test_ai_features.py``): compute features
over ``[0..n]``, append more candles, recompute, and the original rows must come
back bit-identical.  A feature that fails is a leak, not a rounding difference.

Two habits keep it true:

* rolling windows only ever look backwards - no ``center=True``, no negative
  ``shift``;
* anything referencing "the previous bar" uses ``shift(+1)``, never ``shift(-1)``.

NORMALISATION
-------------
Almost every feature is expressed as a multiple of ATR or as a fraction of
price.  Raw prices are deliberately excluded: gold at 1,800 and gold at 2,400
are the same market, and a model keyed to absolute price learns the level rather
than the behaviour.  It also means one feature set works for XAUUSDs at 2,300
and BTCUSDs at 60,000 without rescaling.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

#: Guard against dividing by a zero ATR on a dead-flat series.
EPS = 1e-12

#: Feature block version.  Bumped whenever the column set changes, and recorded
#: in every trained model so a model can never be fed a different feature set
#: from the one it learned on.
FEATURE_VERSION = "m1-v1"


def true_range(frame: pd.DataFrame) -> pd.Series:
    """Wilder's true range, using the PREVIOUS close (causal)."""
    high, low = frame["high"].astype(float), frame["low"].astype(float)
    previous_close = frame["close"].astype(float).shift(1)
    spans = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    )
    return spans.max(axis=1)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range.  ``min_periods=period`` so early rows are NaN.

    Leaving the warm-up as NaN rather than back-filling matters: a back-filled
    ATR is a value computed from candles the agent had not seen yet.
    """
    return true_range(frame).rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI, expressed in [0, 1] rather than [0, 100]."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    average_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    average_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    strength = average_gain / (average_loss + EPS)
    return 1.0 - 1.0 / (1.0 + strength)


def build_m1_features(
    frame: pd.DataFrame,
    atr_period: int = 14,
    rsi_period: int = 14,
    returns_windows: tuple = (1, 3, 5, 15, 30),
    range_windows: tuple = (5, 15, 60),
) -> pd.DataFrame:
    """Attach the causal M1 feature block to a copy of ``frame``.

    ``frame`` must hold **closed** candles, chronologically ordered, with
    ``time/open/high/low/close/tick_volume`` and optionally ``spread``.
    """
    out = frame.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    close = out["close"].astype(float)

    # -- volatility anchor ------------------------------------------------- #
    out["f_atr"] = atr(out, atr_period)
    scale = out["f_atr"].replace(0.0, np.nan)

    # -- candle shape ------------------------------------------------------- #
    candle_range = (high - low)
    body = (close - open_)
    out["f_range_atr"] = candle_range / scale
    out["f_body_atr"] = body / scale
    out["f_upper_wick_atr"] = (high - np.maximum(open_, close)) / scale
    out["f_lower_wick_atr"] = (np.minimum(open_, close) - low) / scale
    # Body as a share of range: 1.0 is a marubozu, 0.0 a doji.  Direction-free.
    out["f_body_ratio"] = body.abs() / candle_range.replace(0.0, np.nan)
    out["f_direction"] = np.sign(body)
    # Where the candle closed inside its own range: 1 = at the high, 0 = the low.
    out["f_close_location"] = (close - low) / candle_range.replace(0.0, np.nan)

    # -- returns, ATR-normalised -------------------------------------------- #
    for window in returns_windows:
        out[f"f_return_{window}_atr"] = (close - close.shift(window)) / scale
    out["f_momentum_atr"] = (close - close.shift(5)) / scale
    # Acceleration: is the recent push faster than the one before it?
    out["f_acceleration"] = out["f_return_3_atr"] - out["f_return_3_atr"].shift(3)

    # -- rolling volatility --------------------------------------------------- #
    log_return = np.log(close / close.shift(1).replace(0.0, np.nan))
    out["f_volatility_15"] = log_return.rolling(15, min_periods=15).std()
    out["f_volatility_60"] = log_return.rolling(60, min_periods=60).std()
    # Volatility regime: is now busier or quieter than the last hour?
    out["f_volatility_ratio"] = out["f_volatility_15"] / (out["f_volatility_60"] + EPS)

    # -- position within recent range ------------------------------------------ #
    for window in range_windows:
        window_high = high.rolling(window, min_periods=window).max()
        window_low = low.rolling(window, min_periods=window).min()
        span = (window_high - window_low).replace(0.0, np.nan)
        out[f"f_dist_high_{window}_atr"] = (window_high - close) / scale
        out[f"f_dist_low_{window}_atr"] = (close - window_low) / scale
        out[f"f_position_{window}"] = (close - window_low) / span

    # -- oscillators ------------------------------------------------------------ #
    out["f_rsi"] = rsi(close, rsi_period)
    fast = close.ewm(span=12, min_periods=12, adjust=False).mean()
    slow = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd = fast - slow
    out["f_macd_atr"] = macd / scale
    out["f_macd_signal_atr"] = macd.ewm(span=9, min_periods=9, adjust=False).mean() / scale
    out["f_macd_hist_atr"] = out["f_macd_atr"] - out["f_macd_signal_atr"]

    # -- short-term trend --------------------------------------------------------- #
    ema_fast = close.ewm(span=9, min_periods=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, min_periods=21, adjust=False).mean()
    out["f_ema_spread_atr"] = (ema_fast - ema_slow) / scale
    out["f_price_vs_ema_atr"] = (close - ema_slow) / scale
    out["f_ema_slope_atr"] = (ema_slow - ema_slow.shift(5)) / scale

    # -- streaks ------------------------------------------------------------------- #
    # How many consecutive candles have closed the same way, signed and capped.
    sign = np.sign(body).fillna(0.0)
    group = (sign != sign.shift(1)).cumsum()
    streak = sign.groupby(group).cumcount() + 1
    out["f_streak"] = (streak.clip(upper=10) * sign) / 10.0

    # -- liquidity and cost ---------------------------------------------------------- #
    volume = pd.to_numeric(out.get("tick_volume", 0), errors="coerce").fillna(0.0)
    median_volume = volume.rolling(60, min_periods=60).median()
    out["f_volume_ratio"] = volume / (median_volume + EPS)
    if "spread" in out.columns:
        spread = pd.to_numeric(out["spread"], errors="coerce").fillna(0.0)
        out["f_spread_raw"] = spread
        median_spread = spread.rolling(60, min_periods=60).median()
        out["f_spread_ratio"] = spread / (median_spread + EPS)
    else:
        out["f_spread_raw"] = 0.0
        out["f_spread_ratio"] = 0.0

    # -- session / time-of-day ---------------------------------------------------------- #
    # Cyclical encoding: 23:59 and 00:01 must be near each other, which a raw
    # hour number gets badly wrong.
    minute_of_day = out["time"].dt.hour * 60 + out["time"].dt.minute
    angle = 2.0 * np.pi * minute_of_day / (24 * 60)
    out["f_tod_sin"] = np.sin(angle)
    out["f_tod_cos"] = np.cos(angle)
    weekday_angle = 2.0 * np.pi * out["time"].dt.dayofweek / 7.0
    out["f_dow_sin"] = np.sin(weekday_angle)
    out["f_dow_cos"] = np.cos(weekday_angle)

    return out


def m1_feature_columns(frame: pd.DataFrame) -> List[str]:
    """Every ``f_*`` column, in a stable order."""
    return sorted(column for column in frame.columns if column.startswith("f_"))


def describe_features() -> Dict[str, str]:
    """Feature name -> one-line meaning, for the design doc and the registry."""
    return {
        "f_atr": "average true range, price units (the normalisation anchor)",
        "f_range_atr": "candle range / ATR",
        "f_body_atr": "signed body / ATR",
        "f_upper_wick_atr": "upper wick / ATR",
        "f_lower_wick_atr": "lower wick / ATR",
        "f_body_ratio": "|body| / range: 1 marubozu, 0 doji",
        "f_direction": "sign of the body",
        "f_close_location": "close within its own range, 0 low .. 1 high",
        "f_return_N_atr": "N-candle return / ATR",
        "f_momentum_atr": "5-candle return / ATR",
        "f_acceleration": "change in 3-candle return over 3 candles",
        "f_volatility_15/60": "rolling std of log returns",
        "f_volatility_ratio": "15-candle vol / 60-candle vol (regime)",
        "f_dist_high_N_atr": "distance below the N-candle high / ATR",
        "f_dist_low_N_atr": "distance above the N-candle low / ATR",
        "f_position_N": "price within the N-candle range, 0..1",
        "f_rsi": "Wilder RSI scaled to 0..1",
        "f_macd_atr / signal / hist": "MACD line, signal and histogram / ATR",
        "f_ema_spread_atr": "EMA9 - EMA21, / ATR",
        "f_price_vs_ema_atr": "close - EMA21, / ATR",
        "f_ema_slope_atr": "EMA21 slope over 5 candles / ATR",
        "f_streak": "signed consecutive same-direction closes, capped and scaled",
        "f_volume_ratio": "tick volume / its 60-candle median",
        "f_spread_raw": "broker spread in points, as recorded",
        "f_spread_ratio": "spread / its 60-candle median",
        "f_tod_sin/cos": "cyclical time of day",
        "f_dow_sin/cos": "cyclical day of week",
    }
