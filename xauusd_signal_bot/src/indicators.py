"""Technical indicator primitives.

Implemented directly on top of pandas/numpy rather than ``pandas-ta``: the
project only needs a dozen classic indicators, and a self-contained
implementation removes a heavy, currently unmaintained dependency while making
every calculation unit-testable.

ANTI-LOOKAHEAD CONTRACT
-----------------------
Every function here is *causal*: the value at index ``i`` is derived only from
rows ``0..i``.  The single exception is :func:`swing_points`, which by
definition needs candles after a pivot to confirm it - that function therefore
returns an explicit ``confirmed_at`` index and callers must respect it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .utils import BoundedCache, frame_fingerprint


# --------------------------------------------------------------------------- #
# moving averages
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.astype(float).ewm(span=max(int(period), 1), adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    period = max(int(period), 1)
    return series.astype(float).rolling(period, min_periods=period).mean()


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (the RMA used by RSI/ATR/ADX)."""
    period = max(int(period), 1)
    return series.astype(float).ewm(alpha=1.0 / period, adjust=False).mean()


def slope(series: pd.Series, lookback: int = 5) -> pd.Series:
    """Average change per bar over ``lookback`` bars (causal)."""
    lookback = max(int(lookback), 1)
    return (series.astype(float) - series.astype(float).shift(lookback)) / lookback


# --------------------------------------------------------------------------- #
# volatility
# --------------------------------------------------------------------------- #
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range."""
    high, low, close = high.astype(float), low.astype(float), close.astype(float)
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average true range (Wilder)."""
    return wilder_smooth(true_range(high, low, close), period)


def bollinger_bands(
    close: pd.Series, period: int = 20, stddev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(upper, middle, lower)`` Bollinger bands."""
    middle = sma(close, period)
    deviation = close.astype(float).rolling(max(int(period), 1), min_periods=max(int(period), 1)).std(ddof=0)
    return middle + stddev * deviation, middle, middle - stddev * deviation


def bollinger_width(close: pd.Series, period: int = 20, stddev: float = 2.0) -> pd.Series:
    """Band width normalised by the middle band (a unitless volatility gauge)."""
    upper, middle, lower = bollinger_bands(close, period, stddev)
    width = (upper - lower) / middle.replace(0.0, np.nan)
    return width.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# momentum
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative strength index (Wilder smoothing)."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken run of gains -> RSI 100
    result = result.where(avg_loss.ne(0.0), 100.0)
    return result.where(avg_gain.ne(0.0) | avg_loss.ne(0.0), 50.0)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(macd_line, signal_line, histogram)``."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    smooth: int = 3,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Return the smoothed ``(%K, %D)`` stochastic oscillator."""
    k_period = max(int(k_period), 1)
    lowest = low.astype(float).rolling(k_period, min_periods=k_period).min()
    highest = high.astype(float).rolling(k_period, min_periods=k_period).max()
    span = (highest - lowest).replace(0.0, np.nan)
    raw_k = 100.0 * (close.astype(float) - lowest) / span
    raw_k = raw_k.fillna(50.0)
    k = raw_k.rolling(max(int(smooth), 1), min_periods=max(int(smooth), 1)).mean()
    d = k.rolling(max(int(d_period), 1), min_periods=max(int(d_period), 1)).mean()
    return k, d


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change, in percent."""
    period = max(int(period), 1)
    previous = close.astype(float).shift(period).replace(0.0, np.nan)
    return 100.0 * (close.astype(float) - previous) / previous


# --------------------------------------------------------------------------- #
# directional movement
# --------------------------------------------------------------------------- #
def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(adx, plus_di, minus_di)`` using Wilder's method."""
    high, low = high.astype(float), low.astype(float)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=high.index
    )

    atr_series = wilder_smooth(true_range(high, low, close), period).replace(0.0, np.nan)
    plus_di = 100.0 * wilder_smooth(plus_dm, period) / atr_series
    minus_di = 100.0 * wilder_smooth(minus_dm, period) / atr_series

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return wilder_smooth(dx.fillna(0.0), period), plus_di, minus_di


# --------------------------------------------------------------------------- #
# swing / pivot detection
# --------------------------------------------------------------------------- #
_SWING_CACHE = BoundedCache(maxsize=48)


def swing_points(
    high: pd.Series,
    low: pd.Series,
    left: int = 2,
    right: int = 2,
) -> pd.DataFrame:
    """Detect fractal swing highs/lows (vectorised).

    A pivot at index ``i`` needs ``left`` bars before and ``right`` bars after,
    so it only becomes *knowable* at index ``i + right``.  The returned frame
    carries that explicitly:

    ``index`` - positional index of the pivot bar
    ``price`` - pivot price
    ``kind``  - ``"high"`` or ``"low"``
    ``confirmed_at`` - positional index at which the pivot became known

    Callers analysing a window ending at position ``n`` must discard rows with
    ``confirmed_at > n``.  :func:`confirmed_swings` does that for you.

    The comparison is strict on the left and non-strict on the right, which
    avoids emitting a pivot twice for a flat top while still confirming it.
    """
    left, right = max(int(left), 1), max(int(right), 1)
    highs = high.to_numpy(dtype="float64")
    lows = low.to_numpy(dtype="float64")
    count = len(highs)
    empty = pd.DataFrame(columns=["index", "price", "kind", "confirmed_at"])
    if count < left + right + 1:
        return empty

    window = left + right + 1
    high_series = pd.Series(highs)
    low_series = pd.Series(lows)

    # value at i = extreme of [i-left, i+right]
    window_max = high_series.rolling(window).max().shift(-right).to_numpy()
    window_min = low_series.rolling(window).min().shift(-right).to_numpy()
    # value at i = extreme of [i-left, i-1] (strictly before the pivot)
    left_max = high_series.rolling(left).max().shift(1).to_numpy()
    left_min = low_series.rolling(left).min().shift(1).to_numpy()

    positions = np.arange(count)
    valid = (positions >= left) & (positions < count - right)
    is_high = valid & (highs >= window_max) & (left_max < highs)
    is_low = valid & (lows <= window_min) & (left_min > lows)

    high_index = positions[is_high]
    low_index = positions[is_low]
    if len(high_index) == 0 and len(low_index) == 0:
        return empty

    records = pd.DataFrame(
        {
            "index": np.concatenate([high_index, low_index]),
            "price": np.concatenate([highs[high_index], lows[low_index]]),
            "kind": ["high"] * len(high_index) + ["low"] * len(low_index),
            "confirmed_at": np.concatenate([high_index, low_index]) + right,
        }
    )
    return records.sort_values(["index", "kind"]).reset_index(drop=True)


def confirmed_swings(
    df: pd.DataFrame, left: int = 2, right: int = 2, as_of: Optional[int] = None
) -> pd.DataFrame:
    """Swings that were already confirmed at position ``as_of`` (default: last bar).

    This is the only swing accessor the engines should use - it is what keeps
    market-structure analysis free of lookahead bias.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["index", "price", "kind", "confirmed_at"])
    as_of = len(df) - 1 if as_of is None else int(as_of)
    key = (frame_fingerprint(df, ("high", "low")), int(left), int(right), as_of)
    cached = _SWING_CACHE.get(key)
    if cached is not None:
        return cached
    swings = swing_points(df["high"], df["low"], left, right)
    if not swings.empty:
        swings = swings[swings["confirmed_at"] <= as_of].reset_index(drop=True)
    return _SWING_CACHE.put(key, swings)


def last_swing(swings: pd.DataFrame, kind: str) -> Optional[dict]:
    """Most recent confirmed swing of the requested ``kind``, or ``None``."""
    if swings is None or swings.empty:
        return None
    subset = swings[swings["kind"] == kind]
    if subset.empty:
        return None
    return subset.iloc[-1].to_dict()


# --------------------------------------------------------------------------- #
# convenience bundle
# --------------------------------------------------------------------------- #
def compute_indicators(df: pd.DataFrame, params) -> pd.DataFrame:
    """Attach the full indicator set to a copy of ``df``.

    ``df`` must contain ``open/high/low/close/tick_volume`` and hold **closed
    candles only**.
    """
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]

    out["ema_fast"] = ema(close, params.ema_fast)
    out["ema_mid"] = ema(close, params.ema_mid)
    out["ema_slow"] = ema(close, params.ema_slow)
    out["ema_trend"] = ema(close, params.ema_trend)
    out["ema_fast_slope"] = slope(out["ema_fast"], 3)
    out["ema_mid_slope"] = slope(out["ema_mid"], 5)
    out["ema_slow_slope"] = slope(out["ema_slow"], 5)

    adx_v, plus_di, minus_di = adx(high, low, close, params.adx_period)
    out["adx"], out["plus_di"], out["minus_di"] = adx_v, plus_di, minus_di

    out["rsi"] = rsi(close, params.rsi_period)
    out["rsi_slope"] = slope(out["rsi"], 3)

    macd_line, macd_signal, macd_hist = macd(
        close, params.macd_fast, params.macd_slow, params.macd_signal
    )
    out["macd"], out["macd_signal"], out["macd_hist"] = macd_line, macd_signal, macd_hist
    out["macd_hist_slope"] = slope(macd_hist, 2)

    stoch_k, stoch_d = stochastic(
        high, low, close, params.stoch_k, params.stoch_smooth, params.stoch_d
    )
    out["stoch_k"], out["stoch_d"] = stoch_k, stoch_d
    out["roc"] = roc(close, params.roc_period)

    out["atr"] = atr(high, low, close, params.atr_period)
    out["atr_pct"] = 100.0 * out["atr"] / close.replace(0.0, np.nan)
    out["atr_history"] = out["atr"].rolling(
        params.atr_history_period, min_periods=max(params.atr_history_period // 2, 2)
    ).mean()
    out["bb_width"] = bollinger_width(close, params.bb_period, params.bb_stddev)
    out["bb_width_ma"] = out["bb_width"].rolling(params.bb_period, min_periods=params.bb_period).mean()

    out["volume_ma"] = sma(out["tick_volume"].astype(float), params.volume_ma_period)
    out["rel_volume"] = out["tick_volume"].astype(float) / out["volume_ma"].replace(0.0, np.nan)

    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = (out["high"] - out["low"]).replace(0.0, np.nan)
    out["body_ratio"] = (out["body"] / out["range"]).fillna(0.0)
    # clip at zero: a malformed bar (high < max(open, close)) must never produce
    # a negative wick that would flip a price-action score.
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)).clip(lower=0.0)
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]).clip(lower=0.0)
    return out
