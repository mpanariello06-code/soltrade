"""Indicator correctness, and the causality property the backtester relies on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    adx,
    atr,
    bollinger_bands,
    compute_indicators,
    confirmed_swings,
    ema,
    macd,
    roc,
    rsi,
    sma,
    stochastic,
    swing_points,
    true_range,
    wilder_smooth,
)
from tests.conftest import make_candles


# --------------------------------------------------------------------------- #
# basic maths
# --------------------------------------------------------------------------- #
def test_sma_matches_manual_average():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(series, 3)
    assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_seeds_on_first_value_and_tracks_a_constant():
    series = pd.Series([10.0] * 50)
    assert ema(series, 9).iloc[-1] == pytest.approx(10.0)


def test_wilder_smoothing_uses_alpha_one_over_period():
    series = pd.Series([1.0, 2.0])
    # first value seeds, second = prev + (x - prev)/period
    assert wilder_smooth(series, 2).iloc[1] == pytest.approx(1.0 + (2.0 - 1.0) / 2.0)


def test_true_range_uses_the_widest_of_the_three_measures():
    high = pd.Series([10.0, 12.0])
    low = pd.Series([9.0, 11.0])
    close = pd.Series([9.5, 11.5])
    result = true_range(high, low, close)
    # bar 2: high-low=1, |high-prev_close|=2.5, |low-prev_close|=1.5 -> 2.5
    assert result.iloc[1] == pytest.approx(2.5)


def test_rsi_is_100_for_an_unbroken_rally_and_0_for_a_collapse():
    rising = pd.Series(np.arange(1.0, 60.0))
    falling = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert rsi(rising, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(falling, 14).iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_rsi_stays_within_bounds_on_random_data():
    close = make_candles(500, seed=3)["close"]
    values = rsi(close, 14).dropna()
    assert values.between(0.0, 100.0).all()


def test_macd_histogram_is_line_minus_signal():
    close = make_candles(400, seed=4)["close"]
    line, signal, hist = macd(close, 12, 26, 9)
    assert np.allclose((line - signal).to_numpy(), hist.to_numpy(), equal_nan=True)


def test_atr_is_positive_and_finite():
    frame = make_candles(400, seed=5)
    values = atr(frame["high"], frame["low"], frame["close"], 14).dropna()
    assert (values > 0).all() and np.isfinite(values).all()


def test_adx_and_di_stay_within_bounds():
    frame = make_candles(600, seed=6)
    adx_values, plus_di, minus_di = adx(frame["high"], frame["low"], frame["close"], 14)
    for series in (adx_values, plus_di, minus_di):
        clean = series.dropna()
        assert clean.between(0.0, 100.0).all()


def test_stochastic_stays_within_bounds():
    frame = make_candles(400, seed=8)
    k, d = stochastic(frame["high"], frame["low"], frame["close"], 14, 3, 3)
    assert k.dropna().between(0.0, 100.0).all()
    assert d.dropna().between(0.0, 100.0).all()


def test_bollinger_bands_are_ordered():
    close = make_candles(300, seed=9)["close"]
    upper, middle, lower = bollinger_bands(close, 20, 2.0)
    mask = upper.notna()
    assert (upper[mask] >= middle[mask]).all()
    assert (middle[mask] >= lower[mask]).all()


def test_roc_is_zero_for_a_flat_series():
    assert roc(pd.Series([100.0] * 40), 10).iloc[-1] == pytest.approx(0.0)


def test_indicators_survive_a_constant_series_without_nan_or_inf():
    """A dead-flat market must not produce inf/NaN through division by zero."""
    count = 300
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=count, freq="5min", tz="UTC"),
            "open": 2000.0, "high": 2000.0, "low": 2000.0, "close": 2000.0,
            "tick_volume": 100.0,
        }
    )
    from config import IndicatorParams

    result = compute_indicators(frame, IndicatorParams())
    row = result.iloc[-1]
    for column in ("rsi", "atr", "adx", "macd_hist", "body_ratio", "stoch_k"):
        assert np.isfinite(float(row[column])), f"{column} is not finite on a flat series"


# --------------------------------------------------------------------------- #
# anti-lookahead
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cut", [400, 700, 1100])
def test_indicators_are_causal(config, cut):
    """The value at bar i must not depend on anything after bar i.

    This is what allows the backtester to pre-compute indicators over the whole
    history and then slice, instead of recomputing on every prefix.
    """
    frame = make_candles(1500, seed=11)
    full = compute_indicators(frame, config.indicators)
    prefix = compute_indicators(frame.iloc[: cut + 1], config.indicators)

    for column in (
        "ema_fast", "ema_trend", "adx", "plus_di", "rsi", "macd", "macd_hist",
        "stoch_k", "roc", "atr", "atr_history", "bb_width", "rel_volume",
    ):
        expected = float(full[column].iloc[cut])
        actual = float(prefix[column].iloc[cut])
        if np.isnan(expected) and np.isnan(actual):
            continue
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9), column


def test_swing_confirmation_lag_is_respected():
    """A pivot must not be visible before its confirmation bar has closed."""
    frame = make_candles(400, seed=13)
    right = 3
    swings = swing_points(frame["high"], frame["low"], 2, right)
    assert not swings.empty
    assert (swings["confirmed_at"] == swings["index"] + right).all()

    as_of = 200
    visible = confirmed_swings(frame.iloc[: as_of + 1], 2, right)
    assert (visible["confirmed_at"] <= as_of).all()
    # the pivot bar itself is always at least `right` bars in the past
    assert (visible["index"] <= as_of - right).all()


def test_confirmed_swings_do_not_repaint():
    """Swings known at bar N must still be present, unchanged, at bar N+k."""
    frame = make_candles(600, seed=17)
    early = confirmed_swings(frame.iloc[:301], 2, 2)
    later = confirmed_swings(frame.iloc[:451], 2, 2)
    merged = later.set_index(["index", "kind"])["price"]
    for _, pivot in early.iterrows():
        key = (int(pivot["index"]), pivot["kind"])
        assert key in merged.index, "a previously confirmed swing disappeared"
        assert float(merged.loc[key]) == pytest.approx(float(pivot["price"]))


def test_swing_points_matches_a_naive_reference_implementation():
    """Guards the vectorised implementation against the obvious loop version."""
    frame = make_candles(300, seed=19)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    left, right = 2, 2

    expected = []
    for i in range(left, len(highs) - right):
        window_high = highs[i - left : i + right + 1]
        window_low = lows[i - left : i + right + 1]
        if highs[i] == window_high.max() and (window_high[:left] < highs[i]).all():
            expected.append((i, "high"))
        if lows[i] == window_low.min() and (window_low[:left] > lows[i]).all():
            expected.append((i, "low"))

    result = swing_points(frame["high"], frame["low"], left, right)
    actual = sorted((int(r["index"]), r["kind"]) for _, r in result.iterrows())
    assert actual == sorted(expected)


def test_compute_indicators_leaves_no_nan_on_the_final_row(enriched):
    assert enriched.iloc[-1].isna().sum() == 0


def test_wicks_are_never_negative(config):
    """Malformed OHLC must not produce a negative wick."""
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=30, freq="5min", tz="UTC"),
            "open": 2000.0, "high": 1999.0, "low": 1998.0, "close": 2000.0,
            "tick_volume": 100.0,
        }
    )
    result = compute_indicators(frame, config.indicators)
    assert (result["upper_wick"] >= 0).all()
    assert (result["lower_wick"] >= 0).all()
