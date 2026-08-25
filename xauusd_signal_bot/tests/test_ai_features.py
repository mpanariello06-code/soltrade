"""Feature causality, leakage protection, resampling and dataset splitting.

The single most valuable test file in the RL stack.  A leaking feature does not
crash - it produces an excellent backtest that never reproduces live, which is
far worse than a crash because it is believed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.features.feature_pipeline import (
    FeatureSpec, build_dataset, observation_matrix, split_chronologically,
)
from ai.features.m1_features import atr, build_m1_features, m1_feature_columns, rsi
from ai.features.mtf_features import attach_context, build_context_frame, context_feature_columns


def make_candles(n=1200, seed=7, volatility=0.16, start=2300.0, spread=20):
    """A valid synthetic M1 series."""
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, volatility, n))
    open_ = np.concatenate([[start], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, volatility * 0.6, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, volatility * 0.6, n))
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "tick_volume": rng.integers(20, 400, n).astype(float),
        "spread": spread,
    })


# --------------------------------------------------------------------------- #
# the future-append stability test  (spec section 7)
# --------------------------------------------------------------------------- #
def test_m1_features_do_not_change_when_future_data_arrives():
    """THE leakage test: appending later candles must not alter earlier rows.

    A feature that uses future information will shift when that future changes.
    One that is causal cannot.
    """
    candles = make_candles(1400)
    cut = 1000

    full = build_m1_features(candles)
    prefix = build_m1_features(candles.iloc[:cut])
    columns = m1_feature_columns(full)

    later = full[columns].iloc[cut - 300:cut].to_numpy(dtype=float)
    earlier = prefix[columns].tail(300).to_numpy(dtype=float)

    leaking = [
        columns[i] for i in range(len(columns))
        if not np.allclose(earlier[:, i], later[:, i],
                           rtol=1e-9, atol=1e-12, equal_nan=True)
    ]
    assert not leaking, f"these features changed when the future arrived: {leaking}"


def test_context_features_do_not_change_when_future_data_arrives():
    """The same test for M5/M15, where leakage is far easier to introduce."""
    candles = make_candles(2400)
    cut = 1800

    full = attach_context(build_m1_features(candles))
    prefix = attach_context(build_m1_features(candles.iloc[:cut]))
    columns = context_feature_columns(full, ("M5", "M15"))
    assert columns, "no context features were produced"

    later = full[columns].iloc[cut - 200:cut].to_numpy(dtype=float)
    earlier = prefix[columns].tail(200).to_numpy(dtype=float)
    leaking = [
        columns[i] for i in range(len(columns))
        if not np.allclose(earlier[:, i], later[:, i],
                           rtol=1e-9, atol=1e-12, equal_nan=True)
    ]
    assert not leaking, f"leaking context features: {leaking}"


def test_a_context_bucket_is_invisible_until_it_closes():
    """The classic MTF mistake, tested directly.

    At 08:01 the M5 bucket labelled 08:00 has not closed. Its data must not be
    visible: three of its five minutes have not happened yet.
    """
    candles = make_candles(600)
    context = build_context_frame(build_m1_features(candles), "M5")

    labels = pd.to_datetime(context["available_at"], utc=True)
    # available_at is the CLOSE, so it is always strictly after the bucket's
    # opening label - five minutes after, for M5.
    assert (labels.dt.minute % 5 == 0).all()

    attached = attach_context(build_m1_features(candles), ("M5",))
    at_0801 = attached[attached["time"] == pd.Timestamp("2024-01-01 08:01", tz="UTC")]
    assert len(at_0801) == 1
    # Rebuilding with nothing after 08:01 must give the identical value.
    truncated = attach_context(
        build_m1_features(candles[candles["time"] <= pd.Timestamp("2024-01-01 08:01", tz="UTC")]),
        ("M5",),
    )
    for column in context_feature_columns(attached, ("M5",)):
        a = float(at_0801.iloc[0][column])
        b = float(truncated.iloc[-1][column])
        assert np.isclose(a, b, equal_nan=True), column


def test_rolling_indicators_never_look_forward():
    """ATR and RSI at row t must equal the same computed on [0..t] alone."""
    candles = make_candles(400)
    full_atr = atr(candles, 14)
    full_rsi = rsi(candles["close"], 14)

    for index in (100, 250, 399):
        window = candles.iloc[: index + 1]
        assert np.isclose(
            float(full_atr.iloc[index]), float(atr(window, 14).iloc[-1]), equal_nan=True
        ), f"ATR leaked at {index}"
        assert np.isclose(
            float(full_rsi.iloc[index]), float(rsi(window["close"], 14).iloc[-1]),
            equal_nan=True,
        ), f"RSI leaked at {index}"


# --------------------------------------------------------------------------- #
# warm-up and normalisation
# --------------------------------------------------------------------------- #
def test_warmup_rows_are_dropped_not_filled():
    """A back-filled indicator is a value the agent could not have had."""
    frame, spec = build_dataset(make_candles(1000))
    assert frame.attrs.get("warmup_dropped", 0) > 0
    assert not frame[spec.columns].isna().any().any(), "NaNs survived into the dataset"
    # the surviving rows start later than the raw series
    assert frame["time"].iloc[0] > pd.Timestamp("2024-01-01", tz="UTC")


def test_the_atr_anchor_is_not_fed_to_the_network():
    """`f_atr` is in raw price units and would smuggle the price level back in."""
    frame, spec = build_dataset(make_candles(800))
    assert "f_atr" in frame.columns, "the environment still needs it for geometry"
    assert "f_atr" not in spec.columns, "the price-scale anchor reached the observation"


def test_features_are_price_scale_free():
    """Gold at 2,300 and Bitcoin at 60,000 must produce comparable features.

    A model keyed to absolute price learns the level, not the behaviour - and
    the same feature set has to serve both markets.
    """
    gold, gold_spec = build_dataset(make_candles(800, seed=3, volatility=0.16, start=2300.0))
    btc, _ = build_dataset(make_candles(800, seed=3, volatility=30.0, start=60_000.0),
                           spec=gold_spec)

    for column in ("f_range_atr", "f_body_atr", "f_rsi", "f_position_15"):
        a = float(gold[column].abs().mean())
        b = float(btc[column].abs().mean())
        assert np.isclose(a, b, rtol=0.35), (
            f"{column} differs by price level: gold {a:.3f} vs BTC {b:.3f}"
        )


def test_the_observation_matrix_is_finite_and_ordered():
    frame, spec = build_dataset(make_candles(600))
    matrix = observation_matrix(frame, spec)
    assert matrix.shape == (len(frame), spec.size)
    assert matrix.dtype == np.float32
    assert np.all(np.isfinite(matrix))
    assert np.all(np.abs(matrix) <= spec.clip + 1e-6)


# --------------------------------------------------------------------------- #
# the feature contract
# --------------------------------------------------------------------------- #
def test_the_spec_fingerprint_changes_with_the_columns():
    """A changed feature set must produce a different fingerprint.

    This is what stops a model being served columns it never trained on -
    a failure that is silent and ruins every result after it.
    """
    a = FeatureSpec(columns=["f_one", "f_two"])
    b = FeatureSpec(columns=["f_one", "f_two"])
    c = FeatureSpec(columns=["f_one", "f_three"])
    d = FeatureSpec(columns=["f_two", "f_one"])

    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert a.fingerprint() != d.fingerprint(), "column ORDER must matter"


def test_a_spec_round_trips_through_disk(tmp_path):
    _frame, spec = build_dataset(make_candles(500))
    path = tmp_path / "feature_spec.json"
    spec.save(path)
    restored = FeatureSpec.load(path)
    assert restored.columns == spec.columns
    assert restored.fingerprint() == spec.fingerprint()


def test_reusing_a_spec_rejects_incompatible_data():
    """Better to fail loudly than to serve a model the wrong columns."""
    _frame, spec = build_dataset(make_candles(400))
    spec.columns = spec.columns + ["f_does_not_exist"]
    with pytest.raises(ValueError, match="absent from the data"):
        build_dataset(make_candles(400), spec=spec)


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def test_chronological_splits_never_overlap_or_reorder():
    """No shuffling, ever: a shuffled split lets Thursday predict Monday."""
    frame, _spec = build_dataset(make_candles(3000))
    train, validation, test = split_chronologically(
        frame, ["2024-01-02", "2024-01-02 12:00"]
    )

    assert len(train) and len(validation) and len(test)
    assert train["time"].max() < validation["time"].min()
    assert validation["time"].max() < test["time"].min()
    assert len(train) + len(validation) + len(test) == len(frame)


def test_split_boundaries_must_increase():
    frame, _spec = build_dataset(make_candles(500))
    with pytest.raises(ValueError, match="strictly increasing"):
        split_chronologically(frame, ["2024-01-02", "2024-01-01"])
