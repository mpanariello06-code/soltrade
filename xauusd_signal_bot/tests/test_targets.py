"""Entry / stop-loss / take-profit construction and R:R arithmetic."""

from __future__ import annotations

import pytest

from src.indicators import compute_indicators
from src.targets import build_targets, compute_stop_loss, compute_take_profits
from src.utils import round_price
from tests.conftest import make_candles


@pytest.fixture
def frame(config):
    return compute_indicators(make_candles(1200, seed=51), config.indicators)


# --------------------------------------------------------------------------- #
# entry
# --------------------------------------------------------------------------- #
def test_entry_defaults_to_the_signal_candle_close(config, frame):
    targets, _ = build_targets(frame, config, "BUY")
    assert targets is not None
    assert targets.entry == round_price(float(frame["close"].iloc[-1]), config.digits)


def test_entry_can_be_overridden(config, frame):
    targets, _ = build_targets(frame, config, "BUY", entry=2345.67)
    assert targets is not None and targets.entry == pytest.approx(2345.67)


# --------------------------------------------------------------------------- #
# stop loss
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direction", ["BUY", "SELL"])
@pytest.mark.parametrize("mode", ["ATR", "STRUCTURE", "HYBRID"])
def test_stop_is_on_the_correct_side_in_every_mode(config, frame, direction, mode):
    config.sl_mode = mode
    entry = float(frame["close"].iloc[-1])
    stop, risk, used = compute_stop_loss(frame, config, direction, entry)
    assert risk > 0
    assert used in ("ATR", "STRUCTURE", "HYBRID")
    if direction == "BUY":
        assert stop < entry
    else:
        assert stop > entry


@pytest.mark.parametrize("direction", ["BUY", "SELL"])
def test_stop_distance_is_clamped_to_the_configured_atr_band(config, frame, direction):
    atr_value = float(frame["atr"].iloc[-1])
    entry = float(frame["close"].iloc[-1])
    _stop, risk, _mode = compute_stop_loss(frame, config, direction, entry)
    assert risk >= config.sl_min_atr_multiplier * atr_value - 1e-6
    assert risk <= config.sl_max_atr_multiplier * atr_value + 1e-6


def test_hybrid_is_never_tighter_than_the_atr_stop(config, frame):
    """HYBRID takes the wider of ATR and structure, so it cannot sit inside noise."""
    entry = float(frame["close"].iloc[-1])
    config.sl_mode = "ATR"
    _s, atr_risk, _m = compute_stop_loss(frame, config, "BUY", entry)
    config.sl_mode = "HYBRID"
    _s, hybrid_risk, _m = compute_stop_loss(frame, config, "BUY", entry)
    assert hybrid_risk >= atr_risk - 1e-9


def test_wider_atr_multiplier_produces_a_wider_stop(config, frame):
    config.sl_mode = "ATR"
    entry = float(frame["close"].iloc[-1])
    config.sl_atr_multiplier = 1.0
    _s, narrow, _m = compute_stop_loss(frame, config, "BUY", entry)
    config.sl_atr_multiplier = 2.5
    _s, wide, _m = compute_stop_loss(frame, config, "BUY", entry)
    assert wide > narrow


# --------------------------------------------------------------------------- #
# take profits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direction", ["BUY", "SELL"])
def test_take_profit_ladder_is_strictly_ordered(config, frame, direction):
    targets, reason = build_targets(frame, config, direction)
    assert targets is not None, reason
    if direction == "BUY":
        assert targets.stop_loss < targets.entry < targets.tp1 < targets.tp2 < targets.tp3
    else:
        assert targets.stop_loss > targets.entry > targets.tp1 > targets.tp2 > targets.tp3


def test_unobstructed_targets_match_the_configured_r_multiples(config, frame):
    """With no opposing zone in the way the ladder is exactly 1.0/1.8/2.8R."""
    entry = float(frame["close"].iloc[-1])
    risk = 5.0
    # push every zone far away by asking for a level weight nothing can reach
    config.min_tp_block_zone_weight = 1e9
    prices, obstructed = compute_take_profits(frame, config, "BUY", entry, risk)
    assert not any(obstructed.values())
    for price, multiple in zip(prices, config.tp_r_multiples):
        assert price == pytest.approx(entry + multiple * risk, abs=0.01)


def test_targets_are_pulled_back_in_front_of_major_structure(config, frame):
    """A zone inside the projected range must shorten the target, not be ignored."""
    entry = float(frame["close"].iloc[-1])
    risk = 5.0
    config.min_tp_block_zone_weight = 0.0  # let every zone count as a wall
    prices, obstructed = compute_take_profits(frame, config, "BUY", entry, risk)
    unobstructed = [entry + m * risk for m in config.tp_r_multiples]
    for index, (price, raw) in enumerate(zip(prices, unobstructed)):
        if obstructed.get(f"tp{index + 1}"):
            assert price <= raw + 1e-6
    assert prices[0] < prices[1] < prices[2]


def test_targets_never_fall_below_the_configured_floor(config, frame):
    entry = float(frame["close"].iloc[-1])
    risk = 5.0
    config.min_tp_block_zone_weight = 0.0
    prices, _ = compute_take_profits(frame, config, "BUY", entry, risk)
    for price, floor in zip(prices, config.tp_min_r_after_adjustment):
        assert price >= entry + floor * risk - 0.5


# --------------------------------------------------------------------------- #
# risk / reward
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direction", ["BUY", "SELL"])
def test_risk_reward_matches_the_published_prices(config, frame, direction):
    """R multiples must be derived from the rounded prices actually sent out."""
    targets, _ = build_targets(frame, config, direction)
    assert targets is not None
    risk = abs(targets.entry - targets.stop_loss)
    sign = 1.0 if direction == "BUY" else -1.0
    assert targets.rr1 == pytest.approx(sign * (targets.tp1 - targets.entry) / risk, abs=0.01)
    assert targets.rr2 == pytest.approx(sign * (targets.tp2 - targets.entry) / risk, abs=0.01)
    assert targets.rr3 == pytest.approx(sign * (targets.tp3 - targets.entry) / risk, abs=0.01)


def test_risk_rewards_increase_across_the_ladder(config, frame):
    targets, _ = build_targets(frame, config, "BUY")
    assert targets is not None
    assert targets.rr1 < targets.rr2 < targets.rr3


def test_risk_equals_the_entry_to_stop_distance(config, frame):
    targets, _ = build_targets(frame, config, "BUY")
    assert targets is not None
    assert targets.risk == pytest.approx(abs(targets.entry - targets.stop_loss), abs=1e-6)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_invalid_direction_is_rejected(config, frame):
    targets, reason = build_targets(frame, config, "HOLD")
    assert targets is None and "invalid direction" in reason


def test_insufficient_data_is_rejected(config, frame):
    targets, reason = build_targets(frame.iloc[:3], config, "BUY")
    assert targets is None and "insufficient data" in reason


def test_prices_are_rounded_to_the_instrument_precision(config, frame):
    targets, _ = build_targets(frame, config, "BUY")
    assert targets is not None
    for price in (targets.entry, targets.stop_loss, targets.tp1, targets.tp2, targets.tp3):
        assert price == round(price, config.digits)
