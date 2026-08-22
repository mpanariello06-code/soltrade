"""M1 scalping: configuration invariants, the cost model, target geometry,
timeout behaviour and ambiguous-candle resolution."""

from __future__ import annotations

import pandas as pd
import pytest

from src.filters import adaptive_threshold, check_risk_reward, check_spread
from src.indicators import compute_indicators
from src.runtime_state import JsonStateStore, RuntimeState, effective_config
from src.signal_tracker import (
    STATUS_SL,
    STATUS_TIMEOUT,
    STATUS_TP3,
    PositionState,
    build_outcome_row,
    evaluate_progress,
)
from src.targets import build_targets, compute_stop_loss, compute_target_distances
from src.timeframes import MODE_SCALPING, SIGNAL_TIMEFRAME, expected_hold_label
from tests.conftest import isolate, make_bars, make_candles, make_ticks

SIGNAL = {
    "signal_id": "s1",
    "timestamp": "2024-05-01T12:00:00+00:00",
    "direction": "BUY",
    "timeframe": "M1",
    "entry": 2300.00,
    "sl": 2299.60,     # 4.0 pips
    "tp1": 2300.36,    # 3.6 pips
    "tp2": 2300.65,
    "tp3": 2301.08,
    "cost_r": 0.60,
    "spread_points": 20.0,
}


@pytest.fixture
def m1(config) -> pd.DataFrame:
    return compute_indicators(make_candles(2000, seed=31), config.indicators)


# --------------------------------------------------------------------------- #
# the build is M1 / SCALPING only
# --------------------------------------------------------------------------- #
def test_only_one_timeframe_and_one_mode(config):
    assert config.signal_timeframe == "M1"
    assert SIGNAL_TIMEFRAME == "M1"
    assert config.mode == MODE_SCALPING == "SCALPING"


def test_config_rejects_any_other_signal_timeframe(config):
    config.signal_timeframe = "M5"
    with pytest.raises(ValueError, match="M1-only"):
        config.validate()


def test_no_mode_or_timeframe_switching_remains(config, tmp_path):
    config.state_file = tmp_path / "state.json"
    runtime = RuntimeState.load(config)
    for removed in ("set_mode", "set_timeframe", "confirmation_timeframes", "micro_timeframe"):
        assert not hasattr(runtime, removed), f"{removed} should be gone"
    described = runtime.describe()
    assert described["mode"] == "SCALPING"
    assert described["signal_timeframe"] == "M1"


def test_weights_still_sum_to_one_hundred(config):
    assert config.weights.total() == pytest.approx(100.0)


def test_m1_weights_favour_microstructure(config):
    """Momentum and price action must outrank the slow-trend components."""
    weights = config.weights
    assert weights.momentum > weights.trend
    assert weights.price_action > weights.trend
    assert weights.liquidity > weights.htf
    assert weights.htf < 10.0, "context should be secondary at a 15-minute horizon"


def test_indicator_periods_are_short_enough_for_m1(config):
    params = config.indicators
    assert params.ema_fast <= 8
    assert params.ema_trend <= 120, "a 200-minute EMA says nothing about the next 3 minutes"
    assert params.swing_left == 1 and params.swing_right == 1


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #
def test_round_trip_cost_sums_every_component(config):
    config.assumed_spread_points = 20.0
    config.slippage_points_entry = 2.0
    config.slippage_points_exit = 3.0
    config.commission_points_per_side = 1.0
    expected = (20.0 + 2.0 + 3.0 + 2.0) * config.point_value
    assert config.round_trip_cost(20.0) == pytest.approx(expected)


def test_live_spread_is_used_when_known(config):
    cheap = config.round_trip_cost(6.0)
    dear = config.round_trip_cost(30.0)
    assert cheap < dear
    assert dear - cheap == pytest.approx(24.0 * config.point_value)


def test_unknown_spread_falls_back_to_the_assumption(config):
    """A backtest must still pay costs rather than trading for free."""
    assumed = config.round_trip_cost(config.assumed_spread_points)
    for missing in (None, float("nan"), "", -1.0):
        assert config.round_trip_cost(missing) == pytest.approx(assumed)


def test_pip_conversion(config):
    assert config.pip_value == pytest.approx(0.10)
    assert config.pips(0.30) == pytest.approx(3.0)
    assert config.pips(config.round_trip_cost(20.0)) == pytest.approx(2.4)


# --------------------------------------------------------------------------- #
# target geometry
# --------------------------------------------------------------------------- #
def test_targets_scale_with_volatility(config):
    """Targets come from market conditions, not fixed pip distances."""
    quiet = compute_target_distances(config, atr_value=0.20, cost_price=0.0)
    busy = compute_target_distances(config, atr_value=0.80, cost_price=0.0)
    assert all(b > q for b, q in zip(busy, quiet))
    assert busy[0] == pytest.approx(0.80 * config.tp_atr_multiples[0])


def test_targets_never_fall_below_the_pip_floor(config):
    distances = compute_target_distances(config, atr_value=0.001, cost_price=0.0)
    for distance, floor in zip(distances, config.min_tp_pips):
        assert distance >= floor * config.pip_value - 1e-9


def test_tp1_is_floored_by_the_cost_of_trading(config):
    cost = config.round_trip_cost(20.0)
    distances = compute_target_distances(config, atr_value=0.10, cost_price=cost)
    assert distances[0] >= config.min_tp1_cost_multiple * cost - 1e-9


def test_cost_floor_lifts_the_whole_ladder_not_just_tp1(config):
    """Otherwise a wide spread would silently turn 1:2 into 1:1."""
    free = compute_target_distances(config, atr_value=0.30, cost_price=0.0)
    costly = compute_target_distances(config, atr_value=0.30, cost_price=0.40)
    assert costly[0] > free[0]
    assert costly[1] > free[1], "TP2 must move up with TP1"
    assert costly[2] > free[2], "TP3 must move up with TP1"
    assert costly[0] < costly[1] < costly[2]


def test_target_distances_are_strictly_increasing(config):
    for atr in (0.05, 0.2, 0.5, 1.5):
        distances = compute_target_distances(config, atr, config.round_trip_cost(20.0))
        assert distances[0] < distances[1] < distances[2]


@pytest.mark.parametrize("direction", ["BUY", "SELL"])
def test_full_target_set_is_ordered_and_priced(config, m1, direction):
    targets, reason = build_targets(m1, config, direction, spread_points=8.0)
    assert targets is not None, reason
    if direction == "BUY":
        assert targets.stop_loss < targets.entry < targets.tp1 < targets.tp2 < targets.tp3
    else:
        assert targets.stop_loss > targets.entry > targets.tp1 > targets.tp2 > targets.tp3
    for price in (targets.entry, targets.stop_loss, targets.tp1, targets.tp2, targets.tp3):
        assert price == round(price, config.digits), "prices must be the rounded ones published"


@pytest.mark.parametrize("direction", ["BUY", "SELL"])
def test_risk_reward_uses_the_exact_rounded_prices(config, m1, direction):
    targets, _ = build_targets(m1, config, direction, spread_points=8.0)
    assert targets is not None
    risk = abs(targets.entry - targets.stop_loss)
    assert targets.risk == pytest.approx(risk, abs=1e-6)
    sign = 1.0 if direction == "BUY" else -1.0
    for rr, price in ((targets.rr1, targets.tp1), (targets.rr2, targets.tp2), (targets.rr3, targets.tp3)):
        assert rr == pytest.approx(sign * (price - targets.entry) / risk, abs=0.01)


def test_net_reward_is_raw_minus_cost(config, m1):
    targets, _ = build_targets(m1, config, "BUY", spread_points=8.0)
    assert targets is not None
    assert targets.cost_r == pytest.approx(targets.cost_price / targets.risk, abs=1e-3)
    for raw, net in ((targets.rr1, targets.net_rr1), (targets.rr2, targets.net_rr2),
                     (targets.rr3, targets.net_rr3)):
        assert net == pytest.approx(raw - targets.cost_r, abs=0.011)
        assert net < raw, "net reward must always be below raw"


def test_a_wider_spread_produces_worse_net_reward(config, m1):
    cheap, _ = build_targets(m1, config, "BUY", spread_points=6.0)
    dear, _ = build_targets(m1, config, "BUY", spread_points=30.0)
    assert cheap is not None and dear is not None
    assert dear.cost_r > cheap.cost_r
    assert dear.cost_pips > cheap.cost_pips


def test_setup_is_rejected_when_the_target_cannot_clear_costs(config, m1):
    """The decisive scalping filter."""
    config.max_tp3_pips = 4.0          # forbid the ladder from growing
    config.assumed_spread_points = 120.0
    targets, reason = build_targets(m1, config, "BUY", spread_points=120.0)
    assert targets is None
    assert "cost" in reason.lower() or "scalp range" in reason.lower()


def test_a_scalp_that_needs_a_huge_move_is_rejected(config, m1):
    config.max_tp3_pips = 1.0
    targets, reason = build_targets(m1, config, "BUY", spread_points=8.0)
    assert targets is None
    assert "scalp range" in reason


def test_stop_clears_ordinary_spread_noise(config, m1):
    """A stop inside a couple of spreads is taken out by quotes, not by price."""
    entry = float(m1["close"].iloc[-1])
    _stop, risk, _mode = compute_stop_loss(m1, config, "BUY", entry, spread_points=40.0)
    spread_price = 40.0 * config.point_value
    assert risk >= config.min_sl_cost_multiple * spread_price - 1e-9


def test_stop_stays_inside_the_atr_band_when_costs_are_small(config, m1):
    atr_value = float(m1["atr"].iloc[-1])
    entry = float(m1["close"].iloc[-1])
    _stop, risk, _mode = compute_stop_loss(m1, config, "BUY", entry, spread_points=1.0)
    assert risk <= config.sl_max_atr_multiplier * atr_value + 1e-9


def test_expected_hold_label_scales_with_the_target(config):
    assert expected_hold_label(3.0, 4.0) == "VERY SHORT"
    assert expected_hold_label(8.0, 4.0) == "SHORT"
    assert expected_hold_label(20.0, 4.0) == "MEDIUM"


# --------------------------------------------------------------------------- #
# spread and net-R filters
# --------------------------------------------------------------------------- #
def _filter_input(config, frame, spread, targets=None):
    from datetime import datetime, timezone

    from src.filters import FilterInput, GateState
    from src.scoring import compute_scorecard
    from src.utils import ComponentScore

    card = compute_scorecard({"trend": ComponentScore("trend", 10, 0, 10)}, config)
    return FilterInput(
        df=frame, card=card, direction="BUY", regime="WEAK_TREND",
        volatility_band="NORMAL", session="LONDON", spread_points=spread,
        candle_time=datetime(2024, 5, 1, 12, tzinfo=timezone.utc),
        htf_alignment="ALIGNED", timeframe_minutes=1, gate=GateState(), targets=targets,
    )


def test_absolute_spread_cap(config, m1):
    reason = check_spread(_filter_input(config, m1, config.max_spread_points + 1), config)
    assert reason and "excessive spread" in reason


def test_spread_is_judged_against_the_holding_window_move(config, m1):
    """Not against a single M1 candle - they are nearly the same size.

    The gate binds when the market is too quiet for the spread: shortening the
    holding window shrinks the move price can plausibly make, which is exactly
    the situation the filter exists to catch.
    """
    atr_value = float(m1["atr"].iloc[-1])
    spread_points = 20.0

    # Normal 15-minute window: sqrt(15) * ATR of room, so a 2-pip spread passes.
    assert check_spread(_filter_input(config, m1, spread_points), config) is None

    # One-minute window: the spread is now a large fraction of the likely move.
    config.max_holding_candles = 1
    reason = check_spread(_filter_input(config, m1, spread_points), config)
    assert reason and "expected move" in reason
    assert config.pips(spread_points * config.point_value) > 0.35 * config.pips(atr_value)


def test_spread_ratio_gate_is_configurable(config, m1):
    config.max_spread_to_expected_move = 0.01
    reason = check_spread(_filter_input(config, m1, 20.0), config)
    assert reason and "expected move" in reason

    config.max_spread_to_expected_move = 5.0
    assert check_spread(_filter_input(config, m1, 20.0), config) is None


def test_unknown_spread_still_charges_the_assumption(config, m1):
    config.assumed_spread_points = config.max_spread_points + 5
    reason = check_spread(_filter_input(config, m1, float("nan")), config)
    assert reason and "excessive spread" in reason


def test_net_reward_gate_rejects_a_raw_winner_that_loses_after_costs(config, m1):
    from src.targets import Targets

    targets = Targets(
        entry=2300.0, stop_loss=2299.6, tp1=2300.3, tp2=2300.6, tp3=2301.0,
        risk=0.4, rr1=0.75, rr2=1.5, rr3=2.5, sl_mode="HYBRID", obstructed={},
        cost_r=1.4, net_rr1=-0.65, net_rr2=0.1, net_rr3=1.1,
    )
    reason = check_risk_reward(_filter_input(config, m1, 20.0, targets), config)
    assert reason and "NET R:R" in reason

    targets.net_rr2 = 0.9
    assert check_risk_reward(_filter_input(config, m1, 20.0, targets), config) is None


# --------------------------------------------------------------------------- #
# timeout
# --------------------------------------------------------------------------- #
def test_flat_market_times_out_at_the_configured_limit(config):
    config.max_holding_candles = 8
    flat = make_bars([(2300, 2300.05, 2299.95, 2300.0)] * 20)
    progress = evaluate_progress(SIGNAL, flat, config)
    assert progress.status == STATUS_TIMEOUT
    assert progress.timed_out is True
    assert progress.bars == 8
    assert progress.closed is True


def test_timeout_marks_the_position_to_market(config):
    config.max_holding_candles = 3
    drifting = make_bars([(2300, 2300.15, 2299.95, 2300.10)] * 3)
    progress = evaluate_progress(SIGNAL, drifting, config)
    assert progress.status == STATUS_TIMEOUT
    # exit at the last close, so R reflects where it actually was
    assert progress.exit_price == pytest.approx(2300.10)
    assert progress.r_multiple == pytest.approx(0.25, abs=0.01)
    assert progress.net_r == pytest.approx(progress.r_multiple - progress.cost_r, abs=1e-6)


def test_no_signal_stays_active_beyond_the_holding_window(config):
    config.max_holding_candles = 5
    flat = make_bars([(2300, 2300.05, 2299.95, 2300.0)] * 60)
    progress = evaluate_progress(SIGNAL, flat, config)
    assert progress.closed is True, "a scalp must never sit active indefinitely"
    assert progress.bars <= 5


def test_timeout_is_configurable_from_the_runtime(config, tmp_path):
    config = isolate(config, tmp_path)
    runtime = RuntimeState.load(config)
    runtime.active.set_max_holding(30)
    assert effective_config(config, runtime).max_holding_candles == 30
    restored = RuntimeState.load(config, JsonStateStore(config.global_state_file))
    assert restored.describe()["max_holding_candles"] == 30
    assert restored.market("BTCUSD").max_holding_candles is None, (
        "changing gold's holding time must not touch Bitcoin's"
    )


# --------------------------------------------------------------------------- #
# detailed outcome tracking
# --------------------------------------------------------------------------- #
def test_milestone_timing_is_recorded(config):
    bars = make_bars(
        [
            (2300, 2300.20, 2299.95, 2300.10),
            (2300.10, 2300.40, 2300.00, 2300.35),   # TP1 on bar 2
            (2300.35, 2300.70, 2300.30, 2300.66),   # TP2 on bar 3
        ]
    )
    progress = evaluate_progress(SIGNAL, bars, config)
    assert progress.bars_to_tp1 == 2
    assert progress.bars_to_tp2 == 3
    assert progress.bars_to_tp3 is None
    assert progress.minutes_to(progress.bars_to_tp1) == "2"
    assert progress.minutes_to(None) == ""


def test_excursions_are_recorded_in_r_and_price(config):
    bars = make_bars([(2300, 2300.30, 2299.70, 2300.0)])
    progress = evaluate_progress(SIGNAL, bars, config)
    assert progress.mfe_r == pytest.approx(0.75, abs=0.01)
    assert progress.mae_r == pytest.approx(-0.75, abs=0.01)
    assert progress.mfe_price == pytest.approx(2300.30)
    assert progress.mae_price == pytest.approx(2299.70)


def test_outcome_row_carries_the_full_scalp_record(config):
    bars = make_bars([(2300, 2301.20, 2299.95, 2301.10)])
    progress = evaluate_progress(SIGNAL, bars, config)
    row = build_outcome_row(SIGNAL, progress, config.digits, config)
    assert row["result"] == STATUS_TP3
    assert row["raw_r"] == progress.r_multiple
    assert row["net_r"] == pytest.approx(progress.r_multiple - SIGNAL["cost_r"], abs=1e-6)
    assert row["cost_r"] == pytest.approx(SIGNAL["cost_r"])
    assert row["timeout"] == 0
    assert row["minutes_to_tp1"] == "1"
    assert float(row["mfe_pips"]) == pytest.approx(12.0, abs=0.01)
    assert float(row["mae_pips"]) == pytest.approx(-0.5, abs=0.01)
    for column in ("bars_to_tp1", "bars_to_sl", "spread_points", "ambiguous_bars"):
        assert column in row


def test_net_r_is_always_below_raw_r(config):
    for bars in (
        [(2300, 2301.20, 2299.95, 2301.10)],
        [(2300, 2300.10, 2299.50, 2299.60)],
        [(2300, 2300.05, 2299.95, 2300.00)] * config.max_holding_candles,
    ):
        progress = evaluate_progress(SIGNAL, make_bars(bars), config)
        assert progress.net_r < progress.r_multiple


def test_cost_r_is_derived_when_the_signal_did_not_store_it(config):
    row = dict(SIGNAL)
    row.pop("cost_r")
    state = PositionState(row, config)
    expected = config.round_trip_cost(row["spread_points"]) / abs(row["entry"] - row["sl"])
    assert state.cost_r == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# ambiguous candles
# --------------------------------------------------------------------------- #
def test_ambiguous_candle_is_pessimistic_without_ticks(config):
    """One candle touching both TP and SL is scored as the stop."""
    both = make_bars([(2300, 2300.40, 2299.55, 2300.0)])
    progress = evaluate_progress(SIGNAL, both, config)
    assert progress.status == STATUS_SL
    assert progress.tp_hits == 0
    assert progress.r_multiple == pytest.approx(-1.0)
    assert progress.ambiguous_bars == 1


def test_ticks_resolve_a_target_first_candle(config):
    """Price ran to TP1 before coming back to the stop."""
    both = make_bars([(2300, 2300.40, 2299.55, 2300.0)])
    ticks = make_ticks([2300.05, 2300.40, 2300.20, 2299.55])
    progress = evaluate_progress(SIGNAL, both, config, intrabar=ticks)
    assert progress.tp_hits == 1, "the target was reached first and must be credited"
    assert progress.r_multiple > -1.0


def test_ticks_confirm_a_stop_first_candle(config):
    both = make_bars([(2300, 2300.40, 2299.55, 2300.0)])
    ticks = make_ticks([2300.05, 2299.55, 2300.20, 2300.40])
    progress = evaluate_progress(SIGNAL, both, config, intrabar=ticks)
    assert progress.status == STATUS_SL
    assert progress.tp_hits == 0


def test_ticks_outside_the_candle_window_are_ignored(config):
    """A tick buffer covering other minutes must not decide this candle."""
    both = make_bars([(2300, 2300.40, 2299.55, 2300.0)])
    ticks = make_ticks([2300.40] * 4, start="2024-05-01 12:30")
    progress = evaluate_progress(SIGNAL, both, config, intrabar=ticks)
    assert progress.status == STATUS_SL, "must stay pessimistic when ticks do not cover the bar"


def test_ambiguity_still_pessimistic_when_ticks_are_also_ambiguous(config):
    both = make_bars([(2300, 2300.40, 2299.55, 2300.0)])
    # a single tick that is simultaneously at both levels cannot order them
    ticks = pd.DataFrame(
        {"time": pd.date_range("2024-05-01 12:01", periods=1, freq="5s", tz="UTC"),
         "high": [2300.40], "low": [2299.55]}
    )
    progress = evaluate_progress(SIGNAL, both, config, intrabar=ticks)
    assert progress.status == STATUS_SL


# --------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------- #
def test_threshold_is_a_single_configurable_value(config):
    assert config.base_threshold == 68.0
    assert not hasattr(config, "mode_timeframe_thresholds")


def test_regime_still_adjusts_the_threshold(config):
    strong = adaptive_threshold("STRONG_BULL_TREND", "NORMAL", "ALIGNED", config)
    normal = adaptive_threshold("WEAK_TREND", "NORMAL", "ALIGNED", config)
    ranging = adaptive_threshold("RANGE", "NORMAL", "ALIGNED", config)
    assert strong < normal < ranging
    assert normal == pytest.approx(config.base_threshold)
