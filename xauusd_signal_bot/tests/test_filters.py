"""Adaptive thresholds, conflict/fakeout/session/spread/cooldown filters."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from src.filters import (
    FilterInput,
    GateState,
    adaptive_threshold,
    check_cooldown,
    check_fakeout,
    check_limits,
    check_risk_reward,
    check_separation,
    check_session,
    check_spread,
    check_threshold,
    check_volatility,
    run_pre_target_filters,
)
from src.indicators import compute_indicators
from src.scoring import compute_scorecard
from src.targets import Targets
from src.utils import ComponentScore, detect_session, session_allowed
from tests.conftest import make_candles

CANDLE_TIME = datetime(2024, 5, 1, 13, 0, tzinfo=timezone.utc)


def _card(config, bull: float, bear: float):
    """A scorecard with the requested totals, spread evenly over components."""
    maxima = {
        "trend": 20.0, "htf": 15.0, "momentum": 15.0, "structure": 15.0,
        "liquidity": 10.0, "support_resistance": 10.0, "volume": 5.0,
        "volatility": 5.0, "price_action": 5.0,
    }
    components = {
        name: ComponentScore(
            name,
            maximum * bull / 100.0,
            maximum * bear / 100.0,
            maximum,
            {"state": "NEUTRAL", "strong_body": True, "upper_wick_ratio": 0.1,
             "lower_wick_ratio": 0.1, "last_swing_high": None, "last_swing_low": None},
        )
        for name, maximum in maxima.items()
    }
    return compute_scorecard(components, config)


def _input(config, frame, bull=85.0, bear=40.0, direction="BUY", **overrides):
    data = dict(
        df=frame,
        card=_card(config, bull, bear),
        direction=direction,
        regime="WEAK_TREND",
        volatility_band="NORMAL",
        session="LONDON",
        spread_points=20.0,
        candle_time=CANDLE_TIME,
        htf_alignment="ALIGNED",
        timeframe_minutes=5,
        gate=GateState(),
        targets=None,
    )
    data.update(overrides)
    return FilterInput(**data)


@pytest.fixture
def frame(config):
    return compute_indicators(make_candles(1000, seed=41), config.indicators)


# --------------------------------------------------------------------------- #
# adaptive threshold  (spec 20/21)
# --------------------------------------------------------------------------- #
def test_threshold_varies_by_regime(config):
    strong = adaptive_threshold("STRONG_BULL_TREND", "NORMAL", "ALIGNED", config)
    normal = adaptive_threshold("WEAK_TREND", "NORMAL", "ALIGNED", config)
    ranging = adaptive_threshold("RANGE", "NORMAL", "ALIGNED", config)
    assert strong < normal < ranging
    assert normal == pytest.approx(config.base_threshold)


def test_high_volatility_demands_more_than_a_normal_market(config):
    assert adaptive_threshold("HIGH_VOLATILITY", "NORMAL", "ALIGNED", config) > config.base_threshold


def test_extreme_volatility_blocks_signals_entirely(config):
    assert adaptive_threshold("WEAK_TREND", "EXTREME", "ALIGNED", config) is None


def test_extreme_volatility_can_be_configured_to_raise_the_bar_instead(config):
    config.block_on_extreme_volatility = False
    threshold = adaptive_threshold("WEAK_TREND", "EXTREME", "ALIGNED", config)
    assert threshold is not None
    assert threshold >= config.base_threshold + config.extreme_volatility_extra_score - 1e-9


def test_counter_trend_setups_need_a_higher_score(config):
    aligned = adaptive_threshold("WEAK_TREND", "NORMAL", "ALIGNED", config)
    counter = adaptive_threshold("WEAK_TREND", "NORMAL", "COUNTER", config)
    assert counter == pytest.approx(aligned + config.counter_trend_extra_score)


def test_threshold_is_clamped_to_the_configured_ceiling(config):
    """No combination of offsets may push the bar outside the safe band."""
    config.regime_threshold_offsets["RANGE"] = 40.0
    config.counter_trend_extra_score = 20.0
    assert adaptive_threshold("RANGE", "NORMAL", "COUNTER", config) == config.max_threshold

    config.regime_threshold_offsets["RANGE"] = -90.0
    config.counter_trend_extra_score = 0.0
    assert adaptive_threshold("RANGE", "NORMAL", "ALIGNED", config) == config.min_threshold


def test_check_threshold_rejects_scores_below_the_bar(config, frame):
    low = _input(config, frame, bull=config.base_threshold - 5, bear=10.0)
    assert "below threshold" in (check_threshold(low, config) or "")
    high = _input(config, frame, bull=config.base_threshold + 5, bear=10.0)
    assert check_threshold(high, config) is None


# --------------------------------------------------------------------------- #
# bull/bear separation  (spec 22)
# --------------------------------------------------------------------------- #
def test_conflicted_market_is_rejected(config, frame):
    """The spec's example: bull 84 / bear 79 must not produce a BUY."""
    conflicted = _input(config, frame, bull=84.0, bear=79.0)
    assert "separation" in (check_separation(conflicted, config) or "")


def test_clear_separation_passes(config, frame):
    clear = _input(config, frame, bull=84.0, bear=60.0)
    assert check_separation(clear, config) is None


def test_separation_boundary_is_inclusive(config, frame):
    exact = _input(config, frame, bull=80.0, bear=80.0 - config.min_score_separation)
    assert check_separation(exact, config) is None
    just_under = _input(config, frame, bull=80.0, bear=80.0 - config.min_score_separation + 0.5)
    assert check_separation(just_under, config) is not None


def test_separation_applies_symmetrically_to_sell(config, frame):
    conflicted = _input(config, frame, bull=79.0, bear=84.0, direction="SELL")
    assert check_separation(conflicted, config) is not None
    clear = _input(config, frame, bull=60.0, bear=84.0, direction="SELL")
    assert check_separation(clear, config) is None


def test_separation_is_configurable(config, frame):
    config.min_score_separation = 2.0
    assert check_separation(_input(config, frame, bull=84.0, bear=79.0), config) is None


# --------------------------------------------------------------------------- #
# spread  (spec 25)
# --------------------------------------------------------------------------- #
def test_excessive_absolute_spread_is_rejected(config, frame):
    data = _input(config, frame, spread_points=config.max_spread_points + 1)
    assert "excessive spread" in (check_spread(data, config) or "")


def test_spread_relative_to_the_expected_move_is_checked(config, frame):
    config.max_spread_to_expected_move = 0.0001
    assert "expected move" in (check_spread(_input(config, frame), config) or "")


def test_unknown_spread_falls_back_to_the_assumed_cost(config, frame):
    """A backtest must still be charged a spread rather than trading for free."""
    config.assumed_spread_points = 5.0
    assert check_spread(_input(config, frame, spread_points=float("nan")), config) is None
    config.assumed_spread_points = config.max_spread_points + 10
    assert "excessive spread" in (
        check_spread(_input(config, frame, spread_points=float("nan")), config) or ""
    )


# --------------------------------------------------------------------------- #
# volatility / session
# --------------------------------------------------------------------------- #
def test_extreme_volatility_filter(config, frame):
    assert check_volatility(_input(config, frame, volatility_band="EXTREME"), config) is not None
    assert check_volatility(_input(config, frame, volatility_band="HIGH"), config) is None


def test_session_filter_allows_everything_by_default(config, frame):
    assert check_session(_input(config, frame, session="ASIAN"), config) is None


def test_session_filter_blocks_disallowed_sessions(config, frame):
    config.allowed_sessions = ("LONDON", "NEW_YORK")
    assert check_session(_input(config, frame, session="ASIAN"), config) is not None
    assert check_session(_input(config, frame, session="LONDON"), config) is None


def test_session_detection_uses_utc_windows(config):
    labels = {
        3: "ASIAN",
        9: "LONDON",
        14: "LONDON_NEW_YORK_OVERLAP",
        19: "NEW_YORK",
    }
    for hour, expected in labels.items():
        moment = datetime(2024, 5, 1, hour, 0, tzinfo=timezone.utc)
        assert detect_session(moment, config.sessions, config.session_priority) == expected


def test_session_allowed_helper():
    assert session_allowed("ASIAN", ("ALL_SESSIONS",))
    assert not session_allowed("ASIAN", ("LONDON",))
    assert session_allowed("ASIAN", ())


# --------------------------------------------------------------------------- #
# cooldown / duplicate prevention  (spec 26)
# --------------------------------------------------------------------------- #
def test_cooldown_blocks_a_signal_too_soon_after_the_last_one(config, frame):
    gate = GateState(last_signal_time=CANDLE_TIME - timedelta(minutes=5 * 3), last_signal_direction="SELL")
    assert "cooldown" in (check_cooldown(_input(config, frame, gate=gate), config) or "")


def test_cooldown_expires_after_the_configured_number_of_candles(config, frame):
    elapsed = timedelta(minutes=5 * (config.cooldown_candles + 1))
    gate = GateState(last_signal_time=CANDLE_TIME - elapsed, last_signal_direction="SELL")
    assert check_cooldown(_input(config, frame, gate=gate), config) is None


def test_same_direction_cooldown_is_longer(config, frame):
    """A repeated BUY needs a longer wait than an opposite-direction signal."""
    elapsed = timedelta(minutes=5 * (config.cooldown_candles + 1))
    gate = GateState(
        last_signal_time=CANDLE_TIME - elapsed,
        last_signal_direction="BUY",
        last_same_direction_time=CANDLE_TIME - elapsed,
    )
    reason = check_cooldown(_input(config, frame, direction="BUY", gate=gate), config)
    assert "same-direction" in (reason or "")


def test_same_direction_cooldown_expires(config, frame):
    elapsed = timedelta(minutes=5 * (config.same_direction_cooldown_candles + 1))
    gate = GateState(
        last_signal_time=CANDLE_TIME - elapsed,
        last_signal_direction="BUY",
        last_same_direction_time=CANDLE_TIME - elapsed,
    )
    assert check_cooldown(_input(config, frame, direction="BUY", gate=gate), config) is None


def test_no_previous_signal_means_no_cooldown(config, frame):
    assert check_cooldown(_input(config, frame, gate=GateState()), config) is None


def test_daily_and_concurrency_limits(config, frame):
    over_daily = GateState(signals_today=config.max_signals_per_day)
    assert "daily signal limit" in (check_limits(_input(config, frame, gate=over_daily), config) or "")
    over_active = GateState(active_count=config.max_concurrent_active_signals)
    assert "active signals" in (check_limits(_input(config, frame, gate=over_active), config) or "")
    assert check_limits(_input(config, frame, gate=GateState()), config) is None


# --------------------------------------------------------------------------- #
# fakeout  (spec 23)
# --------------------------------------------------------------------------- #
def _fakeout_input(config, frame, direction="BUY", **detail_overrides):
    data = _input(config, frame, direction=direction)
    sr = dict(state="NEUTRAL")
    pa = dict(strong_body=True, upper_wick_ratio=0.1, lower_wick_ratio=0.1)
    structure = dict(last_swing_high=None, last_swing_low=None)
    htf = (12.0, 2.0)
    sr.update(detail_overrides.pop("sr", {}))
    pa.update(detail_overrides.pop("pa", {}))
    structure.update(detail_overrides.pop("structure", {}))
    htf = detail_overrides.pop("htf", htf)
    data.card.components["support_resistance"] = ComponentScore("support_resistance", 8, 1, 10, sr)
    data.card.components["price_action"] = ComponentScore("price_action", 4, 0, 5, pa)
    data.card.components["structure"] = ComponentScore("structure", 12, 1, 15, structure)
    data.card.components["htf"] = ComponentScore("htf", htf[0], htf[1], 15, {})
    return data


def test_fakeout_filter_can_be_disabled(config, frame):
    config.enable_fakeout_filter = False
    assert check_fakeout(_fakeout_input(config, frame), config) is None


def test_breakout_without_volume_or_body_is_rejected(config, frame):
    frame = frame.copy()
    frame.loc[frame.index[-1], "rel_volume"] = 0.8
    data = _fakeout_input(config, frame, sr={"state": "BREAKOUT"}, pa={"strong_body": False})
    assert "breakout without" in (check_fakeout(data, config) or "")


def test_breakout_with_volume_passes(config, frame):
    frame = frame.copy()
    frame.loc[frame.index[-1], "rel_volume"] = 1.5
    data = _fakeout_input(config, frame, sr={"state": "BREAKOUT"}, pa={"strong_body": False})
    assert check_fakeout(data, config) is None


def test_close_back_inside_the_broken_range_is_rejected(config, frame):
    """Previous candle closed above the swing high, this one closed back below."""
    frame = frame.copy()
    frame.loc[frame.index[-2], "close"] = 2310.0   # broke out
    frame.loc[frame.index[-1], "close"] = 2299.0   # closed back inside
    data = _fakeout_input(config, frame, structure={"last_swing_high": 2305.0})
    assert "failed breakout" in (check_fakeout(data, config) or "")

    frame.loc[frame.index[-2], "close"] = 2290.0   # broke down
    frame.loc[frame.index[-1], "close"] = 2301.0   # closed back inside
    data = _fakeout_input(
        config, frame, direction="SELL", htf=(2.0, 12.0), structure={"last_swing_low": 2295.0}
    )
    assert "failed breakdown" in (check_fakeout(data, config) or "")


def test_higher_timeframe_disagreement_is_rejected(config, frame):
    data = _fakeout_input(config, frame, htf=(2.0, 12.0))
    assert "higher timeframe" in (check_fakeout(data, config) or "")


def test_thin_participation_is_rejected(config, frame):
    frame = frame.copy()
    frame.loc[frame.index[-1], "rel_volume"] = 0.3
    assert "poor liquidity" in (check_fakeout(_fakeout_input(config, frame), config) or "")


def test_candle_rejected_against_the_signal_direction(config, frame):
    data = _fakeout_input(config, frame, pa={"upper_wick_ratio": 0.8})
    assert "rejected from above" in (check_fakeout(data, config) or "")
    data = _fakeout_input(
        config, frame, direction="SELL", htf=(2.0, 12.0), pa={"lower_wick_ratio": 0.8}
    )
    assert "rejected from below" in (check_fakeout(data, config) or "")


# --------------------------------------------------------------------------- #
# risk / reward  (spec 31)
# --------------------------------------------------------------------------- #
def _targets(rr2: float, net_rr2: Optional[float] = None) -> Targets:
    return Targets(
        entry=2300.0, stop_loss=2299.6, tp1=2300.3, tp2=2300.0 + 0.4 * rr2, tp3=2301.0,
        risk=0.4, rr1=0.75, rr2=rr2, rr3=2.5, sl_mode="HYBRID", obstructed={},
        cost_r=0.4, net_rr1=0.35,
        net_rr2=rr2 - 0.4 if net_rr2 is None else net_rr2,
        net_rr3=2.1,
    )


def test_insufficient_reward_is_rejected(config, frame):
    data = _input(config, frame, targets=_targets(config.min_tp2_rr - 0.2))
    assert "insufficient R:R" in (check_risk_reward(data, config) or "")


def test_sufficient_reward_passes(config, frame):
    data = _input(config, frame, targets=_targets(config.min_tp2_rr + 0.6))
    assert check_risk_reward(data, config) is None


def test_reward_that_survives_raw_but_not_costs_is_rejected(config, frame):
    """The whole reason the cost model exists."""
    data = _input(config, frame, targets=_targets(config.min_tp2_rr + 0.6, net_rr2=0.05))
    reason = check_risk_reward(data, config)
    assert reason and "NET R:R" in reason


def test_missing_targets_are_rejected(config, frame):
    assert check_risk_reward(_input(config, frame, targets=None), config) == "no valid targets"


# --------------------------------------------------------------------------- #
# chain
# --------------------------------------------------------------------------- #
def test_pre_target_chain_reports_the_first_failure(config, frame):
    config.allowed_sessions = ("LONDON",)
    outcome = run_pre_target_filters(_input(config, frame, session="ASIAN"), config)
    assert not outcome.passed
    assert "session" in outcome.reason


def test_pre_target_chain_returns_the_threshold_it_used(config, frame):
    outcome = run_pre_target_filters(_input(config, frame, regime="RANGE"), config)
    expected = config.base_threshold + config.regime_threshold_offsets["RANGE"]
    assert outcome.threshold == pytest.approx(expected)
