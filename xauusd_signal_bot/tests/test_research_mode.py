"""Operating modes, per-timeframe thresholds, near-signal diagnostics and
research-signal data collection."""

from __future__ import annotations

import pandas as pd
import pytest

from performance import SCORE_BANDS, build_report
from src.filters import adaptive_threshold, is_near_signal
from src.indicators import compute_indicators
from src.market_data import resample_candles
from src.runtime_state import JsonStateStore, RuntimeState, config_view, effective_config
from src.signal_engine import DECISION_NEAR, DECISION_NONE, EVALUATION_COLUMNS, SignalEngine
from src.signal_tracker import SIGNAL_COLUMNS, SignalTracker, read_csv_rows
from src.telegram_bot import TelegramNotifier
from src.timeframes import (
    MODES,
    SUPPORTED_TIMEFRAMES,
    confirmation_label,
    confirmation_timeframes,
    micro_timeframe,
)
from tests.conftest import FakeNotifier, make_candles


@pytest.fixture
def runtime(isolated_config):
    return RuntimeState.load(isolated_config)


def build_snapshot(config, candles):
    from src.market_data import MarketSnapshot

    confirm, higher = confirmation_timeframes(config.signal_timeframe)
    return MarketSnapshot(
        symbol=config.symbol,
        m5=compute_indicators(candles, config.indicators),
        m15=compute_indicators(resample_candles(candles, "M5", confirm), config.indicators)
        if confirm else None,
        h1=compute_indicators(resample_candles(candles, "M5", higher), config.indicators)
        if higher else None,
        spread_points=float("nan"),
        signal_timeframe=config.signal_timeframe,
        confirmation=confirmation_label(config.signal_timeframe),
    )


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def test_default_mode_is_standard(runtime):
    assert runtime.describe()["mode"] == "STANDARD"


def test_mode_thresholds_follow_the_specified_order(config):
    """RESEARCH must be the loosest and CONSERVATIVE the strictest."""
    research = config.threshold_for("RESEARCH", "M5")
    standard = config.threshold_for("STANDARD", "M5")
    conservative = config.threshold_for("CONSERVATIVE", "M5")
    assert research == 50.0
    assert standard == 72.0
    assert conservative == 80.0
    assert research < standard < conservative


@pytest.mark.parametrize(
    "timeframe,expected",
    [("M1", 80.0), ("M5", 72.0), ("M15", 70.0), ("M30", 68.0), ("H1", 65.0), ("H4", 65.0)],
)
def test_standard_thresholds_are_timeframe_specific(config, timeframe, expected):
    assert config.threshold_for("STANDARD", timeframe) == expected


@pytest.mark.parametrize(
    "timeframe,expected",
    [("M1", 55.0), ("M5", 50.0), ("M15", 50.0), ("M30", 50.0), ("H1", 50.0), ("H4", 50.0)],
)
def test_research_thresholds_are_timeframe_specific(config, timeframe, expected):
    assert config.threshold_for("RESEARCH", timeframe) == expected


def test_every_mode_covers_every_supported_timeframe(config):
    for mode in MODES:
        for timeframe in SUPPORTED_TIMEFRAMES:
            assert timeframe in config.mode_timeframe_thresholds[mode], (mode, timeframe)


def test_switching_mode_changes_the_active_threshold(runtime):
    runtime.set_mode("RESEARCH")
    assert runtime.active_threshold() == 50.0
    runtime.set_mode("CONSERVATIVE")
    assert runtime.active_threshold() == 80.0
    runtime.set_mode("STANDARD")
    assert runtime.active_threshold() == 72.0


def test_unknown_mode_is_rejected_and_leaves_state_unchanged(runtime):
    runtime.set_mode("RESEARCH")
    runtime.set_mode("SUPER_AGGRESSIVE")
    assert runtime.describe()["mode"] == "RESEARCH"


def test_mode_threshold_flows_through_to_the_filter(config, runtime):
    for mode, expected in (("RESEARCH", 50.0), ("STANDARD", 72.0), ("CONSERVATIVE", 80.0)):
        runtime.set_mode(mode)
        view = effective_config(config, runtime)
        assert adaptive_threshold("WEAK_TREND", "NORMAL", "ALIGNED", view) == expected


# --------------------------------------------------------------------------- #
# timeframe hierarchy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "timeframe,expected",
    [
        ("M1", ("M5", "M15")),
        ("M5", ("M15", "H1")),
        ("M15", ("M30", "H1")),
        ("M30", ("H1", "H4")),
        ("H1", ("H4", None)),
        ("H4", (None, None)),
    ],
)
def test_confirmation_hierarchy_matches_the_specification(timeframe, expected):
    assert confirmation_timeframes(timeframe) == expected


def test_switching_timeframe_moves_the_whole_hierarchy(config, runtime):
    runtime.set_timeframe("M15")
    view = effective_config(config, runtime)
    assert view.signal_timeframe == "M15"
    assert view.intermediate_timeframe == "M30"
    assert view.higher_timeframe == "H1"

    runtime.set_timeframe("M30")
    view = effective_config(config, runtime)
    assert (view.intermediate_timeframe, view.higher_timeframe) == ("H1", "H4")


def test_h4_has_no_confirmation_timeframes(config, runtime):
    runtime.set_timeframe("H4")
    view = effective_config(config, runtime)
    assert view.intermediate_timeframe == ""
    assert view.higher_timeframe == ""
    assert runtime.confirmation_label() == "NONE"


def test_timeframe_switch_also_changes_the_threshold(runtime):
    runtime.set_mode("STANDARD")
    runtime.set_timeframe("M1")
    assert runtime.active_threshold() == 80.0
    runtime.set_timeframe("H1")
    assert runtime.active_threshold() == 65.0


def test_unsupported_timeframe_is_rejected(runtime):
    runtime.set_timeframe("M15")
    runtime.set_timeframe("M7")
    assert runtime.describe()["signal_timeframe"] == "M15"


def test_micro_timeframe_is_never_the_signal_timeframe():
    for timeframe in SUPPORTED_TIMEFRAMES:
        assert micro_timeframe(timeframe) != timeframe


# --------------------------------------------------------------------------- #
# threshold control
# --------------------------------------------------------------------------- #
def test_threshold_steps_apply_and_persist(isolated_config, runtime):
    runtime.set_mode("RESEARCH")
    assert runtime.adjust_threshold(+5) == 55.0
    assert runtime.adjust_threshold(-1) == 54.0

    restored = RuntimeState.load(isolated_config, JsonStateStore(isolated_config.state_file))
    assert restored.active_threshold() == 54.0


def test_threshold_is_clamped_to_the_configured_limits(config, runtime):
    runtime.set_threshold(1000)
    assert runtime.active_threshold() == config.max_threshold
    runtime.set_threshold(-50)
    assert runtime.active_threshold() == config.min_threshold


def test_threshold_reset_restores_the_configured_default(runtime):
    runtime.set_mode("STANDARD")
    runtime.set_threshold(90)
    assert runtime.has_threshold_override()
    assert runtime.reset_threshold() == 72.0
    assert not runtime.has_threshold_override()


def test_threshold_overrides_are_per_mode_and_timeframe(runtime):
    runtime.set_mode("RESEARCH")
    runtime.set_timeframe("M5")
    runtime.set_threshold(58)

    runtime.set_mode("STANDARD")
    assert runtime.active_threshold() == 72.0, "an override must not leak across modes"

    runtime.set_mode("RESEARCH")
    assert runtime.active_threshold() == 58.0

    runtime.set_timeframe("M15")
    assert runtime.active_threshold() == 50.0, "an override must not leak across timeframes"


# --------------------------------------------------------------------------- #
# near-signal diagnostic
# --------------------------------------------------------------------------- #
def test_near_signal_window_is_below_the_threshold_only(config):
    assert is_near_signal(66.0, 72.0, config) is True
    assert is_near_signal(62.0, 72.0, config) is True     # exactly on the margin
    assert is_near_signal(61.9, 72.0, config) is False    # just outside
    assert is_near_signal(72.0, 72.0, config) is False    # would have qualified
    assert is_near_signal(80.0, 72.0, config) is False


def test_near_signal_margin_is_configurable(config):
    config.near_signal_margin = 3.0
    assert is_near_signal(70.0, 72.0, config) is True
    assert is_near_signal(66.0, 72.0, config) is False


def test_near_signal_needs_a_threshold(config):
    assert is_near_signal(70.0, None, config) is False


def test_near_signal_alerts_default_to_off_and_toggle(runtime):
    assert runtime.describe()["near_signal_alerts"] is False
    assert runtime.set_near_signal_alerts(True) is True
    assert runtime.describe()["near_signal_alerts"] is True


def test_engine_marks_near_signal_decisions(isolated_config):
    """Force a threshold just above the achievable score and expect NEAR_SIGNAL."""
    candles = make_candles(4200, seed=91, drift=0.05)
    view = config_view(isolated_config, "RESEARCH", "M5")
    snapshot = build_snapshot(view, candles)

    baseline = SignalEngine(view).evaluate(snapshot, config=view)
    assert baseline.card is not None

    view.base_threshold = _threshold_for_gap(view, snapshot, 5.0)
    evaluation = SignalEngine(view).evaluate(snapshot, config=view)
    assert evaluation.near_signal is True
    assert evaluation.decision == DECISION_NEAR
    assert evaluation.signal is None, "a near signal must never become a trade"

    view.base_threshold = _threshold_for_gap(view, snapshot, 40.0)
    far = SignalEngine(view).evaluate(snapshot, config=view)
    assert far.near_signal is False
    assert far.decision == DECISION_NONE


def _threshold_for_gap(view, snapshot, gap: float) -> float:
    """Base threshold that puts the *effective* bar ``gap`` points above the score.

    The effective threshold is ``base + regime offset (+ counter-trend extra)``,
    so the regime adjustment has to be measured and backed out rather than
    assumed to be zero.
    """
    baseline = SignalEngine(view).evaluate(snapshot, config=view)
    offset = baseline.threshold - view.base_threshold
    return view.clamp_threshold(baseline.best_score + gap - offset)


def test_near_signal_is_recorded_in_the_evaluation_row(isolated_config):
    candles = make_candles(4200, seed=93, drift=0.05)
    view = config_view(isolated_config, "RESEARCH", "M5")
    snapshot = build_snapshot(view, candles)
    view.base_threshold = _threshold_for_gap(view, snapshot, 4.0)

    row = SignalEngine(view).evaluate(snapshot, config=view).to_row()
    assert row["near_signal"] == 1
    assert row["decision"] == DECISION_NEAR
    assert list(row.keys()) == list(EVALUATION_COLUMNS)


# --------------------------------------------------------------------------- #
# research signals: data collection
# --------------------------------------------------------------------------- #
def test_research_mode_produces_more_candidates_than_standard(isolated_config):
    """The whole point of RESEARCH: more candidates to observe, same engine."""
    candles = make_candles(4200, seed=95, drift=0.05)
    counts = {}
    for mode in ("RESEARCH", "STANDARD", "CONSERVATIVE"):
        view = config_view(isolated_config, mode, "M5")
        snapshot = build_snapshot(view, candles)
        evaluation = SignalEngine(view).evaluate(snapshot, config=view)
        counts[mode] = evaluation.threshold
    assert counts["RESEARCH"] < counts["STANDARD"] < counts["CONSERVATIVE"]


def test_research_signal_carries_its_operating_context(isolated_config):
    """Scan a few windows with the gates opened so a candidate is guaranteed."""
    candles = make_candles(4600, seed=97, drift=0.06)
    view = config_view(isolated_config, "RESEARCH", "M5")
    view.base_threshold = view.min_threshold
    view.min_score_separation = 0.0
    view.enable_fakeout_filter = False
    view.min_tp2_rr = 0.0
    view.block_on_extreme_volatility = False
    engine = SignalEngine(view)

    evaluation = None
    for end in range(4200, 4601, 20):
        candidate = engine.evaluate(build_snapshot(view, candles.iloc[:end]), config=view)
        if candidate.has_signal:
            evaluation = candidate
            break
    assert evaluation is not None, "expected at least one research candidate"

    signal = evaluation.signal
    assert signal.mode == "RESEARCH"
    assert signal.is_research is True
    assert signal.timeframe == "M5"
    assert signal.confirmation_timeframes == "M15+H1"
    assert signal.threshold_used > 0

    row = signal.to_row()
    for column in ("mode", "signal_timeframe", "confirmation_timeframes", "threshold_used",
                   "score", "bullish_score", "bearish_score"):
        assert column in row, column
        assert column in SIGNAL_COLUMNS, column
    assert row["mode"] == "RESEARCH"
    assert row["score"] == signal.confidence


def test_research_signals_are_persisted_like_any_other(isolated_config):
    from src.signal_engine import Signal
    from datetime import datetime, timezone

    tracker = SignalTracker(isolated_config, FakeNotifier())
    tracker.load()
    signal = Signal(
        signal_id="research-1", symbol="XAUUSD", timeframe="M5", direction="BUY",
        timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc),
        entry=2300.0, stop_loss=2290.0, tp1=2310.0, tp2=2318.0, tp3=2328.0,
        confidence=56.0, bullish_score=56.0, bearish_score=31.0, regime="WEAK_TREND",
        risk_reward=1.8, session="LONDON", reason_summary="trend", mode="RESEARCH",
        confirmation_timeframes="M15+H1", threshold_used=50.0,
    )
    tracker.record_signal(signal)

    rows = read_csv_rows(isolated_config.signals_csv)
    assert rows[0]["mode"] == "RESEARCH"
    assert rows[0]["confirmation_timeframes"] == "M15+H1"
    assert float(rows[0]["threshold_used"]) == 50.0

    # and it is tracked to an outcome exactly like a standard signal
    times = pd.date_range("2024-05-01 12:05", periods=1, freq="5min", tz="UTC")
    candles = pd.DataFrame(
        {"time": times, "open": [2300.0], "high": [2330.0], "low": [2299.0],
         "close": [2329.0], "tick_volume": [100.0]}
    )
    tracker.update(candles, timeframe="M5")
    outcomes = read_csv_rows(isolated_config.outcomes_csv)
    assert len(outcomes) == 1
    assert outcomes[0]["mode"] == "RESEARCH"
    assert outcomes[0]["result"] == "TP3_HIT"
    assert float(outcomes[0]["R_multiple"]) > 0


def test_research_message_is_clearly_labelled(config):
    from src.signal_engine import Signal
    from datetime import datetime, timezone

    signal = Signal(
        signal_id="r", symbol="XAUUSD", timeframe="M5", direction="BUY",
        timestamp=datetime(2024, 5, 1, tzinfo=timezone.utc),
        entry=3342.5, stop_loss=3339.8, tp1=3345.2, tp2=3347.9, tp3=3351.0,
        confidence=56.0, bullish_score=56.0, bearish_score=31.0, regime="STRONG_BULL_TREND",
        risk_reward=1.8, session="LONDON", reason_summary="Trend + HTF",
        rr1=1.0, rr2=1.8, rr3=2.8, mode="RESEARCH", threshold_used=50.0,
    )
    text = TelegramNotifier(config).format_signal(signal)
    assert "🔬 RESEARCH BUY" in text
    assert "not a validated trading signal" in text
    assert "Score: 56/100" in text
    assert "CONFIRMATIONS" not in text, "research cards must not look like standard cards"


def test_standard_message_is_not_labelled_as_research(config):
    from src.signal_engine import Signal
    from datetime import datetime, timezone

    signal = Signal(
        signal_id="s", symbol="XAUUSD", timeframe="M5", direction="BUY",
        timestamp=datetime(2024, 5, 1, tzinfo=timezone.utc),
        entry=3342.5, stop_loss=3339.8, tp1=3345.2, tp2=3347.9, tp3=3351.0,
        confidence=85.0, bullish_score=85.0, bearish_score=40.0, regime="STRONG_BULL_TREND",
        risk_reward=1.8, session="LONDON", reason_summary="Trend", mode="STANDARD",
    )
    text = TelegramNotifier(config).format_signal(signal)
    assert "RESEARCH" not in text
    assert "CONFIRMATIONS" in text


# --------------------------------------------------------------------------- #
# score bands
# --------------------------------------------------------------------------- #
def test_score_bands_cover_forty_to_one_hundred():
    labels = [band[2] for band in SCORE_BANDS]
    assert labels == ["40-49", "50-59", "60-69", "70-79", "80-89", "90-100"]
    assert SCORE_BANDS[0][0] == 40.0
    assert SCORE_BANDS[-1][1] > 100.0
    for (_, high, _), (low, _, _) in zip(SCORE_BANDS, SCORE_BANDS[1:]):
        assert high == low, "score bands must be contiguous"


def test_score_band_report_groups_by_score_not_confidence():
    signals = pd.DataFrame(
        [
            {"signal_id": "a", "direction": "BUY", "timestamp": "2024-05-01T10:00:00+00:00",
             "status": "TP3_HIT", "mode": "RESEARCH", "timeframe": "M5", "score": 52,
             "confidence": 52, "regime": "WEAK_TREND", "session": "LONDON"},
            {"signal_id": "b", "direction": "SELL", "timestamp": "2024-05-01T11:00:00+00:00",
             "status": "SL_HIT", "mode": "RESEARCH", "timeframe": "M5", "score": 55,
             "confidence": 55, "regime": "RANGE", "session": "LONDON"},
            {"signal_id": "c", "direction": "BUY", "timestamp": "2024-05-01T12:00:00+00:00",
             "status": "TP3_HIT", "mode": "STANDARD", "timeframe": "M15", "score": 74,
             "confidence": 74, "regime": "WEAK_TREND", "session": "ASIAN"},
        ]
    )
    outcomes = pd.DataFrame(
        [
            {"signal_id": "a", "R_multiple": 1.87, "tp_hits": 3, "result": "TP3_HIT", "duration": 90},
            {"signal_id": "b", "R_multiple": -1.0, "tp_hits": 0, "result": "SL_HIT", "duration": 30},
            {"signal_id": "c", "R_multiple": 1.87, "tp_hits": 3, "result": "TP3_HIT", "duration": 60},
        ]
    )
    report = build_report(signals, outcomes)
    bands = {stats.label: stats for stats in report.by_score_band}
    assert set(bands) == {"50-59", "70-79"}
    assert bands["50-59"].closed == 2
    assert bands["50-59"].win_rate == pytest.approx(50.0)
    assert bands["70-79"].closed == 1
    assert bands["70-79"].average_r == pytest.approx(1.87)


def test_performance_splits_by_mode_and_timeframe():
    signals = pd.DataFrame(
        [
            {"signal_id": "a", "direction": "BUY", "timestamp": "2024-05-01T10:00:00+00:00",
             "status": "TP3_HIT", "mode": "RESEARCH", "timeframe": "M5", "score": 52,
             "confidence": 52, "regime": "WEAK_TREND", "session": "LONDON"},
            {"signal_id": "c", "direction": "BUY", "timestamp": "2024-05-01T12:00:00+00:00",
             "status": "SL_HIT", "mode": "STANDARD", "timeframe": "M15", "score": 74,
             "confidence": 74, "regime": "RANGE", "session": "ASIAN"},
        ]
    )
    outcomes = pd.DataFrame(
        [
            {"signal_id": "a", "R_multiple": 1.87, "tp_hits": 3, "result": "TP3_HIT", "duration": 90},
            {"signal_id": "c", "R_multiple": -1.0, "tp_hits": 0, "result": "SL_HIT", "duration": 30},
        ]
    )
    report = build_report(signals, outcomes)
    assert {s.label for s in report.by_mode} == {"RESEARCH", "STANDARD"}
    assert {s.label for s in report.by_timeframe} == {"M5", "M15"}
    assert report.stats_for_mode("RESEARCH").average_r == pytest.approx(1.87)
    assert report.stats_for_mode("MISSING") is None
    assert report.best_regime() == "WEAK_TREND"
    assert report.worst_regime() == "RANGE"
