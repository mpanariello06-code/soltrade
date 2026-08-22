"""End-to-end engine behaviour, signal lifecycle, storage and no-lookahead."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.indicators import compute_indicators
from src.market_data import MarketSnapshot, clean_candles, validate_candles
from src.signal_engine import (
    DECISION_NONE,
    EVALUATION_COLUMNS,
    FEATURE_KEYS,
    Signal,
    SignalEngine,
)
from src.signal_tracker import (
    OUTCOME_COLUMNS,
    SIGNAL_COLUMNS,
    STATUS_ACTIVE,
    STATUS_INVALIDATED,
    STATUS_SL,
    STATUS_TIMEOUT,
    STATUS_TP1,
    STATUS_TP3,
    PositionState,
    SignalTracker,
    append_csv,
    build_gate_state,
    ensure_csv,
    evaluate_progress,
    read_csv_rows,
)
from src.telegram_bot import TelegramNotifier
from src.utils import atomic_write_json, iso, parse_iso, read_json
from tests.conftest import FakeMarket, FakeNotifier, build_snapshot, isolate, make_candles


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def long_candles() -> pd.DataFrame:
    """Enough M1 history to warm up the indicators and the M5 context."""
    return make_candles(2500, seed=61)


@pytest.fixture
def tracker_config(config, tmp_path):
    """XAUUSD config pointed at a throwaway data directory."""
    return isolate(config, tmp_path)


def sample_signal(direction: str = "BUY", when: datetime | None = None) -> Signal:
    when = when or datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    if direction == "BUY":
        entry, stop, tps = 2300.0, 2299.6, (2300.36, 2300.65, 2301.08)
    else:
        entry, stop, tps = 2300.0, 2300.4, (2299.64, 2299.35, 2298.92)
    return Signal(
        signal_id=f"XAUUSD-{direction}-{when:%Y%m%d%H%M%S}",
        symbol="XAUUSD", timeframe="M1", direction=direction, timestamp=when,
        entry=entry, stop_loss=stop, tp1=tps[0], tp2=tps[1], tp3=tps[2],
        confidence=71.0, bullish_score=71.0, bearish_score=30.0,
        regime="WEAK_TREND", risk_reward=1.6, session="LONDON",
        reason_summary="momentum=18.0", confidence_label="STRONG",
        rr1=0.9, rr2=1.6, rr3=2.7, sl_mode="HYBRID",
        confirmations={"momentum": True, "price_action": True},
        mode="SCALPING", threshold_used=68.0, spread_points=20.0,
        cost_pips=2.4, cost_r=0.6, sl_pips=4.0, tp_pips=(3.6, 6.5, 10.8),
        net_rr1=0.3, net_rr2=1.0, net_rr3=2.1, atr=0.30, expected_hold="SHORT",
    )


def outcome_candles(bars, start="2024-05-01 12:01") -> pd.DataFrame:
    times = pd.date_range(start, periods=len(bars), freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "time": times,
            "open": [b[0] for b in bars], "high": [b[1] for b in bars],
            "low": [b[2] for b in bars], "close": [b[3] for b in bars],
            "tick_volume": 100.0,
        }
    )


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def test_engine_produces_a_complete_evaluation(config, long_candles):
    evaluation = SignalEngine(config).evaluate(build_snapshot(config, long_candles))
    assert evaluation.card is not None
    assert evaluation.regime
    assert evaluation.session
    assert 0.0 <= evaluation.card.bullish_score <= 100.0
    assert 0.0 <= evaluation.card.bearish_score <= 100.0


def test_evaluation_row_matches_the_declared_schema(config, long_candles):
    evaluation = SignalEngine(config).evaluate(build_snapshot(config, long_candles))
    row = evaluation.to_row()
    assert list(row.keys()) == list(EVALUATION_COLUMNS)
    for key in FEATURE_KEYS:
        assert f"f_{key}" in row


def test_evaluation_is_logged_even_when_there_is_no_signal(config, long_candles):
    """Spec 36: every evaluated candle produces a row, signal or not."""
    evaluation = SignalEngine(config).evaluate(build_snapshot(config, long_candles))
    row = evaluation.to_row()
    assert row["decision"] in ("BUY", "SELL", DECISION_NONE)
    if row["decision"] == DECISION_NONE:
        assert row["rejection_reason"], "a no-signal evaluation must say why"


def test_insufficient_history_is_rejected_with_a_reason(config):
    snapshot = build_snapshot(config, make_candles(300, seed=63))
    evaluation = SignalEngine(config).evaluate(snapshot)
    assert evaluation.decision == DECISION_NONE
    assert "data validation failed" in evaluation.rejection_reason


def test_engine_is_deterministic(config, long_candles):
    """Same input, same output - no randomness anywhere in V1."""
    engine = SignalEngine(config)
    first = engine.evaluate(build_snapshot(config, long_candles))
    second = engine.evaluate(build_snapshot(config, long_candles))
    assert first.card.bullish_score == second.card.bullish_score
    assert first.card.bearish_score == second.card.bearish_score
    assert first.rejection_reason == second.rejection_reason


def test_evaluation_does_not_change_when_later_candles_arrive(config, long_candles):
    """No repaint: the verdict for a candle is fixed once that candle closes."""
    engine = SignalEngine(config)
    cut = 4000
    early = engine.evaluate(build_snapshot(config, long_candles.iloc[:cut]))
    late = engine.evaluate(build_snapshot(config, long_candles.iloc[:cut]))
    with_more_data = engine.evaluate(build_snapshot(config, long_candles.iloc[:cut]))
    assert early.card.bullish_score == late.card.bullish_score == with_more_data.card.bullish_score
    assert early.timestamp == with_more_data.timestamp


def test_signal_timestamp_is_the_closed_candle_not_wall_clock(config, long_candles):
    snapshot = build_snapshot(config, long_candles)
    evaluation = SignalEngine(config).evaluate(snapshot)
    expected = pd.Timestamp(long_candles["time"].iloc[-1]).to_pydatetime()
    assert evaluation.timestamp.replace(tzinfo=None) == expected.replace(tzinfo=None)


def test_snapshot_exposes_only_closed_candles(config, long_candles):
    """There is no field on the snapshot that can carry a forming candle."""
    snapshot = build_snapshot(config, long_candles)
    assert snapshot.close == pytest.approx(float(snapshot.signal_df["close"].iloc[-1]))
    assert snapshot.signal_timeframe == "M1"
    assert not hasattr(snapshot, "forming_candle")
    assert not hasattr(snapshot, "current_candle")


def test_a_generated_signal_carries_every_required_field(config):
    """Spec 32."""
    signal = sample_signal()
    for field in (
        "signal_id", "symbol", "timeframe", "direction", "timestamp", "entry",
        "stop_loss", "tp1", "tp2", "tp3", "confidence", "bullish_score",
        "bearish_score", "regime", "risk_reward", "session", "reason_summary",
    ):
        assert hasattr(signal, field), field
    row = signal.to_row()
    for column in ("signal_id", "timestamp", "entry", "sl", "tp1", "tp2", "tp3", "status"):
        assert column in row
    assert row["status"] == STATUS_ACTIVE


def test_signal_ids_are_unique_for_the_same_candle(config):
    from src.utils import make_signal_id

    when = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    ids = {make_signal_id("XAUUSD", when, "BUY") for _ in range(200)}
    assert len(ids) == 200


def test_engine_survives_malformed_candles(config, long_candles):
    """A corrupt row must be rejected by validation, not crash the engine."""
    broken = long_candles.copy()
    broken.loc[broken.index[-1], "high"] = float("nan")
    snapshot = MarketSnapshot(
        symbol=config.symbol,
        signal_df=compute_indicators(broken, config.indicators),
        context_df=None,
    )
    evaluation = SignalEngine(config).evaluate(snapshot)
    assert evaluation.decision == DECISION_NONE
    assert evaluation.rejection_reason


def test_duplicate_and_unordered_candles_are_caught(config):
    frame = make_candles(400, seed=65)
    duplicated = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
    ok, reason = validate_candles(duplicated, "M5", 100)
    assert not ok and "duplicate" in reason

    cleaned = clean_candles(duplicated)
    assert not cleaned["time"].duplicated().any()
    ok, _ = validate_candles(cleaned, "M5", 100)
    assert ok


def test_stale_data_is_rejected(config):
    frame = make_candles(400, seed=67)
    reference = pd.Timestamp(frame["time"].iloc[-1]).to_pydatetime() + timedelta(hours=6)
    ok, reason = validate_candles(frame, "M5", 100, 900, reference)
    assert not ok and "stale" in reason


# --------------------------------------------------------------------------- #
# lifecycle / tracking
# --------------------------------------------------------------------------- #
def test_tracker_creates_its_csv_files_with_headers(tracker_config):
    tracker = SignalTracker(tracker_config)
    tracker.load()
    assert tracker_config.signals_csv.exists()
    assert tracker_config.outcomes_csv.exists()
    header = tracker_config.signals_csv.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(SIGNAL_COLUMNS)


def test_recording_a_signal_persists_it_and_updates_state(tracker_config):
    tracker = SignalTracker(tracker_config)
    tracker.load()
    signal = sample_signal()
    tracker.record_signal(signal)

    rows = read_csv_rows(tracker_config.signals_csv)
    assert len(rows) == 1
    assert rows[0]["signal_id"] == signal.signal_id
    assert rows[0]["status"] == STATUS_ACTIVE

    state = read_json(tracker_config.state_file)
    assert state["last_signal_id"] == signal.signal_id
    assert parse_iso(state["last_signal_time"]) == signal.timestamp


def test_signal_lifecycle_advances_to_tp_and_writes_one_outcome(tracker_config):
    tracker = SignalTracker(tracker_config)
    tracker.load()
    signal = sample_signal("BUY")
    tracker.record_signal(signal)

    # a single candle that runs through all three targets
    candles = outcome_candles([(2300, 2301.2, 2299.9, 2301.1)])
    events = tracker.update(candles)
    assert [event.event for event in events][-1] == STATUS_TP3

    rows = read_csv_rows(tracker_config.signals_csv)
    assert rows[0]["status"] == STATUS_TP3
    outcomes = read_csv_rows(tracker_config.outcomes_csv)
    assert len(outcomes) == 1
    assert outcomes[0]["result"] == STATUS_TP3
    assert float(outcomes[0]["R_multiple"]) > 0


def test_outcomes_are_never_written_twice(tracker_config):
    """Re-running update() on a closed signal must not duplicate its outcome."""
    tracker = SignalTracker(tracker_config)
    tracker.load()
    tracker.record_signal(sample_signal("BUY"))
    candles = outcome_candles([(2300, 2301.2, 2299.9, 2301.1)])
    tracker.update(candles)
    tracker.update(candles)
    tracker.update(candles)
    assert len(read_csv_rows(tracker_config.outcomes_csv)) == 1


def test_alerts_are_not_resent_after_a_restart(tracker_config):
    """Status is persisted, so a fresh tracker replays no old notifications."""
    tracker = SignalTracker(tracker_config)
    tracker.load()
    tracker.record_signal(sample_signal("BUY"))
    candles = outcome_candles([(2300, 2300.4, 2299.9, 2300.35)])
    first_events = tracker.update(candles)
    assert [e.event for e in first_events] == [STATUS_TP1]

    restarted = SignalTracker(tracker_config)
    restarted.load()
    assert restarted.update(candles) == []


def test_stop_loss_closes_the_signal_at_minus_one_r(tracker_config):
    tracker = SignalTracker(tracker_config)
    tracker.load()
    tracker.record_signal(sample_signal("BUY"))
    events = tracker.update(outcome_candles([(2300, 2300.1, 2299.5, 2299.55)]))
    assert [e.event for e in events] == [STATUS_SL]
    outcomes = read_csv_rows(tracker_config.outcomes_csv)
    assert float(outcomes[0]["R_multiple"]) == pytest.approx(-1.0)


def test_opposite_signal_invalidates_an_open_one(tracker_config):
    tracker = SignalTracker(tracker_config)
    tracker.load()
    tracker.record_signal(sample_signal("BUY"))
    events = tracker.invalidate_opposite(
        "SELL", datetime(2024, 5, 1, 12, 5, tzinfo=timezone.utc), 2299.9
    )
    assert [e.event for e in events] == [STATUS_INVALIDATED]
    assert read_csv_rows(tracker_config.signals_csv)[0]["status"] == STATUS_INVALIDATED
    assert len(read_csv_rows(tracker_config.outcomes_csv)) == 1


def test_gate_state_reflects_recorded_signals(tracker_config):
    tracker = SignalTracker(tracker_config)
    tracker.load()
    first = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    tracker.record_signal(sample_signal("BUY", first))
    gate = tracker.gate_state(first + timedelta(minutes=10))
    assert gate.last_signal_direction == "BUY"
    assert gate.last_signal_time == first
    assert gate.signals_today == 1
    assert gate.active_count == 1


def test_gate_state_ignores_signals_in_the_future(config):
    """The backtester relies on this to stay free of lookahead."""
    now = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        {"timestamp": iso(now - timedelta(hours=1)), "direction": "BUY"},
        {"timestamp": iso(now + timedelta(hours=1)), "direction": "SELL"},
    ]
    gate = build_gate_state(rows, 0, now)
    assert gate.last_signal_direction == "BUY"
    assert gate.signals_today == 1


def test_timeout_closes_a_stalled_scalp(tracker_config):
    tracker_config.max_holding_candles = 5
    tracker = SignalTracker(tracker_config)
    tracker.load()
    tracker.record_signal(sample_signal("BUY"))
    flat = outcome_candles([(2300, 2300.05, 2299.95, 2300.0)] * 5)
    events = tracker.update(flat)
    assert [e.event for e in events] == [STATUS_TIMEOUT]


def test_progress_ignores_the_signal_candle_itself(config):
    """The entry is that candle's close, so it cannot trade against its own bar."""
    row = sample_signal("BUY").to_row()
    same_bar = pd.DataFrame(
        {
            "time": [pd.Timestamp("2024-05-01 12:00", tz="UTC")],
            "open": [2300.0], "high": [2310.0], "low": [2290.0], "close": [2300.0],
            "tick_volume": [100.0],
        }
    )
    progress = evaluate_progress(row, same_bar, config)
    assert progress.status == STATUS_ACTIVE
    assert progress.bars == 0


def test_position_state_and_evaluate_progress_agree(config):
    """The live path and the backtest path must score outcomes identically."""
    row = sample_signal("BUY").to_row()
    bars = [(2300, 2300.4, 2299.9, 2300.35), (2300.35, 2300.7, 2299.9, 2300.0),
            (2300.0, 2300.05, 2299.5, 2299.55)]
    candles = outcome_candles(bars)

    incremental = PositionState(row, config)
    for _, bar in candles.iterrows():
        if incremental.step(bar, bar["time"].to_pydatetime()):
            break

    batch = evaluate_progress(row, candles, config)
    assert incremental.to_progress().status == batch.status
    assert incremental.to_progress().r_multiple == pytest.approx(batch.r_multiple)
    assert incremental.to_progress().tp_hits == batch.tp_hits


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #
def test_csv_append_creates_the_file_and_survives_restarts(tmp_path):
    path = tmp_path / "sub" / "out.csv"
    append_csv(path, {"a": 1, "b": 2}, ("a", "b"))
    append_csv(path, {"a": 3, "b": 4}, ("a", "b"))
    rows = read_csv_rows(path)
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_csv_append_ignores_unknown_and_fills_missing_columns(tmp_path):
    path = tmp_path / "out.csv"
    ensure_csv(path, ("a", "b"))
    append_csv(path, {"a": 1, "z": 99}, ("a", "b"))
    assert read_csv_rows(path) == [{"a": "1", "b": ""}]


def test_csv_with_a_changed_schema_is_archived_not_corrupted(tmp_path):
    path = tmp_path / "out.csv"
    path.write_text("old,header\n1,2\n", encoding="utf-8")
    ensure_csv(path, ("a", "b", "c"))
    assert path.read_text(encoding="utf-8").splitlines()[0] == "a,b,c"
    assert any(p.name.startswith("out.csv.bak-") for p in tmp_path.iterdir())


def test_csv_files_are_utf8(tmp_path):
    path = tmp_path / "out.csv"
    append_csv(path, {"a": "gold ⭐ 🟢"}, ("a",))
    assert "⭐" in path.read_text(encoding="utf-8")


def test_state_json_round_trips_and_is_corruption_tolerant(tmp_path):
    path = tmp_path / "state.json"
    assert atomic_write_json(path, {"last_processed_candle": "2024-05-01T12:00:00+00:00"})
    assert read_json(path)["last_processed_candle"] == "2024-05-01T12:00:00+00:00"
    path.write_text("{not json", encoding="utf-8")
    assert read_json(path, {"fallback": True}) == {"fallback": True}


# --------------------------------------------------------------------------- #
# telegram formatting (no network)
# --------------------------------------------------------------------------- #
def test_telegram_is_disabled_without_credentials(config):
    config.telegram_bot_token = ""
    notifier = TelegramNotifier(config)
    assert not notifier.enabled
    assert notifier.send_message("hello") is None


def test_scalp_message_matches_the_specified_format(config):
    text = TelegramNotifier(config).format_signal(sample_signal("BUY"))
    assert "🥇 XAUUSD M1 SCALP" in text, "the card is headed with the market's icon"
    assert "Direction: BUY" in text
    assert "Score: 71/100" in text
    assert "Entry: 2300.00" in text
    assert "SL: 2299.60" in text
    assert "TP1: 2300.36" in text and "TP3: 2301.08" in text
    assert "Expected holding period:" in text and "SHORT" in text
    assert "Spread:" in text
    assert "Risk/Reward:" in text
    assert "PAPER TEST ONLY" in text


def test_scalp_message_reports_reward_after_costs(config):
    """Quoting raw R alone on a few-pip target would be misleading."""
    text = TelegramNotifier(config).format_signal(sample_signal("BUY"))
    assert "After costs" in text
    assert "2.4p" in text
    for net in ("0.30R", "1.00R", "2.10R"):
        assert net in text


def test_sell_message_states_the_direction(config):
    text = TelegramNotifier(config).format_signal(sample_signal("SELL"))
    assert "Direction: SELL" in text


def test_outcome_message_reports_raw_and_net(config):
    row = {"symbol": "XAUUSD", "direction": "BUY", "entry": 2300.0, "signal_id": "x"}
    text = TelegramNotifier(config).format_outcome(row, STATUS_TP3, 2301.08, 1.867, 1.267)
    assert "TP3 HIT" in text
    assert "Raw: +1.87R" in text
    assert "Net after costs: +1.27R" in text


def test_timeout_outcome_message(config):
    row = {"symbol": "XAUUSD", "direction": "BUY", "entry": 2300.0, "signal_id": "x"}
    text = TelegramNotifier(config).format_outcome(row, STATUS_TIMEOUT, 2300.05, 0.1, -0.5)
    assert "TIMED OUT" in text


# --------------------------------------------------------------------------- #
# backtester
# --------------------------------------------------------------------------- #
def test_backtester_never_shows_an_unclosed_context_candle(config):
    """The core no-lookahead guarantee of the backtester."""
    from backtest import Backtester

    candles = make_candles(3000, seed=71)
    backtester = Backtester(config)
    prepared = backtester.prepare(candles)

    for index in (2000, 2500, 2999):
        snapshot = backtester._snapshot(prepared, index)
        m1_close = pd.Timestamp(prepared["m1_close"][index])
        if snapshot.context_df is None or snapshot.context_df.empty:
            continue
        minutes = 5 if config.context_timeframe == "M5" else 1
        last_close = pd.Timestamp(snapshot.context_df["time"].iloc[-1]) + timedelta(minutes=minutes)
        assert last_close <= m1_close, "an unclosed context candle leaked into the snapshot"


def test_backtester_window_ends_on_the_evaluated_bar(config):
    from backtest import Backtester

    candles = make_candles(2500, seed=73)
    backtester = Backtester(config)
    prepared = backtester.prepare(candles)
    index = 2400
    snapshot = backtester._snapshot(prepared, index)
    assert pd.Timestamp(snapshot.signal_df["time"].iloc[-1]) == pd.Timestamp(
        candles["time"].iloc[index]
    )
    assert len(snapshot.signal_df) <= config.candles_signal


def test_backtest_run_produces_consistent_signals_and_outcomes(config, tmp_path):
    from backtest import Backtester

    config.data_dir = tmp_path
    candles = make_candles(3000, seed=75, drift=0.02)
    result = Backtester(config).run(candles, log_evaluations=True, progress_every=0)

    assert result.bars_evaluated > 0
    assert len(result.evaluations) == result.bars_evaluated
    assert len(result.outcomes) <= len(result.signals)

    ids = [row["signal_id"] for row in result.signals]
    assert len(ids) == len(set(ids)), "duplicate signal ids"

    for row in result.signals:
        assert set(SIGNAL_COLUMNS).issuperset(row.keys())
    for row in result.outcomes:
        assert set(OUTCOME_COLUMNS).issuperset(row.keys())
        assert row["result"] in ("TP3_HIT", "SL_HIT", "TIMEOUT", "INVALIDATED")


def test_backtest_respects_the_cooldown_between_signals(config, tmp_path):
    from backtest import Backtester

    config.data_dir = tmp_path
    result = Backtester(config).run(
        make_candles(3000, seed=77, drift=0.02), log_evaluations=False, progress_every=0
    )
    times = sorted(parse_iso(row["timestamp"]) for row in result.signals)
    minimum_gap = timedelta(minutes=config.cooldown_candles)
    for earlier, later in zip(times, times[1:]):
        assert later - earlier >= minimum_gap, "cooldown was not enforced"


# --------------------------------------------------------------------------- #
# live loop wiring (main.py) with a stubbed MT5 feed
# --------------------------------------------------------------------------- #
def test_live_loop_processes_each_candle_exactly_once(tracker_config):
    """The whole main.py wiring: new candle -> evaluate -> log -> persist state."""
    from main import SignalRunner

    candles = make_candles(3200, seed=81, drift=0.05)
    runner = SignalRunner(tracker_config)
    runner.market = FakeMarket(tracker_config, candles, start=3100)
    runner.notifier.config.telegram_enabled = False
    ensure_csv(tracker_config.evaluations_csv, EVALUATION_COLUMNS)
    runner.slot("XAUUSD").tracker.load()

    for _ in range(20):
        runner._tick()
        runner.market.advance()

    evaluations = read_csv_rows(tracker_config.evaluations_csv)
    assert len(evaluations) == 20, "one evaluation row per closed candle"
    timestamps = [row["timestamp"] for row in evaluations]
    assert len(timestamps) == len(set(timestamps)), "a candle was evaluated twice"

    state = read_json(tracker_config.state_file)
    assert state["last_processed_candles"]["M1"] == timestamps[-1]


def test_live_loop_skips_a_candle_it_has_already_processed(tracker_config):
    from main import SignalRunner

    candles = make_candles(3200, seed=83)
    runner = SignalRunner(tracker_config)
    runner.market = FakeMarket(tracker_config, candles, start=3150)
    runner.notifier.config.telegram_enabled = False
    ensure_csv(tracker_config.evaluations_csv, EVALUATION_COLUMNS)
    runner.slot("XAUUSD").tracker.load()

    runner._tick()
    runner._tick()   # same candle - must be ignored
    runner._tick()
    assert len(read_csv_rows(tracker_config.evaluations_csv)) == 1


def test_live_loop_records_signals_and_never_duplicates_them(tracker_config):
    from main import SignalRunner

    candles = make_candles(3400, seed=85, drift=0.06)
    runner = SignalRunner(tracker_config)
    runner.market = FakeMarket(tracker_config, candles, start=3000)
    runner.notifier.config.telegram_enabled = False
    ensure_csv(tracker_config.evaluations_csv, EVALUATION_COLUMNS)
    runner.slot("XAUUSD").tracker.load()

    for _ in range(120):
        runner._tick()
        runner.market.advance()

    signals = read_csv_rows(tracker_config.signals_csv)
    ids = [row["signal_id"] for row in signals]
    assert len(ids) == len(set(ids))
    for row in signals:
        assert row["direction"] in ("BUY", "SELL")
        assert float(row["entry"]) > 0
        assert row["status"] in (
            "ACTIVE", "TP1_HIT", "TP2_HIT", "TP3_HIT", "SL_HIT", "EXPIRED", "INVALIDATED"
        )


# --------------------------------------------------------------------------- #
# run state and timeframe switching through the live runner
# --------------------------------------------------------------------------- #
def _runner(config, candles, start):
    """A SignalRunner wired to fake market data and a fake Telegram bot."""
    from main import SignalRunner

    runner = SignalRunner(config)
    runner.market = FakeMarket(config, candles, start=start)
    runner.notifier = FakeNotifier()
    runner.slot(config.symbol).tracker.notifier = runner.notifier
    runner.control.notifier = runner.notifier
    ensure_csv(config.evaluations_csv, EVALUATION_COLUMNS)
    runner.slot(config.symbol).tracker.load()
    return runner


def test_paused_runner_evaluates_nothing(tracker_config):
    candles = make_candles(2600, seed=111)
    runner = _runner(tracker_config, candles, start=2500)
    runner.runtime.pause()

    for _ in range(10):
        runner._tick()
        runner.market.advance()

    assert read_csv_rows(tracker_config.evaluations_csv) == []
    assert runner.runtime.active.last_processed_candle() is None


def test_resuming_restarts_evaluation(tracker_config):
    candles = make_candles(2600, seed=113)
    runner = _runner(tracker_config, candles, start=2500)

    runner.runtime.pause()
    for _ in range(3):
        runner._tick()
        runner.market.advance()
    assert read_csv_rows(tracker_config.evaluations_csv) == []

    runner.runtime.start()
    for _ in range(4):
        runner._tick()
        runner.market.advance()
    assert len(read_csv_rows(tracker_config.evaluations_csv)) == 4


def test_stop_ends_the_loop(tracker_config):
    candles = make_candles(2600, seed=115)
    runner = _runner(tracker_config, candles, start=2500)
    runner.running = True
    runner.runtime.stop()
    runner._tick()
    assert runner.running is False
def test_near_signal_alerts_are_off_by_default(tracker_config):
    candles = make_candles(2600, seed=123)
    runner = _runner(tracker_config, candles, start=2500)
    assert runner.runtime.near_signal_alerts is False

    for _ in range(10):
        runner._tick()
        runner.market.advance()
    assert runner.notifier.near_signals == [], "near-signal alerts fired while disabled"


def test_near_signal_alerts_fire_only_when_enabled(tracker_config):
    """The diagnostic is opt-in, and never sends a tradeable signal card."""
    candles = make_candles(2800, seed=129, drift=0.02)
    runner = _runner(tracker_config, candles, start=2500)
    runner.runtime.set_near_signal_alerts(True)

    for _ in range(60):
        runner._tick()
        runner.market.advance()

    evaluations = read_csv_rows(tracker_config.evaluations_csv)
    near_rows = [row for row in evaluations if row["near_signal"] == "1"]
    assert near_rows, "expected at least one near-signal on this fixture"
    assert len(runner.notifier.near_signals) == len(near_rows)
    for row in near_rows:
        assert row["decision"] == "NEAR_SIGNAL"
        assert row["signal_id"] == "", "a near signal must not create a signal"
    assert runner.notifier.signals == [] or all(
        signal.confidence >= float(runner.runtime.active.active_threshold())
        for signal in runner.notifier.signals
    ), "a near signal leaked into the signal feed"


def test_analyze_now_has_no_side_effects(tracker_config):
    """The ANALYSIS button must not emit, suppress or record anything."""
    candles = make_candles(2600, seed=125)
    runner = _runner(tracker_config, candles, start=2500)

    evaluation = runner.analyze_now()
    assert evaluation is not None
    assert evaluation.card is not None
    assert read_csv_rows(tracker_config.evaluations_csv) == []
    assert read_csv_rows(tracker_config.signals_csv) == []
    assert runner.runtime.active.last_processed_candle() is None
    assert runner.notifier.signals == []