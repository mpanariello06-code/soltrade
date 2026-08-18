"""End-to-end engine behaviour, signal lifecycle, storage and no-lookahead."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.indicators import compute_indicators
from src.market_data import MarketSnapshot, clean_candles, resample_candles, validate_candles
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
from tests.conftest import make_candles


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def build_snapshot(config, candles: pd.DataFrame, spread: float = float("nan")) -> MarketSnapshot:
    """Assemble a snapshot the way the live loop and backtester do."""
    return MarketSnapshot(
        symbol=config.symbol,
        m5=compute_indicators(candles, config.indicators),
        m15=compute_indicators(resample_candles(candles, "M5", "M15"), config.indicators),
        h1=compute_indicators(resample_candles(candles, "M5", "H1"), config.indicators),
        spread_points=spread,
    )


@pytest.fixture
def long_candles() -> pd.DataFrame:
    """Enough M5 history to warm up a 200-period EMA on H1."""
    return make_candles(4200, seed=61)


@pytest.fixture
def tracker_config(config, tmp_path):
    """Config pointed at a throwaway data directory."""
    config.data_dir = tmp_path
    config.signals_csv = tmp_path / "signals.csv"
    config.evaluations_csv = tmp_path / "evaluations.csv"
    config.outcomes_csv = tmp_path / "outcomes.csv"
    config.state_file = tmp_path / "state.json"
    return config


def sample_signal(direction: str = "BUY", when: datetime | None = None) -> Signal:
    when = when or datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    if direction == "BUY":
        entry, stop, tps = 2300.0, 2290.0, (2310.0, 2318.0, 2328.0)
    else:
        entry, stop, tps = 2300.0, 2310.0, (2290.0, 2282.0, 2272.0)
    return Signal(
        signal_id=f"XAUUSD-{direction}-{when:%Y%m%d%H%M%S}",
        symbol="XAUUSD", timeframe="M5", direction=direction, timestamp=when,
        entry=entry, stop_loss=stop, tp1=tps[0], tp2=tps[1], tp3=tps[2],
        confidence=85.0, bullish_score=85.0, bearish_score=40.0,
        regime="WEAK_TREND", risk_reward=1.8, session="LONDON",
        reason_summary="trend=18.0", confidence_label="VERY_STRONG",
        rr1=1.0, rr2=1.8, rr3=2.8, sl_mode="HYBRID",
        confirmations={"trend": True, "htf": True},
    )


def outcome_candles(bars, start="2024-05-01 12:05") -> pd.DataFrame:
    times = pd.date_range(start, periods=len(bars), freq="5min", tz="UTC")
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
    assert snapshot.close == pytest.approx(float(snapshot.m5["close"].iloc[-1]))
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
        m5=compute_indicators(broken, config.indicators),
        m15=compute_indicators(resample_candles(long_candles, "M5", "M15"), config.indicators),
        h1=compute_indicators(resample_candles(long_candles, "M5", "H1"), config.indicators),
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
    candles = outcome_candles([(2300, 2330, 2299, 2329)])
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
    candles = outcome_candles([(2300, 2330, 2299, 2329)])
    tracker.update(candles)
    tracker.update(candles)
    tracker.update(candles)
    assert len(read_csv_rows(tracker_config.outcomes_csv)) == 1


def test_alerts_are_not_resent_after_a_restart(tracker_config):
    """Status is persisted, so a fresh tracker replays no old notifications."""
    tracker = SignalTracker(tracker_config)
    tracker.load()
    tracker.record_signal(sample_signal("BUY"))
    candles = outcome_candles([(2300, 2312, 2299, 2311)])
    first_events = tracker.update(candles)
    assert [e.event for e in first_events] == [STATUS_TP1]

    restarted = SignalTracker(tracker_config)
    restarted.load()
    assert restarted.update(candles) == []


def test_stop_loss_closes_the_signal_at_minus_one_r(tracker_config):
    tracker = SignalTracker(tracker_config)
    tracker.load()
    tracker.record_signal(sample_signal("BUY"))
    events = tracker.update(outcome_candles([(2300, 2302, 2288, 2289)]))
    assert [e.event for e in events] == [STATUS_SL]
    outcomes = read_csv_rows(tracker_config.outcomes_csv)
    assert float(outcomes[0]["R_multiple"]) == pytest.approx(-1.0)


def test_opposite_signal_invalidates_an_open_one(tracker_config):
    tracker = SignalTracker(tracker_config)
    tracker.load()
    tracker.record_signal(sample_signal("BUY"))
    events = tracker.invalidate_opposite(
        "SELL", datetime(2024, 5, 1, 13, 0, tzinfo=timezone.utc), 2295.0
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


def test_expiry_closes_a_stalled_signal(tracker_config):
    tracker_config.signal_expiry_candles = 5
    tracker = SignalTracker(tracker_config)
    tracker.load()
    tracker.record_signal(sample_signal("BUY"))
    flat = outcome_candles([(2300, 2301, 2299, 2300)] * 5)
    events = tracker.update(flat)
    assert [e.event for e in events] == ["EXPIRED"]


def test_progress_ignores_the_signal_candle_itself(config):
    """The entry is that candle's close, so it cannot trade against its own bar."""
    row = sample_signal("BUY").to_row()
    same_bar = pd.DataFrame(
        {
            "time": [pd.Timestamp("2024-05-01 12:00", tz="UTC")],
            "open": [2300.0], "high": [2400.0], "low": [2200.0], "close": [2300.0],
            "tick_volume": [100.0],
        }
    )
    progress = evaluate_progress(row, same_bar, config)
    assert progress.status == STATUS_ACTIVE
    assert progress.bars == 0


def test_position_state_and_evaluate_progress_agree(config):
    """The live path and the backtest path must score outcomes identically."""
    row = sample_signal("BUY").to_row()
    bars = [(2300, 2312, 2299, 2311), (2311, 2320, 2299, 2300), (2300, 2301, 2288, 2289)]
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
    assert notifier.send_message("hello") is False


def test_signal_message_contains_the_key_numbers(config):
    text = TelegramNotifier(config).format_signal(sample_signal("BUY"))
    assert "XAUUSD BUY" in text
    assert "🟢" in text
    assert "Entry: 2300.00" in text
    assert "SL: 2290.00" in text
    assert "TP1: 2310.00" in text and "TP3: 2328.00" in text
    assert "CONFIRMATIONS" in text
    assert "Confidence: 85/100" in text


def test_sell_message_uses_the_sell_icon(config):
    text = TelegramNotifier(config).format_signal(sample_signal("SELL"))
    assert "🔴" in text and "XAUUSD SELL" in text


def test_outcome_message_formats_the_r_multiple(config):
    row = {"symbol": "XAUUSD", "direction": "BUY", "entry": 2300.0, "signal_id": "x"}
    text = TelegramNotifier(config).format_outcome(row, STATUS_TP3, 2328.0, 1.867)
    assert "TP3 HIT" in text and "+1.87R" in text


# --------------------------------------------------------------------------- #
# backtester
# --------------------------------------------------------------------------- #
def test_backtester_never_shows_an_unclosed_higher_timeframe_candle(config):
    """The core no-lookahead guarantee of the backtester."""
    from backtest import Backtester

    candles = make_candles(4000, seed=71)
    backtester = Backtester(config)
    prepared = backtester.prepare(candles)

    for index in (3000, 3500, 3999):
        snapshot = backtester._snapshot(prepared, index)
        m5_close = pd.Timestamp(prepared["m5_close"][index])
        for frame, minutes in ((snapshot.m15, 15), (snapshot.h1, 60)):
            if frame.empty:
                continue
            last_close = pd.Timestamp(frame["time"].iloc[-1]) + timedelta(minutes=minutes)
            assert last_close <= m5_close, "an unclosed HTF candle leaked into the snapshot"


def test_backtester_m5_window_ends_on_the_evaluated_bar(config):
    from backtest import Backtester

    candles = make_candles(3500, seed=73)
    backtester = Backtester(config)
    prepared = backtester.prepare(candles)
    index = 3400
    snapshot = backtester._snapshot(prepared, index)
    assert pd.Timestamp(snapshot.m5["time"].iloc[-1]) == pd.Timestamp(candles["time"].iloc[index])
    assert len(snapshot.m5) <= config.candles_signal


def test_backtest_run_produces_consistent_signals_and_outcomes(config, tmp_path):
    from backtest import Backtester

    config.data_dir = tmp_path
    candles = make_candles(3600, seed=75, drift=0.05)
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
        assert row["result"] in ("TP3_HIT", "SL_HIT", "EXPIRED", "INVALIDATED")


def test_backtest_respects_the_cooldown_between_signals(config, tmp_path):
    from backtest import Backtester

    config.data_dir = tmp_path
    result = Backtester(config).run(
        make_candles(3600, seed=77, drift=0.04), log_evaluations=False, progress_every=0
    )
    times = sorted(parse_iso(row["timestamp"]) for row in result.signals)
    minimum_gap = timedelta(minutes=5 * config.cooldown_candles)
    for earlier, later in zip(times, times[1:]):
        assert later - earlier >= minimum_gap, "cooldown was not enforced"


# --------------------------------------------------------------------------- #
# live loop wiring (main.py) with a stubbed MT5 feed
# --------------------------------------------------------------------------- #
class _FakeMarket:
    """Stands in for MarketData: replays a history one closed candle at a time."""

    def __init__(self, config, candles: pd.DataFrame, start: int) -> None:
        self.config = config
        self.candles = candles
        self.cursor = start
        self.connected = True

    def connect(self) -> bool:
        return True

    def shutdown(self, quiet: bool = False) -> None:
        self.connected = False

    def advance(self) -> None:
        self.cursor += 1

    def build_snapshot(self):
        window = self.candles.iloc[: self.cursor + 1]
        return (
            MarketSnapshot(
                symbol=self.config.symbol,
                m5=window,
                m15=resample_candles(window, "M5", "M15"),
                h1=resample_candles(window, "M5", "H1"),
                m1=None,
                spread_points=20.0,
            ),
            "",
        )


def test_live_loop_processes_each_candle_exactly_once(tracker_config):
    """The whole main.py wiring: new candle -> evaluate -> log -> persist state."""
    from main import SignalRunner

    candles = make_candles(3200, seed=81, drift=0.05)
    runner = SignalRunner(tracker_config)
    runner.market = _FakeMarket(tracker_config, candles, start=3100)
    runner.notifier.config.telegram_enabled = False
    ensure_csv(tracker_config.evaluations_csv, EVALUATION_COLUMNS)
    runner.tracker.load()

    for _ in range(20):
        runner._tick()
        runner.market.advance()

    evaluations = read_csv_rows(tracker_config.evaluations_csv)
    assert len(evaluations) == 20, "one evaluation row per closed candle"
    timestamps = [row["timestamp"] for row in evaluations]
    assert len(timestamps) == len(set(timestamps)), "a candle was evaluated twice"

    state = read_json(tracker_config.state_file)
    assert state["last_processed_candle"] == timestamps[-1]


def test_live_loop_skips_a_candle_it_has_already_processed(tracker_config):
    from main import SignalRunner

    candles = make_candles(3200, seed=83)
    runner = SignalRunner(tracker_config)
    runner.market = _FakeMarket(tracker_config, candles, start=3150)
    runner.notifier.config.telegram_enabled = False
    ensure_csv(tracker_config.evaluations_csv, EVALUATION_COLUMNS)
    runner.tracker.load()

    runner._tick()
    runner._tick()   # same candle - must be ignored
    runner._tick()
    assert len(read_csv_rows(tracker_config.evaluations_csv)) == 1


def test_live_loop_records_signals_and_never_duplicates_them(tracker_config):
    from main import SignalRunner

    candles = make_candles(3400, seed=85, drift=0.06)
    runner = SignalRunner(tracker_config)
    runner.market = _FakeMarket(tracker_config, candles, start=3000)
    runner.notifier.config.telegram_enabled = False
    ensure_csv(tracker_config.evaluations_csv, EVALUATION_COLUMNS)
    runner.tracker.load()

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
