"""Two-market behaviour: configuration, isolation and Telegram switching.

The single question this module exists to answer is: **can activity on one
market ever change the other market's parameters, state, files or open paper
trades?**  Every test here is written so that a leak fails it, rather than
merely reporting a number that happens to match.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from config import Config
from src.markets import (
    BTCUSD,
    BTCUSD_CONFIG,
    DEFAULT_MARKET,
    MARKET_ORDER,
    MARKETS,
    XAUUSD,
    XAUUSD_CONFIG,
    get_market,
    is_supported,
    normalise_market,
)
from src.runtime_state import JsonStateStore, RuntimeState, effective_config
from src.signal_tracker import SignalTracker
from src.targets import compute_target_distances, target_floor
from src.telegram_control import TelegramController
from src.telegram_bot import TelegramNotifier
from src.filters import FilterInput, check_session
from tests.conftest import FakeNotifier, isolate, make_candles


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_exactly_the_two_specified_markets_are_registered():
    assert set(MARKETS) == {XAUUSD, BTCUSD}
    assert MARKET_ORDER == (XAUUSD, BTCUSD)
    assert DEFAULT_MARKET == XAUUSD


def test_symbol_lookup_is_forgiving_but_never_guesses():
    assert normalise_market("btcusd") == BTCUSD
    assert normalise_market(" xauusd ") == XAUUSD
    assert normalise_market(None) == DEFAULT_MARKET
    assert normalise_market("ETHUSD") == DEFAULT_MARKET, "unknown symbols fall back"
    assert is_supported("BTCUSD") and not is_supported("ETHUSD")
    with pytest.raises(KeyError):
        get_market("ETHUSD")


def test_a_third_market_can_be_added_without_touching_the_engine():
    """Extensibility is a registry entry, not an engine change (spec 2)."""
    from dataclasses import replace

    from src import markets as markets_module

    ethusd = replace(BTCUSD_CONFIG, symbol="ETHUSD", key="ethusd", display="ETHUSD")
    original_order = markets_module.MARKET_ORDER
    try:
        markets_module.register_market(ethusd)
        assert is_supported("ETHUSD")
        assert markets_module.MARKET_ORDER == original_order + ("ETHUSD",)
        view = Config().for_market("ETHUSD")
        assert view.symbol == "ETHUSD"
        assert view.market_dir.name == "ethusd"
        assert view.assumed_spread_points == BTCUSD_CONFIG.assumed_spread_points
    finally:
        MARKETS.pop("ETHUSD", None)
        markets_module.MARKET_ORDER = original_order
    assert not is_supported("ETHUSD")


# --------------------------------------------------------------------------- #
# XAUUSD is preserved (spec 3)
# --------------------------------------------------------------------------- #
def test_gold_keeps_its_existing_scalping_parameters():
    """The single-market build's numbers, unchanged."""
    market = XAUUSD_CONFIG
    assert market.threshold == 68.0
    assert market.tp_atr_multiples == (0.45, 1.00, 1.70)
    assert market.min_tp_pips == (1.0, 1.8, 3.0)
    assert market.max_tp3_pips == 12.0
    assert market.sl_atr_multiplier == 0.70
    assert market.assumed_spread_points == 20.0
    assert market.cooldown_candles == 10
    assert market.max_holding_candles == 15
    assert market.is_24h is False
    # gold's floors stay absolute: the percentage floors must not bite
    assert market.min_tp_pct == (0.0, 0.0, 0.0)
    assert market.max_tp3_pct == 0.0


def test_the_default_config_view_is_gold(config):
    assert config.symbol == XAUUSD
    assert config.base_threshold == XAUUSD_CONFIG.threshold
    assert config.market_dir.name == "xauusd"


# --------------------------------------------------------------------------- #
# BTCUSD does not inherit gold's numbers (spec 4, 5, 6)
# --------------------------------------------------------------------------- #
def test_bitcoin_does_not_copy_gold_absolute_distances():
    assert BTCUSD_CONFIG.min_tp_pips == (0.0, 0.0, 0.0)
    assert BTCUSD_CONFIG.max_tp3_pips == 0.0
    assert BTCUSD_CONFIG.min_tp_pct > (0.0, 0.0, 0.0)
    assert BTCUSD_CONFIG.max_tp3_pct > 0.0
    assert BTCUSD_CONFIG.assumed_spread_points != XAUUSD_CONFIG.assumed_spread_points
    assert BTCUSD_CONFIG.cost_model_name != XAUUSD_CONFIG.cost_model_name


def test_bitcoin_parameters_are_labelled_as_research_only():
    """Spec 4: nothing here may be presented as an optimised value."""
    note = BTCUSD_CONFIG.note.upper()
    assert "INITIAL RESEARCH PARAMETERS" in note
    assert "NOT PROVEN PROFITABLE" in note


def test_target_floors_scale_with_price_on_bitcoin(btc_config):
    """A percentage floor must give a bigger distance at a bigger price."""
    cheap = target_floor(btc_config, 0, 30_000.0)
    dear = target_floor(btc_config, 0, 90_000.0)
    assert dear == pytest.approx(cheap * 3.0)
    assert cheap == pytest.approx(30_000.0 * BTCUSD_CONFIG.min_tp_pct[0])


def test_gold_target_floors_ignore_price(config):
    """Gold keeps absolute floors, so its behaviour cannot drift."""
    assert target_floor(config, 0, 2300.0) == target_floor(config, 0, 4600.0)
    assert target_floor(config, 0, 2300.0) == pytest.approx(1.0 * config.pip_value)


def test_the_two_markets_produce_different_target_distances(config, btc_config):
    """Spec 5: never assume gold's TP distance is Bitcoin's."""
    gold = compute_target_distances(config, atr_value=0.40, cost_price=0.024, price=2300.0)
    btc = compute_target_distances(btc_config, atr_value=35.0, cost_price=14.0, price=60_000.0)
    assert btc[0] > gold[0] * 10, "Bitcoin's TP1 must not be a gold distance"
    assert len(gold) == len(btc) == 3


def test_the_same_atr_ladder_is_used_on_both_markets(config, btc_config):
    """The ATR model itself transfers - only the floors and costs differ."""
    assert config.tp_atr_multiples == btc_config.tp_atr_multiples
    gold = compute_target_distances(config, atr_value=1.0, cost_price=0.0, price=2300.0)
    btc = compute_target_distances(btc_config, atr_value=1.0, cost_price=0.0, price=1.0)
    # with the floors out of the way, the ladder is the same shape
    assert gold == pytest.approx(btc)


def test_cost_models_are_market_specific(config, btc_config):
    """Spec 6: each market pays its own spread, slippage and commission."""
    assert config.round_trip_cost(None) != btc_config.round_trip_cost(None)
    assert config.cost_model_name == "XAUUSD_RETAIL"
    assert btc_config.cost_model_name == BTCUSD_CONFIG.cost_model_name
    # the assumed BTC round trip must be worth more than a couple of dollars,
    # otherwise a "profitable" micro-move would be an artefact of a free trade
    assert btc_config.round_trip_cost(None) > 10.0


def test_raw_and_net_reward_stay_separate_on_both_markets(config, btc_config):
    from src.targets import build_targets

    for cfg, entry, atr in ((config, 2300.0, 0.40), (btc_config, 60_000.0, 35.0)):
        targets, reason = build_targets(
            _enriched(cfg, entry), cfg, "BUY", spread_points=cfg.assumed_spread_points
        )
        if targets is None:
            pytest.skip(f"no valid geometry for {cfg.symbol}: {reason}")
        assert targets.cost_r > 0, "a trade that costs nothing is not being modelled"
        assert targets.net_rr1 == pytest.approx(targets.rr1 - targets.cost_r, abs=0.011)
        assert targets.net_rr1 < targets.rr1


def _enriched(cfg, start_price):
    """Indicator-enriched candles at roughly ``start_price`` for ``cfg``."""
    from src.indicators import compute_indicators

    volatility = 0.16 if cfg.symbol == XAUUSD else 30.0
    candles = make_candles(600, seed=5, volatility=volatility, start_price=start_price)
    return compute_indicators(candles, cfg.indicators)


# --------------------------------------------------------------------------- #
# sessions (spec 8)
# --------------------------------------------------------------------------- #
def _filter_input(session: str):
    """The minimum FilterInput ``check_session`` reads."""
    return FilterInput(
        df=None, card=None, direction="BUY", regime="", volatility_band="",
        session=session, spread_points=0.0, candle_time=_utc(12), htf_alignment="",
        timeframe_minutes=1,
    )


def test_bitcoin_is_never_blocked_by_the_session_filter(btc_config):
    btc_config.allowed_sessions = ("LONDON",)
    for session in ("LONDON", "ASIAN", "NEW_YORK", "OFF_SESSION"):
        assert check_session(_filter_input(session), btc_config) is None, session


def test_gold_still_honours_its_session_filter(config):
    config.allowed_sessions = ("LONDON",)
    assert check_session(_filter_input("LONDON"), config) is None
    assert check_session(_filter_input("ASIAN"), config) is not None, (
        "the gold session filter stopped working"
    )


def test_the_session_label_is_still_recorded_for_bitcoin(btc_config):
    """Spec 8: 24/7 does not mean the session is unknown, only unfiltered."""
    from src.utils import detect_session

    labels = {
        detect_session(_utc(hour), btc_config.sessions, btc_config.session_priority)
        for hour in range(24)
    }
    assert len(labels) > 1, "session labels must still be computed for analysis"


def _utc(hour: int):
    from datetime import datetime, timezone

    return datetime(2024, 5, 1, hour, 30, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# storage separation (spec 13)
# --------------------------------------------------------------------------- #
def test_each_market_writes_to_its_own_directory(tmp_path):
    gold = isolate(Config().for_market(XAUUSD), tmp_path)
    btc = isolate(Config().for_market(BTCUSD), tmp_path)

    assert gold.market_dir != btc.market_dir
    for attribute in ("signals_csv", "outcomes_csv", "evaluations_csv", "state_file"):
        assert getattr(gold, attribute) != getattr(btc, attribute), attribute
    assert gold.market_dir.name == "xauusd" and btc.market_dir.name == "btcusd"
    # one shared global state file, one per-market file each
    assert gold.global_state_file == btc.global_state_file == tmp_path / "state.json"


def test_records_identify_their_own_market(tmp_path):
    """Spec 13/14: every row says which market and timeframe it came from."""
    from src.signal_tracker import SIGNAL_COLUMNS, OUTCOME_COLUMNS, read_csv_rows

    for column in ("symbol", "timeframe", "timestamp"):
        assert column in SIGNAL_COLUMNS, column
    for column in ("symbol", "timeframe", "timestamp", "estimated_slippage"):
        assert column in OUTCOME_COLUMNS, column

    gold = isolate(Config().for_market(XAUUSD), tmp_path)
    tracker = SignalTracker(gold, FakeNotifier())
    tracker.load()
    tracker.record_signal(_signal(XAUUSD))

    rows = read_csv_rows(gold.signals_csv)
    assert [row["symbol"] for row in rows] == [XAUUSD]
    assert [row["timeframe"] for row in rows] == ["M1"]
    assert rows[0]["estimated_slippage"] != ""


def _signal(symbol: str, direction: str = "BUY"):
    from datetime import datetime, timezone

    from src.signal_engine import Signal

    market = get_market(symbol)
    price = 2300.0 if symbol == XAUUSD else 60_000.0
    step = 0.30 if symbol == XAUUSD else 20.0
    sign = 1.0 if direction == "BUY" else -1.0
    when = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    return Signal(
        signal_id=f"{symbol}-{direction}-20240501120000",
        symbol=symbol, timeframe="M1", direction=direction, timestamp=when,
        entry=price,
        stop_loss=price - sign * step,
        tp1=price + sign * step, tp2=price + sign * step * 2, tp3=price + sign * step * 3,
        confidence=71.0, bullish_score=71.0, bearish_score=20.0,
        regime="WEAK_TREND", risk_reward=2.0, session="LONDON",
        reason_summary="fixture", threshold_used=market.threshold,
        spread_points=market.assumed_spread_points,
        estimated_slippage=market.slippage_points_entry + market.slippage_points_exit,
    )


# --------------------------------------------------------------------------- #
# state isolation (spec 10, 16, 17, 18)
# --------------------------------------------------------------------------- #
@pytest.fixture
def runtime(tmp_path):
    return RuntimeState.load(isolate(Config().for_market(XAUUSD), tmp_path))


def test_a_threshold_change_on_one_market_never_moves_the_other(runtime):
    before = runtime.market(BTCUSD).active_threshold()
    runtime.market(XAUUSD).set_threshold(80)

    assert runtime.market(XAUUSD).active_threshold() == 80
    assert runtime.market(BTCUSD).active_threshold() == before
    assert runtime.market(BTCUSD).has_threshold_override() is False


def test_every_editable_setting_is_isolated(runtime):
    gold, btc = runtime.market(XAUUSD), runtime.market(BTCUSD)
    gold.set_cooldown(45)
    gold.set_max_holding(99)
    gold.set_min_rr(3.3)
    gold.set_sessions(["LONDON"])

    assert btc.cooldown_candles is None
    assert btc.max_holding_candles is None
    assert btc.min_tp2_rr is None
    assert btc.allowed_sessions is None
    assert btc.describe()["cooldown_candles"] == BTCUSD_CONFIG.cooldown_candles
    assert btc.describe()["max_holding_candles"] == BTCUSD_CONFIG.max_holding_candles


def test_the_candle_marker_is_per_market(runtime):
    runtime.market(XAUUSD).mark_candle_processed("2024-05-01T12:00:00+00:00")
    assert runtime.market(XAUUSD).last_processed_candle() is not None
    assert runtime.market(BTCUSD).last_processed_candle() is None, (
        "gold's candle marker leaked into Bitcoin - switching would skip candles"
    )


def test_state_survives_a_restart_per_market(tmp_path):
    config = isolate(Config().for_market(XAUUSD), tmp_path)
    runtime = RuntimeState.load(config)
    runtime.market(XAUUSD).set_threshold(75)
    runtime.market(BTCUSD).set_threshold(60)
    runtime.set_active_market(BTCUSD)

    restored = RuntimeState.load(config, JsonStateStore(config.global_state_file))
    assert restored.market(XAUUSD).active_threshold() == 75
    assert restored.market(BTCUSD).active_threshold() == 60
    assert restored.active_market == BTCUSD, "spec 20: the active market is persisted"


def test_switching_markets_changes_nothing_but_the_selection(runtime):
    """Spec 10, the critical one: a switch must not reset the other market."""
    gold = runtime.market(XAUUSD)
    gold.set_threshold(74)
    gold.set_cooldown(33)
    gold.mark_candle_processed("2024-05-01T12:00:00+00:00")
    snapshot = dict(gold.describe())
    marker = gold.last_processed_candle()

    runtime.set_active_market(BTCUSD)
    runtime.set_active_market(XAUUSD)
    runtime.set_active_market(BTCUSD)

    assert runtime.market(XAUUSD).describe() == snapshot
    assert runtime.market(XAUUSD).last_processed_candle() == marker
    assert runtime.active_market == BTCUSD


def test_only_the_active_market_is_evaluated(runtime):
    assert runtime.evaluation_markets() == (XAUUSD,)
    runtime.set_active_market(BTCUSD)
    assert runtime.evaluation_markets() == (BTCUSD,)


def test_effective_config_folds_the_right_market(tmp_path):
    config = isolate(Config().for_market(XAUUSD), tmp_path)
    runtime = RuntimeState.load(config)
    runtime.market(BTCUSD).set_threshold(59)

    gold_view = effective_config(config, runtime, XAUUSD)
    btc_view = effective_config(config, runtime, BTCUSD)

    assert gold_view.symbol == XAUUSD
    assert gold_view.base_threshold == XAUUSD_CONFIG.threshold
    assert btc_view.symbol == BTCUSD
    assert btc_view.base_threshold == 59
    assert btc_view.assumed_spread_points == BTCUSD_CONFIG.assumed_spread_points
    assert btc_view.signals_csv != gold_view.signals_csv


# --------------------------------------------------------------------------- #
# tracker / cooldown / duplicate isolation (spec 15, 16, 17)
# --------------------------------------------------------------------------- #
def _trackers(tmp_path):
    gold = isolate(Config().for_market(XAUUSD), tmp_path)
    btc = isolate(Config().for_market(BTCUSD), tmp_path)
    gold_tracker = SignalTracker(gold, FakeNotifier())
    btc_tracker = SignalTracker(btc, FakeNotifier())
    gold_tracker.load()
    btc_tracker.load()
    return (gold, gold_tracker), (btc, btc_tracker)


def test_a_signal_on_one_market_is_invisible_to_the_other(tmp_path):
    (gold, gold_tracker), (btc, btc_tracker) = _trackers(tmp_path)
    gold_tracker.record_signal(_signal(XAUUSD))

    assert len(gold_tracker.active_signals()) == 1
    assert btc_tracker.active_signals() == []

    reloaded = SignalTracker(btc, FakeNotifier())
    reloaded.load()
    assert reloaded.signals == [], "gold's signal appeared in Bitcoin's file"
    assert not btc.signals_csv.read_text(encoding="utf-8").count("XAUUSD")


def test_the_cooldown_is_counted_per_market(tmp_path):
    from datetime import timedelta

    (gold, gold_tracker), (btc, btc_tracker) = _trackers(tmp_path)
    signal = _signal(XAUUSD)
    gold_tracker.record_signal(signal)

    just_after = signal.timestamp + timedelta(minutes=1)
    gold_gate = gold_tracker.gate_state(just_after, "M1")
    btc_gate = btc_tracker.gate_state(just_after, "M1")

    assert gold_gate.last_signal_time == signal.timestamp
    assert gold_gate.signals_today == 1
    assert btc_gate.last_signal_time is None, "a gold signal put Bitcoin on cooldown"
    assert btc_gate.signals_today == 0
    # and the cooldown windows themselves are the markets' own
    assert gold.cooldown_candles == XAUUSD_CONFIG.cooldown_candles
    assert btc.cooldown_candles == BTCUSD_CONFIG.cooldown_candles


def test_duplicate_prevention_is_per_market(tmp_path):
    """The same minute may signal on both markets; neither may signal twice."""
    (gold, gold_tracker), (btc, btc_tracker) = _trackers(tmp_path)
    gold_signal = _signal(XAUUSD)
    btc_signal = _signal(BTCUSD)

    gold_tracker.record_signal(gold_signal)
    btc_tracker.record_signal(btc_signal)
    assert len(gold_tracker.signals) == 1
    assert len(btc_tracker.signals) == 1
    assert gold_signal.signal_id != btc_signal.signal_id

    # each market's own file knows only its own ids
    gold_ids = {row["signal_id"] for row in gold_tracker.signals}
    btc_ids = {row["signal_id"] for row in btc_tracker.signals}
    assert gold_ids.isdisjoint(btc_ids)
    assert gold_signal.signal_id in gold_ids and gold_signal.signal_id not in btc_ids

    # the id itself carries the market, so the same minute cannot collide
    assert gold_signal.signal_id.startswith(XAUUSD)
    assert btc_signal.signal_id.startswith(BTCUSD)


def test_outcome_tracking_is_scoped_to_its_own_market_config(tmp_path):
    """Each tracker carries its own market's geometry, costs and timeout."""
    (gold, gold_tracker), (btc, btc_tracker) = _trackers(tmp_path)

    assert gold_tracker.config.symbol == XAUUSD
    assert btc_tracker.config.symbol == BTCUSD
    assert gold_tracker.config.signals_csv != btc_tracker.config.signals_csv
    assert gold_tracker.config.point_value == XAUUSD_CONFIG.point_value
    assert btc_tracker.config.pip_value == BTCUSD_CONFIG.pip_value
    assert gold.max_holding_candles == XAUUSD_CONFIG.max_holding_candles
    assert btc.max_holding_candles == BTCUSD_CONFIG.max_holding_candles


def test_a_gold_trade_is_never_advanced_by_bitcoin_candles(tmp_path):
    """Spec 15, at the level that actually routes the candles: the live loop.

    Bitcoin's prices are three orders of magnitude away from gold's, so if a
    frame ever crossed markets every gold target would "hit" on the first bar.
    This records what each tracker was handed and asserts it stayed in range.
    """
    seen = {}

    runner, gold_config, btc_config = _runner(tmp_path)
    for symbol, slot in runner.slots.items():
        original = slot.tracker.update

        def spy(frame, ticks=None, timeframe=None, _symbol=symbol, _original=original):
            seen.setdefault(_symbol, []).append(
                (float(frame["low"].min()), float(frame["high"].max()))
            )
            return _original(frame, ticks, timeframe=timeframe)

        slot.tracker.update = spy

    runner.slot(XAUUSD).tracker.record_signal(_gold_signal_at(runner, gold_config))
    runner.control.handle_callback("mkt:BTCUSD")
    for _ in range(30):
        runner._tick()
        runner.market.advance()

    assert seen.get(XAUUSD), "the open gold trade was not tracked after the switch"
    for low, high in seen[XAUUSD]:
        assert 1_000 < low <= high < 10_000, (
            f"gold was tracked against a frame priced {low}-{high}"
        )
    for low, high in seen.get(BTCUSD, []):
        assert high > 10_000, "Bitcoin was tracked against gold prices"


# --------------------------------------------------------------------------- #
# Telegram (spec 9, 11, 12, 19, 21)
# --------------------------------------------------------------------------- #
@pytest.fixture
def controller(tmp_path):
    config = isolate(Config().for_market(XAUUSD), tmp_path)
    runtime = RuntimeState.load(config)
    return TelegramController(config, runtime, FakeNotifier())


def test_the_panel_offers_both_markets_and_marks_the_active_one(controller):
    data = [b["callback_data"] for row in controller.main_keyboard() for b in row]
    text = [b["text"] for row in controller.main_keyboard() for b in row]
    assert data[:2] == ["mkt:XAUUSD", "mkt:BTCUSD"]
    assert any("🥇" in label for label in text)
    assert any("₿" in label for label in text)
    assert text[0].startswith("●"), "the active market is not marked"
    assert text[1].startswith("○")


def test_switching_market_is_reported_prominently(controller):
    text, _keyboard, toast = controller.handle_callback("mkt:BTCUSD")
    assert "BTCUSD" in toast
    assert "Market: ₿ BTCUSD" in text
    assert controller.runtime.active_market == BTCUSD

    text, _keyboard, _toast = controller.handle_callback("mkt:XAUUSD")
    assert "Market: 🥇 XAUUSD" in text


def test_an_unknown_market_button_does_not_change_the_selection(controller):
    controller.handle_callback("mkt:DOGECOIN")
    assert controller.runtime.active_market == XAUUSD


def test_settings_act_on_the_active_market_only(controller):
    """Spec 19: pressing a settings button must not touch the other market."""
    runtime = controller.runtime
    controller.handle_callback("mkt:BTCUSD")
    text = controller.render_settings()
    assert "⚙️ BTCUSD SETTINGS" in text

    before = runtime.market(XAUUSD).active_threshold()
    controller.handle_callback("thr:5")
    assert runtime.market(BTCUSD).active_threshold() == BTCUSD_CONFIG.threshold + 5
    assert runtime.market(XAUUSD).active_threshold() == before, (
        "a BTC threshold press moved gold"
    )

    controller.handle_callback("mkt:XAUUSD")
    assert "⚙️ XAUUSD SETTINGS" in controller.render_settings()
    controller.handle_callback("thr:-5")
    assert runtime.market(XAUUSD).active_threshold() == before - 5
    assert runtime.market(BTCUSD).active_threshold() == BTCUSD_CONFIG.threshold + 5


def test_the_threshold_panel_names_the_market_it_edits(controller):
    controller.handle_callback("mkt:BTCUSD")
    text, _keyboard, _toast = controller.handle_callback("menu:threshold")
    assert "BTCUSD" in text
    assert "does not affect the other market" in text


def test_thresholds_stay_inside_the_configured_bounds(controller):
    """Spec 18: sensible hard bounds, on both markets."""
    config = controller.config
    for symbol in MARKET_ORDER:
        controller.handle_callback(f"mkt:{symbol}")
        for _ in range(40):
            controller.handle_callback("thr:5")
        assert controller.runtime.market(symbol).active_threshold() == config.max_threshold
        for _ in range(60):
            controller.handle_callback("thr:-5")
        assert controller.runtime.market(symbol).active_threshold() == config.min_threshold


def test_analysis_follows_the_selected_market(controller):
    """Spec 11."""
    asked = []

    class Engine:
        def analyze_now(self, symbol=None):
            asked.append(symbol)
            return None

    controller.engine = Engine()
    controller.handle_callback("mkt:BTCUSD")
    text, _keyboard, _toast = controller.handle_callback("view:analysis")
    assert asked == [BTCUSD]
    assert "BTCUSD" in text


def test_performance_is_per_market_and_combined_is_opt_in(controller):
    """Spec 12: markets are never merged by default."""
    requested = []

    class Report:
        total_signals = 0
        total_evaluations = 0

    def loader(symbol=None):
        requested.append(symbol)
        return Report()

    controller._report_loader = loader
    data = [b["callback_data"] for row in controller.performance_keyboard() for b in row]
    assert data[:3] == ["perf:XAUUSD", "perf:BTCUSD", "perf:COMBINED"]

    controller.handle_callback("view:performance")
    assert requested[-1] == XAUUSD, "the default view merged the markets"

    text, _keyboard, _toast = controller.handle_callback("perf:COMBINED")
    assert requested[-1] == "COMBINED"
    assert "COMBINED" in text


def test_the_panel_reports_the_other_markets_open_trades(controller):
    class Engine:
        def open_signals(self, symbol=None):
            return 2 if symbol == BTCUSD else 0

    controller.engine = Engine()
    text = controller.render_panel()
    assert "Also tracking" in text
    assert "₿ BTCUSD 2 open" in text


def test_no_button_can_place_an_order(controller):
    """Spec 21: the control surface is paper-only, end to end."""
    keyboards = [
        controller.main_keyboard(),
        controller.settings_keyboard(),
        controller.threshold_keyboard(),
        controller.performance_keyboard(),
    ]
    forbidden = ("buy", "sell", "order", "trade:", "execute", "leverage", "position:")
    for keyboard in keyboards:
        for button in [b for row in keyboard for b in row]:
            assert not any(word in button["callback_data"].lower() for word in forbidden), (
                button["callback_data"]
            )
    assert "PAPER TEST ONLY" in controller.render_panel()


# --------------------------------------------------------------------------- #
# signal cards
# --------------------------------------------------------------------------- #
def test_the_card_is_rendered_in_the_signals_own_market(config):
    """A BTC card must not be printed with gold's icon or pip unit."""
    notifier = TelegramNotifier(config)          # notifier configured for gold
    text = notifier.format_signal(_signal(BTCUSD))
    assert "₿ BTCUSD M1 SCALP" in text
    assert "$)" in text, "BTC distances are quoted in dollars"

    gold_text = notifier.format_signal(_signal(XAUUSD))
    assert "🥇 XAUUSD M1 SCALP" in gold_text
    assert "p)" in gold_text


# --------------------------------------------------------------------------- #
# live loop (spec 7, 10, 15)
# --------------------------------------------------------------------------- #
def _runner(tmp_path, gold_start=2500, btc_start=2500):
    from main import SignalRunner
    from tests.conftest import FakeMarket, MultiMarket

    config = isolate(Config(), tmp_path, XAUUSD)
    runner = SignalRunner(config)
    gold_config = config.for_market(XAUUSD)
    btc_config = config.for_market(BTCUSD)
    runner.market = MultiMarket(
        {
            XAUUSD: FakeMarket(
                gold_config,
                make_candles(2700, seed=61, drift=0.04, volatility=0.16, start_price=2300.0),
                start=gold_start,
            ),
            BTCUSD: FakeMarket(
                btc_config,
                make_candles(2700, seed=67, drift=8.0, volatility=30.0, start_price=60_000.0),
                start=btc_start,
                spread_points=BTCUSD_CONFIG.assumed_spread_points,
            ),
        }
    )
    runner.notifier = FakeNotifier()
    for slot in runner.slots.values():
        slot.tracker.notifier = runner.notifier
    runner.control.notifier = runner.notifier
    return runner, gold_config, btc_config


def test_only_the_active_market_generates_signals(tmp_path):
    from src.signal_tracker import read_csv_rows

    runner, gold_config, btc_config = _runner(tmp_path)
    for _ in range(30):
        runner._tick()
        runner.market.advance()

    assert read_csv_rows(gold_config.evaluations_csv), "gold was not evaluated"
    assert read_csv_rows(btc_config.evaluations_csv) == [], (
        "the unselected market was evaluated"
    )


def test_switching_markets_moves_evaluation_and_keeps_the_other_intact(tmp_path):
    from src.signal_tracker import read_csv_rows

    runner, gold_config, btc_config = _runner(tmp_path)
    for _ in range(20):
        runner._tick()
        runner.market.advance()

    gold_rows = read_csv_rows(gold_config.evaluations_csv)
    gold_marker = runner.runtime.market(XAUUSD).last_processed_candle()
    assert gold_rows and gold_marker is not None

    runner.control.handle_callback("mkt:BTCUSD")
    for _ in range(20):
        runner._tick()
        runner.market.advance()

    assert read_csv_rows(btc_config.evaluations_csv), "BTC was not evaluated after switching"
    assert read_csv_rows(gold_config.evaluations_csv) == gold_rows, (
        "gold's evaluation file changed after switching away"
    )
    assert runner.runtime.market(XAUUSD).last_processed_candle() == gold_marker


def test_each_market_evaluates_each_closed_candle_exactly_once(tmp_path):
    from src.signal_tracker import read_csv_rows

    runner, gold_config, btc_config = _runner(tmp_path)
    for _ in range(15):
        runner._tick()
        runner._tick()          # same candle twice - must be ignored
        runner.market.advance()

    runner.control.handle_callback("mkt:BTCUSD")
    for _ in range(15):
        runner._tick()
        runner._tick()
        runner.market.advance()

    for path in (gold_config.evaluations_csv, btc_config.evaluations_csv):
        stamps = [row["timestamp"] for row in read_csv_rows(path)]
        assert len(stamps) == len(set(stamps)) == 15, path


def test_an_open_trade_keeps_being_tracked_after_switching_away(tmp_path):
    """Spec 10, the exact scenario the brief calls out."""
    runner, gold_config, _btc = _runner(tmp_path)
    gold_slot = runner.slot(XAUUSD)
    row = gold_slot.tracker.record_signal(_gold_signal_at(runner, gold_config))
    assert gold_slot.has_open_signals()

    runner.control.handle_callback("mkt:BTCUSD")
    assert runner.runtime.active_market == BTCUSD

    for _ in range(40):
        runner._tick()
        runner.market.advance()

    assert (XAUUSD, "M1") in runner.market.fetches, (
        "gold candles were not fetched while its trade was open"
    )
    tracked = gold_slot.tracker.signals[0]
    assert tracked["signal_id"] == row["signal_id"]
    assert float(tracked.get("mfe_r") or 0) != 0 or tracked["status"] != "ACTIVE", (
        "the open gold trade stopped being scored after the switch"
    )


def _gold_signal_at(runner, gold_config):
    """A gold signal anchored to the live fixture's current price."""
    snapshot, _reason = runner.market.build_snapshot(gold_config)
    price = float(snapshot.close)
    signal = _signal(XAUUSD)
    step = 0.30
    signal.entry = price
    signal.stop_loss = price - step
    signal.tp1, signal.tp2, signal.tp3 = price + step, price + 2 * step, price + 3 * step
    signal.timestamp = snapshot.candle_time
    return signal


def test_bitcoin_evaluates_closed_candles_only(btc_config):
    """Spec 7: the anti-lookahead guarantees hold for the new market too."""
    from src.signal_engine import SignalEngine
    from tests.conftest import build_snapshot

    candles = make_candles(1200, seed=71, volatility=30.0, start_price=60_000.0)
    snapshot = build_snapshot(btc_config, candles, spread_points=1000.0)
    evaluation = SignalEngine(btc_config).evaluate(snapshot)

    expected = pd.Timestamp(candles["time"].iloc[-1]).to_pydatetime()
    assert evaluation.timestamp.replace(tzinfo=None) == expected.replace(tzinfo=None), (
        "the signal is stamped with wall clock rather than the closed candle"
    )
    assert evaluation.symbol == BTCUSD
    assert evaluation.timeframe == "M1"
    assert not hasattr(snapshot, "forming_candle")
    # nothing after the evaluated candle is reachable from the snapshot
    assert snapshot.signal_df["time"].max() == candles["time"].iloc[-1]


def test_a_bitcoin_signal_carries_every_required_field(tmp_path):
    """Spec 14, for BTCUSD: no field is left blank or fabricated."""
    from src.signal_tracker import read_csv_rows

    btc = isolate(Config().for_market(BTCUSD), tmp_path)
    tracker = SignalTracker(btc, FakeNotifier())
    tracker.load()
    tracker.record_signal(_signal(BTCUSD))
    row = read_csv_rows(btc.signals_csv)[0]

    assert row["symbol"] == BTCUSD
    assert row["timeframe"] == "M1"
    for column in ("timestamp", "direction", "entry", "tp1", "tp2", "tp3", "sl",
                   "score", "bullish_score", "bearish_score", "threshold_used",
                   "regime", "session", "spread_points", "estimated_slippage"):
        assert row[column] != "", column
    assert float(row["spread_points"]) == BTCUSD_CONFIG.assumed_spread_points
    assert float(row["estimated_slippage"]) == (
        BTCUSD_CONFIG.slippage_points_entry + BTCUSD_CONFIG.slippage_points_exit
    )


def test_the_global_state_file_records_the_active_market(tmp_path):
    runner, _gold, _btc = _runner(tmp_path)
    runner.control.handle_callback("mkt:BTCUSD")
    saved = json.loads(runner.config.global_state_file.read_text(encoding="utf-8"))
    assert saved["runtime"]["active_market"] == BTCUSD
