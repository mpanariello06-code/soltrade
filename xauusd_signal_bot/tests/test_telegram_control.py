"""Telegram control panel: buttons, run state, timeframe switching, safety and
duplicate prevention across restarts."""

from __future__ import annotations

import pandas as pd
import pytest

from src.markets import BTCUSD, XAUUSD
from src.runtime_state import JsonStateStore, RuntimeState, effective_config
from src.telegram_control import TelegramController
from src.timeframes import STATUS_PAUSED, STATUS_RUNNING, STATUS_STOPPED
from src.utils import read_json
from tests.conftest import FakeMarket, FakeNotifier, make_candles


@pytest.fixture
def runtime(isolated_config):
    return RuntimeState.load(isolated_config)


@pytest.fixture
def notifier():
    return FakeNotifier()


@pytest.fixture
def controller(isolated_config, runtime, notifier):
    return TelegramController(isolated_config, runtime, notifier)


def button_data(keyboard):
    """Flatten a keyboard into the list of callback payloads it offers."""
    return [button["callback_data"] for row in keyboard for button in row]


def button_text(keyboard):
    return [button["text"] for row in keyboard for button in row]


# --------------------------------------------------------------------------- #
# panel rendering
# --------------------------------------------------------------------------- #
def test_main_panel_matches_the_specified_layout(controller):
    text = controller.render_panel()
    for expected in ("⚡ M1 SCALPER", f"Market: 🥇 {XAUUSD}", "Status:", "Mode: SCALPING",
                     "Timeframe: M1",
                     "Threshold:", "Signals today:", "Open signals:", "Paper Net R:"):
        assert expected in text, expected


def test_main_keyboard_has_only_the_scalping_controls(controller):
    data = button_data(controller.main_keyboard())
    assert data == [
        f"mkt:{XAUUSD}", f"mkt:{BTCUSD}",
        "view:analysis", "view:performance", "view:trades",
        "run:pause",                     # single toggle: the engine is RUNNING
        "demo:confirm",                  # OFF, so the button asks to confirm ON
        "menu:settings", "panel:refresh",
    ]


def test_no_mode_or_timeframe_buttons_remain(controller):
    """The selector was removed: this build is M1 SCALPING only."""
    keyboards = [
        controller.main_keyboard(),
        controller.threshold_keyboard(),
        controller.settings_keyboard(),
        controller.performance_keyboard(),
    ]
    for keyboard in keyboards:
        for data in button_data(keyboard):
            assert not data.startswith("mode:"), data
            assert not data.startswith("tf:"), data


def test_panel_carries_the_paper_disclaimer(controller):
    assert "SIGNAL ONLY - no orders are placed." in controller.render_panel()


def test_panel_reports_open_signals_and_net_r(isolated_config, runtime, notifier):
    from performance import build_report

    signals = pd.DataFrame(
        [{"signal_id": "a", "direction": "BUY", "timestamp": "2024-05-01T10:00:00+00:00",
          "status": "TP3_HIT", "mode": "SCALPING", "timeframe": "M1", "score": 71,
          "confidence": 71, "regime": "WEAK_TREND", "session": "LONDON"}]
    )
    outcomes = pd.DataFrame(
        [{"signal_id": "a", "R_multiple": 1.74, "net_r": 1.14, "cost_r": 0.6,
          "tp_hits": 3, "result": "TP3_HIT", "duration": 5}]
    )

    class Engine:
        def open_signals(self):
            return 2

        def signals_today(self):
            return 7

    controller = TelegramController(
        isolated_config, runtime, notifier, engine=Engine(),
        report_loader=lambda: build_report(signals, outcomes, isolated_config),
    )
    text = controller.render_panel()
    assert "Signals today: 7" in text
    assert "Open signals: 2" in text
    assert "Paper Net R: +1.14" in text


# --------------------------------------------------------------------------- #
# run state
# --------------------------------------------------------------------------- #
def test_start_pause_stop_change_the_run_state(controller, runtime):
    controller.handle_callback("run:pause")
    assert runtime.describe()["status"] == STATUS_PAUSED
    assert not runtime.is_running

    controller.handle_callback("run:start")
    assert runtime.describe()["status"] == STATUS_RUNNING
    assert runtime.is_running

    controller.handle_callback("run:stop")
    assert runtime.describe()["status"] == STATUS_STOPPED
    assert runtime.is_stopped
def test_threshold_menu_offers_the_documented_steps(controller):
    _text, keyboard, _toast = controller.handle_callback("menu:threshold")
    data = button_data(keyboard)
    # BACK returns to SETTINGS, which is where the threshold menu is reached from.
    for expected in ("thr:-5", "thr:-1", "thr:1", "thr:5", "thr:reset", "menu:settings"):
        assert expected in data, expected


def test_threshold_buttons_adjust_and_report_the_new_value(controller, runtime):
    base = runtime.active.active_threshold()
    text, _keyboard, toast = controller.handle_callback("thr:5")
    assert runtime.active.active_threshold() == base + 5
    assert f"{base + 5:.0f}" in toast
    assert f"Current threshold: {base + 5:.0f}" in text

    controller.handle_callback("thr:-1")
    assert runtime.active.active_threshold() == base + 4


def test_threshold_buttons_respect_the_limits(controller, runtime, isolated_config):
    for _ in range(30):
        controller.handle_callback("thr:5")
    assert runtime.active.active_threshold() == isolated_config.max_threshold
    for _ in range(40):
        controller.handle_callback("thr:-5")
    assert runtime.active.active_threshold() == isolated_config.min_threshold


def test_threshold_reset_button_restores_the_default(controller, runtime, isolated_config):
    controller.handle_callback("thr:5")
    controller.handle_callback("thr:reset")
    assert runtime.active.active_threshold() == isolated_config.base_threshold
    assert not runtime.active.has_threshold_override()


def test_threshold_change_persists(isolated_config, controller):
    controller.handle_callback("thr:-5")
    restored = RuntimeState.load(
        isolated_config, JsonStateStore(isolated_config.global_state_file)
    )
    assert restored.active.active_threshold() == isolated_config.base_threshold - 5


# --------------------------------------------------------------------------- #
# settings menu
# --------------------------------------------------------------------------- #
def test_settings_menu_lists_the_editable_settings(controller):
    text, keyboard, _toast = controller.handle_callback("menu:settings")
    for expected in ("Mode: SCALPING", "Timeframe: M1", "Threshold:",
                     "Max holding period:", "Cooldown:", "Minimum R:R",
                     "Session filter:", "Near-signal alerts:", "Costs assumed"):
        assert expected in text, expected
    data = button_data(keyboard)
    for expected in ("set:cooldown", "set:rr", "set:hold", "set:session", "set:near", "nav:main"):
        assert expected in data, expected


def test_max_holding_can_be_changed_from_settings(controller, runtime, isolated_config):
    before = runtime.describe()["max_holding_candles"]
    controller.handle_callback("set:hold")
    after = runtime.describe()["max_holding_candles"]
    assert after != before
    assert after in isolated_config.holding_choices
    assert effective_config(isolated_config, runtime).max_holding_candles == after


def test_settings_buttons_cycle_their_values(controller, runtime, isolated_config):
    controller.handle_callback("set:cooldown")
    assert runtime.describe()["cooldown_candles"] in isolated_config.cooldown_choices

    controller.handle_callback("set:rr")
    assert runtime.describe()["min_tp2_rr"] in isolated_config.min_rr_choices

    controller.handle_callback("set:session")
    assert runtime.describe()["allowed_sessions"][0] in isolated_config.session_choices

    before = runtime.describe()["near_signal_alerts"]
    controller.handle_callback("set:near")
    assert runtime.describe()["near_signal_alerts"] is not before


def test_settings_changes_reach_the_engine_config(isolated_config, controller, runtime):
    controller.handle_callback("set:cooldown")
    controller.handle_callback("set:rr")
    view = effective_config(isolated_config, runtime)
    assert view.cooldown_candles == runtime.describe()["cooldown_candles"]
    assert view.min_tp2_rr == runtime.describe()["min_tp2_rr"]
    # a user-set cooldown must also move the stricter same-direction cooldown
    assert view.same_direction_cooldown_candles == view.cooldown_candles * 2


# --------------------------------------------------------------------------- #
# configuration safety (spec section 16)
# --------------------------------------------------------------------------- #
def test_no_button_can_reach_credentials_or_paths(controller):
    """Every callback the panel offers must map to an editable setting."""
    keyboards = [
        controller.main_keyboard(),
        controller.threshold_keyboard(),
        controller.settings_keyboard(),
        controller.performance_keyboard(),
    ]
    forbidden = ("token", "password", "login", "server", "symbol", "path", "csv", "secret")
    for keyboard in keyboards:
        for data in button_data(keyboard):
            assert not any(word in data.lower() for word in forbidden), data


def test_credentials_are_never_rendered_into_a_message(isolated_config, runtime, notifier):
    isolated_config.mt5_password = "hunter2-secret"
    isolated_config.telegram_bot_token = "123456:SECRET-TOKEN"
    isolated_config.mt5_login = 99887766
    controller = TelegramController(isolated_config, runtime, notifier)
    rendered = "\n".join(
        [
            controller.render_panel(),
            controller.render_settings(),
            controller.render_threshold_panel(),
            controller.render_performance(),
            controller.render_analysis(),
        ]
    )
    assert "hunter2-secret" not in rendered
    assert "SECRET-TOKEN" not in rendered
    assert "99887766" not in rendered


def test_unknown_callback_falls_back_to_the_main_panel(controller):
    text, keyboard, _toast = controller.handle_callback("danger:rm-rf")
    assert "M1 SCALPER" in text
    assert "panel:refresh" in button_data(keyboard)


def test_updates_from_another_chat_are_ignored(controller, runtime):
    controller.process_update(
        {
            "callback_query": {
                "id": "1",
                "data": "run:pause",
                "message": {"message_id": 5, "chat": {"id": 999999}},
            }
        }
    )
    assert runtime.describe()["status"] == STATUS_RUNNING, "an unauthorised chat paused the engine"


def test_updates_from_the_configured_chat_are_applied(controller, runtime, notifier):
    controller.process_update(
        {
            "callback_query": {
                "id": "1",
                "data": "run:pause",
                "message": {"message_id": 5, "chat": {"id": 4242}},
            }
        }
    )
    assert runtime.describe()["status"] == STATUS_PAUSED
    assert notifier.answers, "the button press was not acknowledged"
    assert notifier.edits[-1]["id"] == 5, "the panel was not edited in place"


def test_slash_command_summons_the_panel(controller, notifier):
    controller.process_update({"message": {"chat": {"id": 4242}, "text": "/panel"}})
    assert notifier.messages
    assert "M1 SCALPER" in notifier.messages[-1]["text"]


def test_slash_command_from_another_chat_is_ignored(controller, notifier):
    controller.process_update({"message": {"chat": {"id": 1}, "text": "/panel"}})
    assert not notifier.messages


# --------------------------------------------------------------------------- #
# analysis / performance views
# --------------------------------------------------------------------------- #
def test_analysis_view_reports_missing_data_gracefully(controller):
    text = controller.render_analysis()
    assert "ANALYSIS" in text
    assert "No market data" in text


def test_analysis_view_renders_the_component_breakdown(isolated_config, runtime, notifier):
    from src.signal_engine import SignalEngine

    candles = make_candles(2500, seed=101, drift=0.02)
    market = FakeMarket(isolated_config, candles, start=len(candles) - 1)
    engine = SignalEngine(isolated_config)

    class Runner:
        def analyze_now(self):
            view = effective_config(isolated_config, runtime)
            snapshot, _ = market.build_snapshot(view)
            return engine.evaluate(snapshot, config=view)

        def signals_today(self):
            return 3

        def connection_state(self):
            return "CONNECTED"

    controller = TelegramController(isolated_config, runtime, notifier, engine=Runner())
    text = controller.render_analysis()
    for expected in ("ANALYSIS", "Bullish Score:", "Bearish Score:", "Regime:",
                     "Momentum:", "Price Action:", "Liquidity:", "Structure:",
                     "S/R:", "Trend:", "Context:", "Volatility:", "Volume:",
                     "Threshold:", "ATR:", "Spread:", "Cost:", "Decision:"):
        assert expected in text, expected
    assert "Closed M1 candles only" in text
    assert "Signals today: 3" in controller.render_panel()
def test_performance_view_reports_no_data_without_csvs(controller):
    assert "No signals recorded yet" in controller.render_performance()


def test_performance_views_come_from_the_csv_data(isolated_config, runtime, notifier):
    from performance import build_report

    signals = pd.DataFrame(
        [
            {"signal_id": "a", "direction": "BUY", "timestamp": "2024-05-01T10:00:00+00:00",
             "status": "TP3_HIT", "mode": "SCALPING", "timeframe": "M1", "score": 56,
             "confidence": 56, "regime": "WEAK_TREND", "session": "LONDON"},
            {"signal_id": "b", "direction": "SELL", "timestamp": "2024-05-01T11:00:00+00:00",
             "status": "SL_HIT", "mode": "SCALPING", "timeframe": "M1", "score": 61,
             "confidence": 61, "regime": "RANGE", "session": "ASIAN"},
        ]
    )
    outcomes = pd.DataFrame(
        [
            {"signal_id": "a", "R_multiple": 1.87, "net_r": 1.27, "cost_r": 0.6,
             "tp_hits": 3, "result": "TP3_HIT", "duration": 6},
            {"signal_id": "b", "R_multiple": -1.0, "net_r": -1.6, "cost_r": 0.6,
             "tp_hits": 0, "result": "SL_HIT", "duration": 4},
        ]
    )
    controller = TelegramController(
        isolated_config, runtime, notifier,
        report_loader=lambda: build_report(signals, outcomes, isolated_config),
    )
    summary = controller.render_performance()
    assert "Signals: 2" in summary
    assert "Avg NET R" in summary and "Avg RAW R" in summary
    assert "Timed out:" in summary
    assert "Best Regime:" in summary and "WEAK_TREND" in summary
    assert "RANGE" in summary

    assert "BY SCORE BAND" in controller.render_performance("score")
    assert "BY REGIME" in controller.render_performance("regime")
    assert "BY SESSION" in controller.render_performance("session")
    assert "BY OUTCOME" in controller.render_performance("outcome")

    data = button_data(controller.performance_keyboard())
    for expected in ("perfv:score", "perfv:regime", "perfv:session", "perfv:hour",
                     "perfv:outcome"):
        assert expected in data


def test_a_failing_report_loader_does_not_break_the_panel(isolated_config, runtime, notifier):
    def boom():
        raise RuntimeError("csv exploded")

    controller = TelegramController(isolated_config, runtime, notifier, report_loader=boom)
    assert "No signals recorded yet" in controller.render_performance()


# --------------------------------------------------------------------------- #
# duplicate prevention across timeframes and restarts
# --------------------------------------------------------------------------- #
def test_processed_candle_is_tracked_for_m1(isolated_config, runtime):
    runtime.active.mark_candle_processed("2024-05-01T12:00:00+00:00")
    assert runtime.active.last_processed_candle().hour == 12
    stored = read_json(isolated_config.state_file)["last_processed_candles"]
    assert set(stored) == {"M1"}, "M1 is the only timeframe this build evaluates"


def test_processed_candles_survive_a_restart(isolated_config, runtime):
    runtime.active.mark_candle_processed("2024-05-01T12:00:00+00:00")
    restored = RuntimeState.load(
        isolated_config, JsonStateStore(isolated_config.global_state_file)
    )
    assert restored.active.last_processed_candle() is not None


def test_legacy_single_candle_marker_is_migrated(isolated_config):
    """State written by the single-market build must still be honoured.

    That build kept one ``data/state.json``; everything in it was gold's, so it
    is adopted into XAUUSD's own file rather than being dropped.
    """
    store = JsonStateStore(isolated_config.global_state_file)
    store.update(last_processed_candle="2024-05-01T12:00:00+00:00")
    runtime = RuntimeState.load(isolated_config, store)
    assert runtime.market("XAUUSD").last_processed_candle() is not None
    assert runtime.market("BTCUSD").last_processed_candle() is None, (
        "legacy gold state must never be applied to Bitcoin"
    )


def test_state_store_is_shared_without_clobbering(isolated_config):
    """The tracker and the runtime must not overwrite each other's keys."""
    from src.signal_tracker import SignalTracker

    runtime = RuntimeState.load(
        isolated_config, JsonStateStore(isolated_config.global_state_file)
    )
    # One writer per file: the tracker shares the market's store rather than
    # opening a second handle on the same path, which is what used to clobber.
    tracker = SignalTracker(isolated_config, store=runtime.active.store)
    tracker.load()

    tracker.save_state(last_signal_id="sig-1")
    runtime.active.set_threshold(77)
    runtime.active.mark_candle_processed("2024-05-01T12:00:00+00:00")

    saved = read_json(isolated_config.state_file)
    assert saved["last_signal_id"] == "sig-1"
    assert saved["runtime"]["threshold_override"] == 77.0
    assert saved["last_processed_candles"]["M1"] == "2024-05-01T12:00:00+00:00"


def test_poller_skips_updates_queued_before_startup(isolated_config, runtime, notifier):
    """A restart must not replay button presses made while the engine was down."""
    notifier.updates = [
        {"update_id": 7, "callback_query": {"id": "x", "data": "run:stop",
                                            "message": {"message_id": 1, "chat": {"id": 4242}}}}
    ]
    controller = TelegramController(isolated_config, runtime, notifier)
    controller._drain_backlog()
    assert controller._offset == 8
    assert runtime.describe()["status"] == STATUS_RUNNING, "a stale STOP was replayed"
