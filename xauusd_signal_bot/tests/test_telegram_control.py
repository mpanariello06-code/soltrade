"""Telegram control panel: buttons, run state, timeframe switching, safety and
duplicate prevention across restarts."""

from __future__ import annotations

import pandas as pd
import pytest

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
def test_main_panel_shows_the_required_fields(controller):
    text = controller.render_panel()
    for expected in ("XAUUSD SIGNAL ENGINE", "Status:", "Mode:", "Signal TF:",
                     "Threshold:", "Signals today:"):
        assert expected in text, expected


def test_main_keyboard_offers_every_documented_control(controller):
    data = button_data(controller.main_keyboard())
    for expected in (
        "run:start", "run:pause", "run:stop",
        "mode:RESEARCH", "mode:STANDARD", "mode:CONSERVATIVE",
        "tf:M1", "tf:M5", "tf:M15", "tf:M30", "tf:H1", "tf:H4",
        "menu:threshold", "view:analysis", "view:performance",
        "menu:settings", "panel:refresh",
    ):
        assert expected in data, expected


def test_active_mode_and_timeframe_are_marked_in_the_keyboard(controller, runtime):
    runtime.set_mode("RESEARCH")
    runtime.set_timeframe("M15")
    labels = button_text(controller.main_keyboard())
    assert any(label.startswith("●") and "RESEARCH" in label for label in labels)
    assert "●M15" in labels
    assert "●M5" not in labels


def test_research_mode_panel_carries_the_disclaimer(controller, runtime):
    runtime.set_mode("RESEARCH")
    assert "RESEARCH MODE" in controller.render_panel()
    runtime.set_mode("STANDARD")
    assert "RESEARCH MODE" not in controller.render_panel()


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


def test_run_state_survives_a_restart(isolated_config, controller, runtime):
    controller.handle_callback("run:pause")
    restored = RuntimeState.load(isolated_config, JsonStateStore(isolated_config.state_file))
    assert restored.describe()["status"] == STATUS_PAUSED


def test_pause_does_not_disconnect_or_reset_anything(controller, runtime):
    runtime.set_mode("RESEARCH")
    runtime.set_timeframe("M15")
    controller.handle_callback("run:pause")
    state = runtime.describe()
    assert state["status"] == STATUS_PAUSED
    assert state["mode"] == "RESEARCH"
    assert state["signal_timeframe"] == "M15"


# --------------------------------------------------------------------------- #
# mode / timeframe buttons
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["RESEARCH", "STANDARD", "CONSERVATIVE"])
def test_mode_buttons_switch_mode_and_threshold(controller, runtime, mode):
    _text, _keyboard, toast = controller.handle_callback(f"mode:{mode}")
    assert runtime.describe()["mode"] == mode
    assert mode in toast
    assert runtime.active_threshold() == controller.config.threshold_for(mode, "M5")


@pytest.mark.parametrize(
    "timeframe,confirmation",
    [("M1", "M5+M15"), ("M5", "M15+H1"), ("M15", "M30+H1"),
     ("M30", "H1+H4"), ("H1", "H4"), ("H4", "NONE")],
)
def test_timeframe_buttons_move_the_whole_hierarchy(controller, runtime, timeframe, confirmation):
    controller.handle_callback(f"tf:{timeframe}")
    assert runtime.describe()["signal_timeframe"] == timeframe
    assert runtime.confirmation_label() == confirmation


def test_timeframe_change_clears_the_candle_cache(isolated_config, runtime, notifier):
    class Engine:
        def __init__(self):
            self.calls = []

        def on_timeframe_changed(self, previous, current):
            self.calls.append((previous, current))

    engine = Engine()
    controller = TelegramController(isolated_config, runtime, notifier, engine=engine)
    controller.handle_callback("tf:M15")
    assert engine.calls == [("M5", "M15")]

    controller.handle_callback("tf:M15")  # same timeframe again
    assert engine.calls == [("M5", "M15")], "no cache flush when nothing changed"


def test_timeframe_and_mode_survive_a_restart(isolated_config, controller):
    controller.handle_callback("mode:RESEARCH")
    controller.handle_callback("tf:M30")
    restored = RuntimeState.load(isolated_config, JsonStateStore(isolated_config.state_file))
    assert restored.describe()["mode"] == "RESEARCH"
    assert restored.describe()["signal_timeframe"] == "M30"
    assert restored.confirmation_label() == "H1+H4"


# --------------------------------------------------------------------------- #
# threshold controls
# --------------------------------------------------------------------------- #
def test_threshold_menu_offers_the_documented_steps(controller):
    _text, keyboard, _toast = controller.handle_callback("menu:threshold")
    data = button_data(keyboard)
    for expected in ("thr:-5", "thr:-1", "thr:1", "thr:5", "thr:reset", "nav:main"):
        assert expected in data, expected


def test_threshold_buttons_adjust_and_report_the_new_value(controller, runtime):
    runtime.set_mode("RESEARCH")
    text, _keyboard, toast = controller.handle_callback("thr:5")
    assert runtime.active_threshold() == 55.0
    assert "55" in toast
    assert "Current threshold: 55" in text

    controller.handle_callback("thr:-1")
    assert runtime.active_threshold() == 54.0


def test_threshold_buttons_respect_the_limits(controller, runtime, isolated_config):
    for _ in range(30):
        controller.handle_callback("thr:5")
    assert runtime.active_threshold() == isolated_config.max_threshold
    for _ in range(40):
        controller.handle_callback("thr:-5")
    assert runtime.active_threshold() == isolated_config.min_threshold


def test_threshold_reset_button_restores_the_default(controller, runtime):
    controller.handle_callback("thr:5")
    controller.handle_callback("thr:reset")
    assert runtime.active_threshold() == 72.0
    assert not runtime.has_threshold_override()


def test_threshold_change_persists(isolated_config, controller):
    controller.handle_callback("thr:-5")
    restored = RuntimeState.load(isolated_config, JsonStateStore(isolated_config.state_file))
    assert restored.active_threshold() == 67.0


# --------------------------------------------------------------------------- #
# settings menu
# --------------------------------------------------------------------------- #
def test_settings_menu_lists_the_editable_settings(controller):
    text, keyboard, _toast = controller.handle_callback("menu:settings")
    for expected in ("Mode:", "Signal timeframe:", "Threshold:", "Cooldown:",
                     "Minimum R:R", "Session filter:", "Near-signal alerts:"):
        assert expected in text, expected
    data = button_data(keyboard)
    for expected in ("set:cooldown", "set:rr", "set:session", "set:near", "nav:main"):
        assert expected in data, expected


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
    assert "XAUUSD SIGNAL ENGINE" in text
    assert "panel:refresh" in button_data(keyboard)


def test_updates_from_another_chat_are_ignored(controller, runtime):
    controller.process_update(
        {
            "callback_query": {
                "id": "1",
                "data": "mode:RESEARCH",
                "message": {"message_id": 5, "chat": {"id": 999999}},
            }
        }
    )
    assert runtime.describe()["mode"] == "STANDARD", "an unauthorised chat changed the mode"


def test_updates_from_the_configured_chat_are_applied(controller, runtime, notifier):
    controller.process_update(
        {
            "callback_query": {
                "id": "1",
                "data": "mode:RESEARCH",
                "message": {"message_id": 5, "chat": {"id": 4242}},
            }
        }
    )
    assert runtime.describe()["mode"] == "RESEARCH"
    assert notifier.answers, "the button press was not acknowledged"
    assert notifier.edits[-1]["id"] == 5, "the panel was not edited in place"


def test_slash_command_summons_the_panel(controller, notifier):
    controller.process_update({"message": {"chat": {"id": 4242}, "text": "/panel"}})
    assert notifier.messages
    assert "XAUUSD SIGNAL ENGINE" in notifier.messages[-1]["text"]


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

    candles = make_candles(4200, seed=101, drift=0.05)
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
                     "Trend:", "HTF:", "Momentum:", "Structure:", "Liquidity:",
                     "S/R:", "Volume:", "Volatility:", "Price Action:",
                     "Threshold:", "Decision:"):
        assert expected in text, expected
    assert "Closed candles only" in text
    assert "Signals today: 3" in controller.render_panel()


def test_analysis_marks_a_non_applicable_component(isolated_config, runtime, notifier):
    """On H4 there is no confirmation timeframe, so HTF must read n/a."""
    from src.signal_engine import SignalEngine

    runtime.set_timeframe("H4")
    candles = make_candles(30000, seed=103)
    market = FakeMarket(isolated_config, candles, start=len(candles) - 1)
    engine = SignalEngine(isolated_config)

    class Runner:
        def analyze_now(self):
            view = effective_config(isolated_config, runtime)
            snapshot, reason = market.build_snapshot(view)
            return engine.evaluate(snapshot, config=view) if snapshot else None

    controller = TelegramController(isolated_config, runtime, notifier, engine=Runner())
    text = controller.render_analysis()
    assert "HTF:" in text
    assert "n/a" in text


def test_performance_view_reports_no_data_without_csvs(controller):
    assert "No signals recorded yet" in controller.render_performance()


def test_performance_views_come_from_the_csv_data(isolated_config, runtime, notifier):
    from performance import build_report

    signals = pd.DataFrame(
        [
            {"signal_id": "a", "direction": "BUY", "timestamp": "2024-05-01T10:00:00+00:00",
             "status": "TP3_HIT", "mode": "RESEARCH", "timeframe": "M5", "score": 56,
             "confidence": 56, "regime": "WEAK_TREND", "session": "LONDON"},
            {"signal_id": "b", "direction": "SELL", "timestamp": "2024-05-01T11:00:00+00:00",
             "status": "SL_HIT", "mode": "RESEARCH", "timeframe": "M15", "score": 61,
             "confidence": 61, "regime": "RANGE", "session": "ASIAN"},
        ]
    )
    outcomes = pd.DataFrame(
        [
            {"signal_id": "a", "R_multiple": 1.87, "tp_hits": 3, "result": "TP3_HIT", "duration": 90},
            {"signal_id": "b", "R_multiple": -1.0, "tp_hits": 0, "result": "SL_HIT", "duration": 45},
        ]
    )
    controller = TelegramController(
        isolated_config, runtime, notifier,
        report_loader=lambda: build_report(signals, outcomes, isolated_config),
    )
    summary = controller.render_performance()
    assert "Signals: 2" in summary
    assert "Win Rate: 50.0%" in summary
    assert "Best Regime:" in summary and "WEAK_TREND" in summary
    assert "RANGE" in summary

    assert "BY TIMEFRAME" in controller.render_performance("timeframe")
    assert "BY SCORE BAND" in controller.render_performance("score")
    assert "BY REGIME" in controller.render_performance("regime")
    assert "BY MODE" in controller.render_performance("mode")

    data = button_data(controller.performance_keyboard())
    for expected in ("perf:timeframe", "perf:score", "perf:regime", "perf:mode"):
        assert expected in data


def test_a_failing_report_loader_does_not_break_the_panel(isolated_config, runtime, notifier):
    def boom():
        raise RuntimeError("csv exploded")

    controller = TelegramController(isolated_config, runtime, notifier, report_loader=boom)
    assert "No signals recorded yet" in controller.render_performance()


# --------------------------------------------------------------------------- #
# duplicate prevention across timeframes and restarts
# --------------------------------------------------------------------------- #
def test_processed_candles_are_tracked_per_timeframe(isolated_config, runtime):
    runtime.mark_candle_processed("M5", "2024-05-01T12:00:00+00:00")
    runtime.mark_candle_processed("M15", "2024-05-01T11:45:00+00:00")

    assert runtime.last_processed_candle("M5").hour == 12
    assert runtime.last_processed_candle("M15").minute == 45
    assert runtime.last_processed_candle("H1") is None

    stored = read_json(isolated_config.state_file)["last_processed_candles"]
    assert set(stored) == {"M5", "M15"}


def test_processed_candles_survive_a_restart(isolated_config, runtime):
    runtime.mark_candle_processed("M5", "2024-05-01T12:00:00+00:00")
    restored = RuntimeState.load(isolated_config, JsonStateStore(isolated_config.state_file))
    assert restored.last_processed_candle("M5") is not None


def test_legacy_single_candle_marker_is_migrated(isolated_config):
    """State written by the pre-multi-timeframe version must still be honoured."""
    store = JsonStateStore(isolated_config.state_file)
    store.update(last_processed_candle="2024-05-01T12:00:00+00:00")
    runtime = RuntimeState.load(isolated_config, store)
    assert runtime.last_processed_candle("M5") is not None


def test_state_store_is_shared_without_clobbering(isolated_config):
    """The tracker and the runtime must not overwrite each other's keys."""
    from src.signal_tracker import SignalTracker

    store = JsonStateStore(isolated_config.state_file)
    runtime = RuntimeState.load(isolated_config, store)
    tracker = SignalTracker(isolated_config, store=store)
    tracker.load()

    tracker.save_state(last_signal_id="sig-1")
    runtime.set_mode("RESEARCH")
    runtime.mark_candle_processed("M5", "2024-05-01T12:00:00+00:00")

    saved = read_json(isolated_config.state_file)
    assert saved["last_signal_id"] == "sig-1"
    assert saved["runtime"]["mode"] == "RESEARCH"
    assert saved["last_processed_candles"]["M5"] == "2024-05-01T12:00:00+00:00"


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
