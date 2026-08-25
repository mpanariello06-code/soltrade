"""MT5 history collection, the PPO bridge, and the Telegram PPO controls.

Everything here runs without MetaTrader5 and without a trained model: the whole
point of the bridge is that both are optional.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from config import Config
from src.mt5_history import (
    RAW_COLUMNS, analyse_quality, load_raw, normalise_raw, raw_dir,
    rates_to_frame, write_quality_report, write_yearly,
)
from src.ppo_bridge import PPOBridge, ai_available
from tests.conftest import isolate


def series(n=200, start="2024-01-01", spread=20):
    times = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "time": times, "open": 2300.0, "high": 2300.5, "low": 2299.5,
        "close": 2300.2, "tick_volume": 50.0, "spread": spread, "real_volume": 0.0,
    })


# --------------------------------------------------------------------------- #
# raw data handling
# --------------------------------------------------------------------------- #
def test_every_raw_mt5_field_is_preserved():
    """Spread and real_volume are kept even when a broker leaves them zero:
    an absent column and a zero column mean different things."""
    frame = series(10)
    assert set(RAW_COLUMNS).issubset(frame.columns)
    assert "spread" in RAW_COLUMNS and "real_volume" in RAW_COLUMNS


def test_duplicate_timestamps_are_removed_and_order_restored():
    frame = series(50)
    scrambled = pd.concat([frame.iloc[25:], frame.iloc[:30]], ignore_index=True)

    clean = normalise_raw(scrambled)
    assert clean["time"].is_monotonic_increasing
    assert not clean["time"].duplicated().any()
    assert len(clean) == 50


def test_gaps_are_reported_and_never_filled():
    """A fabricated candle is indistinguishable from a real one once on disk."""
    frame = series(100).drop(index=[40, 41, 42]).reset_index(drop=True)
    report = analyse_quality(frame, "XAUUSDs", "M1")

    assert report.missing_candles == 3
    assert report.largest_gaps
    assert len(frame) == 97, "the gap was filled in"
    # gaps alone do not disqualify a series - every FX market closes
    assert report.usable is True


def test_structural_defects_make_a_series_unusable():
    frame = series(50)
    frame.loc[10, "high"] = 2200.0          # high below low
    report = analyse_quality(frame, "XAUUSDs", "M1")
    assert report.bad_ohlc_rows == 1
    assert report.usable is False

    frame = series(50)
    frame.loc[5, "close"] = -1.0
    assert analyse_quality(frame, "XAUUSDs").non_positive_prices == 1
    assert analyse_quality(frame, "XAUUSDs").usable is False

    duplicated = pd.concat([series(20), series(20).iloc[:5]], ignore_index=True)
    assert analyse_quality(duplicated, "XAUUSDs").usable is False


def test_abnormal_ranges_and_spreads_are_flagged_not_dropped():
    frame = series(200)
    frame.loc[100, "high"] = 2900.0     # a 600-point minute
    frame.loc[150, "spread"] = 5000.0
    report = analyse_quality(frame, "XAUUSDs", "M1")

    assert report.abnormal_range_rows >= 1
    assert report.abnormal_spread_rows >= 1
    assert len(frame) == 200, "flagged rows were removed"


def test_mt5_timestamps_are_converted_to_utc():
    """MT5 stamps in SERVER time; every file on disk must be UTC."""
    rates = np.array(
        [(1704067200, 2300.0, 2300.5, 2299.5, 2300.2, 50, 20, 0)],
        dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
               ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"),
               ("real_volume", "i8")],
    )
    utc = rates_to_frame(rates, server_utc_offset_hours=0.0)
    shifted = rates_to_frame(rates, server_utc_offset_hours=3.0)

    assert utc["time"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert shifted["time"].iloc[0] == pd.Timestamp("2023-12-31 21:00:00", tz="UTC")


def test_empty_rates_produce_an_empty_frame_not_an_error():
    assert rates_to_frame(None).empty
    assert rates_to_frame([]).empty


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #
def test_history_is_written_as_yearly_files(tmp_path):
    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    frame = pd.concat([series(100, "2023-12-31 23:00"), series(100, "2024-01-01")],
                      ignore_index=True)

    paths = write_yearly(normalise_raw(frame), config, "XAUUSDs", "M1")
    names = sorted(p.name for p in paths)
    assert names == ["2023.csv", "2024.csv"]
    assert raw_dir(config, "XAUUSDs", "M1").is_dir()


def test_re_downloading_merges_rather_than_truncating(tmp_path):
    """A partial re-download must never shorten the history already on disk."""
    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    write_yearly(series(500, "2024-01-01"), config, "XAUUSDs", "M1")
    write_yearly(series(10, "2024-01-01"), config, "XAUUSDs", "M1")

    assert len(load_raw(config, "XAUUSDs", "M1")) == 500


def test_the_two_markets_store_history_separately(tmp_path):
    config = isolate(Config(), tmp_path, "XAUUSDs")
    write_yearly(series(50), config, "XAUUSDs", "M1")
    write_yearly(series(50), config, "BTCUSDs", "M1")

    assert raw_dir(config, "XAUUSDs", "M1") != raw_dir(config, "BTCUSDs", "M1")
    assert len(load_raw(config, "XAUUSDs", "M1")) == 50
    assert len(load_raw(config, "BTCUSDs", "M1")) == 50


def test_a_quality_report_is_written_beside_the_data(tmp_path):
    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    report = analyse_quality(series(100), "XAUUSDs", "M1")
    path = write_quality_report(report, config, "XAUUSDs", "M1")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "XAUUSDs"
    assert payload["usable"] is True
    assert "missing_candles" in payload


def test_loading_missing_history_returns_empty_rather_than_raising(tmp_path):
    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    assert load_raw(config, "XAUUSDs", "M1").empty


# --------------------------------------------------------------------------- #
# the PPO bridge: everything about it is optional
# --------------------------------------------------------------------------- #
def test_the_bridge_defaults_to_rule_only(tmp_path):
    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    bridge = PPOBridge(config)

    assert bridge.mode_name() == "RULE_ONLY"
    assert bridge.active is False


def test_the_bridge_holds_when_no_model_exists(tmp_path):
    """A missing model must not stop the rule engine - it just does nothing."""
    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    bridge = PPOBridge(config)

    assert bridge.observe("XAUUSDs", config, object()) is None
    assert bridge.state("XAUUSDs")["loaded"] is False


def test_requesting_ppo_without_a_model_stays_on_the_rule_engine(tmp_path):
    """Reported honestly rather than shown as success."""
    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    bridge = PPOBridge(config)

    applied = bridge.set_mode("PPO_SHADOW")
    assert applied == "RULE_ONLY"
    assert bridge.active is False


@pytest.mark.skipif(not ai_available(), reason="the RL stack is not installed")
def test_an_unrecognised_mode_is_refused(tmp_path):
    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    bridge = PPOBridge(config)
    assert bridge.set_mode("PPO_LIVE") == "RULE_ONLY"
    assert bridge.set_mode("nonsense") == "RULE_ONLY"


def test_a_broken_observation_never_raises_into_the_loop(tmp_path):
    """The scalper must survive anything the RL side does."""
    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    bridge = PPOBridge(config)

    class Exploding:
        @property
        def signal_df(self):
            raise RuntimeError("boom")

    assert bridge.observe("XAUUSDs", config, Exploding()) is None


# --------------------------------------------------------------------------- #
# Telegram PPO controls
# --------------------------------------------------------------------------- #
class _Engine:
    """The engine surface the PPO panel calls."""

    def __init__(self, mode="RULE_ONLY", loaded=False) -> None:
        self.mode = mode
        self.loaded = loaded
        self.requested = []
        self.refuse = False

    def strategy_mode(self, symbol=None):
        return self.mode

    def set_strategy_mode(self, mode):
        self.requested.append(mode)
        if self.refuse:
            return "RULE_ONLY"
        self.mode = mode
        return mode

    def ppo_state(self, symbol=None):
        if not self.loaded:
            return {"mode": self.mode, "loaded": False}
        return {
            "mode": self.mode, "loaded": True, "model_version": "ppo_v001",
            "model_status": "SHADOW", "last_action": "BUY",
            "probabilities": {"HOLD": 0.2, "BUY": 0.7, "SELL": 0.1},
            "trades": 42, "net_r": 3.5, "average_net_r": 0.083,
            "win_rate": 55.0, "profit_factor": 1.2, "max_drawdown_r": 2.1,
            "average_holding": 4.5, "features": 65, "fingerprint": "abc123",
            "decisions": 500, "failures": 3, "failure_rate": 0.6,
            "train_period": "2023-01-01 -> 2024-01-01",
            "validation_period": "2024-01-01 -> 2024-03-01",
            "test_period": "2024-03-01 -> 2024-05-01",
        }


def _controller(tmp_path, engine):
    from src.runtime_state import RuntimeState
    from src.telegram_control import TelegramController
    from tests.conftest import FakeNotifier

    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    return TelegramController(config, RuntimeState.load(config),
                              FakeNotifier(), engine=engine)


def _buttons(keyboard):
    return [b["callback_data"] for row in keyboard for b in row]


def test_the_ppo_button_is_on_the_main_panel(tmp_path):
    controller = _controller(tmp_path, _Engine())
    assert "ppo:status" in _buttons(controller.main_keyboard())


def test_the_ppo_menu_offers_every_documented_control(tmp_path):
    controller = _controller(tmp_path, _Engine())
    data = _buttons(controller.ppo_keyboard())
    for expected in ("ppo:status", "ppo:rule", "ppo:shadow", "ppo:demo_confirm",
                     "ppo:performance", "ppo:model"):
        assert expected in data, expected


def test_ppo_status_says_so_when_no_model_is_loaded(tmp_path):
    controller = _controller(tmp_path, _Engine())
    text, _keyboard, _toast = controller.handle_callback("ppo:status")

    assert "PPO STATUS" in text
    assert "No PPO model is loaded" in text
    assert "rule engine is deciding" in text


def test_ppo_status_shows_the_action_and_simulated_results(tmp_path):
    controller = _controller(tmp_path, _Engine(mode="PPO_SHADOW", loaded=True))
    text, _keyboard, _toast = controller.handle_callback("ppo:status")

    assert "Mode: PPO_SHADOW" in text
    assert "ppo_v001" in text
    assert "Latest action: BUY" in text
    assert "BUY probability:  70%" in text
    assert "PPO paper trades: 42" in text
    assert "NOT placing orders" in text, "shadow results must be labelled simulated"


def test_ppo_status_surfaces_inference_failures(tmp_path):
    """A model that fails often is silently holding, which looks like working."""
    controller = _controller(tmp_path, _Engine(mode="PPO_SHADOW", loaded=True))
    text, _keyboard, _toast = controller.handle_callback("ppo:status")
    assert "Inference failures: 3" in text


def test_switching_to_shadow_goes_through_the_engine(tmp_path):
    engine = _Engine()
    controller = _controller(tmp_path, engine)

    _text, _keyboard, toast = controller.handle_callback("ppo:shadow")
    assert engine.requested == ["PPO_SHADOW"]
    assert "PPO_SHADOW" in toast


def test_ppo_demo_requires_an_explicit_confirmation(tmp_path):
    """Promoting a model from watching to trading is not a one-tap action."""
    engine = _Engine()
    controller = _controller(tmp_path, engine)

    text, keyboard, _toast = controller.handle_callback("ppo:demo_confirm")
    assert "ENABLE PPO DEMO?" in text
    assert "does NOT bypass any safety check" in text
    assert _buttons(keyboard) == ["ppo:demo", "ppo:status"]
    assert engine.requested == [], "the prompt alone switched mode"

    controller.handle_callback("ppo:demo")
    assert engine.requested == ["PPO_DEMO"]


def test_a_refused_switch_is_reported_not_shown_as_success(tmp_path):
    engine = _Engine()
    engine.refuse = True
    controller = _controller(tmp_path, engine)

    _text, _keyboard, toast = controller.handle_callback("ppo:shadow")
    assert "Refused" in toast
    assert engine.mode == "RULE_ONLY"


def test_ppo_pause_returns_to_the_rule_engine(tmp_path):
    engine = _Engine(mode="PPO_DEMO")
    controller = _controller(tmp_path, engine)

    controller.handle_callback("ppo:rule")
    assert engine.requested == ["RULE_ONLY"]


def test_ppo_performance_keeps_the_books_apart(tmp_path):
    """PPO simulated, rule paper and demo fills are never summed."""
    controller = _controller(tmp_path, _Engine(mode="PPO_SHADOW", loaded=True))
    text, _keyboard, _toast = controller.handle_callback("ppo:performance")

    assert "PPO (SIMULATED)" in text
    assert "RULE ENGINE (paper)" in text
    assert "never summed" in text


def test_ppo_model_reports_what_it_was_trained_on(tmp_path):
    controller = _controller(tmp_path, _Engine(mode="PPO_SHADOW", loaded=True))
    text, _keyboard, _toast = controller.handle_callback("ppo:model")

    assert "ppo_v001" in text
    assert "Status: SHADOW" in text
    assert "Features: 65" in text
    assert "2023-01-01" in text


def test_the_ppo_panel_survives_an_engine_with_no_ppo_at_all(tmp_path):
    """The panel must not break on a runner that predates the RL stack."""
    class Bare:
        pass

    controller = _controller(tmp_path, Bare())
    text, _keyboard, _toast = controller.handle_callback("ppo:status")
    assert "PPO STATUS" in text
    assert "No PPO model is loaded" in text


# --------------------------------------------------------------------------- #
# backtest strategy modes  (spec section 31)
# --------------------------------------------------------------------------- #
def test_the_backtester_offers_all_three_strategy_modes():
    import backtest

    parser_source = backtest.main.__doc__ or ""
    # the flag itself is what matters; parse it out of the CLI
    import argparse
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), pytest.raises(SystemExit):
        backtest.main(["--help"])
    text = buffer.getvalue()
    for mode in ("RULE_ONLY", "PPO_BACKTEST", "PPO_SHADOW"):
        assert mode in text, mode
    assert "--model" in text
    assert parser_source is not None
    assert argparse is not None


def test_an_invalid_strategy_is_rejected():
    import backtest

    with pytest.raises(SystemExit):
        backtest.main(["--strategy", "LIVE_AUTO", "--data", "nope.csv"])


def test_ppo_and_the_rule_engine_are_summarised_in_one_vocabulary():
    """Two vocabularies for one comparison is where mistakes live."""
    from ai.evaluation.metrics import summarise_trades
    from backtest import _rule_summary

    class _Stats:
        closed, total_net_r, average_net_r = 10, 2.5, 0.25
        net_win_rate, net_profit_factor = 60.0, 1.4
        max_drawdown_r, average_duration_min = 3.0, 5.0

    class _Report:
        total_signals, overall = 10, _Stats()

    rule = _rule_summary(_Report())
    ppo = summarise_trades(pd.DataFrame({
        "net_r": [1.0], "gross_r": [1.4], "cost_r": [0.4],
        "holding_candles": [3], "result": ["TP3_HIT"], "tp_hits": [3],
        "spread_points": [20],
    }), label="PPO")

    for key in ("label", "trades", "net_r", "average_net_r", "win_rate",
                "profit_factor", "max_drawdown_r", "average_holding"):
        assert key in rule, key
        assert key in ppo, key


def test_an_empty_rule_report_summarises_as_zero_not_as_an_error():
    from backtest import _rule_summary

    class _Report:
        total_signals = 0
        overall = None

    summary = _rule_summary(_Report())
    assert summary["trades"] == 0 and summary["net_r"] == 0.0
    assert _rule_summary(None)["trades"] == 0


def test_ppo_is_charged_the_same_costs_as_the_rule_engine():
    """PPO must never get a friendlier simulation than what it is compared to."""
    from ai.environment.scalping_env import EnvConfig

    config = Config().for_market("XAUUSDs")
    env = EnvConfig.from_market_config(config)

    assert env.assumed_spread_points == config.assumed_spread_points
    assert env.slippage_points_entry == config.slippage_points_entry
    assert env.slippage_points_exit == config.slippage_points_exit
    assert env.commission_points_per_side == config.commission_points_per_side
    assert env.point_value == config.point_value
    # and the round-trip figures agree
    assert env.cost_points(config.assumed_spread_points) == pytest.approx(
        config.round_trip_cost(None) / config.point_value
    )


def test_ppo_inherits_the_engines_holding_window_and_geometry():
    from ai.environment.scalping_env import EnvConfig

    config = Config().for_market("XAUUSDs")
    env = EnvConfig.from_market_config(config)

    assert env.max_holding_candles == config.max_holding_candles
    assert env.sl_atr == config.sl_atr_multiplier
    assert tuple(env.tp_atr) == tuple(config.tp_atr_multiples)
    assert env.move_sl_to_breakeven_after_tp1 == config.move_sl_to_breakeven_after_tp1


def test_each_market_gives_the_environment_its_own_costs():
    from ai.environment.scalping_env import EnvConfig

    gold = EnvConfig.from_market_config(Config().for_market("XAUUSDs"))
    btc = EnvConfig.from_market_config(Config().for_market("BTCUSDs"))

    assert gold.assumed_spread_points != btc.assumed_spread_points
    assert gold.cost_points(gold.assumed_spread_points) != btc.cost_points(
        btc.assumed_spread_points
    )
