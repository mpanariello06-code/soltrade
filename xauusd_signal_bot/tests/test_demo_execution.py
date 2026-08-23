"""Demo execution layer: safety gates, order lifecycle, recovery and reporting.

EVERY TEST RUNS AGAINST A SCRIPTED FAKE BROKER.  Nothing here can reach a real
account: the manager talks to the :class:`~src.demo_broker.DemoBroker` port, and
these tests supply :class:`~src.demo_broker.FakeDemoBroker`.

The tests are written so a safety regression FAILS rather than merely producing
a different number - a blocked order is asserted by checking that no order
request was transmitted at all, not just that a flag came back false.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from config import Config
from src.demo_broker import (
    BUY,
    SELL,
    BrokerPosition,
    FakeDemoBroker,
    MT5DemoBroker,
    OrderResult,
    Quote,
)
from src.demo_execution import (
    EXECUTION_COLUMNS,
    RESULT_SL,
    RESULT_TIMEOUT,
    RESULT_TP3,
    DemoTrade,
    ExecutionManager,
)
from src.execution_config import (
    DEMO_EXECUTION_BLOCKED,
    AccountInfo,
    DemoExecutionBlocked,
    DemoRiskModel,
    ExecutionMode,
    ExecutionSettings,
    assert_no_live_mode,
    classify_account,
    load_execution_settings,
    parse_execution_mode,
)
from src.markets import BTCUSD, XAUUSD, get_market
from src.runtime_state import JsonStateStore
from src.signal_tracker import read_csv_rows
from tests.conftest import isolate
from tests.test_markets import _signal


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
GOLD_QUOTE = Quote(bid=2300.00, ask=2300.20, spread_points=20)
BTC_QUOTE = Quote(bid=60_000.00, ask=60_010.00, spread_points=1000)

DEMO_ACCOUNT = AccountInfo(
    login=5_000_001, server="Demo-Server", currency="USD", balance=10_000.0,
    trade_mode="DEMO", is_demo=True, verified=True,
)
LIVE_ACCOUNT = AccountInfo(
    login=7_000_001, server="Live-Server", currency="USD", balance=10_000.0,
    trade_mode="LIVE", is_demo=False, verified=True,
)
UNVERIFIED_ACCOUNT = AccountInfo(verified=False)


def armed_settings(**overrides) -> ExecutionSettings:
    """Execution settings with both switches deliberately turned on."""
    settings = ExecutionSettings(
        mode=ExecutionMode.DEMO_AUTO,
        demo_trading_enabled=True,
        demo_login=5_000_001,
        demo_password="demo-password",
        demo_server="Demo-Server",
        risk=DemoRiskModel(account_balance=10_000.0, risk_per_trade=0.005),
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    settings.validate()
    return settings


def build(tmp_path, symbol=XAUUSD, account=DEMO_ACCOUNT, settings=None, **broker_kwargs):
    """A manager wired to a fake broker, on an isolated data directory."""
    config = isolate(Config().for_market(symbol), tmp_path, symbol)
    quote = GOLD_QUOTE if symbol == XAUUSD else BTC_QUOTE
    broker = FakeDemoBroker(
        quotes={get_market(symbol).feed_symbol(): quote},
        point_value=config.point_value,
        **broker_kwargs,
    )
    broker.account_info = account
    manager = ExecutionManager(
        config, settings or armed_settings(), broker,
        store=JsonStateStore(config.state_file),
    )
    return config, manager, broker


def gold_signal(direction: str = BUY, signal_id: str = ""):
    """A signal with sane gold geometry around the fixture quote."""
    signal = _signal(XAUUSD, direction)
    if direction == BUY:
        signal.entry, signal.stop_loss = 2300.00, 2299.60
        signal.tp1, signal.tp2, signal.tp3 = 2300.36, 2300.65, 2301.08
    else:
        signal.entry, signal.stop_loss = 2300.00, 2300.40
        signal.tp1, signal.tp2, signal.tp3 = 2299.64, 2299.35, 2298.92
    if signal_id:
        signal.signal_id = signal_id
    return signal


def move_to(broker, symbol, price, spread=0.20):
    """Move the fake market to ``price``."""
    broker.quotes[get_market(symbol).feed_symbol()] = Quote(
        bid=price, ask=price + spread, spread_points=spread * 100,
    )


# --------------------------------------------------------------------------- #
# 28. there is no live mode  (spec sections 2, 28)
# --------------------------------------------------------------------------- #
def test_there_are_exactly_two_execution_modes_and_neither_is_live():
    names = {mode.name for mode in ExecutionMode}
    assert names == {"SIGNAL_ONLY", "DEMO_AUTO"}
    for mode in ExecutionMode:
        for word in ("LIVE", "REAL", "PROD"):
            assert word not in mode.value.upper()
    assert_no_live_mode()


def test_the_default_mode_is_signal_only():
    assert ExecutionSettings().mode is ExecutionMode.SIGNAL_ONLY
    assert ExecutionSettings().executes is False
    assert parse_execution_mode(None) is ExecutionMode.SIGNAL_ONLY
    assert parse_execution_mode("") is ExecutionMode.SIGNAL_ONLY


@pytest.mark.parametrize("attempt", ["LIVE_AUTO", "REAL_AUTO", "PRODUCTION_TRADING", "typo"])
def test_no_spelling_of_the_mode_variable_enables_live_trading(attempt):
    """An unrecognised mode falls back to the safe one, never to trading."""
    assert parse_execution_mode(attempt) is ExecutionMode.SIGNAL_ONLY


def test_both_switches_are_required(monkeypatch):
    """Mode alone is not enough, and the flag alone is not enough."""
    mode_only = ExecutionSettings(mode=ExecutionMode.DEMO_AUTO)
    assert mode_only.executes is False
    assert "DEMO_TRADING_ENABLED" in mode_only.blocking_reason()

    flag_only = ExecutionSettings(demo_trading_enabled=True)
    assert flag_only.executes is False
    assert "EXECUTION_MODE" in flag_only.blocking_reason()

    both = ExecutionSettings(mode=ExecutionMode.DEMO_AUTO, demo_trading_enabled=True)
    assert both.executes is True
    # ...but a dedicated demo account is still required before anything happens
    assert "demo account" in both.blocking_reason()


def test_a_dedicated_demo_account_is_required(tmp_path):
    """The data feed's credentials are deliberately not reused."""
    settings = ExecutionSettings(
        mode=ExecutionMode.DEMO_AUTO, demo_trading_enabled=True,
    )
    _config, manager, broker = build(tmp_path, settings=settings)
    decision = manager.execute_signal(gold_signal())
    assert not decision.executed
    assert "demo account" in decision.reason
    assert broker.sent == [], "an order was sent without a configured demo account"


def test_the_environment_defaults_to_signal_only(monkeypatch):
    for name in ("EXECUTION_MODE", "DEMO_TRADING_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    settings = load_execution_settings()
    assert settings.mode is ExecutionMode.SIGNAL_ONLY
    assert settings.executes is False


# --------------------------------------------------------------------------- #
# 3, 4, 27. account verification and live rejection
# --------------------------------------------------------------------------- #
def test_a_live_account_is_refused_and_no_order_is_sent(tmp_path):
    """Spec section 27: the explicit live-account safety test."""
    _config, manager, broker = build(tmp_path, account=LIVE_ACCOUNT)

    with pytest.raises(DemoExecutionBlocked) as caught:
        manager.execute_signal(gold_signal())

    assert DEMO_EXECUTION_BLOCKED in str(caught.value)
    assert "LIVE" in caught.value.reason
    assert broker.sent == [], "an order request reached a LIVE account"
    assert manager.open_trades() == []


def test_an_unverifiable_account_is_refused_and_no_order_is_sent(tmp_path):
    """"Could not verify" must behave exactly like "live", never like "demo"."""
    _config, manager, broker = build(tmp_path, account=UNVERIFIED_ACCOUNT)

    with pytest.raises(DemoExecutionBlocked) as caught:
        manager.execute_signal(gold_signal())

    assert "could not be verified" in caught.value.reason
    assert broker.sent == []


def test_a_contest_account_is_not_treated_as_demo():
    """Only trade mode 0 is demo.  A contest account is not."""
    class Raw:
        login, server, currency, balance = 1, "Contest", "USD", 100.0
        trade_mode = 1

    info = classify_account(Raw())
    assert info.verified is True
    assert info.is_demo is False
    assert info.trade_mode == "CONTEST"


def test_a_missing_account_object_is_never_assumed_demo():
    info = classify_account(None)
    assert info.is_demo is False and info.verified is False


def test_the_account_is_re_verified_on_every_execution(tmp_path):
    """A terminal can be re-pointed mid-session; a cached 'yes' would be fatal."""
    _config, manager, broker = build(tmp_path)
    assert manager.execute_signal(gold_signal(signal_id="first")).executed

    broker.account_info = LIVE_ACCOUNT
    with pytest.raises(DemoExecutionBlocked):
        manager.execute_signal(gold_signal(signal_id="second"))
    assert len(broker.sent) == 1, "a second order was sent after the account changed"


# --------------------------------------------------------------------------- #
# 4, 5. symbol and market validation
# --------------------------------------------------------------------------- #
def test_the_brokers_own_symbol_is_used_for_the_order(tmp_path):
    _config, manager, broker = build(tmp_path)
    manager.execute_signal(gold_signal())
    request = broker.sent[0]
    assert request.broker_symbol == get_market(XAUUSD).feed_symbol()
    assert request.symbol == XAUUSD


def test_an_unavailable_symbol_blocks_the_trade(tmp_path):
    _config, manager, broker = build(tmp_path)
    broker.known_symbols = {"SOMETHING-ELSE"}

    decision = manager.execute_signal(gold_signal())
    assert not decision.executed
    assert "does not exist" in decision.reason
    assert broker.sent == []


def test_a_missing_quote_blocks_the_trade(tmp_path):
    _config, manager, broker = build(tmp_path)
    broker.known_symbols = {get_market(XAUUSD).feed_symbol()}
    broker.quotes.clear()

    decision = manager.execute_signal(gold_signal())
    assert not decision.executed
    assert "no quote" in decision.reason
    assert broker.sent == []


# --------------------------------------------------------------------------- #
# 5, 7. spread and hard limits
# --------------------------------------------------------------------------- #
def test_a_blown_out_spread_blocks_the_trade(tmp_path):
    _config, manager, broker = build(
        tmp_path, settings=armed_settings(max_spread_points=25.0)
    )
    broker.quotes[get_market(XAUUSD).feed_symbol()] = Quote(
        bid=2300.00, ask=2301.00, spread_points=100,
    )
    decision = manager.execute_signal(gold_signal())
    assert not decision.executed
    assert "spread" in decision.reason
    assert broker.sent == []


def test_the_open_position_limit_blocks_new_trades(tmp_path):
    _config, manager, broker = build(
        tmp_path, settings=armed_settings(max_open_positions=1)
    )
    assert manager.execute_signal(gold_signal(signal_id="one")).executed
    decision = manager.execute_signal(gold_signal(signal_id="two"))

    assert not decision.executed
    assert "open-position limit" in decision.reason
    assert len(broker.sent) == 1


def test_the_daily_trade_limit_blocks_new_trades(tmp_path):
    _config, manager, broker = build(
        tmp_path, settings=armed_settings(max_trades_per_day=2, max_open_positions=9)
    )
    for index in range(2):
        assert manager.execute_signal(gold_signal(signal_id=f"s{index}")).executed
    decision = manager.execute_signal(gold_signal(signal_id="s2"))

    assert not decision.executed
    assert "trades-per-day limit" in decision.reason
    assert len(broker.sent) == 2


def test_the_daily_loss_limit_blocks_new_trades(tmp_path):
    _config, manager, broker = build(
        tmp_path, settings=armed_settings(max_daily_loss=5.0, max_open_positions=9)
    )
    manager._realised_today = -10.0

    decision = manager.execute_signal(gold_signal())
    assert not decision.executed
    assert "daily loss limit" in decision.reason
    assert broker.sent == []


def test_a_blocked_limit_still_lets_open_trades_be_managed(tmp_path):
    """Spec section 7: limits stop NEW trades, never existing management."""
    _config, manager, broker = build(
        tmp_path, settings=armed_settings(max_open_positions=1)
    )
    trade = manager.execute_signal(gold_signal(signal_id="open")).trade
    assert not manager.execute_signal(gold_signal(signal_id="blocked")).executed

    move_to(broker, XAUUSD, 2299.50)
    manager.manage()
    assert trade.result == RESULT_SL, "the open trade stopped being managed"


# --------------------------------------------------------------------------- #
# 6. position sizing
# --------------------------------------------------------------------------- #
def test_position_size_comes_from_the_stop_distance():
    """A wider stop must produce a smaller position, at constant risk."""
    risk = DemoRiskModel(
        account_balance=10_000.0, risk_per_trade=0.01,
        minimum_lot=0.01, maximum_lot=100.0, point_value_per_lot=1.0,
    )
    tight = risk.size_for(0.40, 0.01)     # 40 points
    wide = risk.size_for(0.80, 0.01)      # 80 points

    assert tight.ok and wide.ok
    assert wide.approved_lots == pytest.approx(tight.approved_lots / 2, rel=0.02)
    assert tight.risk_amount == 100.0


def test_position_size_is_clamped_to_the_maximum():
    risk = DemoRiskModel(
        account_balance=1_000_000.0, risk_per_trade=0.5,
        minimum_lot=0.01, maximum_lot=0.10, point_value_per_lot=1.0,
    )
    result = risk.size_for(0.40, 0.01)
    assert result.approved_lots == 0.10
    assert result.requested_lots > result.approved_lots
    assert "clamped" in result.reason


def test_a_size_below_the_minimum_lot_is_rejected_not_rounded_up():
    risk = DemoRiskModel(
        account_balance=10.0, risk_per_trade=0.001,
        minimum_lot=0.01, maximum_lot=1.0, point_value_per_lot=1.0,
    )
    result = risk.size_for(5.0, 0.01)
    assert not result.ok
    assert result.approved_lots == 0.0


def test_an_invalid_stop_distance_is_rejected_with_the_arithmetic():
    risk = DemoRiskModel(point_value_per_lot=1.0)
    for distance in (0.0, -1.0, float("nan"), float("inf")):
        result = risk.size_for(distance, 0.01)
        assert not result.ok, distance
        assert result.reason, "a rejection must say why"
        assert "requested" in result.describe() and "approved" in result.describe()


def test_an_unsizable_signal_blocks_the_trade(tmp_path):
    settings = armed_settings()
    settings.risk = DemoRiskModel(
        account_balance=1.0, risk_per_trade=0.001,
        minimum_lot=1.0, maximum_lot=10.0,
    )
    settings.max_order_lots = 10.0
    _config, manager, broker = build(tmp_path, settings=settings)

    decision = manager.execute_signal(gold_signal())
    assert not decision.executed
    assert "position size rejected" in decision.reason
    assert broker.sent == []


def test_the_order_never_exceeds_the_hard_lot_ceiling(tmp_path):
    settings = armed_settings(max_order_lots=0.02)
    settings.risk = DemoRiskModel(
        account_balance=1_000_000.0, risk_per_trade=0.5, maximum_lot=50.0,
    )
    _config, manager, broker = build(tmp_path, settings=settings)
    manager.execute_signal(gold_signal())
    assert broker.sent[0].volume <= 0.02


# --------------------------------------------------------------------------- #
# 9, 10. entry price, SL and TP
# --------------------------------------------------------------------------- #
def test_a_buy_enters_at_the_ask_and_a_sell_at_the_bid(tmp_path):
    """Spec section 9: the candle close is NOT the execution price."""
    _config, manager, broker = build(tmp_path)
    manager.execute_signal(gold_signal(BUY, "buy"))
    assert broker.sent[-1].requested_price == GOLD_QUOTE.ask

    manager.settings.max_open_positions = 9
    manager.execute_signal(gold_signal(SELL, "sell"))
    assert broker.sent[-1].requested_price == GOLD_QUOTE.bid


def test_all_three_prices_are_recorded(tmp_path):
    """signal entry, requested entry and actual fill are all different numbers."""
    _config, manager, broker = build(tmp_path, slippage_points=3.0)
    trade = manager.execute_signal(gold_signal()).trade

    assert trade.signal_entry == 2300.00          # what the engine said
    assert trade.requested_price == GOLD_QUOTE.ask  # what we asked for
    assert trade.fill_price > trade.requested_price  # what we got
    assert trade.slippage_points == pytest.approx(3.0, abs=0.01)


def test_the_signals_own_levels_are_used_verbatim(tmp_path):
    """Spec section 10: execution must not recalculate stops or targets."""
    _config, manager, broker = build(tmp_path)
    signal = gold_signal()
    trade = manager.execute_signal(signal).trade

    assert trade.stop_loss == signal.stop_loss
    assert (trade.tp1, trade.tp2, trade.tp3) == (signal.tp1, signal.tp2, signal.tp3)
    assert broker.sent[0].stop_loss == signal.stop_loss
    # the broker holds the FINAL target; TP1/TP2 are managed as partial closes
    assert broker.sent[0].take_profit == signal.tp3


def test_missing_broker_stops_are_attached_after_the_fill(tmp_path):
    """An accepted order with no protection is the dangerous case."""
    _config, manager, broker = build(tmp_path)
    broker.scripted_results = [
        OrderResult(accepted=True, ticket=4242, fill_price=2300.20, volume=0.05,
                    retcode=MT5DemoBroker.RETCODE_DONE, requested_volume=0.05)
    ]
    broker.open_positions.append(
        BrokerPosition(ticket=4242, symbol=get_market(XAUUSD).feed_symbol(),
                       direction=BUY, volume=0.05, open_price=2300.20,
                       stop_loss=0.0, take_profit=0.0, magic=770_101)
    )
    manager.execute_signal(gold_signal())

    assert broker.modifications, "SL/TP were never attached"
    assert broker.open_positions[0].stop_loss != 0.0
    assert broker.open_positions[0].take_profit != 0.0


@pytest.mark.parametrize("direction,stop,targets", [
    (BUY, 2301.00, (2300.36, 2300.65, 2301.08)),        # stop above entry
    (BUY, 2299.60, (2299.00, 2300.65, 2301.08)),        # target below entry
    (SELL, 2299.00, (2299.64, 2299.35, 2298.92)),       # stop below entry
])
def test_incoherent_levels_block_the_trade(tmp_path, direction, stop, targets):
    _config, manager, broker = build(tmp_path)
    signal = gold_signal(direction)
    signal.stop_loss = stop
    signal.tp1, signal.tp2, signal.tp3 = targets

    decision = manager.execute_signal(signal)
    assert not decision.executed
    assert broker.sent == []


def test_an_absurdly_distant_stop_blocks_the_trade(tmp_path):
    """A symptom of a bad quote, not a real setup."""
    # the fixture's stop is 0.60 on a 2300 price = 0.026%, so a 0.01% ceiling
    # is what makes it "absurdly distant"
    _config, manager, broker = build(
        tmp_path, settings=armed_settings(max_sl_distance_pct=0.0001)
    )
    decision = manager.execute_signal(gold_signal())
    assert not decision.executed
    assert "stop is" in decision.reason
    assert broker.sent == []


# --------------------------------------------------------------------------- #
# 8. duplicate prevention
# --------------------------------------------------------------------------- #
def test_one_signal_creates_exactly_one_order(tmp_path):
    _config, manager, broker = build(tmp_path)
    signal = gold_signal()

    assert manager.execute_signal(signal).executed
    for _ in range(5):
        repeat = manager.execute_signal(signal)
        assert not repeat.executed
        assert repeat.reason == "signal already processed"
    assert len(broker.sent) == 1


def test_duplicate_prevention_survives_a_restart(tmp_path):
    """Spec section 8: a restart must not re-execute a signal."""
    config, manager, broker = build(tmp_path)
    signal = gold_signal()
    assert manager.execute_signal(signal).executed

    # a completely fresh manager over the same state file
    restarted = ExecutionManager(
        config, armed_settings(), broker, store=JsonStateStore(config.state_file),
    )
    decision = restarted.execute_signal(signal)

    assert not decision.executed
    assert decision.reason == "signal already processed"
    assert len(broker.sent) == 1, "the restart duplicated the order"


def test_a_signal_is_marked_processed_even_when_it_is_refused(tmp_path):
    """A refused signal must not be retried on the next cycle either."""
    settings = armed_settings()
    settings.risk = DemoRiskModel(
        account_balance=1.0, risk_per_trade=0.001, minimum_lot=1.0, maximum_lot=10.0,
    )
    settings.max_order_lots = 10.0
    _config, manager, _broker = build(tmp_path, settings=settings)

    signal = gold_signal()
    manager.execute_signal(signal)
    assert manager.already_processed(signal.signal_id)


# --------------------------------------------------------------------------- #
# 10, 11, 12. TP ladder, stop, timeout
# --------------------------------------------------------------------------- #
def test_the_partial_ladder_closes_tp1_tp2_then_tp3(tmp_path):
    config, manager, broker = build(tmp_path)
    trade = manager.execute_signal(gold_signal()).trade
    opened = trade.volume

    move_to(broker, XAUUSD, 2300.40)
    manager.manage()
    assert trade.tp_hits == 1 and trade.remaining_volume < opened and trade.is_open

    move_to(broker, XAUUSD, 2300.70)
    manager.manage()
    assert trade.tp_hits == 2 and trade.is_open

    move_to(broker, XAUUSD, 2301.10)
    manager.manage()
    assert trade.tp_hits == 3
    assert trade.result == RESULT_TP3
    assert trade.remaining_volume == 0.0

    rows = read_csv_rows(config.executions_csv)
    assert len(rows) == 1
    assert rows[0]["result"] == RESULT_TP3
    assert rows[0]["tp1_filled"] == "1" and rows[0]["tp3_filled"] == "1"


def test_the_tp_fractions_are_configurable(tmp_path):
    settings = armed_settings(tp_fractions=(0.5, 0.25, 0.25))
    settings.risk = DemoRiskModel(
        account_balance=100_000.0, risk_per_trade=0.01, maximum_lot=10.0,
    )
    settings.max_order_lots = 10.0
    _config, manager, broker = build(tmp_path, settings=settings)
    trade = manager.execute_signal(gold_signal()).trade
    opened = trade.volume

    move_to(broker, XAUUSD, 2300.40)
    manager.manage()
    assert trade.closed_volume == pytest.approx(opened * 0.5, abs=0.011)


def test_a_stop_out_closes_everything_and_is_recorded(tmp_path):
    config, manager, broker = build(tmp_path)
    trade = manager.execute_signal(gold_signal()).trade

    move_to(broker, XAUUSD, 2299.50)
    manager.manage()

    assert trade.result == RESULT_SL
    assert trade.remaining_volume == 0.0
    rows = read_csv_rows(config.executions_csv)
    assert rows[0]["result"] == RESULT_SL
    assert float(rows[0]["net_profit"]) < 0


def test_the_stop_wins_when_a_target_is_also_reachable(tmp_path):
    """Pessimistic resolution: never flatter the result."""
    _config, manager, broker = build(tmp_path)
    trade = manager.execute_signal(gold_signal()).trade

    # a quote below the stop, even though the ask is above TP1
    broker.quotes[get_market(XAUUSD).feed_symbol()] = Quote(
        bid=2299.50, ask=2300.40, spread_points=90,
    )
    manager.manage()
    assert trade.result == RESULT_SL


def test_a_position_times_out_using_the_existing_holding_window(tmp_path):
    """Spec section 12: reuse the scalper's own timeout, do not invent one."""
    config, manager, broker = build(tmp_path)
    trade = manager.execute_signal(gold_signal()).trade
    limit = config.max_holding_candles

    manager.manage(when=trade.opened_at + timedelta(minutes=limit - 1))
    assert trade.is_open, "closed before the configured holding window"

    manager.manage(when=trade.opened_at + timedelta(minutes=limit + 1))
    assert trade.result == RESULT_TIMEOUT
    rows = read_csv_rows(config.executions_csv)
    assert rows[0]["result"] == RESULT_TIMEOUT
    assert float(rows[0]["holding_seconds"]) > 0


def test_breakeven_follows_the_existing_strategy_setting(tmp_path):
    """Execution must not introduce breakeven, nor drop it (spec section 11)."""
    config, manager, broker = build(tmp_path)
    config.move_sl_to_breakeven_after_tp1 = True
    trade = manager.execute_signal(gold_signal()).trade
    move_to(broker, XAUUSD, 2300.40)
    manager.manage()
    assert trade.breakeven_applied is True
    assert trade.stop_loss == trade.fill_price

    config2, manager2, broker2 = build(tmp_path / "off")
    config2.move_sl_to_breakeven_after_tp1 = False
    trade2 = manager2.execute_signal(gold_signal()).trade
    original_stop = trade2.stop_loss
    move_to(broker2, XAUUSD, 2300.40)
    manager2.manage()
    assert trade2.breakeven_applied is False
    assert trade2.stop_loss == original_stop


def test_r_is_measured_against_the_risk_at_the_actual_fill(tmp_path):
    """A breakeven move must not rewrite the R denominator."""
    config, manager, broker = build(tmp_path, slippage_points=2.0)
    config.move_sl_to_breakeven_after_tp1 = True
    trade = manager.execute_signal(gold_signal()).trade
    risk_at_entry = trade.initial_risk

    move_to(broker, XAUUSD, 2300.40)
    manager.manage()
    assert trade.stop_loss == trade.fill_price      # breakeven applied
    assert trade.risk_per_unit == risk_at_entry     # denominator unchanged

    move_to(broker, XAUUSD, 2301.10)
    manager.manage()
    rows = read_csv_rows(config.executions_csv)
    assert rows[0]["R_multiple"] not in ("", None), "R was lost to a zero denominator"


# --------------------------------------------------------------------------- #
# 22, 23, 24. recovery, connection loss, order failure
# --------------------------------------------------------------------------- #
def test_restart_restores_open_trades_and_keeps_managing_them(tmp_path):
    config, manager, broker = build(tmp_path)
    trade = manager.execute_signal(gold_signal()).trade
    ticket = trade.ticket

    restarted = ExecutionManager(
        config, armed_settings(), broker, store=JsonStateStore(config.state_file),
    )
    assert len(restarted.open_trades()) == 1
    assert restarted.open_trades()[0].ticket == ticket

    assert restarted.reconcile() is True
    move_to(broker, XAUUSD, 2299.50)
    restarted.manage()
    assert restarted.open_trades() == []
    assert read_csv_rows(config.executions_csv)[0]["result"] == RESULT_SL


def test_reconciliation_records_a_position_that_closed_while_offline(tmp_path):
    config, manager, broker = build(tmp_path)
    manager.execute_signal(gold_signal()).trade
    broker.open_positions.clear()          # closed at the broker while we were down

    restarted = ExecutionManager(
        config, armed_settings(), broker, store=JsonStateStore(config.state_file),
    )
    assert restarted.reconcile() is True
    assert restarted.open_trades() == []
    rows = read_csv_rows(config.executions_csv)
    assert len(rows) == 1
    assert "offline" in rows[0]["notes"]


def test_reconciliation_adopts_an_untracked_position(tmp_path):
    """A position with our magic number that we have no record of."""
    config, manager, broker = build(tmp_path)
    broker.open_positions.append(
        BrokerPosition(ticket=555, symbol=get_market(XAUUSD).feed_symbol(),
                       direction=BUY, volume=0.03, open_price=2300.00,
                       stop_loss=2299.50, take_profit=2301.00, magic=770_101)
    )
    assert manager.reconcile() is True
    adopted = manager.open_trades()
    assert len(adopted) == 1
    assert adopted[0].ticket == 555
    assert "adopted" in adopted[0].notes


def test_a_failed_reconciliation_halts_execution(tmp_path):
    config, manager, broker = build(tmp_path)

    def explode(magic=None):
        raise ConnectionError("terminal gone")

    broker.positions = explode
    assert manager.reconcile() is False
    assert manager.halted is True

    broker.positions = lambda magic=None: []
    decision = manager.execute_signal(gold_signal())
    assert not decision.executed
    assert "halted" in decision.reason


def test_a_rejected_order_is_reported_and_not_retried(tmp_path):
    _config, manager, broker = build(tmp_path)
    broker.scripted_results = [
        OrderResult(accepted=False, retcode=10019, comment="No money",
                    requested_volume=0.05)
    ]
    signal = gold_signal()
    decision = manager.execute_signal(signal)

    assert not decision.executed
    assert "rejected" in decision.reason
    assert manager.open_trades() == []
    # the signal is spent: a retry must not fire a second request
    assert not manager.execute_signal(signal).executed
    assert len(broker.sent) == 1


def test_an_indeterminate_reply_halts_instead_of_retrying(tmp_path):
    """Spec section 24: never retry an order whose outcome is unknown."""
    _config, manager, broker = build(tmp_path)
    broker.scripted_results = [
        OrderResult(accepted=False, indeterminate=True, comment="timeout",
                    requested_volume=0.05)
    ]
    decision = manager.execute_signal(gold_signal())

    assert not decision.executed
    assert manager.halted is True
    assert len(broker.sent) == 1, "an unknown-outcome order was retried"


def test_a_partial_fill_is_recorded_at_the_filled_size(tmp_path):
    _config, manager, broker = build(tmp_path)
    broker.scripted_results = [
        OrderResult(accepted=True, ticket=99, fill_price=2300.20, volume=0.02,
                    retcode=MT5DemoBroker.RETCODE_DONE_PARTIAL,
                    partial=True, requested_volume=0.10)
    ]
    broker.open_positions.append(
        BrokerPosition(ticket=99, symbol=get_market(XAUUSD).feed_symbol(),
                       direction=BUY, volume=0.02, open_price=2300.20,
                       stop_loss=2299.60, take_profit=2301.08, magic=770_101)
    )
    trade = manager.execute_signal(gold_signal()).trade

    assert trade.volume == 0.02
    assert trade.remaining_volume == 0.02
    assert "partial fill" in trade.notes


def test_execution_is_off_when_the_broker_cannot_connect():
    settings = armed_settings()
    broker = FakeDemoBroker(fail_connect=True)
    assert broker.connect() is False
    assert settings.executes is True, "settings alone do not make a connection"


# --------------------------------------------------------------------------- #
# 20, 21, 25. isolation, logging and reporting
# --------------------------------------------------------------------------- #
def test_the_two_markets_execute_in_complete_isolation(tmp_path):
    """Spec section 20/21: gold and Bitcoin share nothing but the connection."""
    gold_config, gold, gold_broker = build(tmp_path, XAUUSD)
    btc_config, btc, btc_broker = build(tmp_path, BTCUSD)

    assert gold_config.executions_csv != btc_config.executions_csv
    assert gold_config.executions_csv.parent.name == "xauusds"
    assert btc_config.executions_csv.parent.name == "btcusds"

    gold.execute_signal(gold_signal())
    assert len(gold.open_trades()) == 1
    assert btc.open_trades() == [], "a gold trade appeared on Bitcoin"
    assert btc.trades_today() == 0
    assert not btc.already_processed(gold_signal().signal_id)

    # gold stopping out must not touch Bitcoin's books
    move_to(gold_broker, XAUUSD, 2299.50)
    gold.manage()
    assert read_csv_rows(gold_config.executions_csv)
    assert read_csv_rows(btc_config.executions_csv) == []


def test_an_execution_row_carries_signal_execution_and_outcome(tmp_path):
    """Spec section 13: all three blocks in one traceable row."""
    config, manager, broker = build(tmp_path, slippage_points=2.0)
    signal = gold_signal()
    manager.execute_signal(signal)
    move_to(broker, XAUUSD, 2299.50)
    manager.manage()

    row = read_csv_rows(config.executions_csv)[0]
    assert set(EXECUTION_COLUMNS).issuperset(row.keys())

    # signal block
    assert row["signal_id"] == signal.signal_id
    assert row["symbol"] == XAUUSD and row["timeframe"] == "M1"
    assert float(row["signal_entry"]) == signal.entry
    assert float(row["tp1"]) == signal.tp1
    # execution block
    assert int(row["broker_ticket"]) > 0
    assert float(row["requested_price"]) > 0
    assert float(row["actual_fill_price"]) > 0
    assert float(row["slippage_points"]) != 0
    assert float(row["position_size"]) > 0
    assert row["account_type"] == "DEMO"
    # outcome block
    assert row["result"] == RESULT_SL
    assert float(row["holding_seconds"]) >= 0
    assert row["net_profit"] and row["R_multiple"]


def test_signal_and_execution_results_are_reported_separately(tmp_path):
    """Spec section 14: execution results never overwrite signal results."""
    from performance import analyse_executions, compare_signal_and_execution

    config, manager, broker = build(tmp_path)
    manager.execute_signal(gold_signal())
    move_to(broker, XAUUSD, 2301.10)
    manager.manage()

    stats = analyse_executions(config.executions_csv, XAUUSD)
    assert stats.trades == 1
    assert stats.average_r != 0.0

    comparison = compare_signal_and_execution(config, XAUUSD)
    assert comparison.symbol == XAUUSD
    # the paper file is untouched by execution, so the two are independent
    assert comparison.signal_trades == 0
    assert comparison.execution_trades == 1
    assert comparison.comparable is False, "1 fill is not a comparison"


def test_an_empty_execution_file_reports_zero_rather_than_failing(tmp_path):
    from performance import analyse_executions

    config, _manager, _broker = build(tmp_path)
    stats = analyse_executions(config.executions_csv, XAUUSD)
    assert stats.trades == 0 and stats.average_r == 0.0


def test_open_trades_are_described_for_the_panel(tmp_path):
    _config, manager, _broker = build(tmp_path)
    manager.execute_signal(gold_signal())
    lines = manager.describe_open()
    text = "\n".join(lines)

    for expected in ("Entry:", "Current:", "SL:", "TP1:", "TP2:", "TP3:", "Time Open:"):
        assert expected in text, expected


# --------------------------------------------------------------------------- #
# state round trip
# --------------------------------------------------------------------------- #
def test_a_trade_survives_a_state_round_trip():
    trade = DemoTrade(
        signal_id="x", symbol=XAUUSD, broker_symbol="XAUUSDs", direction=BUY,
        ticket=1, fill_price=2300.2, stop_loss=2299.6, tp1=2300.4, tp2=2300.7,
        tp3=2301.1, volume=0.05, remaining_volume=0.05, initial_risk=0.6,
        tp_hits=1, breakeven_applied=True,
    )
    restored = DemoTrade.from_state(trade.state())

    assert restored.signal_id == trade.signal_id
    assert restored.ticket == trade.ticket
    assert restored.initial_risk == trade.initial_risk
    assert restored.tp_hits == 1
    assert restored.breakeven_applied is True


def test_unreadable_stored_trades_do_not_stop_start_up(tmp_path):
    config = isolate(Config().for_market(XAUUSD), tmp_path, XAUUSD)
    store = JsonStateStore(config.state_file)
    store.update_section("execution", {"open_trades": [{"nonsense": True}], })

    manager = ExecutionManager(config, armed_settings(), FakeDemoBroker(), store=store)
    assert manager.open_trades() == []


# --------------------------------------------------------------------------- #
# 15, 16, 19, 21. Telegram controls
# --------------------------------------------------------------------------- #
class _StubEngine:
    """The engine surface the Telegram panel actually calls."""

    def __init__(self, state="SIGNAL_ONLY") -> None:
        self.state = state
        self.toggles = []
        self.refuse = False
        self.open_lines = []

    def execution_state(self, symbol=None):
        return self.state

    def set_demo_auto(self, enabled):
        self.toggles.append(enabled)
        if enabled and self.refuse:
            return False
        self.state = "DEMO_AUTO" if enabled else "SIGNAL_ONLY"
        return enabled

    def demo_open_trades(self, symbol=None):
        return 1 if self.state == "DEMO_AUTO" else 0

    def demo_trades_today(self, symbol=None):
        return 7 if self.state == "DEMO_AUTO" else 0

    def demo_net_today(self, symbol=None):
        return 12.5 if self.state == "DEMO_AUTO" else 0.0

    def open_trade_lines(self, symbol=None):
        return self.open_lines


def _controller(tmp_path, engine=None):
    from src.runtime_state import RuntimeState
    from src.telegram_control import TelegramController
    from tests.conftest import FakeNotifier

    config = isolate(Config().for_market(XAUUSD), tmp_path, XAUUSD)
    runtime = RuntimeState.load(config)
    return TelegramController(
        config, runtime, FakeNotifier(), engine=engine or _StubEngine()
    )


def _buttons(keyboard):
    return [b["callback_data"] for row in keyboard for b in row]


def test_the_panel_shows_execution_status_and_demo_counters(tmp_path):
    engine = _StubEngine("DEMO_AUTO")
    controller = _controller(tmp_path, engine)
    text = controller.render_panel()

    assert "Execution: DEMO_AUTO" in text
    assert "Open Demo Trades: 1" in text
    assert "Today's Demo Trades: 7" in text
    assert "Today's Net P/L: +12.50" in text
    assert "DEMO AUTO - orders go to the DEMO account only." in text


def test_the_panel_says_signal_only_when_execution_is_off(tmp_path):
    controller = _controller(tmp_path)
    text = controller.render_panel()
    assert "Execution: SIGNAL_ONLY" in text
    assert "SIGNAL ONLY - no orders are placed." in text


def test_demo_auto_requires_an_explicit_confirmation(tmp_path):
    """Spec section 16: never enable implicitly."""
    engine = _StubEngine()
    controller = _controller(tmp_path, engine)

    assert "demo:confirm" in _buttons(controller.main_keyboard())

    text, keyboard, _toast = controller.handle_callback("demo:confirm")
    assert "ENABLE DEMO AUTO?" in text
    assert "automatically place orders" in text
    assert _buttons(keyboard) == ["demo:on", "nav:main"]
    assert engine.toggles == [], "the confirmation prompt already armed execution"


def test_only_the_explicit_enable_arms_execution(tmp_path):
    engine = _StubEngine()
    controller = _controller(tmp_path, engine)

    _text, _keyboard, toast = controller.handle_callback("demo:on")
    assert engine.toggles == [True]
    assert "DEMO AUTO ON" in toast
    assert "demo:off" in _buttons(controller.main_keyboard())


def test_cancelling_the_confirmation_leaves_execution_off(tmp_path):
    engine = _StubEngine()
    controller = _controller(tmp_path, engine)
    controller.handle_callback("demo:confirm")
    controller.handle_callback("nav:main")

    assert engine.toggles == []
    assert engine.state == "SIGNAL_ONLY"


def test_turning_demo_auto_off_is_immediate(tmp_path):
    engine = _StubEngine("DEMO_AUTO")
    controller = _controller(tmp_path, engine)

    _text, _keyboard, toast = controller.handle_callback("demo:off")
    assert engine.toggles == [False]
    assert "DEMO AUTO OFF" in toast


def test_a_refused_enable_is_reported_rather_than_shown_as_on(tmp_path):
    """If the engine cannot verify the account, the panel must not claim ON."""
    engine = _StubEngine()
    engine.refuse = True
    controller = _controller(tmp_path, engine)

    _text, _keyboard, toast = controller.handle_callback("demo:on")
    assert "refused" in toast.lower()
    assert engine.state == "SIGNAL_ONLY"
    assert "demo:confirm" in _buttons(controller.main_keyboard())


def test_the_open_trades_panel_lists_positions(tmp_path):
    engine = _StubEngine("DEMO_AUTO")
    engine.open_lines = ["XAUUSDs", "BUY", "Entry: 2300.20", "Time Open: 42 sec", ""]
    controller = _controller(tmp_path, engine)

    assert "view:trades" in _buttons(controller.main_keyboard())
    text, keyboard, _toast = controller.handle_callback("view:trades")

    assert "OPEN DEMO TRADES" in text
    assert "Entry: 2300.20" in text
    assert "view:trades" in _buttons(keyboard), "no REFRESH button"


def test_the_open_trades_panel_handles_nothing_open(tmp_path):
    controller = _controller(tmp_path)
    text, _keyboard, _toast = controller.handle_callback("view:trades")
    assert "No open demo trades." in text


def test_the_execution_performance_view_is_reachable(tmp_path):
    from src.telegram_control import TelegramController
    from src.runtime_state import RuntimeState
    from tests.conftest import FakeNotifier

    config = isolate(Config().for_market(XAUUSD), tmp_path, XAUUSD)
    runtime = RuntimeState.load(config)
    controller = TelegramController(config, runtime, FakeNotifier(), engine=_StubEngine())

    assert "perfv:execution" in _buttons(controller.performance_keyboard())
    text = controller.render_performance("execution")
    assert "DEMO EXECUTION vs SIGNALS" in text
    assert "No demo trades recorded yet." in text


def test_switching_market_does_not_disturb_demo_trades(tmp_path):
    """Spec section 21: switching is a selection, not an intervention."""
    gold_config, gold, gold_broker = build(tmp_path, XAUUSD)
    _btc_config, btc, _btc_broker = build(tmp_path, BTCUSD)

    trade = gold.execute_signal(gold_signal()).trade
    before = trade.state()

    from src.runtime_state import RuntimeState
    runtime = RuntimeState.load(gold_config)
    runtime.set_active_market(BTCUSD)

    assert gold.open_trades()[0].state() == before
    assert btc.open_trades() == []

    # and the gold position keeps being managed while Bitcoin is selected
    move_to(gold_broker, XAUUSD, 2299.50)
    gold.manage()
    assert gold.open_trades() == []
    assert read_csv_rows(gold_config.executions_csv)[0]["result"] == RESULT_SL


# --------------------------------------------------------------------------- #
# 22. runner-level restart recovery
# --------------------------------------------------------------------------- #
def _runner(tmp_path, broker=None, settings=None):
    """A SignalRunner with fake market data and a fake demo broker."""
    from main import SignalRunner
    from tests.conftest import FakeMarket, FakeNotifier, MultiMarket, make_candles

    config = isolate(Config(), tmp_path, XAUUSD)
    config.telegram_enabled = False
    runner = SignalRunner(config)

    runner.market = MultiMarket({
        XAUUSD: FakeMarket(
            config.for_market(XAUUSD),
            make_candles(1500, seed=41, volatility=0.16, start_price=2300.0),
            start=1400,
        ),
        BTCUSD: FakeMarket(
            config.for_market(BTCUSD),
            make_candles(1500, seed=43, volatility=30.0, start_price=60_000.0),
            start=1400,
        ),
    })
    runner.notifier = FakeNotifier()
    if broker is not None:
        runner.broker = broker
        for slot in runner.slots.values():
            if slot.execution:
                slot.execution.broker = broker
    if settings is not None:
        runner.execution_settings = settings
        for slot in runner.slots.values():
            if slot.execution:
                slot.execution.settings = settings
    return runner, config


def test_the_runner_defaults_to_signal_only(tmp_path):
    runner, _config = _runner(tmp_path)
    assert runner.execution_state() == "SIGNAL_ONLY"
    assert runner.execution_settings.executes is False


def test_the_runner_never_executes_in_signal_only_mode(tmp_path):
    broker = FakeDemoBroker(quotes={get_market(XAUUSD).feed_symbol(): GOLD_QUOTE})
    runner, _config = _runner(tmp_path, broker=broker)

    runner._execute_signal(runner.slot(XAUUSD), gold_signal())
    assert broker.sent == [], "an order was sent while in SIGNAL_ONLY"


def test_the_runner_executes_once_armed(tmp_path):
    broker = FakeDemoBroker(quotes={get_market(XAUUSD).feed_symbol(): GOLD_QUOTE})
    runner, _config = _runner(tmp_path, broker=broker, settings=armed_settings())

    runner._execute_signal(runner.slot(XAUUSD), gold_signal())
    assert len(broker.sent) == 1


def test_a_live_account_disarms_the_whole_runner(tmp_path):
    broker = FakeDemoBroker(quotes={get_market(XAUUSD).feed_symbol(): GOLD_QUOTE})
    broker.account_info = LIVE_ACCOUNT
    runner, _config = _runner(tmp_path, broker=broker, settings=armed_settings())

    runner._execute_signal(runner.slot(XAUUSD), gold_signal(signal_id="one"))
    assert broker.sent == []
    assert runner.execution_settings.demo_trading_enabled is False
    assert runner.execution_state() == "SIGNAL_ONLY"

    # and the next signal does not even try
    runner._execute_signal(runner.slot(XAUUSD), gold_signal(signal_id="two"))
    assert broker.sent == []


def test_start_execution_refuses_a_live_account(tmp_path):
    broker = FakeDemoBroker(quotes={get_market(XAUUSD).feed_symbol(): GOLD_QUOTE})
    broker.account_info = LIVE_ACCOUNT
    runner, _config = _runner(tmp_path, broker=broker, settings=armed_settings())

    assert runner.start_execution() is False
    assert runner.execution_settings.demo_trading_enabled is False
    alerts = [m["text"] for m in runner.notifier.messages]
    assert any("DEMO EXECUTION BLOCKED" in text for text in alerts)


def test_start_execution_refuses_an_unverifiable_account(tmp_path):
    broker = FakeDemoBroker(quotes={get_market(XAUUSD).feed_symbol(): GOLD_QUOTE})
    broker.account_info = UNVERIFIED_ACCOUNT
    runner, _config = _runner(tmp_path, broker=broker, settings=armed_settings())

    assert runner.start_execution() is False
    alerts = [m["text"] for m in runner.notifier.messages]
    assert any("could not be verified" in text for text in alerts)


def test_a_restart_recovers_open_trades_without_duplicating_them(tmp_path):
    """Spec section 22, end to end through the runner."""
    broker = FakeDemoBroker(quotes={get_market(XAUUSD).feed_symbol(): GOLD_QUOTE})
    settings = armed_settings()
    runner, config = _runner(tmp_path, broker=broker, settings=settings)

    signal = gold_signal()
    runner._execute_signal(runner.slot(XAUUSD), signal)
    assert len(broker.sent) == 1
    ticket = runner.slot(XAUUSD).execution.open_trades()[0].ticket

    # restart: brand-new runner over the SAME data directory and broker
    restarted, _config = _runner(tmp_path, broker=broker, settings=armed_settings())
    assert restarted.start_execution() is True

    recovered = restarted.slot(XAUUSD).execution.open_trades()
    assert len(recovered) == 1
    assert recovered[0].ticket == ticket

    # the same signal must not create a second order
    restarted._execute_signal(restarted.slot(XAUUSD), signal)
    assert len(broker.sent) == 1, "the restart duplicated the order"

    # and the recovered position is still managed to its exit
    move_to(broker, XAUUSD, 2299.50)
    restarted._manage_demo_trades(restarted.slot(XAUUSD))
    assert restarted.slot(XAUUSD).execution.open_trades() == []


def test_a_restart_does_not_replay_telegram_commands(tmp_path):
    """The old command-replay bug must stay fixed with execution added."""
    from tests.conftest import FakeNotifier

    broker = FakeDemoBroker(quotes={get_market(XAUUSD).feed_symbol(): GOLD_QUOTE})
    runner, _config = _runner(tmp_path, broker=broker, settings=armed_settings())

    notifier = FakeNotifier()
    notifier.updates = [
        {"update_id": 1, "callback_query": {
            "id": "a", "data": "demo:on",
            "message": {"message_id": 1, "chat": {"id": 4242}}}},
    ]
    runner.control.notifier = notifier
    runner.control._drain_backlog()

    assert broker.sent == [], "a queued DEMO AUTO press was replayed on start-up"


def test_a_market_with_an_open_demo_trade_is_always_ticked(tmp_path):
    """Even when the user has switched away from it (spec section 21)."""
    broker = FakeDemoBroker(quotes={
        get_market(XAUUSD).feed_symbol(): GOLD_QUOTE,
        get_market(BTCUSD).feed_symbol(): BTC_QUOTE,
    })
    runner, _config = _runner(tmp_path, broker=broker, settings=armed_settings())
    runner._execute_signal(runner.slot(XAUUSD), gold_signal())
    runner.runtime.set_active_market(BTCUSD)

    assert runner.slot(XAUUSD).has_open_demo_trades()
    move_to(broker, XAUUSD, 2299.50)
    runner._tick()
    assert runner.slot(XAUUSD).execution.open_trades() == [], (
        "the gold position stopped being managed after switching to Bitcoin"
    )


def test_the_runner_counts_demo_activity_per_market(tmp_path):
    broker = FakeDemoBroker(quotes={get_market(XAUUSD).feed_symbol(): GOLD_QUOTE})
    runner, _config = _runner(tmp_path, broker=broker, settings=armed_settings())
    runner._execute_signal(runner.slot(XAUUSD), gold_signal())

    assert runner.demo_open_trades(XAUUSD) == 1
    assert runner.demo_open_trades(BTCUSD) == 0
    assert runner.demo_trades_today(XAUUSD) == 1
    assert runner.demo_trades_today(BTCUSD) == 0
    assert runner.demo_open_trades() == 1, "the all-markets total is wrong"
