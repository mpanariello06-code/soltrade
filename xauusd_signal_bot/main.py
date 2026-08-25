"""M1 micro-scalping research runner, one engine over every configured market.

Connects to MetaTrader 5 for **market data only**, evaluates one candidate per
newly closed M1 candle of the active market, notifies Telegram and logs
everything to that market's own CSV files.

This program never places, modifies or closes an order.

TWO MARKETS, ONE ENGINE
-----------------------
The scalping engine is market-agnostic: it reads plain config attributes and
never mentions a symbol.  Each market supplies its own price scale, cost
assumptions, thresholds, floors and session behaviour through
:mod:`src.markets`, which :meth:`config.Config.for_market` folds onto the
config the engine runs on.

MARKET ISOLATION
----------------
Each market has its own tracker, its own CSV files, its own state file, its own
cooldown and its own processed-candle marker.  Switching the active market in
Telegram changes only which market *generates* signals - an open paper trade on
the market being left keeps being tracked against **its own** candles until it
resolves.
"""

from __future__ import annotations

import signal as os_signal
import sys
import time
from datetime import date, datetime
from typing import Dict, List, Optional
from config import Config, load_config
from src.logger import get_logger, setup_logging
from src.market_data import MT5_AVAILABLE, MarketData
from src.runtime_state import JsonStateStore, RuntimeState, effective_config
from src.signal_engine import EVALUATION_COLUMNS, SignalEngine
from src.signal_tracker import SignalTracker, append_csv, ensure_csv
from src.telegram_bot import TelegramNotifier
from src.telegram_control import TelegramController
from src.demo_broker import MT5DemoBroker
from src.demo_execution import ExecutionManager
from src.execution_config import DemoExecutionBlocked, load_execution_settings
from src.markets import MARKET_ORDER, get_market
from src.ppo_bridge import PPOBridge
from src.timeframes import MODE_SCALPING, SIGNAL_TIMEFRAME
from src.utils import is_weekend, iso, now_utc, parse_iso

LOGGER = get_logger("main")

BANNER_WIDTH = 46


class MarketSlot:
    """Everything the runner needs for one market: tracker, config, state.

    One slot per market, created once at startup.  Nothing is shared between
    slots except the MT5 connection and the Telegram notifier, which is what
    keeps the markets isolated.
    """

    def __init__(
        self, symbol: str, config: Config, runtime, notifier,
        execution_settings=None, broker=None,
    ) -> None:
        self.symbol = symbol
        self.market = get_market(symbol)
        self.base_config = config
        self.runtime = runtime
        self.notifier = notifier
        self.market_runtime = runtime.market(symbol)
        view = config.for_market(symbol)
        view.market_dir.mkdir(parents=True, exist_ok=True)
        ensure_csv(view.evaluations_csv, EVALUATION_COLUMNS)
        self.tracker = SignalTracker(view, notifier, store=self.market_runtime.store)
        self.tracker.load()

        # The demo execution manager is per-market, exactly like the tracker,
        # so a gold position and a Bitcoin position share no state or limits.
        # It exists even in SIGNAL_ONLY mode: it is then simply never asked to
        # execute anything, and still manages positions left open by a previous
        # DEMO_AUTO session.
        self.execution: Optional[ExecutionManager] = None
        if execution_settings is not None and broker is not None:
            self.execution = ExecutionManager(
                view, execution_settings, broker, notifier,
                store=self.market_runtime.store,
            )

    def config_view(self) -> Config:
        """This market's config with its live Telegram settings folded in."""
        return effective_config(self.base_config, self.runtime, self.symbol)

    def has_open_signals(self) -> bool:
        return bool(self.tracker.active_signals())

    def has_open_demo_trades(self) -> bool:
        return bool(self.execution and self.execution.open_trades())

    def signals_today(self) -> int:
        today = now_utc().date()
        count = 0
        for row in self.tracker.signals:
            timestamp = parse_iso(str(row.get("timestamp", "")))
            if timestamp is not None and timestamp.date() == today:
                count += 1
        return count


class SignalRunner:
    """Owns the live loop across every configured market.

    Also serves the Telegram control panel: ``analyze_now``, ``signals_today``,
    ``open_signals`` and ``connection_state`` all take an optional symbol and
    default to the active market.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = JsonStateStore(config.global_state_file)
        self.runtime = RuntimeState.load(config, self.store)
        self.market = MarketData(config)
        self.notifier = TelegramNotifier(config)
        self.engine = SignalEngine(config)

        # Execution is opt-in and off by default.  The broker object is built
        # unconditionally so positions from a previous DEMO_AUTO session are
        # still reconciled and managed, but nothing connects or trades unless
        # both EXECUTION_MODE and DEMO_TRADING_ENABLED say so.
        self.execution_settings = load_execution_settings()
        try:
            self.execution_settings.validate()
        except ValueError as exc:
            LOGGER.error("Execution settings are invalid (%s) - execution disabled", exc)
            self.execution_settings.demo_trading_enabled = False
        self.broker = MT5DemoBroker(self.execution_settings)

        # The PPO layer is entirely optional and defaults to RULE_ONLY.  The
        # bridge owns every import of ai/, so a missing torch, a missing model
        # or a broken checkpoint degrades to the rule engine rather than to a
        # crash.  main.py never imports ai/ directly, which is what keeps the
        # dependency arrow one-way.
        self.ppo = PPOBridge(config)
        # Every market gets its own slot up front - tracker, runtime and CSV
        # files included - so a market keeps being tracked whether or not it is
        # the one currently selected in Telegram.
        self.slots: Dict[str, MarketSlot] = {
            symbol: MarketSlot(
                symbol, config, self.runtime, self.notifier,
                execution_settings=self.execution_settings, broker=self.broker,
            )
            for symbol in MARKET_ORDER
        }
        self.control = TelegramController(
            config, self.runtime, self.notifier, engine=self, report_loader=self._load_report
        )
        self.running = False
        self.today: date = now_utc().date()

    def slot(self, symbol: Optional[str] = None) -> MarketSlot:
        """The slot for ``symbol``, defaulting to the active market."""
        return self.slots[self.runtime.market(symbol).symbol]

    #: internal alias kept for readability at call sites
    _slot = slot

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> bool:
        """Connect everything and print the startup dashboard."""
        # Launching the process is an implicit START: a persisted STOP would
        # otherwise make the program exit immediately with no explanation.
        # A persisted PAUSE is meaningful and is deliberately kept.
        if self.runtime.is_stopped:
            LOGGER.info("Persisted state was STOPPED - starting in RUNNING state")
            self.runtime.start()

        mt5_ok = self.market.connect()
        telegram_ok = self.notifier.test_connection()
        execution_ok = self.start_execution()
        self._print_dashboard(mt5_ok, telegram_ok, execution_ok)

        if not mt5_ok:
            LOGGER.error("Cannot start without an MT5 data connection")
            if not MT5_AVAILABLE:
                LOGGER.error(
                    "The MetaTrader5 package is Windows-only. Use backtest.py on other platforms."
                )
            return False

        if telegram_ok:
            self.control.start()
            self.control.send_panel()

        self.running = True
        LOGGER.info("Scalping engine started (markets: %s)", ", ".join(MARKET_ORDER))
        return True

    def stop(self) -> None:
        """Shut down cleanly."""
        if not self.running:
            return
        self.running = False
        LOGGER.info("Shutting down")
        try:
            self.control.stop()
            if self.notifier.connected:
                self.notifier.send_text("⏹️ M1 scalper stopped")
        finally:
            self.market.shutdown()

    # -- Telegram control-panel hooks --------------------------------------- #
    def signals_today(self, symbol: Optional[str] = None) -> int:
        return self._slot(symbol).signals_today()

    def open_signals(self, symbol: Optional[str] = None) -> int:
        return len(self._slot(symbol).tracker.active_signals())

    def connection_state(self) -> str:
        return "CONNECTED" if self.market.connected else "DISCONNECTED"

    def analyze_now(self, symbol: Optional[str] = None):
        """Evaluate the latest **closed** candle on demand, with no side effects.

        Used by the ANALYSIS button.  It records nothing, sends nothing and does
        not advance the processed-candle marker, so pressing it can never emit
        or suppress a signal - on any market.
        """
        slot = self._slot(symbol)
        cfg = slot.config_view()
        snapshot, reason = self.market.build_snapshot(cfg, use_cache=True)
        if snapshot is None:
            LOGGER.info("[%s] on-demand analysis unavailable: %s", slot.symbol, reason)
            return None
        gate = slot.tracker.gate_state(snapshot.candle_time, SIGNAL_TIMEFRAME)
        return self.engine.evaluate(snapshot, gate, config=cfg)

    # -- demo execution ----------------------------------------------------- #
    def start_execution(self) -> bool:
        """Connect the demo broker, verify the account and reconcile positions.

        Returns True only when demo execution is fully armed.  Every failure
        path leaves execution OFF rather than half-on: an unverified account, a
        failed connection or a failed reconciliation all mean no orders.
        """
        settings = self.execution_settings
        blocking = settings.blocking_reason()
        if blocking:
            LOGGER.info("Demo execution is off: %s", blocking)
            return False

        if not self.broker.connect():
            LOGGER.error("Demo broker would not connect - execution stays off")
            self.notifier.send_text(
                "🚨 DEMO EXECUTION BLOCKED\n\n"
                "Could not connect to the demo account.\n\nNo orders will be placed."
            )
            settings.demo_trading_enabled = False
            return False

        account = self.broker.account()
        if not account.verified or not account.is_demo:
            reason = (
                "Account type could not be verified." if not account.verified
                else f"The connected account is {account.trade_mode}, not DEMO."
            )
            LOGGER.error("DEMO EXECUTION BLOCKED: %s", reason)
            self.notifier.send_text(
                f"🚨 DEMO EXECUTION BLOCKED\n\n{reason}\n\n"
                "Execution has been disabled. No order was placed."
            )
            # Belt and braces: the per-order guard would refuse anyway, but the
            # switch is turned off so nothing even tries.
            settings.demo_trading_enabled = False
            self.broker.shutdown()
            return False

        LOGGER.info("Demo execution armed on %s", account.describe())

        # Reconcile BEFORE any new signal can be executed, so a restart can
        # never duplicate an order for a position that is already open.
        for slot in self.slots.values():
            if slot.execution and not slot.execution.reconcile():
                LOGGER.error("[%s] reconciliation failed - execution halted", slot.symbol)
                return False
        return True

    def execution_state(self) -> str:
        """One-word execution status for the dashboard and Telegram panel."""
        settings = self.execution_settings
        if not settings.executes:
            return "SIGNAL_ONLY"
        if any(s.execution and s.execution.halted for s in self.slots.values()):
            return "HALTED"
        return "DEMO_AUTO"

    def demo_trades_today(self, symbol: Optional[str] = None) -> int:
        if symbol is None:
            return sum(
                s.execution.trades_today() for s in self.slots.values() if s.execution
            )
        slot = self._slot(symbol)
        return slot.execution.trades_today() if slot.execution else 0

    def demo_open_trades(self, symbol: Optional[str] = None) -> int:
        if symbol is None:
            return sum(
                len(s.execution.open_trades()) for s in self.slots.values() if s.execution
            )
        slot = self._slot(symbol)
        return len(slot.execution.open_trades()) if slot.execution else 0

    def demo_net_today(self, symbol: Optional[str] = None) -> float:
        if symbol is None:
            return round(sum(
                s.execution.realised_today() for s in self.slots.values() if s.execution
            ), 2)
        slot = self._slot(symbol)
        return slot.execution.realised_today() if slot.execution else 0.0

    def open_trade_lines(self, symbol: Optional[str] = None) -> List[str]:
        """OPEN TRADES panel body, across every market or just one."""
        slots = (
            list(self.slots.values()) if symbol is None else [self._slot(symbol)]
        )
        lines: List[str] = []
        for slot in slots:
            if slot.execution:
                lines.extend(slot.execution.describe_open())
        return lines

    def execution_blocking_reason(self, symbol: Optional[str] = None) -> str:
        """Why demo execution cannot be armed right now, or ``""``.

        Surfaced in Telegram so a refused toggle explains itself on the panel.
        Sending the user to the console to find out why a button did nothing is
        not an acceptable answer.
        """
        settings = self.execution_settings
        # Deployment-level only: turning DEMO AUTO off from Telegram must not
        # make the panel report the deployment as unconfigured, or there would
        # be no way to turn it back on.
        reason = settings.deployment_blocking_reason()
        if reason:
            return reason
        halted = [
            slot for slot in self.slots.values()
            if slot.execution and slot.execution.halted
        ]
        if halted:
            return f"execution halted: {halted[0].execution.halt_reason}"
        return ""

    def set_demo_auto(self, enabled: bool) -> bool:
        """Turn DEMO AUTO on or off from Telegram.

        Enabling re-runs the full arming sequence - connect, verify, reconcile -
        so the toggle can never arm execution against an unverified account just
        because it was verified earlier in the session.

        ``EXECUTION_MODE`` is deliberately NOT changed here.  It is the
        deployment-level arming switch, set once in ``.env`` by whoever
        configured the demo account; a Telegram button that could flip it would
        make that switch meaningless.  When it is not set, this refuses and the
        panel says exactly which line is missing.
        """
        settings = self.execution_settings
        if not enabled:
            settings.demo_trading_enabled = False
            LOGGER.info("DEMO AUTO disabled from Telegram")
            return False

        if not settings.mode.executes:
            LOGGER.warning(
                "DEMO AUTO requested but EXECUTION_MODE is %s. "
                "Set EXECUTION_MODE=DEMO_AUTO in .env and restart.",
                settings.mode.value,
            )
            return False

        settings.demo_trading_enabled = True
        if not settings.has_demo_account:
            LOGGER.warning(
                "DEMO AUTO requested but no demo account is configured: %s",
                settings.blocking_reason(),
            )
            settings.demo_trading_enabled = False
            return False
        if not self.start_execution():
            return False
        LOGGER.info("DEMO AUTO enabled from Telegram")
        return True

    # -- PPO hooks (read by the Telegram panel) ----------------------------- #
    def strategy_mode(self, symbol: Optional[str] = None) -> str:
        return self.ppo.mode_name()

    def set_strategy_mode(self, mode: str) -> str:
        """Switch RULE_ONLY / PPO_SHADOW / PPO_DEMO.  Returns what took effect."""
        return self.ppo.set_mode(mode)

    def ppo_state(self, symbol: Optional[str] = None) -> Dict[str, object]:
        return self.ppo.state(self.runtime.market(symbol).symbol)

    def on_market_changed(self, previous: str, current: str) -> None:
        """Hook fired by the Telegram panel after the active market changes.

        Switching is a SELECTION, not a reset.  Nothing belonging to
        ``previous`` is cleared: its tracker, CSV files, cooldowns, processed
        candle marker and open paper trades all stay exactly as they were, and
        its open trades keep being tracked against its OWN candles on every
        subsequent cycle.  The only work here is dropping cached candle frames
        so the newly selected market is read fresh.
        """
        self.market.clear_cache()
        LOGGER.info(
            "Active market %s -> %s (%s keeps %s open paper trade(s))",
            previous, current, previous, len(self._slot(previous).tracker.active_signals()),
        )

    def _load_report(self, symbol: Optional[str] = None):
        """Performance report for one market, or every market combined."""
        from performance import analyse, analyse_combined

        if symbol == "COMBINED":
            return analyse_combined(self.config, MARKET_ORDER)
        view = self.config.for_market(self.runtime.market(symbol).symbol)
        return analyse(view.signals_csv, view.outcomes_csv, view)

    # -- dashboard ---------------------------------------------------------- #
    def _print_dashboard(
        self, mt5_ok: bool, telegram_ok: bool, execution_ok: bool = False
    ) -> None:
        """Print the startup console dashboard."""
        cfg = self.config
        state = self.runtime.describe()
        lines = [
            "=" * BANNER_WIDTH,
            f"M1 SCALPER - {' / '.join(MARKET_ORDER)}",
            f"Status: {state['status']}" if mt5_ok else "Status: NOT CONNECTED",
            "=" * BANNER_WIDTH,
            f"MT5:      {'CONNECTED' if mt5_ok else 'DISCONNECTED'}",
            f"Telegram: {'CONNECTED' if telegram_ok else 'DISABLED / UNAVAILABLE'}",
            f"Execution: {self.execution_state()}"
            + ("   (armed on a verified DEMO account)" if execution_ok else ""),
            "",
            f"Active market: {state['label']}",
            f"Mode:      {MODE_SCALPING}      Timeframe: {SIGNAL_TIMEFRAME}",
            f"Broker UTC offset: {cfg.mt5_server_utc_offset_hours:+.1f}h",
            "",
        ]
        for symbol in MARKET_ORDER:
            market = get_market(symbol)
            view = self.config.for_market(symbol)
            described = self.runtime.market(symbol).describe()
            slot = self.slots.get(symbol)
            cost = view.pips(view.round_trip_cost(None))
            lines += [
                f"{market.label()}"
                f"{'   <- active' if symbol == state['active_market'] else ''}",
                f"   threshold {described['threshold']:.0f}"
                f"   hold {described['max_holding_candles']}m"
                f"   cooldown {described['cooldown_candles']}m"
                f"   24/7 {'yes' if market.is_24h else 'no'}",
                f"   cost model {market.cost_model_name}: "
                f"{cost:.1f}{market.pip_name} round trip",
                f"   data {view.market_dir.name}/  "
                f"signals today {slot.signals_today() if slot else 0}, "
                f"open {len(slot.tracker.active_signals()) if slot else 0}",
            ]
            if slot and slot.execution:
                lines.append(
                    f"   demo trades today {slot.execution.trades_today()}, "
                    f"open {len(slot.execution.open_trades())}, "
                    f"net {slot.execution.realised_today():+.2f}"
                )
        execution_note = (
            "DEMO AUTO: orders are placed on the configured DEMO account only."
            if execution_ok else
            "SIGNAL ONLY - no orders are sent."
        )
        lines += [
            "",
            execution_note,
            "This system has no live-trading mode.",
            *[
                f"{symbol} parameters are INITIAL RESEARCH PARAMETERS."
                for symbol in MARKET_ORDER
                if "INITIAL RESEARCH PARAMETERS" in get_market(symbol).note.upper()
            ],
            "=" * BANNER_WIDTH,
        ]
        print("\n".join(lines), flush=True)

    # -- main loop ----------------------------------------------------------- #
    def run(self) -> None:
        """Poll until interrupted; evaluate once per newly closed candle."""
        while self.running:
            try:
                self._tick()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                LOGGER.exception("Unhandled error in main loop: %s", exc)
            if self.running:
                time.sleep(max(self.config.poll_seconds, 1))

    def _tick(self) -> None:
        """One poll cycle, across every market that needs attention."""
        if self.runtime.is_stopped:
            LOGGER.info("STOP requested from Telegram - shutting down")
            self.running = False
            return

        evaluating = set(self.runtime.evaluation_markets())
        # A market is worth fetching data for when it is being evaluated, or
        # when it still has an open paper trade.  That second case is what keeps
        # a signal alive after the user switches away from its market.
        needed: List[str] = [
            symbol for symbol in MARKET_ORDER
            if symbol in evaluating
            or self.slots[symbol].has_open_signals()
            or self.slots[symbol].has_open_demo_trades()
        ]
        for symbol in needed:
            try:
                self._tick_market(self.slots[symbol], evaluate=symbol in evaluating)
            except Exception as exc:  # noqa: BLE001 - one market must not stop the rest
                LOGGER.exception("[%s] cycle failed: %s", symbol, exc)

    def _tick_market(self, slot: MarketSlot, evaluate: bool) -> None:
        """Track and (optionally) evaluate ONE market.

        Everything here reads and writes only ``slot``'s own config view,
        tracker, candles and state file.
        """
        cfg = slot.config_view()
        market_runtime = slot.market_runtime
        last_processed = market_runtime.last_processed_candle()

        # Cheap probe first: M1 closes once a minute but we poll every few
        # seconds, so most polls have nothing new to look at.
        latest = self.market.latest_closed_candle_time(cfg.symbol, SIGNAL_TIMEFRAME)
        has_new_candle = latest is not None and (
            last_processed is None or latest > last_processed
        )
        # An open DEMO position is managed on EVERY cycle, not only when a
        # candle closes: a stop or a target can be reached mid-candle, and
        # waiting a full minute to notice would misreport the exit.
        self._manage_demo_trades(slot)

        if (
            latest is not None
            and not has_new_candle
            and not slot.has_open_signals()
        ):
            return

        snapshot, reason = self.market.build_snapshot(cfg, use_cache=not has_new_candle)
        if snapshot is None:
            # Gold does not trade at the weekend; Bitcoin always does, so a gap
            # there is a real problem rather than an expected one.
            if is_weekend(now_utc()) and not cfg.is_24h:
                LOGGER.debug("[%s] snapshot unavailable (market closed): %s", cfg.symbol, reason)
            else:
                LOGGER.warning("[%s] snapshot unavailable: %s", cfg.symbol, reason)
            return

        # 1. Outcome tracking runs even while PAUSED, and even for a market the
        #    user has switched away from: an open scalp must still reach its
        #    target, its stop or its timeout, against its OWN candles.
        self._track_open_signals(slot, cfg, snapshot)

        if not evaluate or not self.runtime.is_running:
            return

        candle_time = snapshot.candle_time
        if last_processed is not None and candle_time <= last_processed:
            return  # no new closed M1 candle on this market yet

        LOGGER.debug(
            "[%s] new closed M1 candle %s (close %.2f)",
            cfg.symbol, iso(candle_time), snapshot.close,
        )
        self._roll_day(candle_time)

        # 2. evaluate the new candle
        gate = slot.tracker.gate_state(candle_time, SIGNAL_TIMEFRAME)
        evaluation = self.engine.evaluate(snapshot, gate, config=cfg)
        append_csv(cfg.evaluations_csv, evaluation.to_row(), EVALUATION_COLUMNS)

        if evaluation.has_signal and evaluation.signal is not None:
            slot.tracker.invalidate_opposite(
                evaluation.signal.direction, candle_time, snapshot.close
            )
            # Paper recording happens FIRST and unconditionally.  Execution is a
            # downstream consumer: whether or not a demo order is placed, the
            # signal's own outcome keeps being tracked, so signal performance
            # and execution performance stay independently measurable.
            slot.tracker.record_signal(evaluation.signal)
            self._execute_signal(slot, evaluation.signal)
        elif evaluation.near_signal:
            LOGGER.info(
                "[%s] NEAR SIGNAL | best %.1f vs threshold %.1f | %s",
                cfg.symbol, evaluation.best_score, evaluation.threshold,
                evaluation.rejection_reason,
            )
            if self.runtime.near_signal_alerts:
                self.notifier.send_near_signal(evaluation)

        # 3. PPO observes EVERY evaluated candle, signal or not.  Knowing when
        #    the agent declined is as informative as knowing when it acted, and
        #    a recorder that only sees signal candles cannot be compared with
        #    one that sees all of them.
        self._observe_ppo(slot, cfg, snapshot, evaluation)

        # 4. remember the candle so it is never processed twice - per market
        market_runtime.mark_candle_processed(iso(candle_time))

    def _execute_signal(self, slot: MarketSlot, signal) -> None:
        """Hand a recorded signal to the demo execution layer, if it is armed.

        Never raises into the signal loop: a refused or failed demo order must
        not stop the engine from evaluating the next candle.
        """
        if slot.execution is None or not self.execution_settings.executes:
            return
        try:
            decision = slot.execution.execute_signal(signal)
        except DemoExecutionBlocked as exc:
            # The account is not a verified demo account.  Turn execution off
            # for the whole process rather than letting the next signal retry.
            LOGGER.error("[%s] %s", slot.symbol, exc)
            self.execution_settings.demo_trading_enabled = False
            return
        except Exception as exc:  # noqa: BLE001 - execution must not kill the loop
            LOGGER.exception("[%s] demo execution failed: %s", slot.symbol, exc)
            return
        if not decision.executed and decision.reason:
            LOGGER.info("[%s] no demo order: %s", slot.symbol, decision.reason)

    def _observe_ppo(self, slot: MarketSlot, cfg: Config, snapshot, evaluation) -> None:
        """Let PPO see this candle.  Never raises into the signal loop.

        In RULE_ONLY this is a no-op.  In PPO_SHADOW it records and simulates.
        In PPO_DEMO the bridge may return a signal, which is then executed
        through the SAME path and the same ten gates a rule signal uses -
        PPO gets no privileged route to the broker.
        """
        if not self.ppo.active:
            return
        try:
            ppo_signal = self.ppo.observe(
                slot.symbol, cfg, snapshot, evaluation.signal if evaluation else None
            )
        except Exception as exc:  # noqa: BLE001 - PPO must never stop the scalper
            LOGGER.exception("[%s] PPO observation failed: %s", slot.symbol, exc)
            return
        if ppo_signal is not None:
            LOGGER.info("[%s] PPO signal %s - routing through the standard gates",
                        slot.symbol, ppo_signal.direction)
            self._execute_signal(slot, ppo_signal)

    def _manage_demo_trades(self, slot: MarketSlot) -> None:
        """Advance this market's open demo positions.

        Runs for every market with an open position regardless of which one is
        selected in Telegram and regardless of PAUSE, because a live position
        must always reach its exit (spec sections 11, 12, 21).
        """
        if slot.execution is None or not slot.execution.open_trades():
            return
        try:
            slot.execution.manage()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("[%s] demo trade management failed: %s", slot.symbol, exc)

    def _track_open_signals(self, slot: MarketSlot, cfg: Config, snapshot) -> None:
        """Advance this market's open scalps against this market's candles.

        Ticks are replayed where available so a candle which traded through both
        the target and the stop is resolved by what actually happened first.
        """
        if not slot.has_open_signals():
            return
        ticks = None
        if cfg.use_ticks_for_ambiguous_candles:
            try:
                ticks = self.market.refresh_tick_buffer(
                    cfg.symbol, cfg.max_holding_candles + 2
                )
            except Exception as exc:  # noqa: BLE001 - ticks are an optimisation
                LOGGER.debug("[%s] tick refresh failed: %s", cfg.symbol, exc)
        try:
            slot.tracker.update(snapshot.signal_df, ticks, timeframe=SIGNAL_TIMEFRAME)
        except Exception as exc:  # noqa: BLE001 - tracking must not kill the loop
            LOGGER.exception("[%s] outcome tracking failed: %s", cfg.symbol, exc)

    def _roll_day(self, candle_time: datetime) -> None:
        """Note the day rollover (the counter itself is derived from the CSV)."""
        if candle_time.date() != self.today:
            self.today = candle_time.date()


def main() -> int:
    """Entry point."""
    try:
        config = load_config()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(config.log_file, config.log_level)
    LOGGER.info("=" * 60)
    LOGGER.info("XAUUSD signal engine starting (paper signal mode)")

    runner = SignalRunner(config)

    def handle_stop(signum, _frame):  # noqa: ANN001 - signal handler signature
        LOGGER.info("Received signal %s - stopping", signum)
        runner.running = False

    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        try:
            os_signal.signal(sig, handle_stop)
        except (ValueError, AttributeError, OSError):
            pass  # not available on every platform/thread

    if not runner.start():
        return 1
    try:
        runner.run()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        runner.stop()
    LOGGER.info("XAUUSD signal engine stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
