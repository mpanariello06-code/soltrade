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
from src.markets import MARKET_ORDER, get_market
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

    def __init__(self, symbol: str, config: Config, runtime, notifier) -> None:
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

    def config_view(self) -> Config:
        """This market's config with its live Telegram settings folded in."""
        return effective_config(self.base_config, self.runtime, self.symbol)

    def has_open_signals(self) -> bool:
        return bool(self.tracker.active_signals())

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
        # Every market gets its own slot up front - tracker, runtime and CSV
        # files included - so a market keeps being tracked whether or not it is
        # the one currently selected in Telegram.
        self.slots: Dict[str, MarketSlot] = {
            symbol: MarketSlot(symbol, config, self.runtime, self.notifier)
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
        self._print_dashboard(mt5_ok, telegram_ok)

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
    def _print_dashboard(self, mt5_ok: bool, telegram_ok: bool) -> None:
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
        lines += [
            "",
            "PAPER TEST ONLY - no orders are ever sent.",
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
            if symbol in evaluating or self.slots[symbol].has_open_signals()
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
        if latest is not None and not has_new_candle and not slot.has_open_signals():
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
            slot.tracker.record_signal(evaluation.signal)
        elif evaluation.near_signal:
            LOGGER.info(
                "[%s] NEAR SIGNAL | best %.1f vs threshold %.1f | %s",
                cfg.symbol, evaluation.best_score, evaluation.threshold,
                evaluation.rejection_reason,
            )
            if self.runtime.near_signal_alerts:
                self.notifier.send_near_signal(evaluation)

        # 3. remember the candle so it is never processed twice - per market
        market_runtime.mark_candle_processed(iso(candle_time))

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
