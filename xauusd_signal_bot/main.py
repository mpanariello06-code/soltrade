"""XAUUSD M1 micro-scalping research runner.

Connects to MetaTrader 5 for **market data only**, evaluates one candidate per
newly closed M1 candle, notifies Telegram and logs everything to CSV.

This program never places, modifies or closes an order.

SINGLE PURPOSE
--------------
One timeframe (M1) and one mode (SCALPING).  The multi-timeframe selector and
the RESEARCH/STANDARD/CONSERVATIVE modes were removed; what remains is a
dedicated micro-scalping research system.

Threshold, holding period, cooldown, minimum R:R, session filter and run state
are controllable from the Telegram panel while the process runs and are
persisted in ``data/state.json``.

Every reward figure is reported twice - RAW and NET of the spread, slippage and
commission assumed at signal time.  At a target of a few pips the cost is a
large fraction of the move, so a raw number on its own would be misleading.
"""

from __future__ import annotations

import signal as os_signal
import sys
import time
from datetime import date, datetime
from config import Config, load_config
from src.logger import get_logger, setup_logging
from src.market_data import MT5_AVAILABLE, MarketData
from src.runtime_state import JsonStateStore, RuntimeState, effective_config
from src.signal_engine import EVALUATION_COLUMNS, SignalEngine
from src.signal_tracker import SignalTracker, append_csv, ensure_csv
from src.telegram_bot import TelegramNotifier
from src.telegram_control import TelegramController
from src.timeframes import MODE_SCALPING, SIGNAL_TIMEFRAME
from src.utils import is_weekend, iso, now_utc, parse_iso

LOGGER = get_logger("main")

BANNER_WIDTH = 46


class SignalRunner:
    """Owns the live loop: poll -> detect new candle -> evaluate -> notify.

    Also serves the Telegram control panel: it exposes ``analyze_now()``,
    ``signals_today()``, ``connection_state()`` and ``on_timeframe_changed()``.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = JsonStateStore(config.state_file)
        self.runtime = RuntimeState.load(config, self.store)
        self.market = MarketData(config)
        self.notifier = TelegramNotifier(config)
        self.tracker = SignalTracker(config, self.notifier, store=self.store)
        self.engine = SignalEngine(config)
        self.control = TelegramController(
            config, self.runtime, self.notifier, engine=self, report_loader=self._load_report
        )
        self.running = False
        self.today: date = now_utc().date()

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> bool:
        """Connect everything and print the startup dashboard."""
        cfg = self.config
        ensure_csv(cfg.evaluations_csv, EVALUATION_COLUMNS)
        self.tracker.load()

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
        LOGGER.info("Signal engine started")
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
                self.notifier.send_text("⏹️ XAUUSD signal engine stopped")
        finally:
            self.market.shutdown()

    # -- Telegram control-panel hooks --------------------------------------- #
    def signals_today(self) -> int:
        """Signals recorded so far today, across every timeframe."""
        today = now_utc().date()
        count = 0
        for row in self.tracker.signals:
            timestamp = parse_iso(str(row.get("timestamp", "")))
            if timestamp is not None and timestamp.date() == today:
                count += 1
        return count

    def open_signals(self) -> int:
        """Scalps still being tracked."""
        return len(self.tracker.active_signals())

    def connection_state(self) -> str:
        """Human-readable MT5 connection state for the panel."""
        return "CONNECTED" if self.market.connected else "DISCONNECTED"

    def analyze_now(self):
        """Evaluate the latest **closed** candle on demand, with no side effects.

        Used by the ANALYSIS button.  It records nothing, sends nothing and does
        not advance the processed-candle marker, so pressing it can never emit
        or suppress a signal.
        """
        cfg = effective_config(self.config, self.runtime)
        snapshot, reason = self.market.build_snapshot(cfg, use_cache=True)
        if snapshot is None:
            LOGGER.info("On-demand analysis unavailable: %s", reason)
            return None
        gate = self.tracker.gate_state(snapshot.candle_time, cfg.signal_timeframe)
        return self.engine.evaluate(snapshot, gate, config=cfg)

    def _load_report(self):
        """Build the performance report from the CSVs for the panel."""
        from performance import analyse

        return analyse(self.config.signals_csv, self.config.outcomes_csv, self.config)

    # -- dashboard ---------------------------------------------------------- #
    def _print_dashboard(self, mt5_ok: bool, telegram_ok: bool) -> None:
        """Print the startup console dashboard."""
        cfg = self.config
        state = self.runtime.describe()
        last_signal = "none yet"
        if self.tracker.signals:
            latest = self.tracker.signals[-1]
            last_signal = (
                f"{latest.get('direction')} @ {latest.get('entry')} "
                f"({latest.get('timestamp')}) status={latest.get('status')}"
            )
        last_candle = self.runtime.last_processed_candle()
        cost_pips = cfg.pips(cfg.round_trip_cost(None))
        lines = [
            "=" * BANNER_WIDTH,
            "XAUUSD M1 SCALPER",
            f"Status: {state['status']}" if mt5_ok else "Status: NOT CONNECTED",
            "=" * BANNER_WIDTH,
            f"MT5:      {'CONNECTED' if mt5_ok else 'DISCONNECTED'}",
            f"Telegram: {'CONNECTED' if telegram_ok else 'DISABLED / UNAVAILABLE'}",
            "",
            f"Symbol:    {cfg.symbol}",
            f"Mode:      {MODE_SCALPING}",
            f"Timeframe: {SIGNAL_TIMEFRAME}   Context: {state['context_timeframe']}",
            f"Threshold: {state['threshold']:.0f} (adaptive by regime)",
            f"Max hold:  {state['max_holding_candles']} minutes",
            f"Sessions:  {', '.join(state['allowed_sessions'])}",
            f"Broker UTC offset: {cfg.mt5_server_utc_offset_hours:+.1f}h",
            "",
            "Assumed round-trip cost:",
            f"  {cost_pips:.1f} pips "
            f"(spread {cfg.assumed_spread_points:.0f} + slippage "
            f"{cfg.slippage_points_entry:.0f}+{cfg.slippage_points_exit:.0f} pts)",
            "",
            f"Last candle: {iso(last_candle) if last_candle else 'none yet'}",
            f"Last signal: {last_signal}",
            f"Signals today: {self.signals_today()}",
            "",
            "PAPER TEST ONLY - no orders are ever sent.",
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
        """One poll cycle."""
        if self.runtime.is_stopped:
            LOGGER.info("STOP requested from Telegram - shutting down")
            self.running = False
            return

        cfg = effective_config(self.config, self.runtime)
        last_processed = self.runtime.last_processed_candle()

        # Cheap probe first: M1 closes once a minute but we poll every few
        # seconds, so most polls have nothing new to look at.
        latest = self.market.latest_closed_candle_time(cfg.symbol, SIGNAL_TIMEFRAME)
        has_new_candle = latest is not None and (
            last_processed is None or latest > last_processed
        )
        has_open_signals = bool(self.tracker.active_signals())
        if latest is not None and not has_new_candle and not has_open_signals:
            return

        snapshot, reason = self.market.build_snapshot(cfg, use_cache=not has_new_candle)
        if snapshot is None:
            # Gold does not trade at the weekend; stale data then is expected.
            if is_weekend(now_utc()):
                LOGGER.debug("Snapshot unavailable (market closed): %s", reason)
            else:
                LOGGER.warning("Snapshot unavailable: %s", reason)
            return

        # 1. Outcome tracking runs even while PAUSED: an open scalp must still
        #    reach its target, its stop or its timeout.
        self._track_open_signals(cfg, snapshot)

        if not self.runtime.is_running:
            return

        candle_time = snapshot.candle_time
        if last_processed is not None and candle_time <= last_processed:
            return  # no new closed M1 candle yet

        LOGGER.debug("New closed M1 candle %s (close %.2f)", iso(candle_time), snapshot.close)
        self._roll_day(candle_time)

        # 2. evaluate the new candle
        gate = self.tracker.gate_state(candle_time, SIGNAL_TIMEFRAME)
        evaluation = self.engine.evaluate(snapshot, gate, config=cfg)
        append_csv(self.config.evaluations_csv, evaluation.to_row(), EVALUATION_COLUMNS)

        if evaluation.has_signal and evaluation.signal is not None:
            self.tracker.invalidate_opposite(
                evaluation.signal.direction, candle_time, snapshot.close
            )
            self.tracker.record_signal(evaluation.signal)
        elif evaluation.near_signal:
            LOGGER.info(
                "NEAR SIGNAL | best %.1f vs threshold %.1f | %s",
                evaluation.best_score, evaluation.threshold, evaluation.rejection_reason,
            )
            if self.runtime.near_signal_alerts:
                self.notifier.send_near_signal(evaluation)

        # 3. remember the candle so it is never processed twice
        self.runtime.mark_candle_processed(SIGNAL_TIMEFRAME, iso(candle_time))

    def _track_open_signals(self, cfg: Config, snapshot) -> None:
        """Advance every open scalp against the newest M1 candles.

        Ticks are replayed where available so that a candle which traded through
        both the target and the stop is resolved by what actually happened
        first, rather than by the pessimistic default.  At this scale that
        distinction decides a meaningful share of outcomes.
        """
        if not self.tracker.active_signals():
            return
        ticks = None
        if cfg.use_ticks_for_ambiguous_candles:
            try:
                ticks = self.market.refresh_tick_buffer(
                    cfg.symbol, cfg.max_holding_candles + 2
                )
            except Exception as exc:  # noqa: BLE001 - ticks are an optimisation
                LOGGER.debug("Tick refresh failed, staying pessimistic: %s", exc)
        try:
            self.tracker.update(snapshot.signal_df, ticks, timeframe=SIGNAL_TIMEFRAME)
        except Exception as exc:  # noqa: BLE001 - tracking must not kill the loop
            LOGGER.exception("Outcome tracking failed: %s", exc)

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
