"""XAUUSD signal engine - live paper-signal runner.

Connects to MetaTrader 5 for **market data only**, evaluates one signal per
newly closed candle of the active signal timeframe, notifies Telegram and logs
everything to CSV.

This program never places, modifies or closes an order.

OPERATING MODES
---------------
``RESEARCH``     collects far more candidate setups for observation and future
                 ML training.  Every message is labelled as research and is
                 explicitly *not* a validated trading signal.
``STANDARD``     the default.
``CONSERVATIVE`` demands more confirmation than STANDARD.

Mode, signal timeframe, threshold, cooldown, minimum R:R, session filter and
run state are all controllable from the Telegram panel while the process runs,
and are persisted in ``data/state.json``.
"""

from __future__ import annotations

import signal as os_signal
import sys
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd

from config import Config, load_config
from src.logger import get_logger, setup_logging
from src.market_data import MT5_AVAILABLE, MarketData
from src.runtime_state import JsonStateStore, RuntimeState, effective_config
from src.signal_engine import EVALUATION_COLUMNS, SignalEngine
from src.signal_tracker import SignalTracker, append_csv, ensure_csv
from src.telegram_bot import TelegramNotifier
from src.telegram_control import TelegramController
from src.timeframes import MODE_ICONS, micro_timeframe
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

    def connection_state(self) -> str:
        """Human-readable MT5 connection state for the panel."""
        return "CONNECTED" if self.market.connected else "DISCONNECTED"

    def on_timeframe_changed(self, previous: str, current: str) -> None:
        """Drop cached candles belonging to the previous hierarchy.

        Without this, the first evaluation after a switch could mix frames from
        the old and new confirmation sets.
        """
        self.market.clear_cache()
        LOGGER.info("Signal timeframe changed %s -> %s; candle cache cleared", previous, current)

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
        """Print the simple console dashboard (spec section 40)."""
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
        lines = [
            "=" * BANNER_WIDTH,
            "XAUUSD SIGNAL ENGINE",
            f"Status: {state['status']}" if mt5_ok else "Status: NOT CONNECTED",
            "=" * BANNER_WIDTH,
            f"MT5:      {'CONNECTED' if mt5_ok else 'DISCONNECTED'}",
            f"Telegram: {'CONNECTED' if telegram_ok else 'DISABLED / UNAVAILABLE'}",
            "",
            f"Symbol:    {cfg.symbol}",
            f"Mode:      {MODE_ICONS.get(state['mode'], '')} {state['mode']}",
            f"Signal TF: {state['signal_timeframe']}",
            f"Confirm:   {state['confirmation']}",
            f"Threshold: {state['threshold']:.0f} (adaptive by regime)",
            f"Sessions:  {', '.join(state['allowed_sessions'])}",
            f"Broker UTC offset: {cfg.mt5_server_utc_offset_hours:+.1f}h",
            "",
            f"Last candle: {iso(last_candle) if last_candle else 'none yet'}",
            f"Last signal: {last_signal}",
            f"Signals today: {self.signals_today()}",
            "",
            "MODE: PAPER SIGNAL ONLY - no orders are ever sent.",
        ]
        if state["mode"] == "RESEARCH":
            lines.append("RESEARCH MODE: candidates are collected for study,")
            lines.append("not validated trading signals.")
        lines.append("=" * BANNER_WIDTH)
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
        timeframe = cfg.signal_timeframe
        last_processed = self.runtime.last_processed_candle(timeframe)

        # Cheap probe first: most polls happen between candle closes, and there
        # is no reason to re-download several hundred candles on every one of
        # them.  A full fetch only happens when there is a new candle to
        # evaluate, or an open signal whose TP/SL needs checking.
        latest = self.market.latest_closed_candle_time(cfg.symbol, timeframe)
        has_new_candle = latest is not None and (
            last_processed is None or latest > last_processed
        )
        has_open_signals = bool(self.tracker.active_timeframes())
        if latest is not None and not has_new_candle and not has_open_signals:
            return

        snapshot, reason = self.market.build_snapshot(cfg, use_cache=not has_new_candle)
        if snapshot is None:
            # Gold does not trade at the weekend; stale data then is expected and
            # would otherwise fill the log with warnings for two days.
            if is_weekend(now_utc()):
                LOGGER.debug("Snapshot unavailable (market closed): %s", reason)
            else:
                LOGGER.warning("Snapshot unavailable: %s", reason)
            return

        # 1. Outcome tracking runs even while PAUSED: an already-open signal
        #    must still reach its TP or SL, and those alerts are not new signals.
        self._track_open_signals(cfg, snapshot)

        if not self.runtime.is_running:
            return

        candle_time = snapshot.candle_time
        if last_processed is not None and candle_time <= last_processed:
            return  # no new closed candle on this timeframe yet

        LOGGER.info(
            "New closed %s candle %s (close %.2f)", timeframe, iso(candle_time), snapshot.close
        )
        self._roll_day(candle_time)

        # 2. evaluate the new candle
        gate = self.tracker.gate_state(candle_time, timeframe)
        evaluation = self.engine.evaluate(snapshot, gate, config=cfg)
        append_csv(self.config.evaluations_csv, evaluation.to_row(), EVALUATION_COLUMNS)

        if evaluation.has_signal and evaluation.signal is not None:
            self.tracker.invalidate_opposite(
                evaluation.signal.direction, candle_time, snapshot.close
            )
            self.tracker.record_signal(evaluation.signal)
        else:
            if evaluation.near_signal:
                LOGGER.info(
                    "NEAR SIGNAL | best %.1f vs threshold %.1f | %s",
                    evaluation.best_score, evaluation.threshold, evaluation.rejection_reason,
                )
                if self.runtime.near_signal_alerts:
                    self.notifier.send_near_signal(evaluation)
            LOGGER.info(
                "No signal | bull %.1f / bear %.1f | %s | %s",
                evaluation.card.bullish_score if evaluation.card else 0.0,
                evaluation.card.bearish_score if evaluation.card else 0.0,
                evaluation.regime or "-",
                evaluation.rejection_reason or "-",
            )

        # 3. remember the candle so it is never processed twice, per timeframe
        self.runtime.mark_candle_processed(timeframe, iso(candle_time))

    def _track_open_signals(self, cfg: Config, snapshot) -> None:
        """Advance every open signal using candles of *its own* timeframe.

        Signals raised on a timeframe the engine has since moved away from are
        still tracked to completion - switching timeframe must not orphan them.
        """
        active = self.tracker.active_timeframes()
        if not active:
            return

        for timeframe in active:
            try:
                if timeframe == cfg.signal_timeframe:
                    candles, intrabar = snapshot.m5, snapshot.m1
                else:
                    candles = self.market.get_candles(
                        cfg.symbol, timeframe, cfg.candles_signal, use_cache=True
                    )
                    if candles.empty:
                        continue
                    intrabar = self._micro_candles(cfg, timeframe)
                self.tracker.update(candles, intrabar, timeframe=timeframe)
            except Exception as exc:  # noqa: BLE001 - one bad timeframe must not stop the rest
                LOGGER.exception("Outcome tracking failed for %s: %s", timeframe, exc)

    def _micro_candles(self, cfg: Config, timeframe: str) -> Optional[pd.DataFrame]:
        """Finer candles used to resolve ambiguous TP/SL bars, if any exist."""
        micro = micro_timeframe(timeframe)
        if not micro:
            return None
        frame = self.market.get_candles(cfg.symbol, micro, cfg.candles_micro, use_cache=True)
        return None if frame.empty else frame

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
