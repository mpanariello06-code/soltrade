"""XAUUSD signal engine - live paper-signal runner.

Connects to MetaTrader 5 for **market data only**, evaluates one signal per
newly closed M5 candle, notifies Telegram and logs everything to CSV.

This program never places, modifies or closes an order.
"""

from __future__ import annotations

import signal as os_signal
import sys
import time
from datetime import datetime
from typing import Optional

from config import Config, load_config
from src.logger import get_logger, setup_logging
from src.market_data import MarketData, MT5_AVAILABLE
from src.signal_engine import EVALUATION_COLUMNS, SignalEngine
from src.signal_tracker import SignalTracker, append_csv, ensure_csv
from src.telegram_bot import TelegramNotifier
from src.utils import is_weekend, iso, now_utc

LOGGER = get_logger("main")

BANNER_WIDTH = 46


class SignalRunner:
    """Owns the live loop: poll -> detect new candle -> evaluate -> notify."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.market = MarketData(config)
        self.notifier = TelegramNotifier(config)
        self.tracker = SignalTracker(config, self.notifier)
        self.engine = SignalEngine(config)
        self.running = False
        self.signals_today = 0
        self.today = now_utc().date()
        self.last_evaluation: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> bool:
        """Connect everything and print the startup dashboard."""
        cfg = self.config
        ensure_csv(cfg.evaluations_csv, EVALUATION_COLUMNS)
        self.tracker.load()

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
            self.notifier.send_text(
                f"✅ XAUUSD signal engine started\nSymbol: {cfg.symbol}\n"
                f"Signal TF: {cfg.signal_timeframe}\nMode: PAPER SIGNAL (no trades placed)"
            )
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
            if self.notifier.connected:
                self.notifier.send_text("⏹️ XAUUSD signal engine stopped")
        finally:
            self.market.shutdown()

    # -- dashboard ---------------------------------------------------------- #
    def _print_dashboard(self, mt5_ok: bool, telegram_ok: bool) -> None:
        """Print the simple console dashboard (spec section 40)."""
        cfg = self.config
        last_signal = "none yet"
        if self.tracker.signals:
            latest = self.tracker.signals[-1]
            last_signal = (
                f"{latest.get('direction')} @ {latest.get('entry')} "
                f"({latest.get('timestamp')}) status={latest.get('status')}"
            )
        last_candle = self.tracker.last_processed_candle
        lines = [
            "=" * BANNER_WIDTH,
            "XAUUSD SIGNAL ENGINE",
            "Status: RUNNING" if mt5_ok else "Status: NOT CONNECTED",
            "=" * BANNER_WIDTH,
            f"MT5:      {'CONNECTED' if mt5_ok else 'DISCONNECTED'}",
            f"Telegram: {'CONNECTED' if telegram_ok else 'DISABLED / UNAVAILABLE'}",
            "",
            f"Symbol:    {cfg.symbol}",
            f"Signal TF: {cfg.signal_timeframe}",
            f"HTF:       {cfg.intermediate_timeframe} / {cfg.higher_timeframe}",
            f"Sessions:  {', '.join(cfg.allowed_sessions)}",
            f"Threshold: {cfg.base_threshold:.0f} (adaptive by regime)",
            f"Broker UTC offset: {cfg.mt5_server_utc_offset_hours:+.1f}h",
            "",
            f"Last candle: {iso(last_candle) if last_candle else 'none yet'}",
            f"Last signal: {last_signal}",
            f"Signals today: {self.signals_today}",
            "",
            "MODE: PAPER SIGNAL ONLY - no orders are ever sent.",
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
            time.sleep(max(self.config.poll_seconds, 1))

    def _tick(self) -> None:
        """One poll cycle."""
        snapshot, reason = self.market.build_snapshot()
        if snapshot is None:
            # Gold does not trade at the weekend; stale data then is expected and
            # would otherwise fill the log with warnings for two days.
            if is_weekend(now_utc()):
                LOGGER.debug("Snapshot unavailable (market closed): %s", reason)
            else:
                LOGGER.warning("Snapshot unavailable: %s", reason)
            return

        candle_time = snapshot.candle_time
        last_processed = self.tracker.last_processed_candle
        if last_processed is not None and candle_time <= last_processed:
            return  # no new closed candle yet

        LOGGER.info("New closed candle %s (close %.2f)", iso(candle_time), snapshot.close)
        self._roll_day(candle_time)

        # 1. update open signals first so the gate sees current active counts
        self.tracker.update(snapshot.m5, snapshot.m1)

        # 2. evaluate the new candle
        gate = self.tracker.gate_state(candle_time)
        evaluation = self.engine.evaluate(snapshot, gate)
        append_csv(self.config.evaluations_csv, evaluation.to_row(), EVALUATION_COLUMNS)

        if evaluation.has_signal and evaluation.signal is not None:
            self.tracker.invalidate_opposite(
                evaluation.signal.direction, candle_time, snapshot.close
            )
            self.tracker.record_signal(evaluation.signal)
            self.signals_today += 1
        else:
            LOGGER.info(
                "No signal | bull %.1f / bear %.1f | %s | %s",
                evaluation.card.bullish_score if evaluation.card else 0.0,
                evaluation.card.bearish_score if evaluation.card else 0.0,
                evaluation.regime or "-",
                evaluation.rejection_reason or "-",
            )

        # 3. remember the candle so it is never processed twice
        self.tracker.save_state(last_processed_candle=iso(candle_time))

    def _roll_day(self, candle_time: datetime) -> None:
        """Reset the per-day counter shown on the console."""
        if candle_time.date() != self.today:
            self.today = candle_time.date()
            self.signals_today = 0


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
