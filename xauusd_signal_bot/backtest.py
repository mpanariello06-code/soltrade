"""Historical backtester for the XAUUSD M1 scalping engine.

Feeds M1 candles one at a time into the **same** SignalEngine that runs live,
and tracks outcomes with the same PositionState.

NO-LOOKAHEAD GUARANTEES
-----------------------
1. The engine is handed a *slice* ending at the bar being evaluated; nothing
   later is reachable.
2. The context frame is cut by **close time**: an M5 candle is only visible
   once its close time is at or before the M1 candle's close time.
3. The cooldown context is rebuilt from signals dated on or before the bar.
4. Outcome tracking starts on the candle *after* the signal candle, because the
   entry is that candle's close.
5. Indicators are pre-computed once over the whole history and then sliced.
   Every indicator is causal, so the value at bar ``i`` is identical to
   computing it on ``0..i`` - ``tests/test_indicators.py`` asserts this.

COSTS
-----
A backtest has no live spread, so ``assumed_spread_points`` is charged on every
trade unless ``--spread`` overrides it.  NET R is what the report leads with;
raw R is shown for reference only.

AMBIGUOUS CANDLES
-----------------
With no tick feed, any M1 candle that trades through both the target and the
stop is scored **pessimistically** (stop first).  Live tracking replays ticks
and is more accurate, so live and backtested outcomes are not strictly
comparable - the backtest is the conservative one.

Input data
----------
A CSV of M1 candles with columns ``time, open, high, low, close, tick_volume``.

Usage::

    python backtest.py --data history/XAUUSD_M1.csv
    python backtest.py --data history/XAUUSD_M1.csv --spread 12
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import Config, load_config
from performance import build_report, render_report
from src.indicators import compute_indicators
from src.logger import get_logger, setup_logging
from src.market_data import (
    MarketSnapshot,
    clean_candles,
    resample_candles,
    timeframe_minutes,
    validate_candles,
)
from src.timeframes import SIGNAL_TIMEFRAME
from src.signal_engine import EVALUATION_COLUMNS, SignalEngine
from src.signal_tracker import (
    OUTCOME_COLUMNS,
    SIGNAL_COLUMNS,
    STATUS_INVALIDATED,
    PositionState,
    build_gate_state,
    build_outcome_row,
    rewrite_csv,
)
from src.utils import as_utc, iso

LOGGER = get_logger("backtest")

#: accepted column aliases in a history CSV
COLUMN_ALIASES = {
    "date": "time", "datetime": "time", "timestamp": "time",
    "vol": "tick_volume", "volume": "tick_volume", "tickvol": "tick_volume",
    "o": "open", "h": "high", "l": "low", "c": "close",
}


@dataclass
class BacktestResult:
    """Signals, outcomes and evaluation rows produced by one run."""

    signals: List[Dict[str, Any]] = field(default_factory=list)
    outcomes: List[Dict[str, Any]] = field(default_factory=list)
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    bars_evaluated: int = 0
    open_at_end: int = 0
    start: str = ""
    end: str = ""
    elapsed_seconds: float = 0.0

    def signals_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.signals, columns=list(SIGNAL_COLUMNS)) if self.signals else pd.DataFrame()

    def outcomes_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.outcomes, columns=list(OUTCOME_COLUMNS)) if self.outcomes else pd.DataFrame()


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_history(path: Path) -> pd.DataFrame:
    """Load and normalise an M1 history CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"history file not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8")
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    frame = frame.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in frame.columns})

    missing = [c for c in ("time", "open", "high", "low", "close") if c not in frame.columns]
    if missing:
        raise ValueError(f"history file is missing column(s): {', '.join(missing)}")

    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce", format="mixed")
    if "tick_volume" not in frame.columns:
        LOGGER.warning("No volume column found - using a constant; the volume engine will be neutral")
        frame["tick_volume"] = 1.0
    for column in ("open", "high", "low", "close", "tick_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    cleaned = clean_candles(frame)
    LOGGER.info(
        "Loaded %s candles from %s (%s -> %s)",
        len(cleaned), path.name,
        cleaned["time"].iloc[0] if len(cleaned) else "-",
        cleaned["time"].iloc[-1] if len(cleaned) else "-",
    )
    return cleaned


# --------------------------------------------------------------------------- #
# backtester
# --------------------------------------------------------------------------- #
class Backtester:
    """Replays history bar by bar through the live signal engine."""

    def __init__(
        self,
        config: Config,
        spread_points: Optional[float] = None,
    ) -> None:
        self.config = config
        self.engine = SignalEngine(config)
        # A backtest has no live quote, so every trade pays the configured
        # assumption rather than trading for free.
        self.spread_points = (
            config.assumed_spread_points if spread_points is None else float(spread_points)
        )

    # -- preparation -------------------------------------------------------- #
    def prepare(self, source: pd.DataFrame) -> Dict[str, Any]:
        """Pre-compute indicators for M1 and, if enabled, the context timeframe.

        The context frame is resampled from the same M1 file, so no second data
        file is needed.
        """
        cfg = self.config
        params = cfg.indicators

        ok, reason = validate_candles(source, SIGNAL_TIMEFRAME, cfg.min_candles_required)
        if not ok:
            raise ValueError(f"history failed validation: {reason}")

        prepared: Dict[str, Any] = {
            "m1": compute_indicators(source, params),
            "m1_close": (
                pd.to_datetime(source["time"], utc=True) + timedelta(minutes=1)
            ).to_numpy(),
            "context": None,
            "context_close": None,
        }

        if cfg.context_timeframe:
            frame = resample_candles(source, SIGNAL_TIMEFRAME, cfg.context_timeframe)
            prepared["context"] = compute_indicators(frame, params)
            prepared["context_close"] = (
                pd.to_datetime(frame["time"], utc=True)
                + timedelta(minutes=timeframe_minutes(cfg.context_timeframe))
            ).to_numpy()
        return prepared

    def _warmup_index(self, prepared: Dict[str, Any]) -> int:
        """First bar index at which every active timeframe has enough history."""
        cfg = self.config
        m1_close = prepared["m1_close"]
        context_close = prepared["context_close"]

        for index in range(cfg.min_candles_required, len(m1_close)):
            if context_close is None:
                return index
            if (
                np.searchsorted(context_close, m1_close[index], side="right")
                >= cfg.min_context_candles
            ):
                return index
        needed = cfg.min_context_candles * timeframe_minutes(cfg.context_timeframe or "M1")
        raise ValueError(
            f"history is too short: need roughly {max(needed, cfg.min_candles_required)} "
            f"M1 candles to warm up"
        )

    def _snapshot(self, prepared: Dict[str, Any], index: int) -> MarketSnapshot:
        """Build the snapshot visible at the close of M1 bar ``index``."""
        cfg = self.config
        cutoff = prepared["m1_close"][index]

        start = max(0, index + 1 - cfg.candles_signal)
        signal_window = prepared["m1"].iloc[start : index + 1]

        context_window = None
        if prepared["context"] is not None:
            end = int(np.searchsorted(prepared["context_close"], cutoff, side="right"))
            context_window = prepared["context"].iloc[
                max(0, end - cfg.candles_context) : end
            ]

        return MarketSnapshot(
            symbol=cfg.symbol,
            signal_df=signal_window,
            context_df=context_window,
            spread_points=self.spread_points,
            evaluated_at=as_utc(pd.Timestamp(cutoff).to_pydatetime()),
            signal_timeframe=SIGNAL_TIMEFRAME,
            context_timeframe=cfg.context_timeframe or "",
        )

    # -- main loop ----------------------------------------------------------- #
    def run(
        self,
        m5: pd.DataFrame,
        log_evaluations: bool = True,
        progress_every: int = 2000,
    ) -> BacktestResult:
        """Replay ``m5`` and return the resulting signals and outcomes."""
        started = time.time()
        prepared = self.prepare(m5)
        raw = prepared["m1"]
        first = self._warmup_index(prepared)
        total = len(raw)

        result = BacktestResult()
        result.start = iso(as_utc(pd.Timestamp(raw["time"].iloc[first]).to_pydatetime()))
        result.end = iso(as_utc(pd.Timestamp(raw["time"].iloc[-1]).to_pydatetime()))
        LOGGER.info("Replaying bars %s..%s (%s evaluations)", first, total - 1, total - first)

        open_positions: List[Tuple[Dict[str, Any], PositionState]] = []

        # Pre-extract the OHLC arrays so per-bar outcome tracking never touches
        # the DataFrame (which dominated the profile).
        highs = raw["high"].to_numpy(dtype="float64")
        lows = raw["low"].to_numpy(dtype="float64")
        closes = raw["close"].to_numpy(dtype="float64")
        times = pd.to_datetime(raw["time"], utc=True)

        for index in range(first, total):
            bar_time = as_utc(times.iloc[index].to_pydatetime())
            bar = {"high": highs[index], "low": lows[index], "close": closes[index]}

            # 1. advance open signals with this newly closed candle
            still_open: List[Tuple[Dict[str, Any], PositionState]] = []
            for row, state in open_positions:
                if state.step(bar, bar_time):
                    self._close_position(result, row, state)
                else:
                    row["status"] = state.status
                    still_open.append((row, state))
            open_positions = still_open

            # 2. evaluate this candle
            snapshot = self._snapshot(prepared, index)
            gate = build_gate_state(
                result.signals, len(open_positions), bar_time, SIGNAL_TIMEFRAME
            )
            evaluation = self.engine.evaluate(snapshot, gate, config=self.config)
            result.bars_evaluated += 1
            if log_evaluations:
                result.evaluations.append(evaluation.to_row())

            # 3. record any signal
            if evaluation.has_signal and evaluation.signal is not None:
                signal = evaluation.signal
                if self.config.invalidate_on_opposite_signal:
                    remaining = []
                    for row, state in open_positions:
                        if str(row.get("direction")) != signal.direction:
                            state.close_now(STATUS_INVALIDATED, closes[index], bar_time)
                            self._close_position(result, row, state)
                        else:
                            remaining.append((row, state))
                    open_positions = remaining

                row = signal.to_row()
                row.update({"status_updated_at": iso(bar_time), "tp_hits": 0, "mfe_r": 0.0, "mae_r": 0.0})
                result.signals.append(row)
                open_positions.append((row, PositionState(row, self.config)))

            if progress_every and (index - first) % progress_every == 0 and index > first:
                done = index - first
                rate = done / max(time.time() - started, 1e-9)
                remaining_bars = total - index
                print(
                    f"  {done}/{total - first} bars | {len(result.signals)} signals | "
                    f"{rate:.0f} bars/s | ~{remaining_bars / max(rate, 1e-9) / 60:.1f} min left",
                    flush=True,
                )

        result.open_at_end = len(open_positions)
        for row, state in open_positions:
            row["status"] = state.status
            row["tp_hits"] = state.tp_hits
        result.elapsed_seconds = time.time() - started
        LOGGER.info(
            "Backtest finished: %s bars, %s signals, %s closed, %s still open (%.1fs)",
            result.bars_evaluated, len(result.signals), len(result.outcomes),
            result.open_at_end, result.elapsed_seconds,
        )
        return result

    def _close_position(
        self, result: BacktestResult, row: Dict[str, Any], state: PositionState
    ) -> None:
        """Finalise a closed signal into the result set."""
        progress = state.to_progress()
        row["status"] = progress.status
        row["tp_hits"] = progress.tp_hits
        row["mfe_r"] = progress.mfe_r
        row["mae_r"] = progress.mae_r
        result.outcomes.append(
            build_outcome_row(row, progress, self.config.digits, self.config)
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def slice_history(
    m5: pd.DataFrame, start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    """Restrict the history to a date range (inclusive)."""
    frame = m5
    times = pd.to_datetime(frame["time"], utc=True)
    if start:
        frame = frame[times >= pd.Timestamp(start, tz="UTC")]
        times = pd.to_datetime(frame["time"], utc=True)
    if end:
        frame = frame[times <= pd.Timestamp(end, tz="UTC") + timedelta(days=1)]
    return frame.reset_index(drop=True)


def write_outputs(result: BacktestResult, out_dir: Path, prefix: str = "backtest") -> None:
    """Persist the run's signals, outcomes and evaluations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rewrite_csv(out_dir / f"{prefix}_signals.csv", result.signals, SIGNAL_COLUMNS)
    rewrite_csv(out_dir / f"{prefix}_outcomes.csv", result.outcomes, OUTCOME_COLUMNS)
    if result.evaluations:
        rewrite_csv(out_dir / f"{prefix}_evaluations.csv", result.evaluations, EVALUATION_COLUMNS)
    print(f"\nWrote {prefix}_signals.csv / {prefix}_outcomes.csv to {out_dir}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    config = load_config()
    parser = argparse.ArgumentParser(description="XAUUSD M1 scalping backtester")
    parser.add_argument("--data", type=Path, required=True, help="M1 history CSV")
    parser.add_argument("--start", type=str, default=None, help="start date, e.g. 2024-01-01")
    parser.add_argument("--end", type=str, default=None, help="end date, e.g. 2024-06-30")
    parser.add_argument("--out", type=Path, default=config.data_dir, help="output directory")
    parser.add_argument("--prefix", type=str, default="backtest", help="output filename prefix")
    parser.add_argument(
        "--spread", type=float, default=None,
        help="constant spread in points (default: ASSUMED_SPREAD_POINTS from config)",
    )
    parser.add_argument(
        "--no-evaluations", action="store_true",
        help="skip writing the per-candle evaluations file (it is large)",
    )
    args = parser.parse_args(argv)

    setup_logging(config.log_file, config.log_level)
    try:
        history = slice_history(load_history(args.data), args.start, args.end)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        backtester = Backtester(config, spread_points=args.spread)
        result = backtester.run(history, log_evaluations=not args.no_evaluations)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    cost_pips = config.pips(config.round_trip_cost(backtester.spread_points))
    print(
        f"\nMode: SCALPING | timeframe: M1 | context: {config.context_timeframe or '-'} | "
        f"threshold: {config.base_threshold:.0f}"
        f"\nCosts charged: spread {backtester.spread_points:.0f} pts + slippage "
        f"{config.slippage_points_entry:.0f}+{config.slippage_points_exit:.0f} pts "
        f"= {cost_pips:.1f} pips round trip"
    )

    write_outputs(result, args.out, args.prefix)

    print()
    print(f"Period evaluated : {result.start} -> {result.end}")
    print(f"Bars evaluated   : {result.bars_evaluated}")
    print(f"Still open at end: {result.open_at_end}")
    report = build_report(result.signals_frame(), result.outcomes_frame(), config)
    if report.total_signals == 0:
        print("\nNo signals were generated for this period and configuration.")
        return 0
    print(render_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
