"""Historical backtester for the XAUUSD signal engine.

Feeds candles one at a time into the **same** :class:`~src.signal_engine.SignalEngine`
that runs live, and tracks outcomes with the same
:class:`~src.signal_tracker.PositionState`.

NO-LOOKAHEAD GUARANTEES
-----------------------
1. The engine is handed a *slice* ending at the bar being evaluated; nothing
   later is reachable.
2. Higher-timeframe frames are cut by **close time**: an H1 candle is only
   visible once its close time is at or before the M5 candle's close time.  A
   forming H1 candle is never shown.
3. The cooldown/limit context is rebuilt from signals dated on or before the
   current bar only.
4. Outcome tracking starts on the candle *after* the signal candle, because the
   entry is that candle's close.
5. Indicators are pre-computed once over the whole history and then sliced.
   Every indicator in :mod:`src.indicators` is causal, so the value at bar ``i``
   is identical to computing it on ``0..i`` - ``tests/test_indicators.py``
   asserts this.

Input data
----------
A CSV of M5 candles with columns ``time, open, high, low, close, tick_volume``
(``volume``/``vol`` and ``date``/``datetime``/``timestamp`` are accepted as
aliases).  M15 and H1 are derived by resampling, so only one file is needed.

Usage::

    python backtest.py --data history/XAUUSD_M5.csv
    python backtest.py --data history/XAUUSD_M5.csv --start 2024-01-01 --end 2024-06-30
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
    """Load and normalise an M5 history CSV."""
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

    def __init__(self, config: Config, spread_points: float = float("nan")) -> None:
        self.config = config
        self.engine = SignalEngine(config)
        self.spread_points = spread_points
        self.signal_minutes = timeframe_minutes(config.signal_timeframe)
        self.intermediate_minutes = timeframe_minutes(config.intermediate_timeframe)
        self.higher_minutes = timeframe_minutes(config.higher_timeframe)

    # -- preparation -------------------------------------------------------- #
    def prepare(self, m5: pd.DataFrame) -> Dict[str, Any]:
        """Derive the HTF series and pre-compute indicators for every timeframe."""
        params = self.config.indicators
        m15 = resample_candles(m5, self.config.signal_timeframe, self.config.intermediate_timeframe)
        h1 = resample_candles(m5, self.config.signal_timeframe, self.config.higher_timeframe)

        ok, reason = validate_candles(m5, self.config.signal_timeframe, self.config.min_candles_required)
        if not ok:
            raise ValueError(f"history failed validation: {reason}")

        return {
            "m5": compute_indicators(m5, params),
            "m15": compute_indicators(m15, params),
            "h1": compute_indicators(h1, params),
            # close times drive the "what was visible then" cut
            "m15_close": (
                pd.to_datetime(m15["time"], utc=True)
                + timedelta(minutes=self.intermediate_minutes)
            ).to_numpy(),
            "h1_close": (
                pd.to_datetime(h1["time"], utc=True)
                + timedelta(minutes=self.higher_minutes)
            ).to_numpy(),
            "m5_close": (
                pd.to_datetime(m5["time"], utc=True) + timedelta(minutes=self.signal_minutes)
            ).to_numpy(),
        }

    def _warmup_index(self, prepared: Dict[str, Any]) -> int:
        """First bar index at which every timeframe has enough closed history."""
        cfg = self.config
        m5_close = prepared["m5_close"]
        m15_close, h1_close = prepared["m15_close"], prepared["h1_close"]

        for index in range(cfg.min_candles_required, len(m5_close)):
            cutoff = m5_close[index]
            if np.searchsorted(m15_close, cutoff, side="right") < cfg.min_htf_candles_required:
                continue
            if np.searchsorted(h1_close, cutoff, side="right") < cfg.min_htf_candles_required:
                continue
            return index
        raise ValueError(
            "history is too short: need roughly "
            f"{cfg.min_htf_candles_required * (self.higher_minutes // self.signal_minutes)} "
            f"{cfg.signal_timeframe} candles to warm up the {cfg.higher_timeframe} view"
        )

    def _snapshot(self, prepared: Dict[str, Any], index: int) -> MarketSnapshot:
        """Build the snapshot visible at the close of bar ``index``."""
        cfg = self.config
        cutoff = prepared["m5_close"][index]

        m5_start = max(0, index + 1 - cfg.candles_signal)
        m5_window = prepared["m5"].iloc[m5_start : index + 1]

        m15_end = int(np.searchsorted(prepared["m15_close"], cutoff, side="right"))
        m15_window = prepared["m15"].iloc[max(0, m15_end - cfg.candles_intermediate) : m15_end]

        h1_end = int(np.searchsorted(prepared["h1_close"], cutoff, side="right"))
        h1_window = prepared["h1"].iloc[max(0, h1_end - cfg.candles_higher) : h1_end]

        return MarketSnapshot(
            symbol=cfg.symbol,
            m5=m5_window,
            m15=m15_window,
            h1=h1_window,
            m1=None,  # no intrabar feed in a backtest -> pessimistic tie-breaking
            spread_points=self.spread_points,
            evaluated_at=as_utc(pd.Timestamp(cutoff).to_pydatetime()),
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
        raw = prepared["m5"]
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
            gate = build_gate_state(result.signals, len(open_positions), bar_time)
            evaluation = self.engine.evaluate(snapshot, gate)
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
        result.outcomes.append(build_outcome_row(row, progress, self.config.digits))


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
    parser = argparse.ArgumentParser(description="XAUUSD signal engine backtester")
    parser.add_argument("--data", type=Path, required=True, help="M5 history CSV")
    parser.add_argument("--start", type=str, default=None, help="start date, e.g. 2024-01-01")
    parser.add_argument("--end", type=str, default=None, help="end date, e.g. 2024-06-30")
    parser.add_argument("--out", type=Path, default=config.data_dir, help="output directory")
    parser.add_argument("--prefix", type=str, default="backtest", help="output filename prefix")
    parser.add_argument(
        "--spread", type=float, default=float("nan"),
        help="simulate a constant spread in points (default: unknown -> spread filter skipped)",
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

    backtester = Backtester(config, spread_points=args.spread)
    try:
        result = backtester.run(history, log_evaluations=not args.no_evaluations)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

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
