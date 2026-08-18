"""Signal persistence, lifecycle management and outcome tracking.

Storage layout (all UTF-8, all created on demand):

``data/signals.csv``     one row per emitted signal, ``status`` updated in place
``data/evaluations.csv`` one append-only row per evaluated candle
``data/outcomes.csv``    one append-only row per closed signal
``data/state.json``      last processed candle and last signal (restart safety)

OUTCOME MODEL
-------------
A signal is modelled as three equal partials taken at TP1/TP2/TP3.  After TP1
is reached the stop moves to breakeven (configurable).  The R multiple is the
size-weighted sum of the realised parts plus the remaining size marked out at
the exit price, so:

* stopped out before TP1            -> -1.00R
* TP1 then stopped at breakeven     -> +0.33R
* TP1, TP2 then breakeven           -> +0.93R
* all three targets                 -> +1.87R  (with 1.0/1.8/2.8 R targets)

AMBIGUOUS CANDLES
-----------------
When a single M5 candle touches both the next target and the stop, OHLC alone
cannot say which came first.  If M1 data is available it is replayed to resolve
the order; otherwise the **pessimistic** assumption is used (stop first).  The
backtester has no M1 feed and therefore always assumes the worse case.

RE-DERIVED, NOT ACCUMULATED
---------------------------
Progress is recomputed from the stored signal plus the candle history on every
cycle rather than incrementally accumulated.  That makes the calculation
idempotent and restart-safe: alerts fire on a *change* against the status
persisted in ``signals.csv``, so a restart cannot replay old notifications.
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .filters import GateState
from .logger import get_logger
from .market_data import timeframe_minutes
from .utils import as_utc, atomic_write_json, fnum, iso, now_utc, parse_iso, read_json

LOGGER = get_logger("tracker")

_CSV_LOCK = threading.Lock()

STATUS_ACTIVE = "ACTIVE"
STATUS_TP1 = "TP1_HIT"
STATUS_TP2 = "TP2_HIT"
STATUS_TP3 = "TP3_HIT"
STATUS_SL = "SL_HIT"
STATUS_EXPIRED = "EXPIRED"
STATUS_INVALIDATED = "INVALIDATED"

TERMINAL_STATUSES = (STATUS_TP3, STATUS_SL, STATUS_EXPIRED, STATUS_INVALIDATED)
#: milestone ordering used to decide which alerts are newly due
STATUS_RANK = {
    STATUS_ACTIVE: 0,
    STATUS_TP1: 1,
    STATUS_TP2: 2,
    STATUS_TP3: 3,
    STATUS_SL: 4,
    STATUS_EXPIRED: 4,
    STATUS_INVALIDATED: 4,
}

SIGNAL_COLUMNS: Tuple[str, ...] = (
    "signal_id", "timestamp", "symbol", "timeframe", "direction",
    "entry", "sl", "tp1", "tp2", "tp3",
    "confidence", "bullish_score", "bearish_score", "regime", "status",
    "session", "risk_reward", "rr1", "rr2", "rr3", "sl_mode",
    "confidence_label", "reason_summary", "status_updated_at", "tp_hits",
    "mfe_r", "mae_r",
)

OUTCOME_COLUMNS: Tuple[str, ...] = (
    "signal_id", "entry", "result", "exit_level", "R_multiple", "duration", "timestamp",
    "symbol", "direction", "signal_time", "bars", "tp_hits", "confidence", "regime",
    "session", "mfe_r", "mae_r",
)


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def _archive_mismatched(path: Path, expected: Sequence[str]) -> None:
    """Move a CSV aside when its header no longer matches the schema.

    Appending rows under a stale header would silently misalign columns, so the
    old file is preserved with a timestamped name instead of being corrupted.
    """
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), None)
    except (OSError, UnicodeDecodeError):
        header = None
    if header is None or list(header) == list(expected):
        return
    backup = path.with_suffix(path.suffix + f".bak-{now_utc():%Y%m%d%H%M%S}")
    try:
        shutil.move(str(path), str(backup))
        LOGGER.warning("Schema of %s changed - previous file archived as %s", path.name, backup.name)
    except OSError as exc:
        LOGGER.error("Could not archive %s: %s", path, exc)


def ensure_csv(path: Path, fieldnames: Sequence[str]) -> None:
    """Create ``path`` with a header row if it does not already exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        _archive_mismatched(path, fieldnames)
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(fieldnames)).writeheader()


def append_csv(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> bool:
    """Append one row, creating the file/header when needed.  Never raises."""
    try:
        with _CSV_LOCK:
            ensure_csv(path, fieldnames)
            with open(path, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(fieldnames), extrasaction="ignore", restval=""
                )
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        return True
    except (OSError, UnicodeEncodeError, ValueError) as exc:
        LOGGER.error("Failed to append to %s: %s", path, exc)
        return False


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of dicts; returns ``[]`` on any problem."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        LOGGER.error("Failed to read %s: %s", path, exc)
        return []


def rewrite_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> bool:
    """Rewrite a CSV atomically (temp file + replace) so a crash cannot truncate it."""
    path = Path(path)
    try:
        with _CSV_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(fieldnames), extrasaction="ignore", restval=""
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in fieldnames})
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        return True
    except OSError as exc:
        LOGGER.error("Failed to rewrite %s: %s", path, exc)
        return False


# --------------------------------------------------------------------------- #
# progress calculation
# --------------------------------------------------------------------------- #
@dataclass
class Progress:
    """Deterministic state of one signal given the candles after it."""

    status: str = STATUS_ACTIVE
    tp_hits: int = 0
    closed: bool = False
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    r_multiple: float = 0.0
    bars: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0


def _resolve_touch_order(
    intrabar: Optional[pd.DataFrame],
    candle_time: datetime,
    candle_minutes: int,
    target: float,
    stop: float,
    direction: str,
) -> str:
    """Decide whether the target or the stop was reached first inside one candle.

    Replays M1 candles when available; otherwise returns ``"SL"`` (pessimistic).
    """
    if intrabar is None or intrabar.empty:
        return "SL"
    start = as_utc(candle_time)
    end = start + timedelta(minutes=candle_minutes)
    times = pd.to_datetime(intrabar["time"], utc=True)
    window = intrabar[(times >= start) & (times < end)]
    if window.empty:
        return "SL"

    for _, bar in window.iterrows():
        high, low = fnum(bar["high"]), fnum(bar["low"])
        if direction == "BUY":
            hit_tp, hit_sl = high >= target, low <= stop
        else:
            hit_tp, hit_sl = low <= target, high >= stop
        if hit_tp and hit_sl:
            return "SL"  # still ambiguous at M1 -> stay pessimistic
        if hit_tp:
            return "TP"
        if hit_sl:
            return "SL"
    return "SL"


class PositionState:
    """Incremental per-bar state machine for one open signal.

    Both the live tracker and the backtester drive this same object, so a
    backtested outcome is produced by exactly the code that scores live
    outcomes.  Call :meth:`step` once per closed candle, in order.
    """

    def __init__(self, signal_row: Dict[str, Any], config) -> None:
        self.config = config
        self.direction = str(signal_row.get("direction", "")).upper()
        self.entry = fnum(signal_row.get("entry"))
        self.initial_stop = fnum(signal_row.get("sl"))
        self.targets = [
            fnum(signal_row.get("tp1")), fnum(signal_row.get("tp2")), fnum(signal_row.get("tp3"))
        ]
        self.signal_time = parse_iso(str(signal_row.get("timestamp", "")))
        self.candle_minutes = timeframe_minutes(
            str(signal_row.get("timeframe") or config.signal_timeframe)
        )
        self.risk = abs(self.entry - self.initial_stop)
        self.sign = 1.0 if self.direction == "BUY" else -1.0
        self.fractions = list(config.partial_fractions)
        self.rr = (
            [self.sign * (target - self.entry) / self.risk for target in self.targets]
            if self.risk > 0
            else [0.0, 0.0, 0.0]
        )

        self.current_stop = self.initial_stop
        self.pending_stop: Optional[float] = None
        self.realised = 0.0
        self.remaining = 1.0
        self.tp_hits = 0
        self.bars = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.status = STATUS_ACTIVE
        self.closed = False
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.r_multiple = 0.0

    @property
    def valid(self) -> bool:
        """False when the stored signal is unusable (bad direction/prices)."""
        return (
            self.direction in ("BUY", "SELL")
            and self.signal_time is not None
            and self.entry > 0
            and self.risk > 0
        )

    def step(
        self,
        bar: Any,
        bar_time: datetime,
        intrabar: Optional[pd.DataFrame] = None,
    ) -> bool:
        """Advance one closed candle.  Returns ``True`` once the signal closes."""
        if self.closed:
            return True
        self.bars += 1

        # A breakeven stop only becomes active on the candle *after* TP1: OHLC
        # cannot say whether the same bar's low came before or after the target.
        if self.pending_stop is not None:
            self.current_stop = self.pending_stop
            self.pending_stop = None

        high, low, close = fnum(bar["high"]), fnum(bar["low"]), fnum(bar["close"])
        favourable = self.sign * ((high if self.direction == "BUY" else low) - self.entry) / self.risk
        adverse = self.sign * ((low if self.direction == "BUY" else high) - self.entry) / self.risk
        self.mfe = max(self.mfe, favourable)
        self.mae = min(self.mae, adverse)

        while True:
            next_target = self.targets[self.tp_hits] if self.tp_hits < len(self.targets) else None
            if self.direction == "BUY":
                tp_touched = next_target is not None and high >= next_target
                sl_touched = low <= self.current_stop
            else:
                tp_touched = next_target is not None and low <= next_target
                sl_touched = high >= self.current_stop

            if tp_touched and sl_touched:
                first = _resolve_touch_order(
                    intrabar, bar_time, self.candle_minutes, next_target, self.current_stop, self.direction
                )
                if first == "SL":
                    tp_touched = False
                else:
                    sl_touched = False

            if sl_touched:
                exit_r = self.sign * (self.current_stop - self.entry) / self.risk
                self._close(STATUS_SL, self.current_stop, bar_time, self.realised + self.remaining * exit_r)
                return True

            if tp_touched:
                self.realised += self.fractions[self.tp_hits] * self.rr[self.tp_hits]
                self.remaining = max(0.0, self.remaining - self.fractions[self.tp_hits])
                self.tp_hits += 1
                if self.tp_hits == 1 and self.config.move_sl_to_breakeven_after_tp1:
                    self.pending_stop = self.entry
                if self.tp_hits >= len(self.targets):
                    self._close(
                        STATUS_TP3,
                        self.targets[-1],
                        bar_time,
                        self.realised + self.remaining * self.rr[-1],
                    )
                    return True
                continue
            break

        if self.bars >= self.config.signal_expiry_candles:
            exit_r = self.sign * (close - self.entry) / self.risk
            self._close(STATUS_EXPIRED, close, bar_time, self.realised + self.remaining * exit_r)
            return True

        self.status = {0: STATUS_ACTIVE, 1: STATUS_TP1, 2: STATUS_TP2}.get(self.tp_hits, STATUS_ACTIVE)
        self.r_multiple = round(self.realised, 4)
        return False

    def close_now(self, status: str, price: float, at_time: datetime) -> None:
        """Force-close the signal at ``price`` (used for INVALIDATED)."""
        if self.closed or not self.valid:
            return
        exit_r = self.sign * (price - self.entry) / self.risk
        self._close(status, price, at_time, self.realised + self.remaining * exit_r)

    def _close(self, status: str, price: float, at_time: datetime, r_multiple: float) -> None:
        self.status = status
        self.closed = True
        self.exit_price = price
        self.exit_time = at_time
        self.r_multiple = round(r_multiple, 4)

    def to_progress(self) -> "Progress":
        """Snapshot the current state as a :class:`Progress`."""
        return Progress(
            status=self.status,
            tp_hits=self.tp_hits,
            closed=self.closed,
            exit_price=self.exit_price,
            exit_time=self.exit_time,
            r_multiple=round(self.r_multiple, 4),
            bars=self.bars,
            mfe_r=round(self.mfe, 3),
            mae_r=round(self.mae, 3),
        )


def evaluate_progress(
    signal_row: Dict[str, Any],
    candles: pd.DataFrame,
    config,
    intrabar: Optional[pd.DataFrame] = None,
) -> Progress:
    """Recompute a signal's state from the candles that closed *after* it.

    Only candles strictly after the signal candle are considered: the entry is
    that candle's close, so the trade cannot interact with its own bar.  This is
    the single most important anti-lookahead rule in outcome tracking.
    """
    state = PositionState(signal_row, config)
    if not state.valid or candles is None or candles.empty:
        return Progress()

    times = pd.to_datetime(candles["time"], utc=True)
    future = candles[times > state.signal_time]
    if future.empty:
        return Progress()

    for _, bar in future.iterrows():
        bar_time = as_utc(pd.Timestamp(bar["time"]).to_pydatetime())
        if state.step(bar, bar_time, intrabar):
            break
    return state.to_progress()


# --------------------------------------------------------------------------- #
# cooldown / limit context
# --------------------------------------------------------------------------- #
def build_gate_state(
    signal_rows: Sequence[Dict[str, Any]], active_count: int, reference_time: datetime
) -> GateState:
    """Derive cooldown and daily-limit context from past signals.

    Shared by the live tracker and the backtester so both apply identical
    cooldown rules.  Signals dated after ``reference_time`` are ignored, which
    is what keeps the backtest free of lookahead.
    """
    reference_time = as_utc(reference_time)
    last_time: Optional[datetime] = None
    last_direction = ""
    last_same: Dict[str, Optional[datetime]] = {"BUY": None, "SELL": None}
    signals_today = 0

    for row in signal_rows:
        timestamp = parse_iso(str(row.get("timestamp", "")))
        if timestamp is None or timestamp > reference_time:
            continue
        direction = str(row.get("direction", "")).upper()
        if last_time is None or timestamp > last_time:
            last_time, last_direction = timestamp, direction
        if direction in last_same:
            previous = last_same[direction]
            if previous is None or timestamp > previous:
                last_same[direction] = timestamp
        if timestamp.date() == reference_time.date():
            signals_today += 1

    return GateState(
        last_signal_time=last_time,
        last_signal_direction=last_direction,
        last_same_direction_time=last_same.get(last_direction) if last_direction else None,
        signals_today=signals_today,
        active_count=active_count,
    )


def build_outcome_row(
    signal_row: Dict[str, Any], progress: "Progress", digits: int = 2
) -> Dict[str, Any]:
    """Build an ``outcomes.csv`` row.  Shared by the live tracker and backtester."""
    signal_time = parse_iso(str(signal_row.get("timestamp", "")))
    duration_minutes = 0.0
    if signal_time and progress.exit_time:
        duration_minutes = round(
            (as_utc(progress.exit_time) - signal_time).total_seconds() / 60.0, 1
        )
    return {
        "signal_id": signal_row.get("signal_id", ""),
        "entry": signal_row.get("entry", ""),
        "result": progress.status,
        "exit_level": round(fnum(progress.exit_price), digits),
        "R_multiple": progress.r_multiple,
        "duration": duration_minutes,
        "timestamp": iso(progress.exit_time),
        "symbol": signal_row.get("symbol", ""),
        "direction": signal_row.get("direction", ""),
        "signal_time": signal_row.get("timestamp", ""),
        "bars": progress.bars,
        "tp_hits": progress.tp_hits,
        "confidence": signal_row.get("confidence", ""),
        "regime": signal_row.get("regime", ""),
        "session": signal_row.get("session", ""),
        "mfe_r": progress.mfe_r,
        "mae_r": progress.mae_r,
    }


# --------------------------------------------------------------------------- #
# tracker
# --------------------------------------------------------------------------- #
@dataclass
class TrackerEvent:
    """One status transition worth notifying about."""

    signal_row: Dict[str, Any]
    event: str
    price: float
    r_multiple: Optional[float] = None


class SignalTracker:
    """Owns ``signals.csv``, ``outcomes.csv`` and ``state.json``.

    ``signals.csv`` is deliberately kept small (a handful of rows per day) so it
    can be held in memory and rewritten atomically on status changes.  The large
    ``evaluations.csv`` is append-only and is never read by the live loop.
    """

    def __init__(self, config, notifier=None) -> None:
        self.config = config
        self.notifier = notifier
        self.signals: List[Dict[str, Any]] = []
        self.outcome_ids: set = set()
        self.state: Dict[str, Any] = {}

    # -- loading ----------------------------------------------------------- #
    def load(self) -> None:
        """Load persisted signals, outcomes and state from disk."""
        cfg = self.config
        ensure_csv(cfg.signals_csv, SIGNAL_COLUMNS)
        ensure_csv(cfg.outcomes_csv, OUTCOME_COLUMNS)
        self.signals = read_csv_rows(cfg.signals_csv)
        self.outcome_ids = {row.get("signal_id", "") for row in read_csv_rows(cfg.outcomes_csv)}
        self.state = read_json(cfg.state_file, {})
        LOGGER.info(
            "Loaded %s signals (%s active) and %s recorded outcomes",
            len(self.signals),
            len(self.active_signals()),
            len(self.outcome_ids),
        )

    def active_signals(self) -> List[Dict[str, Any]]:
        """Signals that have not reached a terminal status."""
        return [row for row in self.signals if row.get("status") not in TERMINAL_STATUSES]

    # -- state.json --------------------------------------------------------- #
    def save_state(self, **updates: Any) -> None:
        """Merge ``updates`` into ``state.json`` and persist atomically."""
        self.state.update(updates)
        atomic_write_json(self.config.state_file, self.state)

    @property
    def last_processed_candle(self) -> Optional[datetime]:
        """Open time of the last candle the engine evaluated."""
        return parse_iso(str(self.state.get("last_processed_candle", "")))

    # -- gate --------------------------------------------------------------- #
    def gate_state(self, reference_time: datetime) -> GateState:
        """Build the cooldown/limit context for the filter chain."""
        return build_gate_state(self.signals, len(self.active_signals()), reference_time)

    # -- recording ---------------------------------------------------------- #
    def record_signal(self, signal) -> Dict[str, Any]:
        """Persist a new signal, notify Telegram and update the state file."""
        row = signal.to_row()
        row.update({"status_updated_at": iso(now_utc()), "tp_hits": 0, "mfe_r": 0.0, "mae_r": 0.0})
        append_csv(self.config.signals_csv, row, SIGNAL_COLUMNS)
        self.signals.append(row)
        self.save_state(
            last_signal_id=signal.signal_id,
            last_signal_time=iso(signal.timestamp),
            last_signal_direction=signal.direction,
        )
        if self.notifier is not None:
            self.notifier.send_signal(signal)
        LOGGER.info("Recorded signal %s (%s)", signal.signal_id, signal.direction)
        return row

    def invalidate_opposite(self, direction: str, at_time: datetime, price: float) -> List[TrackerEvent]:
        """Close still-open signals that point the other way.

        A fresh, fully-confirmed signal in the opposite direction is treated as
        invalidating the earlier one (spec section 34).
        """
        if not self.config.invalidate_on_opposite_signal:
            return []
        events: List[TrackerEvent] = []
        for row in self.active_signals():
            if str(row.get("direction", "")).upper() == direction:
                continue
            signal_time = parse_iso(str(row.get("timestamp", "")))
            minutes = timeframe_minutes(
                str(row.get("timeframe") or self.config.signal_timeframe)
            )
            elapsed_bars = 0
            if signal_time is not None:
                elapsed_bars = int(
                    (as_utc(at_time) - signal_time).total_seconds() / 60.0 / max(minutes, 1)
                )
            progress = Progress(
                status=STATUS_INVALIDATED,
                tp_hits=int(fnum(row.get("tp_hits"), 0)),
                closed=True,
                exit_price=price,
                exit_time=at_time,
                r_multiple=self._mark_to_market(row, price),
                bars=max(elapsed_bars, 0),
                mfe_r=float(fnum(row.get("mfe_r"), 0.0)),
                mae_r=float(fnum(row.get("mae_r"), 0.0)),
            )
            events.extend(self._apply_progress(row, progress))
        return events

    def _mark_to_market(self, row: Dict[str, Any], price: float) -> float:
        """R multiple if the remaining size were closed at ``price`` right now."""
        entry, stop = fnum(row.get("entry")), fnum(row.get("sl"))
        risk = abs(entry - stop)
        if risk <= 0:
            return 0.0
        sign = 1.0 if str(row.get("direction", "")).upper() == "BUY" else -1.0
        hits = int(fnum(row.get("tp_hits"), 0))
        fractions = list(self.config.partial_fractions)
        realised = 0.0
        remaining = 1.0
        for index in range(min(hits, len(fractions))):
            target = fnum(row.get(f"tp{index + 1}"))
            realised += fractions[index] * sign * (target - entry) / risk
            remaining -= fractions[index]
        return round(realised + max(remaining, 0.0) * sign * (price - entry) / risk, 4)

    # -- lifecycle ---------------------------------------------------------- #
    def update(
        self, candles: pd.DataFrame, intrabar: Optional[pd.DataFrame] = None
    ) -> List[TrackerEvent]:
        """Re-evaluate every open signal against the latest candles."""
        events: List[TrackerEvent] = []
        for row in list(self.active_signals()):
            try:
                progress = evaluate_progress(row, candles, self.config, intrabar)
            except (KeyError, ValueError, TypeError) as exc:
                LOGGER.exception("Progress calculation failed for %s: %s", row.get("signal_id"), exc)
                continue
            events.extend(self._apply_progress(row, progress))
        return events

    def _apply_progress(self, row: Dict[str, Any], progress: Progress) -> List[TrackerEvent]:
        """Persist a progress result and emit alerts for newly crossed milestones."""
        events: List[TrackerEvent] = []
        previous_status = str(row.get("status", STATUS_ACTIVE))
        previous_rank = STATUS_RANK.get(previous_status, 0)
        new_rank = STATUS_RANK.get(progress.status, 0)

        changed = (
            str(row.get("tp_hits")) != str(progress.tp_hits)
            or str(row.get("mfe_r")) != str(progress.mfe_r)
            or str(row.get("mae_r")) != str(progress.mae_r)
        )
        row["tp_hits"] = progress.tp_hits
        row["mfe_r"] = progress.mfe_r
        row["mae_r"] = progress.mae_r

        if new_rank <= previous_rank and progress.status == previous_status:
            # nothing new happened - only touch the file if a tracked figure moved
            if changed:
                self._persist_signals()
            return events

        # Emit an alert for every milestone crossed since the last update, so a
        # candle that runs TP1 -> TP2 in one move still reports both.
        milestones: List[Tuple[str, float]] = []
        for level, status in ((1, STATUS_TP1), (2, STATUS_TP2), (3, STATUS_TP3)):
            if progress.tp_hits >= level and STATUS_RANK[status] > previous_rank:
                milestones.append((status, fnum(row.get(f"tp{level}"))))
        if progress.closed and progress.status in (STATUS_SL, STATUS_EXPIRED, STATUS_INVALIDATED):
            milestones.append((progress.status, fnum(progress.exit_price)))

        for status, price in milestones:
            final = status == progress.status and progress.closed
            events.append(
                TrackerEvent(
                    signal_row=dict(row),
                    event=status,
                    price=price,
                    r_multiple=progress.r_multiple if final else None,
                )
            )

        row["status"] = progress.status
        row["status_updated_at"] = iso(now_utc())
        self._persist_signals()

        if progress.closed:
            self._write_outcome(row, progress)

        for event in events:
            LOGGER.info(
                "%s -> %s at %.2f (%s)",
                row.get("signal_id"),
                event.event,
                event.price,
                f"{event.r_multiple:+.2f}R" if event.r_multiple is not None else "milestone",
            )
            if self.notifier is not None:
                self.notifier.send_outcome(row, event.event, event.price, event.r_multiple)
        return events

    def _persist_signals(self) -> None:
        """Rewrite ``signals.csv`` with the in-memory rows."""
        rewrite_csv(self.config.signals_csv, self.signals, SIGNAL_COLUMNS)

    def _write_outcome(self, row: Dict[str, Any], progress: Progress) -> None:
        """Append the closed signal to ``outcomes.csv`` exactly once."""
        signal_id = str(row.get("signal_id", ""))
        if signal_id in self.outcome_ids:
            return
        outcome = build_outcome_row(row, progress, self.config.digits)
        if append_csv(self.config.outcomes_csv, outcome, OUTCOME_COLUMNS):
            self.outcome_ids.add(signal_id)
