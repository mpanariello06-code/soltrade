"""Small, dependency-light helpers shared across the whole project."""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

ALL_SESSIONS = "ALL_SESSIONS"
UNKNOWN_SESSION = "OFF_SESSION"


# --------------------------------------------------------------------------- #
# numeric helpers
# --------------------------------------------------------------------------- #
def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide without ever raising ``ZeroDivisionError`` or returning NaN/inf."""
    try:
        if denominator is None or numerator is None:
            return default
        if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
            return default
        if abs(float(denominator)) < 1e-12:
            return default
        result = float(numerator) / float(denominator)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to ``[low, high]``; non-finite input returns ``low``."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def is_finite_number(value: Any) -> bool:
    """True when ``value`` is a real, finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def scale(value: float, low: float, high: float) -> float:
    """Map ``value`` from the range ``[low, high]`` onto ``[0, 1]`` (clamped)."""
    if abs(high - low) < 1e-12:
        return 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


def round_price(price: float, digits: int = 2) -> float:
    """Round a price to the instrument's quote precision."""
    if not is_finite_number(price):
        return 0.0
    return round(float(price), digits)


# --------------------------------------------------------------------------- #
# component score container
# --------------------------------------------------------------------------- #
@dataclass
class ComponentScore:
    """Raw output of one analysis engine.

    Engines score on their own natural scale (``max_score``); :mod:`src.scoring`
    rescales to the configured weight.  Keeping the two separate means weights
    can be re-tuned without touching engine internals.
    """

    name: str
    bull: float = 0.0
    bear: float = 0.0
    max_score: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_score = max(float(self.max_score), 1e-9)
        self.bull = clamp(self.bull, 0.0, self.max_score)
        self.bear = clamp(self.bear, 0.0, self.max_score)

    def normalised(self) -> Tuple[float, float]:
        """Return ``(bull, bear)`` as fractions of ``max_score`` in ``[0, 1]``."""
        return (
            clamp(safe_div(self.bull, self.max_score), 0.0, 1.0),
            clamp(safe_div(self.bear, self.max_score), 0.0, 1.0),
        )


# --------------------------------------------------------------------------- #
# time / session helpers
# --------------------------------------------------------------------------- #
def now_utc() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    """Attach UTC to a naive datetime, or convert an aware one to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def broker_time_to_utc(broker_dt: datetime, server_utc_offset_hours: float) -> datetime:
    """Convert a broker-server timestamp to UTC.

    ``server_utc_offset_hours`` is how far the broker clock is *ahead* of UTC
    (e.g. ``3`` for a UTC+3 server).
    """
    naive = broker_dt.replace(tzinfo=None) if broker_dt.tzinfo else broker_dt
    return (naive - timedelta(hours=server_utc_offset_hours)).replace(tzinfo=timezone.utc)


def _in_window(hour: float, window: Tuple[int, int]) -> bool:
    """Half-open window test that also handles windows wrapping past midnight."""
    start, end = window
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def detect_session(dt_utc: datetime, sessions: Any, priority: Sequence[str]) -> str:
    """Label a UTC timestamp with a trading session name.

    Windows may overlap, so ``priority`` decides which single label wins.
    Returns :data:`UNKNOWN_SESSION` outside every configured window.
    """
    dt_utc = as_utc(dt_utc)
    hour = dt_utc.hour + dt_utc.minute / 60.0
    windows = {
        "ASIAN": sessions.asian,
        "LONDON": sessions.london,
        "NEW_YORK": sessions.new_york,
        "LONDON_NEW_YORK_OVERLAP": sessions.overlap,
    }
    for name in priority:
        window = windows.get(name)
        if window is not None and _in_window(hour, window):
            return name
    for name, window in windows.items():
        if _in_window(hour, window):
            return name
    return UNKNOWN_SESSION


def session_allowed(session: str, allowed: Sequence[str]) -> bool:
    """True when ``session`` passes the configured session filter."""
    if not allowed or ALL_SESSIONS in allowed:
        return True
    return session in allowed


def is_weekend(dt_utc: datetime) -> bool:
    """Rough market-closed check (Saturday, plus Sunday before 21:00 UTC)."""
    dt_utc = as_utc(dt_utc)
    if dt_utc.weekday() == 5:
        return True
    if dt_utc.weekday() == 6 and dt_utc.hour < 21:
        return True
    return False


# --------------------------------------------------------------------------- #
# identifiers
# --------------------------------------------------------------------------- #
def make_signal_id(symbol: str, candle_time: datetime, direction: str) -> str:
    """Deterministic-prefix, collision-safe signal identifier."""
    stamp = as_utc(candle_time).strftime("%Y%m%d-%H%M%S")
    return f"{symbol}-{direction}-{stamp}-{uuid.uuid4().hex[:6]}"


def iso(dt: Optional[datetime]) -> str:
    """Format a datetime as an ISO-8601 UTC string (empty string for ``None``)."""
    if dt is None:
        return ""
    return as_utc(dt).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def parse_iso(text: str) -> Optional[datetime]:
    """Parse an ISO-8601 string produced by :func:`iso`; ``None`` when invalid."""
    if not text:
        return None
    try:
        return as_utc(datetime.fromisoformat(str(text).strip()))
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# small JSON state files
# --------------------------------------------------------------------------- #
def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read a small JSON file, returning ``default`` on any problem."""
    default = {} if default is None else default
    try:
        if not Path(path).exists():
            return dict(default)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else dict(default)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return dict(default)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> bool:
    """Write JSON atomically (temp file + replace) so a crash cannot truncate it."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        return True
    except OSError:
        return False


def fnum(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to a finite float, falling back to ``default``.

    Indicator warm-up leaves NaNs at the head of every series; engines call this
    on every value they read so a NaN can never leak into a score.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


# --------------------------------------------------------------------------- #
# content-addressed memoisation
# --------------------------------------------------------------------------- #
def frame_fingerprint(df, columns: Sequence[str] = ("high", "low", "close")) -> int:
    """Content hash of selected numeric columns of a DataFrame.

    Used to memoise expensive derivations (swing detection, level building)
    that are recomputed several times per evaluation with identical input.
    Hashing the raw bytes means a *different* slice can never collide with a
    cached result, which an identity- or length-based key could.
    """
    parts = [len(df)]
    for column in columns:
        if column in df.columns:
            parts.append(hash(df[column].to_numpy(dtype="float64").tobytes()))
        else:
            parts.append(0)
    return hash(tuple(parts))


class BoundedCache:
    """Tiny FIFO cache - enough for the handful of frames live in one evaluation."""

    def __init__(self, maxsize: int = 32) -> None:
        self.maxsize = maxsize
        self._store: Dict[Any, Any] = {}

    def get(self, key: Any) -> Any:
        return self._store.get(key)

    def put(self, key: Any, value: Any) -> Any:
        if len(self._store) >= self.maxsize:
            # drop the oldest inserted entry
            self._store.pop(next(iter(self._store)), None)
        self._store[key] = value
        return value

    def clear(self) -> None:
        self._store.clear()
