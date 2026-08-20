"""Live-editable runtime settings, persisted across restarts.

Everything the Telegram control panel can change lives here.  The engine itself
is untouched: :func:`effective_config` folds the runtime settings into a *copy*
of :class:`config.Config`, so every analysis engine keeps reading plain config
attributes exactly as before.

CONCURRENCY
-----------
The Telegram poller runs on a background thread while the signal loop runs on
the main thread.  Both reach this object, so every read and write goes through
an ``RLock``.  ``state.json`` has a single owner - :class:`JsonStateStore` -
which both the tracker and the runtime state share, so neither can clobber the
other's keys.

SAFETY
------
Only the settings named in ``Config.telegram_editable_settings`` can be reached
from chat.  Credentials, the bot token, the symbol and all file paths are not
runtime settings at all and have no setter here (spec section 16).
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .logger import get_logger
from .timeframes import (
    MODE_SCALPING,
    RUN_STATES,
    SIGNAL_TIMEFRAME,
    STATUS_PAUSED,
    STATUS_RUNNING,
    STATUS_STOPPED,
)
from .utils import atomic_write_json, parse_iso, read_json

LOGGER = get_logger("runtime")

RUNTIME_KEY = "runtime"


class JsonStateStore:
    """Single-writer, lock-guarded view over ``data/state.json``.

    Both :class:`~src.signal_tracker.SignalTracker` and :class:`RuntimeState`
    share one instance, so a write from the Telegram thread cannot drop keys
    written by the signal loop.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = read_json(self.path, {})

    def snapshot(self) -> Dict[str, Any]:
        """A copy of the whole state document."""
        with self._lock:
            return copy.deepcopy(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def update(self, **values: Any) -> None:
        """Merge top-level keys and persist atomically."""
        with self._lock:
            self._data.update(values)
            self._flush()

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Merge into a nested section and persist atomically."""
        with self._lock:
            current = dict(self._data.get(section) or {})
            current.update(values)
            self._data[section] = current
            self._flush()

    def set_in_map(self, key: str, subkey: str, value: Any) -> None:
        """Set one entry of a top-level dict-valued key and persist."""
        with self._lock:
            current = dict(self._data.get(key) or {})
            current[str(subkey)] = value
            self._data[key] = current
            self._flush()

    def _flush(self) -> None:
        if not atomic_write_json(self.path, self._data):
            LOGGER.error("Could not persist %s - settings may be lost on restart", self.path)

    def reload(self) -> None:
        """Re-read the file from disk (used by tests and after manual edits)."""
        with self._lock:
            self._data = read_json(self.path, {})


@dataclass
class RuntimeState:
    """Mutable, persisted engine settings.

    Instances are created through :meth:`load` so that the previous session's
    mode, timeframe, threshold and run state are restored - which is what stops
    a restart from silently reverting to defaults or re-emitting signals.
    """

    config: Any
    store: JsonStateStore
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    status: str = STATUS_RUNNING
    #: Explicit threshold set from Telegram; ``None`` means "use config".
    threshold_override: Optional[float] = None
    cooldown_candles: Optional[int] = None
    min_tp2_rr: Optional[float] = None
    max_holding_candles: Optional[int] = None
    allowed_sessions: Optional[Tuple[str, ...]] = None
    near_signal_alerts: bool = False

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, config, store: Optional[JsonStateStore] = None) -> "RuntimeState":
        """Restore runtime settings from ``state.json``, falling back to config."""
        store = store or JsonStateStore(config.state_file)
        saved = dict(store.get(RUNTIME_KEY) or {})

        sessions = saved.get("allowed_sessions")
        state = cls(
            config=config,
            store=store,
            status=str(saved.get("status", STATUS_RUNNING)).upper(),
            threshold_override=saved.get("threshold_override"),
            cooldown_candles=saved.get("cooldown_candles"),
            min_tp2_rr=saved.get("min_tp2_rr"),
            max_holding_candles=saved.get("max_holding_candles"),
            allowed_sessions=tuple(sessions) if sessions else None,
            near_signal_alerts=bool(saved.get("near_signal_alerts", config.near_signal_alerts)),
        )
        if state.status not in RUN_STATES:
            state.status = STATUS_RUNNING
        state._migrate_last_processed()
        LOGGER.info(
            "Runtime state: %s | %s %s | threshold=%.0f | max hold=%s candles",
            state.status, MODE_SCALPING, SIGNAL_TIMEFRAME,
            state.active_threshold(), state.describe()["max_holding_candles"],
        )
        return state

    def _migrate_last_processed(self) -> None:
        """Upgrade state written by the earlier multi-timeframe build."""
        legacy = self.store.get("last_processed_candle")
        per_timeframe = self.store.get("last_processed_candles")
        if isinstance(legacy, str) and legacy and not per_timeframe:
            self.store.update(last_processed_candles={SIGNAL_TIMEFRAME: legacy})
            LOGGER.info("Migrated last_processed_candle to the M1 marker")

    # ------------------------------------------------------------------ #
    def persist(self) -> None:
        """Write the current settings back to ``state.json``."""
        with self._lock:
            self.store.update_section(
                RUNTIME_KEY,
                {
                    "status": self.status,
                    "threshold_override": self.threshold_override,
                    "cooldown_candles": self.cooldown_candles,
                    "min_tp2_rr": self.min_tp2_rr,
                    "max_holding_candles": self.max_holding_candles,
                    "allowed_sessions": list(self.allowed_sessions) if self.allowed_sessions else None,
                    "near_signal_alerts": self.near_signal_alerts,
                },
            )

    # -- run state ------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        with self._lock:
            return self.status == STATUS_RUNNING

    @property
    def is_stopped(self) -> bool:
        with self._lock:
            return self.status == STATUS_STOPPED

    def set_status(self, status: str) -> str:
        """START / PAUSE / STOP.  Returns the status actually applied."""
        status = str(status).upper()
        if status not in RUN_STATES:
            return self.status
        with self._lock:
            self.status = status
        self.persist()
        LOGGER.info("Engine status -> %s", status)
        return status

    def start(self) -> str:
        return self.set_status(STATUS_RUNNING)

    def pause(self) -> str:
        return self.set_status(STATUS_PAUSED)

    def stop(self) -> str:
        return self.set_status(STATUS_STOPPED)

    # -- threshold -------------------------------------------------------- #
    def active_threshold(self) -> float:
        """Base threshold before the regime adjustment."""
        with self._lock:
            if self.threshold_override is not None:
                return self.config.clamp_threshold(self.threshold_override)
            return self.config.clamp_threshold(self.config.base_threshold)

    def default_threshold(self) -> float:
        """Configured default, ignoring any override."""
        return self.config.clamp_threshold(self.config.base_threshold)

    def has_threshold_override(self) -> bool:
        with self._lock:
            return self.threshold_override is not None

    def set_threshold(self, value: float) -> float:
        """Set an explicit threshold."""
        clamped = self.config.clamp_threshold(value)
        with self._lock:
            self.threshold_override = clamped
        self.persist()
        LOGGER.info("Threshold -> %.0f", clamped)
        return clamped

    def adjust_threshold(self, delta: float) -> float:
        """Nudge the threshold, respecting the configured limits."""
        return self.set_threshold(self.active_threshold() + float(delta))

    def reset_threshold(self) -> float:
        """Drop the override and fall back to the configured default."""
        with self._lock:
            self.threshold_override = None
        self.persist()
        LOGGER.info("Threshold reset to default %.0f", self.active_threshold())
        return self.active_threshold()

    # -- other editable settings ------------------------------------------ #
    def set_cooldown(self, candles: Optional[int]) -> Optional[int]:
        with self._lock:
            self.cooldown_candles = None if candles is None else max(0, int(candles))
        self.persist()
        return self.cooldown_candles

    def set_min_rr(self, value: Optional[float]) -> Optional[float]:
        with self._lock:
            self.min_tp2_rr = None if value is None else max(0.0, float(value))
        self.persist()
        return self.min_tp2_rr

    def set_sessions(self, sessions: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
        with self._lock:
            self.allowed_sessions = tuple(s.upper() for s in sessions) if sessions else None
        self.persist()
        return self.allowed_sessions

    def set_max_holding(self, candles: Optional[int]) -> Optional[int]:
        """Maximum holding period in M1 candles before a scalp times out."""
        with self._lock:
            self.max_holding_candles = None if candles is None else max(1, int(candles))
        self.persist()
        return self.max_holding_candles

    def set_near_signal_alerts(self, enabled: bool) -> bool:
        with self._lock:
            self.near_signal_alerts = bool(enabled)
        self.persist()
        return self.near_signal_alerts

    # -- candle bookkeeping ------------------------------------------------ #
    def last_processed_candle(self, timeframe: Optional[str] = None) -> Optional[datetime]:
        """Last closed M1 candle already evaluated."""
        timeframe = timeframe or SIGNAL_TIMEFRAME
        table = self.store.get("last_processed_candles") or {}
        return parse_iso(str(table.get(timeframe, "")))

    def mark_candle_processed(self, timeframe: str, when: str) -> None:
        """Record that the candle at ``when`` has been evaluated."""
        self.store.set_in_map("last_processed_candles", timeframe, when)

    # -- display ----------------------------------------------------------- #
    def describe(self) -> Dict[str, Any]:
        """Flat mapping used by the Telegram panel and the console dashboard."""
        with self._lock:
            return {
                "status": self.status,
                "mode": MODE_SCALPING,
                "signal_timeframe": SIGNAL_TIMEFRAME,
                "context_timeframe": self.config.context_timeframe or "NONE",
                "threshold": self.active_threshold(),
                "threshold_is_custom": self.threshold_override is not None,
                "cooldown_candles": (
                    self.config.cooldown_candles if self.cooldown_candles is None
                    else self.cooldown_candles
                ),
                "min_tp2_rr": (
                    self.config.min_tp2_rr if self.min_tp2_rr is None else self.min_tp2_rr
                ),
                "max_holding_candles": (
                    self.config.max_holding_candles if self.max_holding_candles is None
                    else self.max_holding_candles
                ),
                "allowed_sessions": (
                    self.config.allowed_sessions if self.allowed_sessions is None
                    else self.allowed_sessions
                ),
                "near_signal_alerts": self.near_signal_alerts,
            }


# --------------------------------------------------------------------------- #
# folding runtime settings into a config view
# --------------------------------------------------------------------------- #
def effective_config(config, runtime: Optional["RuntimeState"]):
    """Fold the live Telegram-controlled settings into a copy of ``config``.

    This is the seam that keeps the analysis layer unchanged: every engine still
    reads plain config attributes, but the values reflect whatever the panel
    last set.  The copy is shallow - ``weights``, ``indicators`` and ``sessions``
    are shared and never mutated here.
    """
    if runtime is None:
        return config

    describe = runtime.describe()
    view = copy.copy(config)
    view.base_threshold = describe["threshold"]
    view.near_signal_alerts = describe["near_signal_alerts"]
    view.allowed_sessions = tuple(describe["allowed_sessions"])
    view.min_tp2_rr = float(describe["min_tp2_rr"])
    view.max_holding_candles = int(describe["max_holding_candles"])

    cooldown = int(describe["cooldown_candles"])
    view.cooldown_candles = cooldown
    if runtime.cooldown_candles is not None:
        # A user-set cooldown also scales the stricter same-direction cooldown,
        # otherwise raising one silently leaves the other stale.
        view.same_direction_cooldown_candles = cooldown * 2
    return view
