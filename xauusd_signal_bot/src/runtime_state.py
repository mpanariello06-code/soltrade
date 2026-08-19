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
    MODE_STANDARD,
    RUN_STATES,
    STATUS_PAUSED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    confirmation_label,
    confirmation_timeframes,
    micro_timeframe,
    normalise_mode,
    normalise_timeframe,
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
    mode: str = MODE_STANDARD
    signal_timeframe: str = "M5"
    #: "MODE:TF" -> explicit threshold set by the user; absent means "use config"
    threshold_overrides: Dict[str, float] = field(default_factory=dict)
    cooldown_candles: Optional[int] = None
    min_tp2_rr: Optional[float] = None
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
            mode=normalise_mode(saved.get("mode", config.mode)),
            signal_timeframe=normalise_timeframe(
                saved.get("signal_timeframe", config.signal_timeframe)
            ),
            threshold_overrides={
                str(k): float(v) for k, v in dict(saved.get("threshold_overrides") or {}).items()
            },
            cooldown_candles=saved.get("cooldown_candles"),
            min_tp2_rr=saved.get("min_tp2_rr"),
            allowed_sessions=tuple(sessions) if sessions else None,
            near_signal_alerts=bool(saved.get("near_signal_alerts", config.near_signal_alerts)),
        )
        if state.status not in RUN_STATES:
            state.status = STATUS_RUNNING
        state._migrate_last_processed()
        LOGGER.info(
            "Runtime state: %s | mode=%s | TF=%s (confirm %s) | threshold=%.0f",
            state.status, state.mode, state.signal_timeframe,
            confirmation_label(state.signal_timeframe), state.active_threshold(),
        )
        return state

    def _migrate_last_processed(self) -> None:
        """Upgrade the pre-multi-timeframe single ``last_processed_candle`` key."""
        legacy = self.store.get("last_processed_candle")
        per_timeframe = self.store.get("last_processed_candles")
        if isinstance(legacy, str) and legacy and not per_timeframe:
            self.store.update(last_processed_candles={self.signal_timeframe: legacy})
            LOGGER.info("Migrated last_processed_candle to per-timeframe tracking")

    # ------------------------------------------------------------------ #
    def persist(self) -> None:
        """Write the current settings back to ``state.json``."""
        with self._lock:
            self.store.update_section(
                RUNTIME_KEY,
                {
                    "status": self.status,
                    "mode": self.mode,
                    "signal_timeframe": self.signal_timeframe,
                    "threshold_overrides": dict(self.threshold_overrides),
                    "cooldown_candles": self.cooldown_candles,
                    "min_tp2_rr": self.min_tp2_rr,
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

    # -- mode / timeframe ------------------------------------------------ #
    def set_mode(self, mode: str) -> str:
        """Switch operating mode; the threshold follows the new mode."""
        with self._lock:
            self.mode = normalise_mode(mode, self.mode)
        self.persist()
        LOGGER.info("Mode -> %s (threshold %.0f)", self.mode, self.active_threshold())
        return self.mode

    def set_timeframe(self, timeframe: str) -> str:
        """Switch the signal timeframe; the confirmation hierarchy follows."""
        with self._lock:
            self.signal_timeframe = normalise_timeframe(timeframe, self.signal_timeframe)
        self.persist()
        LOGGER.info(
            "Signal timeframe -> %s (confirmation %s, threshold %.0f)",
            self.signal_timeframe,
            confirmation_label(self.signal_timeframe),
            self.active_threshold(),
        )
        return self.signal_timeframe

    def confirmation_timeframes(self) -> Tuple[Optional[str], Optional[str]]:
        with self._lock:
            return confirmation_timeframes(self.signal_timeframe)

    def confirmation_label(self) -> str:
        with self._lock:
            return confirmation_label(self.signal_timeframe)

    def micro_timeframe(self) -> Optional[str]:
        with self._lock:
            return micro_timeframe(self.signal_timeframe)

    # -- threshold -------------------------------------------------------- #
    def _threshold_key(self) -> str:
        return f"{self.mode}:{self.signal_timeframe}"

    def active_threshold(self) -> float:
        """Base threshold for the current mode and timeframe."""
        with self._lock:
            override = self.threshold_overrides.get(self._threshold_key())
            if override is not None:
                return self.config.clamp_threshold(override)
            return self.config.threshold_for(self.mode, self.signal_timeframe)

    def default_threshold(self) -> float:
        """Configured default for the current mode/timeframe, ignoring overrides."""
        with self._lock:
            return self.config.threshold_for(self.mode, self.signal_timeframe)

    def has_threshold_override(self) -> bool:
        with self._lock:
            return self._threshold_key() in self.threshold_overrides

    def set_threshold(self, value: float) -> float:
        """Set an explicit threshold for the current mode/timeframe."""
        clamped = self.config.clamp_threshold(value)
        with self._lock:
            self.threshold_overrides[self._threshold_key()] = clamped
        self.persist()
        LOGGER.info("Threshold -> %.0f (%s)", clamped, self._threshold_key())
        return clamped

    def adjust_threshold(self, delta: float) -> float:
        """Nudge the threshold by ``delta``, respecting the configured limits."""
        return self.set_threshold(self.active_threshold() + float(delta))

    def reset_threshold(self) -> float:
        """Drop the override and fall back to the configured default."""
        with self._lock:
            self.threshold_overrides.pop(self._threshold_key(), None)
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

    def set_near_signal_alerts(self, enabled: bool) -> bool:
        with self._lock:
            self.near_signal_alerts = bool(enabled)
        self.persist()
        return self.near_signal_alerts

    # -- per-timeframe candle bookkeeping --------------------------------- #
    def last_processed_candle(self, timeframe: Optional[str] = None) -> Optional[datetime]:
        """Last closed candle already evaluated on ``timeframe``.

        Tracked *per timeframe* so switching to M15 and back to M5 cannot
        re-evaluate an M5 candle that was already processed.
        """
        timeframe = timeframe or self.signal_timeframe
        table = self.store.get("last_processed_candles") or {}
        return parse_iso(str(table.get(timeframe, "")))

    def mark_candle_processed(self, timeframe: str, when: str) -> None:
        """Record that ``timeframe``'s candle at ``when`` has been evaluated."""
        self.store.set_in_map("last_processed_candles", timeframe, when)

    # -- display ----------------------------------------------------------- #
    def describe(self) -> Dict[str, Any]:
        """Flat mapping used by the Telegram panel and the console dashboard."""
        with self._lock:
            return {
                "status": self.status,
                "mode": self.mode,
                "signal_timeframe": self.signal_timeframe,
                "confirmation": confirmation_label(self.signal_timeframe),
                "threshold": self.active_threshold(),
                "threshold_is_custom": self.has_threshold_override(),
                "cooldown_candles": (
                    self.config.cooldown_candles if self.cooldown_candles is None
                    else self.cooldown_candles
                ),
                "min_tp2_rr": (
                    self.config.min_tp2_rr if self.min_tp2_rr is None else self.min_tp2_rr
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
def config_view(
    config,
    mode: Optional[str] = None,
    timeframe: Optional[str] = None,
    threshold: Optional[float] = None,
):
    """Return a copy of ``config`` set up for one mode / signal timeframe.

    This is the seam that keeps the whole analysis layer unchanged: the engines
    still read ``config.signal_timeframe``, ``config.base_threshold`` and so on,
    but the values now reflect the requested mode and timeframe - along with the
    confirmation hierarchy that follows from it.

    Used by both the live runner (via :func:`effective_config`) and the
    backtester, so an offline run reproduces exactly what live mode would do.

    The copy is shallow - ``weights``, ``indicators`` and ``sessions`` are shared
    and never mutated here.
    """
    mode = normalise_mode(mode or getattr(config, "mode", MODE_STANDARD))
    timeframe = normalise_timeframe(timeframe or config.signal_timeframe)
    intermediate, higher = confirmation_timeframes(timeframe)

    view = copy.copy(config)
    view.mode = mode
    view.signal_timeframe = timeframe
    view.intermediate_timeframe = intermediate or ""
    view.higher_timeframe = higher or ""
    view.micro_timeframe = micro_timeframe(timeframe) or ""
    view.base_threshold = (
        config.clamp_threshold(threshold) if threshold is not None
        else config.threshold_for(mode, timeframe)
    )
    return view


def effective_config(config, runtime: Optional[RuntimeState]):
    """Fold the live Telegram-controlled settings into a config copy."""
    if runtime is None:
        return config

    describe = runtime.describe()
    view = config_view(
        config,
        mode=describe["mode"],
        timeframe=describe["signal_timeframe"],
        threshold=describe["threshold"],
    )
    view.near_signal_alerts = describe["near_signal_alerts"]
    view.allowed_sessions = tuple(describe["allowed_sessions"])
    view.min_tp2_rr = float(describe["min_tp2_rr"])

    cooldown = int(describe["cooldown_candles"])
    view.cooldown_candles = cooldown
    if runtime.cooldown_candles is not None:
        # A user-set cooldown also scales the stricter same-direction cooldown,
        # otherwise raising one silently leaves the other stale.
        view.same_direction_cooldown_candles = cooldown * 2
    return view
