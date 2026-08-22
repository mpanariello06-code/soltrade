"""Live-editable runtime settings, persisted across restarts.

Everything the Telegram control panel can change lives here.  The engine itself
is untouched: :func:`effective_config` folds the runtime settings onto a
market's config view, so every analysis engine keeps reading plain config
attributes exactly as before.

MARKET ISOLATION
----------------
State is split in two, and the split is what guarantees that activity on one
market cannot disturb another:

``data/state.json``            global - active market, run state, alert prefs
``data/<market>/state.json``   per-market - threshold, cooldown, holding period,
                               last processed candle, last signal

Each file has exactly one writer (its own :class:`JsonStateStore`), so a write
for Bitcoin can never clobber a key belonging to gold.  Switching the active
market changes which market *generates* signals; it does not touch any other
market's stored state.

CONCURRENCY
-----------
The Telegram poller runs on a background thread while the signal loop runs on
the main thread.  Both reach this object, so every read and write goes through
an ``RLock``.

SAFETY
------
Only the settings named in ``Config.telegram_editable_settings`` can be reached
from chat.  Credentials, the bot token, the symbol list and all file paths are
not runtime settings at all and have no setter here.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .logger import get_logger
from .markets import DEFAULT_MARKET, MARKET_ORDER, get_market, normalise_market
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
class MarketRuntime:
    """Live-editable settings for ONE market, persisted in its own file.

    Nothing here is shared between markets: a threshold change on Bitcoin
    cannot move gold's, and gold's processed-candle marker lives in gold's file.
    """

    symbol: str
    config: Any
    store: JsonStateStore
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    threshold_override: Optional[float] = None
    cooldown_candles: Optional[int] = None
    min_tp2_rr: Optional[float] = None
    max_holding_candles: Optional[int] = None
    allowed_sessions: Optional[Tuple[str, ...]] = None

    # -- loading / persistence --------------------------------------------- #
    @classmethod
    def load(cls, symbol: str, config, store: JsonStateStore) -> "MarketRuntime":
        saved = dict(store.get(RUNTIME_KEY) or {})
        sessions = saved.get("allowed_sessions")
        return cls(
            symbol=symbol,
            config=config,
            store=store,
            threshold_override=saved.get("threshold_override"),
            cooldown_candles=saved.get("cooldown_candles"),
            min_tp2_rr=saved.get("min_tp2_rr"),
            max_holding_candles=saved.get("max_holding_candles"),
            allowed_sessions=tuple(sessions) if sessions else None,
        )

    def persist(self) -> None:
        with self._lock:
            self.store.update_section(
                RUNTIME_KEY,
                {
                    "symbol": self.symbol,
                    "threshold_override": self.threshold_override,
                    "cooldown_candles": self.cooldown_candles,
                    "min_tp2_rr": self.min_tp2_rr,
                    "max_holding_candles": self.max_holding_candles,
                    "allowed_sessions": (
                        list(self.allowed_sessions) if self.allowed_sessions else None
                    ),
                },
            )

    # -- threshold ---------------------------------------------------------- #
    def active_threshold(self) -> float:
        with self._lock:
            if self.threshold_override is not None:
                return self.config.clamp_threshold(self.threshold_override)
            return self.config.clamp_threshold(self.default_threshold())

    def default_threshold(self) -> float:
        return self.config.clamp_threshold(get_market(self.symbol).threshold)

    def has_threshold_override(self) -> bool:
        with self._lock:
            return self.threshold_override is not None

    def set_threshold(self, value: float) -> float:
        clamped = self.config.clamp_threshold(value)
        with self._lock:
            self.threshold_override = clamped
        self.persist()
        LOGGER.info("[%s] threshold -> %.0f", self.symbol, clamped)
        return clamped

    def adjust_threshold(self, delta: float) -> float:
        return self.set_threshold(self.active_threshold() + float(delta))

    def reset_threshold(self) -> float:
        with self._lock:
            self.threshold_override = None
        self.persist()
        LOGGER.info("[%s] threshold reset to %.0f", self.symbol, self.active_threshold())
        return self.active_threshold()

    # -- other editable settings -------------------------------------------- #
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

    def set_max_holding(self, candles: Optional[int]) -> Optional[int]:
        with self._lock:
            self.max_holding_candles = None if candles is None else max(1, int(candles))
        self.persist()
        return self.max_holding_candles

    def set_sessions(self, sessions: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
        with self._lock:
            self.allowed_sessions = tuple(s.upper() for s in sessions) if sessions else None
        self.persist()
        return self.allowed_sessions

    # -- candle bookkeeping -------------------------------------------------- #
    def last_processed_candle(self) -> Optional[datetime]:
        """Last closed M1 candle already evaluated **on this market**."""
        table = self.store.get("last_processed_candles") or {}
        return parse_iso(str(table.get(SIGNAL_TIMEFRAME, "")))

    def mark_candle_processed(self, when: str) -> None:
        self.store.set_in_map("last_processed_candles", SIGNAL_TIMEFRAME, when)

    def save_signal_marker(self, **values: Any) -> None:
        self.store.update(**values)

    # -- display -------------------------------------------------------------- #
    def describe(self) -> Dict[str, Any]:
        market = get_market(self.symbol)
        with self._lock:
            return {
                "symbol": self.symbol,
                "icon": market.icon,
                "label": market.label(),
                "threshold": self.active_threshold(),
                "threshold_is_custom": self.threshold_override is not None,
                "default_threshold": self.default_threshold(),
                "cooldown_candles": (
                    market.cooldown_candles if self.cooldown_candles is None
                    else self.cooldown_candles
                ),
                "min_tp2_rr": (
                    market.min_tp2_rr if self.min_tp2_rr is None else self.min_tp2_rr
                ),
                "max_holding_candles": (
                    market.max_holding_candles if self.max_holding_candles is None
                    else self.max_holding_candles
                ),
                "allowed_sessions": (
                    market.default_sessions if self.allowed_sessions is None
                    else self.allowed_sessions
                ),
                "is_24h": market.is_24h,
                "cost_model": market.cost_model_name,
                "pip_name": market.pip_name,
            }


@dataclass
class RuntimeState:
    """Global engine state plus one :class:`MarketRuntime` per market."""

    config: Any
    store: JsonStateStore
    markets: Dict[str, MarketRuntime] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    status: str = STATUS_RUNNING
    active_market: str = DEFAULT_MARKET
    near_signal_alerts: bool = False

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, config, store: Optional[JsonStateStore] = None) -> "RuntimeState":
        """Restore global and per-market settings from disk."""
        store = store or JsonStateStore(config.global_state_file)
        saved = dict(store.get(RUNTIME_KEY) or {})

        state = cls(
            config=config,
            store=store,
            status=str(saved.get("status", STATUS_RUNNING)).upper(),
            active_market=normalise_market(saved.get("active_market"), DEFAULT_MARKET),
            near_signal_alerts=bool(
                saved.get("near_signal_alerts", config.near_signal_alerts)
            ),
        )
        if state.status not in RUN_STATES:
            state.status = STATUS_RUNNING

        for symbol in MARKET_ORDER:
            view = config.for_market(symbol)
            view.market_dir.mkdir(parents=True, exist_ok=True)
            state.markets[symbol] = MarketRuntime.load(
                symbol, config, JsonStateStore(view.state_file)
            )

        state._migrate_single_market_state()
        LOGGER.info(
            "Runtime: %s | %s %s | active market %s | thresholds %s",
            state.status, MODE_SCALPING, SIGNAL_TIMEFRAME, state.active_market,
            {s: round(m.active_threshold()) for s, m in state.markets.items()},
        )
        return state

    def _migrate_single_market_state(self) -> None:
        """Adopt state written by the single-market (XAUUSD-only) build.

        That build kept everything in ``data/state.json``; anything found there
        belongs to gold, so it is copied into gold's own file rather than being
        silently dropped or, worse, applied to Bitcoin.
        """
        legacy_runtime = dict(self.store.get(RUNTIME_KEY) or {})
        legacy_candles = self.store.get("last_processed_candles")
        if not legacy_candles:
            # even older: one scalar marker, before the timeframe map existed
            single = self.store.get("last_processed_candle")
            legacy_candles = {SIGNAL_TIMEFRAME: single} if single else None
        gold = self.markets.get(DEFAULT_MARKET)
        if gold is None:
            return

        moved = False
        if legacy_candles and not (gold.store.get("last_processed_candles") or {}):
            gold.store.update(last_processed_candles=dict(legacy_candles))
            moved = True
        carried = {
            key: legacy_runtime[key]
            for key in ("threshold_override", "cooldown_candles", "min_tp2_rr",
                        "max_holding_candles", "allowed_sessions")
            if key in legacy_runtime and legacy_runtime[key] is not None
        }
        if carried and not gold.has_threshold_override():
            for key, value in carried.items():
                setattr(gold, key, tuple(value) if key == "allowed_sessions" else value)
            gold.persist()
            moved = True
        if moved:
            LOGGER.info("Migrated single-market state into %s", DEFAULT_MARKET)

    # ------------------------------------------------------------------ #
    def persist(self) -> None:
        """Write the global settings back to ``data/state.json``."""
        with self._lock:
            self.store.update_section(
                RUNTIME_KEY,
                {
                    "status": self.status,
                    "active_market": self.active_market,
                    "near_signal_alerts": self.near_signal_alerts,
                },
            )

    # -- markets ------------------------------------------------------------ #
    def market(self, symbol: Optional[str] = None) -> MarketRuntime:
        """The :class:`MarketRuntime` for ``symbol`` (default: the active one)."""
        with self._lock:
            key = normalise_market(symbol or self.active_market)
        return self.markets[key]

    @property
    def active(self) -> MarketRuntime:
        return self.market()

    def set_active_market(self, symbol: str) -> str:
        """Switch which market generates signals.

        Purely a selection: no other market's state, data or open paper trades
        are touched, and open positions on the market being left continue to be
        tracked against their own candles.
        """
        with self._lock:
            self.active_market = normalise_market(symbol, self.active_market)
        self.persist()
        LOGGER.info("Active market -> %s", self.active_market)
        return self.active_market

    def evaluation_markets(self) -> Tuple[str, ...]:
        """Markets that should be *evaluated* this cycle."""
        if getattr(self.config, "evaluate_all_markets", False):
            return MARKET_ORDER
        with self._lock:
            return (self.active_market,)

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

    def set_near_signal_alerts(self, enabled: bool) -> bool:
        with self._lock:
            self.near_signal_alerts = bool(enabled)
        self.persist()
        return self.near_signal_alerts

    # -- display ------------------------------------------------------------ #
    def describe(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Flat mapping for the Telegram panel and the console dashboard."""
        market_state = self.market(symbol).describe()
        with self._lock:
            market_state.update(
                {
                    "status": self.status,
                    "mode": MODE_SCALPING,
                    "signal_timeframe": SIGNAL_TIMEFRAME,
                    "active_market": self.active_market,
                    "near_signal_alerts": self.near_signal_alerts,
                }
            )
        return market_state


# --------------------------------------------------------------------------- #
# folding runtime settings into a config view
# --------------------------------------------------------------------------- #
def effective_config(config, runtime: Optional[RuntimeState], symbol: Optional[str] = None):
    """Build the config view the engine runs on for one market.

    Two layers are folded, in order:

    1. the market's own parameters (:meth:`config.Config.for_market`)
    2. whatever the Telegram panel has changed for that market

    Every analysis engine keeps reading plain config attributes; only the values
    differ.  The copy is shallow - ``weights``, ``indicators`` and ``sessions``
    are shared and never mutated here.
    """
    if runtime is None:
        return config.for_market(symbol) if symbol else config

    market_runtime = runtime.market(symbol)
    describe = market_runtime.describe()
    view = config.for_market(market_runtime.symbol)

    view.base_threshold = describe["threshold"]
    view.allowed_sessions = tuple(describe["allowed_sessions"])
    view.min_tp2_rr = float(describe["min_tp2_rr"])
    view.max_holding_candles = int(describe["max_holding_candles"])
    view.near_signal_alerts = runtime.near_signal_alerts

    cooldown = int(describe["cooldown_candles"])
    view.cooldown_candles = cooldown
    if market_runtime.cooldown_candles is not None:
        # A user-set cooldown also scales the stricter same-direction cooldown,
        # otherwise raising one silently leaves the other stale.
        view.same_direction_cooldown_candles = cooldown * 2
    return view
