"""MetaTrader 5 market-data access.

READ-ONLY BY DESIGN.  This module never imports, references or wraps any MT5
order-execution function (``order_send``, ``order_check``, ...).  MT5 is used
purely as a candle/quote feed.

The ``MetaTrader5`` package is Windows-only.  It is imported lazily and
defensively so that the rest of the project (indicators, engines, backtester,
tests) remains importable and runnable on any platform.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .timeframes import SIGNAL_TIMEFRAME
from .utils import (
    BoundedCache,
    as_utc,
    broker_time_to_utc,
    frame_fingerprint,
    is_finite_number,
    now_utc,
    safe_div,
)

LOGGER = logging.getLogger(__name__)

#: The MetaTrader5 module keeps global terminal state and is not thread-safe.
#: The Telegram control thread can ask for an on-demand analysis while the main
#: loop is mid-poll, so every terminal call is serialised through this lock.
_MT5_LOCK = threading.RLock()

try:  # pragma: no cover - depends on the host OS
    import MetaTrader5 as mt5  # type: ignore

    MT5_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import problem must be non-fatal
    mt5 = None  # type: ignore[assignment]
    MT5_AVAILABLE = False


#: Timeframe label -> duration in minutes.  Used for staleness and resampling.
TIMEFRAME_MINUTES: Dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}

REQUIRED_COLUMNS = ("time", "open", "high", "low", "close", "tick_volume")


def timeframe_minutes(timeframe: str) -> int:
    """Duration of one candle of ``timeframe`` in minutes."""
    try:
        return TIMEFRAME_MINUTES[str(timeframe).upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe '{timeframe}'") from exc


def _mt5_timeframe(timeframe: str):
    """Map a timeframe label to the MT5 enum constant."""
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 package is not available on this machine")
    label = str(timeframe).upper()
    constant = getattr(mt5, f"TIMEFRAME_{label}", None)
    if constant is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'")
    return constant


# --------------------------------------------------------------------------- #
# candle hygiene
# --------------------------------------------------------------------------- #
def clean_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time, drop duplicate timestamps and rows with impossible OHLC.

    Keeps the *last* occurrence of a duplicated timestamp: when MT5 re-sends a
    bar it is the fresher copy.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    out = df.copy()
    for column in REQUIRED_COLUMNS:
        if column not in out.columns:
            raise ValueError(f"candle frame is missing required column '{column}'")

    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = out.sort_values("time")
    out = out.drop_duplicates(subset=["time"], keep="last")

    valid = (
        (out["high"] >= out["low"])
        & (out["high"] >= out["open"])
        & (out["high"] >= out["close"])
        & (out["low"] <= out["open"])
        & (out["low"] <= out["close"])
        & (out[["open", "high", "low", "close"]] > 0).all(axis=1)
        & np.isfinite(out[["open", "high", "low", "close"]]).all(axis=1)
    )
    out = out[valid]

    if "tick_volume" not in out.columns:
        out["tick_volume"] = 0.0
    out["tick_volume"] = pd.to_numeric(out["tick_volume"], errors="coerce").fillna(0.0)
    if "spread" not in out.columns:
        out["spread"] = np.nan
    return out.reset_index(drop=True)


_STRUCTURE_CACHE = BoundedCache(maxsize=32)


def validate_candles(
    df: pd.DataFrame,
    timeframe: str,
    min_required: int,
    max_staleness_seconds: Optional[int] = None,
    reference_time: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """Validate a closed-candle frame.  Returns ``(ok, reason)``.

    ``reason`` is an empty string when the frame is usable.  The structural
    checks (ordering, duplicates, OHLC sanity) are memoised on the frame's
    contents; the staleness check depends on wall-clock time and is always
    re-evaluated.
    """
    if df is None or df.empty:
        return False, "no candle data"
    if len(df) < min_required:
        return False, f"insufficient history ({len(df)} < {min_required})"

    cache_key = frame_fingerprint(df, ("open", "high", "low", "close", "time"))
    structural = _STRUCTURE_CACHE.get(cache_key)
    if structural is None:
        structural = ""
        if df["time"].duplicated().any():
            structural = "duplicate candle timestamps"
        elif not df["time"].is_monotonic_increasing:
            structural = "candle timestamps are not ordered"
        elif df[["open", "high", "low", "close"]].isna().any().any():
            structural = "NaN OHLC values"
        elif (df[["open", "high", "low", "close"]] <= 0).any().any():
            structural = "non-positive OHLC values"
        elif (df["high"] < df["low"]).any():
            structural = "high < low in candle data"
        _STRUCTURE_CACHE.put(cache_key, structural)
    if structural:
        return False, structural

    if max_staleness_seconds is not None:
        reference_time = reference_time or now_utc()
        last_open = as_utc(pd.Timestamp(df["time"].iloc[-1]).to_pydatetime())
        expected_close = last_open + timedelta(minutes=timeframe_minutes(timeframe))
        age = (as_utc(reference_time) - expected_close).total_seconds()
        if age > max_staleness_seconds:
            return False, f"stale data (last candle closed {int(age)}s ago)"
    return True, ""


def resample_candles(df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
    """Aggregate lower-timeframe candles into a higher timeframe.

    Only *complete* target buckets are returned, so the result never contains a
    partially-formed candle.  Used by the backtester to derive M15/H1 from an
    M5 history file.
    """
    source_minutes = timeframe_minutes(source_tf)
    target_minutes = timeframe_minutes(target_tf)
    if target_minutes % source_minutes != 0:
        raise ValueError(f"{target_tf} is not a whole multiple of {source_tf}")
    if target_minutes == source_minutes:
        return df.copy()

    work = df.copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work = work.set_index("time")
    aggregated = work.resample(f"{target_minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])

    bars_per_bucket = target_minutes // source_minutes
    counts = work.resample(f"{target_minutes}min", label="left", closed="left").size()
    # A bucket is complete when it holds a full set of source bars.  Gold has
    # gaps (rollover/weekend), so partial buckets in the middle of history are
    # tolerated - only the final bucket must be complete, since that is the one
    # that could still be forming.
    if len(aggregated) and counts.reindex(aggregated.index).iloc[-1] < bars_per_bucket:
        aggregated = aggregated.iloc[:-1]

    aggregated = aggregated.reset_index()
    aggregated["spread"] = np.nan
    return aggregated


# --------------------------------------------------------------------------- #
# snapshot passed to the signal engine
# --------------------------------------------------------------------------- #
@dataclass
class MarketSnapshot:
    """Everything the signal engine is allowed to see for one evaluation.

    Both frames hold **closed candles only**; the last row of ``signal_df`` is
    the M1 signal candle.  There is deliberately no field carrying the forming
    candle.

    ``context_df`` is the short-term context timeframe (M5 by default) and is
    ``None`` when context is switched off - the context component is then marked
    not-applicable and its weight is shared out.
    """

    symbol: str
    signal_df: pd.DataFrame
    context_df: Optional[pd.DataFrame] = None
    spread_points: float = float("nan")
    evaluated_at: datetime = field(default_factory=now_utc)
    signal_timeframe: str = "M1"
    context_timeframe: str = ""

    @property
    def candle_time(self) -> datetime:
        """UTC open time of the M1 signal candle."""
        return as_utc(pd.Timestamp(self.signal_df["time"].iloc[-1]).to_pydatetime())

    @property
    def close(self) -> float:
        """Close price of the signal candle."""
        return float(self.signal_df["close"].iloc[-1])


# --------------------------------------------------------------------------- #
# MT5 connector
# --------------------------------------------------------------------------- #
class MarketData:
    """Thin, resilient, read-only wrapper around the MetaTrader 5 terminal."""

    #: How long a fetched frame may be reused for a *repeat* poll, per timeframe
    #: minute.  A cached frame is never used when building the snapshot for a
    #: newly closed candle - see :meth:`build_snapshot`.
    CACHE_TTL_FRACTION = 0.25
    CACHE_TTL_CAP_SECONDS = 120.0

    def __init__(self, config) -> None:
        self.config = config
        self.connected = False
        self._last_connect_attempt = 0.0
        self._reconnect_backoff = 5.0
        # (symbol, timeframe) -> (fetched_at, count, frame)
        self._cache: Dict[Tuple[str, str], Tuple[float, int, pd.DataFrame]] = {}
        self._cache_lock = threading.RLock()
        self._tick_buffer: Optional[pd.DataFrame] = None
        self._tick_last_time: Optional[datetime] = None

    # -- candle cache ------------------------------------------------------ #
    def _cache_ttl(self, timeframe: str) -> float:
        """Seconds a frame of ``timeframe`` may be reused for."""
        minutes = timeframe_minutes(timeframe)
        return min(minutes * 60.0 * self.CACHE_TTL_FRACTION, self.CACHE_TTL_CAP_SECONDS)

    def _cached(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        with self._cache_lock:
            entry = self._cache.get((symbol, timeframe))
        if entry is None:
            return None
        fetched_at, cached_count, frame = entry
        if cached_count < count:
            return None
        if time.time() - fetched_at > self._cache_ttl(timeframe):
            return None
        return frame

    def _store_cache(self, symbol: str, timeframe: str, count: int, frame: pd.DataFrame) -> None:
        with self._cache_lock:
            self._cache[(symbol, timeframe)] = (time.time(), count, frame)

    def clear_cache(self, timeframe: Optional[str] = None) -> None:
        """Drop cached candles.

        Called when the signal timeframe changes so that no frame belonging to
        the previous hierarchy can leak into the next evaluation.
        """
        with self._cache_lock:
            if timeframe is None:
                self._cache.clear()
            else:
                for key in [k for k in self._cache if k[1] == timeframe]:
                    self._cache.pop(key, None)

    # -- connection ------------------------------------------------------- #
    def connect(self) -> bool:
        """Initialise the MT5 terminal and select the symbol.  Never raises."""
        if not MT5_AVAILABLE:
            LOGGER.error(
                "MetaTrader5 package unavailable - live mode requires Windows with "
                "MT5 installed (`pip install MetaTrader5`)."
            )
            return False

        self._last_connect_attempt = time.time()
        try:
            kwargs = {}
            if self.config.mt5_terminal_path:
                kwargs["path"] = self.config.mt5_terminal_path
            if self.config.mt5_login and self.config.mt5_password and self.config.mt5_server:
                kwargs.update(
                    login=int(self.config.mt5_login),
                    password=self.config.mt5_password,
                    server=self.config.mt5_server,
                )
            with _MT5_LOCK:
                initialised = mt5.initialize(**kwargs)
            if not initialised:
                LOGGER.error("MT5 initialize() failed: %s", mt5.last_error())
                self.connected = False
                return False

            with _MT5_LOCK:
                selected = mt5.symbol_select(self.config.symbol, True)
            if not selected:
                LOGGER.error("Symbol '%s' is unavailable on this account", self.config.symbol)
                with _MT5_LOCK:
                    mt5.shutdown()
                self.connected = False
                return False

            with _MT5_LOCK:
                info = mt5.terminal_info()
            LOGGER.info(
                "MT5 connected (terminal=%s, symbol=%s)",
                getattr(info, "name", "unknown"),
                self.config.symbol,
            )
            self.connected = True
            self._reconnect_backoff = 5.0
            return True
        except Exception as exc:  # noqa: BLE001 - connection must never crash the loop
            LOGGER.exception("MT5 connection error: %s", exc)
            self.connected = False
            return False

    def ensure_connection(self) -> bool:
        """Reconnect if needed, with exponential backoff between attempts."""
        if self.connected and self._terminal_alive():
            return True
        elapsed = time.time() - self._last_connect_attempt
        if elapsed < self._reconnect_backoff:
            return False
        LOGGER.warning("MT5 connection lost - attempting to reconnect")
        self.shutdown(quiet=True)
        if self.connect():
            return True
        self._reconnect_backoff = min(self._reconnect_backoff * 2.0, 300.0)
        return False

    def _terminal_alive(self) -> bool:
        if not MT5_AVAILABLE:
            return False
        try:
            with _MT5_LOCK:
                return mt5.terminal_info() is not None
        except Exception:  # noqa: BLE001
            return False

    def shutdown(self, quiet: bool = False) -> None:
        """Close the MT5 connection."""
        if MT5_AVAILABLE:
            try:
                with _MT5_LOCK:
                    mt5.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.connected = False
        if not quiet:
            LOGGER.info("MT5 connection closed")

    # -- data ------------------------------------------------------------- #
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        closed_only: bool = True,
        use_cache: bool = False,
        cache_result: bool = True,
    ) -> pd.DataFrame:
        """Fetch candles, newest last, timestamps converted to UTC.

        With ``closed_only=True`` (the default and the only mode the signal
        engine may use) the still-forming candle at position 0 is dropped.
        Returns an empty frame on any failure - callers check emptiness.

        ``use_cache`` reuses a recently fetched frame for the same
        ``(symbol, timeframe)``.  It is off by default and is never enabled for
        the frames that feed an actual evaluation, so a signal is always decided
        on freshly fetched candles.
        """
        if use_cache and closed_only:
            cached = self._cached(symbol, timeframe, count)
            if cached is not None:
                return cached
        if not self.ensure_connection():
            return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
        try:
            # +1 because the forming candle is discarded below.
            with _MT5_LOCK:
                rates = mt5.copy_rates_from_pos(
                    symbol, _mt5_timeframe(timeframe), 0, int(count) + 1
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("copy_rates_from_pos failed for %s %s: %s", symbol, timeframe, exc)
            self.connected = False
            return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

        if rates is None or len(rates) == 0:
            LOGGER.warning("No candles returned for %s %s (%s)", symbol, timeframe, mt5.last_error())
            return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

        df = pd.DataFrame(rates)
        offset = self.config.mt5_server_utc_offset_hours
        df["time"] = [
            broker_time_to_utc(datetime.fromtimestamp(int(t), tz=timezone.utc).replace(tzinfo=None), offset)
            for t in df["time"]
        ]
        df["time"] = pd.to_datetime(df["time"], utc=True)

        if closed_only and len(df):
            df = df.iloc[:-1]  # position 0 of MT5 is the forming candle -> last here
        cleaned = clean_candles(df)
        if closed_only and cache_result and not cleaned.empty:
            self._store_cache(symbol, timeframe, count, cleaned)
        return cleaned

    def get_latest_candle(self, symbol: str, timeframe: str) -> Optional[pd.Series]:
        """Most recently *closed* candle, or ``None``."""
        df = self.get_candles(symbol, timeframe, 3, closed_only=True)
        if df.empty:
            return None
        return df.iloc[-1]

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Current mid price from the tick feed, falling back to the last close."""
        if not self.ensure_connection():
            return None
        try:
            with _MT5_LOCK:
                tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            bid, ask = float(tick.bid), float(tick.ask)
            if is_finite_number(bid) and is_finite_number(ask) and bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            return bid if bid > 0 else None
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("symbol_info_tick failed: %s", exc)
            return None

    def get_spread(self, symbol: str) -> float:
        """Current spread in points.  ``nan`` when unavailable."""
        if not self.ensure_connection():
            return float("nan")
        try:
            with _MT5_LOCK:
                info = mt5.symbol_info(symbol)
            if info is None:
                return float("nan")
            if getattr(info, "spread", 0):
                return float(info.spread)
            with _MT5_LOCK:
                tick = mt5.symbol_info_tick(symbol)
            point = float(getattr(info, "point", 0.0) or 0.0)
            if tick is None or point <= 0:
                return float("nan")
            return safe_div(float(tick.ask) - float(tick.bid), point, float("nan"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("symbol_info failed: %s", exc)
            return float("nan")

    # -- composite --------------------------------------------------------- #
    def latest_closed_candle_time(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Open time of the most recently closed candle - a cheap "is there a
        new bar?" probe that avoids downloading full history on every poll.
        """
        # cache_result=False: a 3-candle probe must not evict the full frame
        frame = self.get_candles(symbol, timeframe, 3, closed_only=True, cache_result=False)
        if frame.empty:
            return None
        return as_utc(pd.Timestamp(frame["time"].iloc[-1]).to_pydatetime())

    def build_snapshot(
        self, config=None, use_cache: bool = False
    ) -> Tuple[Optional[MarketSnapshot], str]:
        """Fetch M1 and the context timeframe, and assemble a snapshot.

        Returns ``(snapshot, reason)``; ``snapshot`` is ``None`` when data could
        not be assembled and ``reason`` explains why.
        """
        cfg = config or self.config
        signal_df = self.get_candles(cfg.symbol, SIGNAL_TIMEFRAME, cfg.candles_signal)
        ok, reason = validate_candles(
            signal_df,
            SIGNAL_TIMEFRAME,
            cfg.min_candles_required,
            cfg.max_candle_staleness_seconds,
        )
        if not ok:
            return None, f"{SIGNAL_TIMEFRAME}: {reason}"

        context_df = None
        context_tf = cfg.context_timeframe
        if context_tf:
            frame = self.get_candles(
                cfg.symbol, context_tf, cfg.candles_context, use_cache=use_cache
            )
            ok, reason = validate_candles(frame, context_tf, cfg.min_context_candles)
            if not ok:
                return None, f"{context_tf}: {reason}"
            context_df = frame

        return (
            MarketSnapshot(
                symbol=cfg.symbol,
                signal_df=signal_df,
                context_df=context_df,
                spread_points=self.get_spread(cfg.symbol),
                evaluated_at=now_utc(),
                signal_timeframe=SIGNAL_TIMEFRAME,
                context_timeframe=context_tf or "",
            ),
            "",
        )

    # -- ticks: resolving ambiguous candles -------------------------------- #
    def refresh_tick_buffer(self, symbol: str, minutes: int) -> Optional[pd.DataFrame]:
        """Maintain a rolling buffer of raw ticks for the last ``minutes``.

        At M1 scalping scale a single candle very often trades through both the
        target and the stop, and OHLC cannot say which came first.  Replaying
        ticks resolves that ordering; without them the tracker falls back to the
        pessimistic assumption (stop first), which is a large systematic penalty
        when the whole trade is a few pips wide.

        Only the delta since the previous call is fetched, so the cost per poll
        is roughly one minute of ticks.  Returns ``None`` when ticks are
        unavailable - the caller must treat that as "stay pessimistic".
        """
        if not MT5_AVAILABLE or not self.ensure_connection():
            return self._tick_buffer

        now = now_utc()
        window_start = now - timedelta(minutes=max(int(minutes), 1))
        fetch_from = window_start
        if self._tick_last_time is not None and self._tick_last_time > window_start:
            fetch_from = self._tick_last_time

        offset = self.config.mt5_server_utc_offset_hours
        try:
            with _MT5_LOCK:
                ticks = mt5.copy_ticks_range(
                    symbol,
                    fetch_from + timedelta(hours=offset),
                    now + timedelta(hours=offset),
                    mt5.COPY_TICKS_ALL,
                )
        except Exception as exc:  # noqa: BLE001 - ticks are an optimisation
            LOGGER.debug("copy_ticks_range failed: %s", exc)
            return self._tick_buffer

        if ticks is not None and len(ticks):
            frame = pd.DataFrame(ticks)
            times = pd.to_datetime(frame["time_msc"], unit="ms", utc=True)
            times = times - pd.Timedelta(hours=offset)
            # Bid is used for both extremes: we only need the ORDER in which the
            # levels were touched, not an exact fill price.
            price = pd.to_numeric(frame.get("bid"), errors="coerce")
            fresh = pd.DataFrame({"time": times, "high": price, "low": price}).dropna()
            with self._cache_lock:
                if self._tick_buffer is None or self._tick_buffer.empty:
                    self._tick_buffer = fresh
                else:
                    self._tick_buffer = pd.concat([self._tick_buffer, fresh], ignore_index=True)

        with self._cache_lock:
            buffer = self._tick_buffer
            if buffer is not None and not buffer.empty:
                buffer = buffer[buffer["time"] >= window_start].reset_index(drop=True)
                self._tick_buffer = buffer
                self._tick_last_time = as_utc(
                    pd.Timestamp(buffer["time"].iloc[-1]).to_pydatetime()
                ) if len(buffer) else None
            return self._tick_buffer
