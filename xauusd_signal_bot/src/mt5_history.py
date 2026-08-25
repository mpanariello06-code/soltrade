"""Chunked historical M1 download from MetaTrader 5, with a quality report.

WHY A SEPARATE MODULE FROM ``market_data.py``
---------------------------------------------
``market_data.py`` serves the LIVE loop: a few hundred recent candles, cached,
latency-sensitive.  This serves the RESEARCH pipeline: millions of candles,
written once to disk, where correctness matters far more than speed.  Mixing the
two would put a bulk download behind the live loop's cache and its lock.

RAW DATA IS IMMUTABLE
---------------------
Everything written by this module is the broker's own data plus nothing.  No
gap filling, no interpolation, no smoothing, no dropped outliers.  Problems are
*reported* in ``data_quality_report.json`` and left in place, because a
fabricated candle is indistinguishable from a real one once it is on disk - and
a model trained on invented prices learns invented behaviour.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from .logger import get_logger
from .markets import get_market, normalise_market
from .utils import iso

LOGGER = get_logger("mt5_history")

#: Fields MT5 returns for a rate.  ``spread`` and ``real_volume`` are preserved
#: even when a broker leaves them at zero: an absent column and a zero column
#: mean different things, and only the raw file can tell them apart later.
RAW_COLUMNS: Tuple[str, ...] = (
    "time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume",
)

#: Minutes per timeframe, for gap detection.
TIMEFRAME_MINUTES: Dict[str, int] = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}

#: One request per chunk.  MT5 caps a single ``copy_rates_range`` reply, and a
#: month of M1 (~30k bars) sits comfortably inside every terminal's limit while
#: keeping peak memory to a few MB.
DEFAULT_CHUNK_DAYS = 30


def _mt5_module():
    """The MetaTrader5 module, imported lazily.

    Windows-only, so importing at module scope would make this file - and every
    test that touches it - unimportable anywhere else.
    """
    from . import market_data

    return market_data.mt5


def _mt5_timeframe(mt5, timeframe: str):
    name = f"TIMEFRAME_{str(timeframe).upper()}"
    value = getattr(mt5, name, None)
    if value is None:
        raise ValueError(f"MetaTrader5 has no timeframe {timeframe!r}")
    return value


# --------------------------------------------------------------------------- #
# quality report
# --------------------------------------------------------------------------- #
@dataclass
class QualityReport:
    """What is wrong with a downloaded series, stated rather than repaired."""

    symbol: str = ""
    broker_symbol: str = ""
    timeframe: str = "M1"
    requested_start: str = ""
    requested_end: str = ""
    first_candle: str = ""
    last_candle: str = ""
    rows: int = 0
    duplicate_timestamps: int = 0
    out_of_order_rows: int = 0
    missing_candles: int = 0
    #: Gaps longer than a weekend, as ``(start, end, missing_bars)``.
    largest_gaps: List[Dict[str, Any]] = field(default_factory=list)
    bad_ohlc_rows: int = 0
    non_positive_prices: int = 0
    zero_range_rows: int = 0
    abnormal_range_rows: int = 0
    negative_spread_rows: int = 0
    abnormal_spread_rows: int = 0
    zero_volume_rows: int = 0
    timezone: str = "UTC"
    notes: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether the series is fit to train on.

        Duplicates and disorder are structural and disqualifying.  Gaps are not:
        a market that closes at the weekend has gaps by definition, and refusing
        them would refuse every FX series ever recorded.
        """
        return (
            self.rows > 0
            and self.duplicate_timestamps == 0
            and self.out_of_order_rows == 0
            and self.bad_ohlc_rows == 0
            and self.non_positive_prices == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {key: value for key, value in self.__dict__.items()}
        data["usable"] = self.usable
        return data

    def summary(self) -> str:
        lines = [
            f"{self.symbol} {self.timeframe}: {self.rows} candles",
            f"  period      {self.first_candle or '-'} -> {self.last_candle or '-'}",
            f"  duplicates  {self.duplicate_timestamps}",
            f"  disordered  {self.out_of_order_rows}",
            f"  missing     {self.missing_candles} (gaps are expected when the market closes)",
            f"  bad OHLC    {self.bad_ohlc_rows}",
            f"  non-positive prices {self.non_positive_prices}",
            f"  abnormal range/spread {self.abnormal_range_rows}/{self.abnormal_spread_rows}",
            f"  USABLE: {'yes' if self.usable else 'NO'}",
        ]
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


def analyse_quality(
    frame: pd.DataFrame, symbol: str, timeframe: str = "M1",
    requested_start: str = "", requested_end: str = "",
) -> QualityReport:
    """Inspect a raw series and describe every defect found.

    Nothing here modifies ``frame``.  The caller decides what to do about a
    problem; this only makes sure the problem is visible.
    """
    report = QualityReport(
        symbol=symbol, timeframe=timeframe,
        requested_start=requested_start, requested_end=requested_end,
        rows=int(len(frame)),
    )
    if frame.empty:
        report.notes.append("no candles returned")
        return report

    times = pd.to_datetime(frame["time"], utc=True)
    report.first_candle = iso(times.iloc[0].to_pydatetime())
    report.last_candle = iso(times.iloc[-1].to_pydatetime())
    report.duplicate_timestamps = int(times.duplicated().sum())
    report.out_of_order_rows = int((times.diff().dropna() < pd.Timedelta(0)).sum())

    open_, high = frame["open"].astype(float), frame["high"].astype(float)
    low, close = frame["low"].astype(float), frame["close"].astype(float)

    # High must bound both ends; low must be bounded by them.  A row failing
    # this is not a wide candle, it is corrupt.
    bad = (
        (high < low)
        | (high < open_) | (high < close)
        | (low > open_) | (low > close)
    )
    report.bad_ohlc_rows = int(bad.sum())
    report.non_positive_prices = int(
        ((open_ <= 0) | (high <= 0) | (low <= 0) | (close <= 0)).sum()
    )

    candle_range = (high - low)
    report.zero_range_rows = int((candle_range == 0).sum())
    typical = float(candle_range.median()) if len(candle_range) else 0.0
    if typical > 0:
        # 50x the median range in one minute is a data error far more often than
        # it is a real move; flagged, never dropped.
        report.abnormal_range_rows = int((candle_range > typical * 50).sum())

    if "spread" in frame.columns:
        spread = pd.to_numeric(frame["spread"], errors="coerce")
        report.negative_spread_rows = int((spread < 0).sum())
        median_spread = float(spread.median()) if spread.notna().any() else 0.0
        if median_spread > 0:
            report.abnormal_spread_rows = int((spread > median_spread * 50).sum())
    if "tick_volume" in frame.columns:
        report.zero_volume_rows = int(
            (pd.to_numeric(frame["tick_volume"], errors="coerce").fillna(0) <= 0).sum()
        )

    step = pd.Timedelta(minutes=TIMEFRAME_MINUTES.get(timeframe.upper(), 1))
    deltas = times.diff().dropna()
    gaps = deltas[deltas > step]
    report.missing_candles = int(((gaps / step) - 1).sum()) if len(gaps) else 0
    if len(gaps):
        ranked = gaps.sort_values(ascending=False).head(10)
        for position, delta in ranked.items():
            index = frame.index.get_loc(position)
            report.largest_gaps.append({
                "from": iso(times.iloc[max(index - 1, 0)].to_pydatetime()),
                "to": iso(times.iloc[index].to_pydatetime()),
                "missing_bars": int(delta / step) - 1,
            })
    return report


def normalise_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically and drop exact duplicate timestamps.

    The ONLY two transformations applied to broker data, and both are lossless:
    a duplicate timestamp is a transport artefact (chunk boundaries overlap by
    design), and ordering is a property of time, not of the data.  Prices are
    never touched.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.sort_values("time", kind="mergesort")
    out = out.drop_duplicates(subset=["time"], keep="first")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def _chunks(
    start: datetime, end: datetime, days: int
) -> Iterator[Tuple[datetime, datetime]]:
    """Split a period into half-open windows of at most ``days``."""
    step = timedelta(days=max(int(days), 1))
    cursor = start
    while cursor < end:
        stop = min(cursor + step, end)
        yield cursor, stop
        cursor = stop


def rates_to_frame(rates, server_utc_offset_hours: float = 0.0) -> pd.DataFrame:
    """Convert an MT5 rates array into a UTC-stamped DataFrame.

    MT5 stamps candles in the *server's* timezone.  The offset is subtracted so
    every file on disk is UTC, which is what every downstream component assumes
    and what makes two brokers' data comparable.
    """
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=list(RAW_COLUMNS))
    frame = pd.DataFrame(rates)
    if "time" not in frame.columns:
        return pd.DataFrame(columns=list(RAW_COLUMNS))
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    if server_utc_offset_hours:
        frame["time"] = frame["time"] - pd.Timedelta(hours=float(server_utc_offset_hours))
    for column in RAW_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0
    return frame[list(RAW_COLUMNS)]


class MT5HistoryDownloader:
    """Pulls history from MT5 in chunks and writes immutable yearly files."""

    def __init__(self, config, mt5_module=None) -> None:
        self.config = config
        self._mt5 = mt5_module
        self.connected = False

    @property
    def mt5(self):
        if self._mt5 is None:
            self._mt5 = _mt5_module()
        return self._mt5

    # -- connection ---------------------------------------------------------- #
    def connect(self) -> bool:
        """Initialise the terminal and confirm it is actually usable."""
        mt5 = self.mt5
        if mt5 is None:
            LOGGER.error(
                "MetaTrader5 is not available. It is Windows-only; run the "
                "downloader on the machine hosting your terminal."
            )
            return False
        kwargs: Dict[str, Any] = {}
        if getattr(self.config, "mt5_terminal_path", ""):
            kwargs["path"] = self.config.mt5_terminal_path
        if getattr(self.config, "mt5_login", 0):
            kwargs.update(
                login=int(self.config.mt5_login),
                password=self.config.mt5_password,
                server=self.config.mt5_server,
            )
        try:
            ok = bool(mt5.initialize(**kwargs))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("MT5 initialize() raised: %s", exc)
            return False
        if not ok:
            LOGGER.error("MT5 initialize() failed: %s", self._last_error())
            return False

        terminal = mt5.terminal_info()
        if terminal is None:
            LOGGER.error("MT5 terminal_info() returned nothing - terminal not ready")
            mt5.shutdown()
            return False
        LOGGER.info(
            "MT5 connected (terminal=%s, build=%s)",
            getattr(terminal, "name", "unknown"), getattr(terminal, "build", "?"),
        )
        self.connected = True
        return True

    def shutdown(self) -> None:
        if self.mt5 is not None and self.connected:
            try:
                self.mt5.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.connected = False

    def _last_error(self) -> str:
        try:
            return str(self.mt5.last_error())
        except Exception:  # noqa: BLE001
            return "unknown"

    def ensure_symbol(self, broker_symbol: str) -> bool:
        """Verify the symbol exists and make it visible to the terminal."""
        mt5 = self.mt5
        if mt5 is None:
            return False
        if mt5.symbol_info(broker_symbol) is None:
            LOGGER.error("Symbol '%s' does not exist on this account", broker_symbol)
            return False
        if not mt5.symbol_select(broker_symbol, True):
            LOGGER.error("Could not select symbol '%s'", broker_symbol)
            return False
        return True

    # -- download ------------------------------------------------------------ #
    def download(
        self, symbol: str, start: datetime, end: datetime,
        timeframe: str = "M1", chunk_days: int = DEFAULT_CHUNK_DAYS,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Fetch ``[start, end)`` in chunks and return one normalised frame.

        Chunks overlap at their boundaries by construction; the duplicate
        timestamps that creates are removed by :func:`normalise_raw`, which is
        why overlap is safe and gaps are not silently created.
        """
        symbol = normalise_market(symbol)
        broker_symbol = get_market(symbol).feed_symbol()
        if not self.ensure_symbol(broker_symbol):
            return pd.DataFrame(columns=list(RAW_COLUMNS))

        mt5 = self.mt5
        timeframe_value = _mt5_timeframe(mt5, timeframe)
        offset = float(getattr(self.config, "mt5_server_utc_offset_hours", 0.0) or 0.0)

        collected: List[pd.DataFrame] = []
        windows = list(_chunks(start, end, chunk_days))
        for index, (chunk_start, chunk_end) in enumerate(windows, start=1):
            # Server-local bounds: MT5 compares against its own clock.
            server_start = chunk_start + timedelta(hours=offset)
            server_end = chunk_end + timedelta(hours=offset)
            try:
                rates = mt5.copy_rates_range(
                    broker_symbol, timeframe_value, server_start, server_end
                )
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not lose the rest
                LOGGER.error("copy_rates_range failed for %s..%s: %s",
                             chunk_start.date(), chunk_end.date(), exc)
                continue
            frame = rates_to_frame(rates, offset)
            if not frame.empty:
                collected.append(frame)
            if progress and (index % 12 == 0 or index == len(windows)):
                rows = sum(len(f) for f in collected)
                LOGGER.info("  %s: chunk %d/%d, %d candles so far",
                            symbol, index, len(windows), rows)

        if not collected:
            LOGGER.warning("%s: no candles returned for the requested period", symbol)
            return pd.DataFrame(columns=list(RAW_COLUMNS))
        return normalise_raw(pd.concat(collected, ignore_index=True))


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #
def raw_dir(config, symbol: str, timeframe: str = "M1") -> Path:
    """``data/raw/<SYMBOL>/<TF>/`` for a market."""
    symbol = normalise_market(symbol)
    return Path(config.data_dir) / "raw" / symbol / timeframe.upper()


def write_yearly(
    frame: pd.DataFrame, config, symbol: str, timeframe: str = "M1",
    overwrite: bool = False,
) -> List[Path]:
    """Split a series into ``<year>.csv`` files under the raw directory.

    Yearly files keep any single read small enough to be cheap and make it
    obvious at a glance which periods exist.  An existing year is **merged**
    with, not replaced, unless ``overwrite`` - re-downloading a partial year
    must never truncate the history already on disk.
    """
    if frame.empty:
        return []
    target = raw_dir(config, symbol, timeframe)
    target.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    times = pd.to_datetime(frame["time"], utc=True)
    for year, group in frame.groupby(times.dt.year):
        path = target / f"{int(year)}.csv"
        payload = group
        if path.exists() and not overwrite:
            existing = pd.read_csv(path)
            payload = normalise_raw(pd.concat([existing, group], ignore_index=True))
        payload.to_csv(path, index=False, encoding="utf-8")
        written.append(path)
        LOGGER.info("  wrote %s (%d candles)", path.name, len(payload))
    return written


def load_raw(
    config, symbol: str, timeframe: str = "M1",
    start: Optional[str] = None, end: Optional[str] = None,
) -> pd.DataFrame:
    """Read every yearly file back as one chronological frame."""
    directory = raw_dir(config, symbol, timeframe)
    if not directory.is_dir():
        return pd.DataFrame(columns=list(RAW_COLUMNS))
    frames = [pd.read_csv(path) for path in sorted(directory.glob("*.csv"))]
    if not frames:
        return pd.DataFrame(columns=list(RAW_COLUMNS))
    frame = normalise_raw(pd.concat(frames, ignore_index=True))
    if start:
        frame = frame[frame["time"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        frame = frame[frame["time"] < pd.Timestamp(end, tz="UTC")]
    return frame.reset_index(drop=True)


def write_quality_report(report: QualityReport, config, symbol: str,
                         timeframe: str = "M1") -> Path:
    """Persist the quality report beside the raw data it describes."""
    path = raw_dir(config, symbol, timeframe) / "data_quality_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
