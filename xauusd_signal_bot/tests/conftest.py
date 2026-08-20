"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional  # noqa: E402

from config import Config  # noqa: E402
from src.indicators import compute_indicators  # noqa: E402


def make_candles(
    count: int = 3000,
    seed: int = 7,
    drift: float = 0.0,
    volatility: float = 0.16,
    start_price: float = 2300.0,
    freq: str = "1min",
) -> pd.DataFrame:
    """Build a synthetic but *valid* M1 OHLC series (high >= max(o,c) always).

    The default volatility gives an M1 ATR of roughly 3-4 pips, which is the
    right order for XAUUSD - small enough that the spread genuinely matters,
    which is what the scalping tests need to exercise.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, volatility, count)
    close = start_price + np.cumsum(steps)
    open_ = np.concatenate([[start_price], close[:-1]])
    upper = np.maximum(open_, close) + np.abs(rng.normal(0, volatility * 0.7, count))
    lower = np.minimum(open_, close) - np.abs(rng.normal(0, volatility * 0.7, count))
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-05-01", periods=count, freq=freq, tz="UTC"),
            "open": open_.round(2),
            "high": upper.round(2),
            "low": lower.round(2),
            "close": close.round(2),
            "tick_volume": rng.integers(20, 400, count).astype(float),
            "spread": 20.0,
        }
    )


def make_bars(
    bars, start: str = "2024-05-01 12:01", freq: str = "1min"
) -> pd.DataFrame:
    """Build an explicit candle frame from ``(open, high, low, close)`` tuples."""
    times = pd.date_range(start, periods=len(bars), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "time": times,
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "tick_volume": 50.0,
        }
    )


def make_ticks(prices, start: str = "2024-05-01 12:01", freq: str = "5s") -> pd.DataFrame:
    """Build a tick frame in the shape the ambiguity resolver expects."""
    times = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.DataFrame({"time": times, "high": list(prices), "low": list(prices)})


@pytest.fixture
def config() -> Config:
    """A validated default configuration (no .env dependency)."""
    cfg = Config()
    cfg.validate()
    return cfg


@pytest.fixture
def candles() -> pd.DataFrame:
    """Raw M5 candles."""
    return make_candles()


@pytest.fixture
def enriched(config, candles) -> pd.DataFrame:
    """M5 candles with the full indicator set attached."""
    return compute_indicators(candles, config.indicators)


# --------------------------------------------------------------------------- #
# shared test doubles
# --------------------------------------------------------------------------- #
class FakeNotifier:
    """Records what would have been sent instead of calling Telegram."""

    enabled = True
    connected = True

    def __init__(self) -> None:
        self.messages = []
        self.edits = []
        self.answers = []
        self.signals = []
        self.near_signals = []
        self.outcomes = []
        self.updates = []
        self._next_id = 100

    # transport ---------------------------------------------------------- #
    def send_message(self, text, keyboard=None):
        self._next_id += 1
        self.messages.append({"id": self._next_id, "text": text, "keyboard": keyboard})
        return self._next_id

    def edit_message(self, message_id, text, keyboard=None):
        self.edits.append({"id": message_id, "text": text, "keyboard": keyboard})
        return True

    def answer_callback(self, callback_id, text=""):
        self.answers.append(text)
        return True

    def get_updates(self, offset, timeout=25):
        pending, self.updates = self.updates, []
        return pending

    # notifier API -------------------------------------------------------- #
    def test_connection(self):
        return True

    def send_text(self, text):
        return bool(self.send_message(text))

    def send_signal(self, signal):
        self.signals.append(signal)
        return True

    def send_near_signal(self, evaluation):
        self.near_signals.append(evaluation)
        return True

    def send_outcome(self, signal_row, event, price, r_multiple=None, net_r=None):
        self.outcomes.append((signal_row.get("signal_id"), event, price, r_multiple, net_r))
        return True

    @property
    def last_text(self) -> str:
        if self.edits:
            return self.edits[-1]["text"]
        return self.messages[-1]["text"] if self.messages else ""

    @property
    def last_keyboard(self):
        if self.edits:
            return self.edits[-1]["keyboard"]
        return self.messages[-1]["keyboard"] if self.messages else None


class FakeMarket:
    """Replays a stored M1 history one closed candle at a time."""

    def __init__(self, config, candles: pd.DataFrame, start: int,
                 spread_points: float = 12.0, ticks: Optional[pd.DataFrame] = None) -> None:
        from src.market_data import resample_candles  # local import: avoids cycles

        self._resample = resample_candles
        self.config = config
        self.candles = candles
        self.cursor = start
        self.spread_points = spread_points
        self.ticks = ticks
        self.connected = True
        self.cache_cleared = 0
        self.fetches = []

    def connect(self):
        return True

    def shutdown(self, quiet=False):
        self.connected = False

    def clear_cache(self, timeframe=None):
        self.cache_cleared += 1

    def advance(self, steps: int = 1):
        self.cursor += steps

    def _frame(self, timeframe: str, count: Optional[int] = None):
        """Mirror MarketData: only the most recent ``count`` candles are visible.

        Honouring the window matters - a longer history changes which reference
        levels exist (a previous *day* only appears once the window spans one),
        so a test double that ignores it would not reproduce live behaviour.
        """
        window = self.candles.iloc[: self.cursor + 1]
        if timeframe != "M1":
            window = self._resample(window, "M1", timeframe)
        if count:
            window = window.iloc[-int(count):]
        return window

    def get_candles(self, symbol, timeframe, count, closed_only=True, use_cache=False,
                    cache_result=True):
        self.fetches.append(timeframe)
        return self._frame(timeframe, count)

    def get_spread(self, symbol):
        return self.spread_points

    def refresh_tick_buffer(self, symbol, minutes):
        return self.ticks

    def latest_closed_candle_time(self, symbol, timeframe):
        frame = self._frame(timeframe, 3)
        if frame.empty:
            return None
        return pd.Timestamp(frame["time"].iloc[-1]).to_pydatetime()

    def build_snapshot(self, config=None, use_cache: bool = False):
        from src.market_data import MarketSnapshot

        cfg = config or self.config
        signal = self._frame("M1", cfg.candles_signal)
        if signal.empty:
            return None, "M1: no data"
        context = (
            self._frame(cfg.context_timeframe, cfg.candles_context)
            if cfg.context_timeframe else None
        )
        return (
            MarketSnapshot(
                symbol=cfg.symbol,
                signal_df=signal,
                context_df=context,
                spread_points=self.spread_points,
                signal_timeframe="M1",
                context_timeframe=cfg.context_timeframe or "",
            ),
            "",
        )


@pytest.fixture
def isolated_config(config, tmp_path) -> Config:
    """Config pointed at a throwaway data directory."""
    config.data_dir = tmp_path
    config.signals_csv = tmp_path / "signals.csv"
    config.evaluations_csv = tmp_path / "evaluations.csv"
    config.outcomes_csv = tmp_path / "outcomes.csv"
    config.state_file = tmp_path / "state.json"
    config.telegram_chat_id = "4242"
    return config


def build_snapshot(config, candles: pd.DataFrame, spread_points: float = 12.0):
    """Assemble an M1 snapshot the way the live loop and backtester do."""
    from src.indicators import compute_indicators
    from src.market_data import MarketSnapshot, resample_candles

    context = None
    if config.context_timeframe:
        context = compute_indicators(
            resample_candles(candles, "M1", config.context_timeframe), config.indicators
        )
    return MarketSnapshot(
        symbol=config.symbol,
        signal_df=compute_indicators(candles, config.indicators),
        context_df=context,
        spread_points=spread_points,
        signal_timeframe="M1",
        context_timeframe=config.context_timeframe or "",
    )
