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

from config import Config  # noqa: E402
from src.indicators import compute_indicators  # noqa: E402


def make_candles(
    count: int = 3000,
    seed: int = 7,
    drift: float = 0.0,
    volatility: float = 0.45,
    start_price: float = 2300.0,
    freq: str = "5min",
) -> pd.DataFrame:
    """Build a synthetic but *valid* OHLC series (high >= max(o,c) always)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, volatility, count)
    close = start_price + np.cumsum(steps)
    open_ = np.concatenate([[start_price], close[:-1]])
    upper = np.maximum(open_, close) + np.abs(rng.normal(0, volatility * 0.9, count))
    lower = np.minimum(open_, close) - np.abs(rng.normal(0, volatility * 0.9, count))
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-03-01", periods=count, freq=freq, tz="UTC"),
            "open": open_.round(2),
            "high": upper.round(2),
            "low": lower.round(2),
            "close": close.round(2),
            "tick_volume": rng.integers(60, 1500, count).astype(float),
            "spread": 20.0,
        }
    )


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

    def send_outcome(self, signal_row, event, price, r_multiple=None):
        self.outcomes.append((signal_row.get("signal_id"), event, price, r_multiple))
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
    """Replays a stored history one closed candle at a time.

    Honours whatever signal/confirmation timeframes the passed config asks for,
    so timeframe switching can be tested without MetaTrader 5.
    """

    def __init__(self, config, candles: pd.DataFrame, start: int,
                 source_timeframe: str = "M5",
                 spread_points: float = float("nan")) -> None:
        from src.market_data import resample_candles  # local import: avoids cycles

        self._resample = resample_candles
        self.config = config
        self.candles = candles
        self.cursor = start
        self.source_timeframe = source_timeframe
        # Unknown by default, exactly like a backtest: a hard-coded spread that
        # happens to exceed MAX_SPREAD_ATR_RATIO on the low-volatility fixture
        # would reject every candle and make these tests pass vacuously.
        self.spread_points = spread_points
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

    def _frame(self, timeframe: str):
        window = self.candles.iloc[: self.cursor + 1]
        if timeframe == self.source_timeframe:
            return window
        return self._resample(window, self.source_timeframe, timeframe)

    def get_candles(self, symbol, timeframe, count, closed_only=True, use_cache=False):
        self.fetches.append(timeframe)
        return self._frame(timeframe)

    def get_spread(self, symbol):
        return self.spread_points

    def latest_closed_candle_time(self, symbol, timeframe):
        frame = self._frame(timeframe)
        if frame.empty:
            return None
        return pd.Timestamp(frame["time"].iloc[-1]).to_pydatetime()

    def build_snapshot(self, config=None, use_cache: bool = False):
        from src.market_data import MarketSnapshot

        cfg = config or self.config
        confirm = cfg.intermediate_timeframe or None
        higher = cfg.higher_timeframe or None
        signal = self._frame(cfg.signal_timeframe)
        if signal.empty:
            return None, f"{cfg.signal_timeframe}: no data"
        return (
            MarketSnapshot(
                symbol=cfg.symbol,
                m5=signal,
                m15=self._frame(confirm) if confirm else None,
                h1=self._frame(higher) if higher else None,
                m1=None,
                spread_points=self.spread_points,
                signal_timeframe=cfg.signal_timeframe,
                confirmation="+".join(t for t in (confirm, higher) if t) or "NONE",
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
