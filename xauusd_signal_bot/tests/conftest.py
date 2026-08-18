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
