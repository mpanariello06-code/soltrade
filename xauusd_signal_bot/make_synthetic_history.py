"""Generate a synthetic XAUUSD M1 history for smoke-testing the pipeline.

This produces a regime-switching random walk that *looks* like gold - trending
stretches, ranging stretches, quiet stretches - so every code path in the engine,
backtester and performance report can be exercised without broker data.

IT IS NOT REAL DATA.  Results computed on it say nothing whatsoever about
whether the strategy has an edge; use it to verify the plumbing, then run
``backtest.py`` on genuine XAUUSD M5 history before drawing any conclusion.

Usage::

    python make_synthetic_history.py
    python make_synthetic_history.py --bars 40000 --seed 7 --out history/XAUUSD_M5_synthetic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

#: (drift per bar, volatility per bar) for each synthetic regime
#: (drift per M1 bar, volatility per M1 bar) in price terms.  These give an M1
#: ATR of roughly 2-5 pips, which is the right order for XAUUSD.
REGIMES = {
    "up": (0.018, 0.16),
    "down": (-0.018, 0.16),
    "range": (0.0, 0.18),
    "quiet": (0.0, 0.08),
}
REGIME_PROBABILITIES = (0.30, 0.30, 0.25, 0.15)


def generate(bars: int = 30000, seed: int = 42, start_price: float = 2300.0) -> pd.DataFrame:
    """Build a regime-switching synthetic M1 series with valid OHLC."""
    rng = np.random.default_rng(seed)
    names = list(REGIMES)

    drift = np.zeros(bars)
    volatility = np.zeros(bars)
    cursor = 0
    while cursor < bars:
        length = int(rng.integers(400, 2400))
        regime = rng.choice(names, p=REGIME_PROBABILITIES)
        drift[cursor : cursor + length], volatility[cursor : cursor + length] = REGIMES[regime]
        cursor += length

    close = start_price + np.cumsum(rng.normal(drift, volatility))
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, volatility * 0.9))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, volatility * 0.9))

    return pd.DataFrame(
        {
            "time": pd.date_range("2024-03-01", periods=bars, freq="1min", tz="UTC"),
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "tick_volume": rng.integers(20, 400, bars).astype(float),
        }
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic XAUUSD M1 history")
    parser.add_argument("--bars", type=int, default=30000, help="number of M1 candles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("history/XAUUSD_M1_synthetic.csv"))
    args = parser.parse_args(argv)

    frame = generate(args.bars, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, encoding="utf-8")
    print(
        f"Wrote {len(frame)} synthetic candles to {args.out} "
        f"({frame['close'].min():.2f} - {frame['close'].max():.2f})"
    )
    print("NOTE: synthetic data. Backtest results on it are NOT evidence of an edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
