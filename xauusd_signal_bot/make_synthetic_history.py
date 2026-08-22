"""Generate a synthetic M1 history for smoke-testing the pipeline.

This produces a regime-switching random walk with the rough character of the
chosen market - trending stretches, ranging stretches, quiet stretches - so
every code path in the engine, backtester and performance report can be
exercised without broker data.

IT IS NOT REAL DATA.  Results computed on it say nothing whatsoever about
whether the strategy has an edge - for EITHER market.  Use it to verify the
plumbing, then run ``backtest.py`` on genuine history before drawing any
conclusion.  The BTCUSD profile in particular is a plausible-looking guess at
crypto M1 behaviour, not a calibration against exchange data.

Usage::

    python make_synthetic_history.py
    python make_synthetic_history.py --symbol BTCUSD
    python make_synthetic_history.py --bars 40000 --seed 7 --out history/custom.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

#: (drift per M1 bar, volatility per M1 bar) in PRICE terms, per regime.
#: XAUUSD: an M1 ATR of roughly 2-5 pips, the right order for gold.
REGIMES = {
    "up": (0.018, 0.16),
    "down": (-0.018, 0.16),
    "range": (0.0, 0.18),
    "quiet": (0.0, 0.08),
}
REGIME_PROBABILITIES = (0.30, 0.30, 0.25, 0.15)

#: BTCUSD is quoted three orders of magnitude higher and moves a larger
#: FRACTION of price per minute, so its regimes are expressed as fractions and
#: scaled by the starting price.  ~0.05% per M1 bar in the active regimes.
#: ASSUMED, not fitted to exchange data.
BTC_REGIMES_PCT = {
    "up": (0.000030, 0.00055),
    "down": (-0.000030, 0.00055),
    "range": (0.0, 0.00060),
    "quiet": (0.0, 0.00025),
}

#: symbol -> (start price, regime table in price terms)
PROFILES = {
    "XAUUSD": (2300.0, REGIMES),
    "BTCUSD": (
        60000.0,
        {
            name: (drift * 60000.0, vol * 60000.0)
            for name, (drift, vol) in BTC_REGIMES_PCT.items()
        },
    ),
}


def generate(
    bars: int = 30000,
    seed: int = 42,
    start_price: Optional[float] = None,
    symbol: str = "XAUUSD",
) -> pd.DataFrame:
    """Build a regime-switching synthetic M1 series with valid OHLC."""
    default_price, regimes = PROFILES[symbol.upper()]
    if start_price is None:
        start_price = default_price
    rng = np.random.default_rng(seed)
    names = list(regimes)

    drift = np.zeros(bars)
    volatility = np.zeros(bars)
    cursor = 0
    while cursor < bars:
        length = int(rng.integers(400, 2400))
        regime = rng.choice(names, p=REGIME_PROBABILITIES)
        drift[cursor : cursor + length], volatility[cursor : cursor + length] = regimes[regime]
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
    parser = argparse.ArgumentParser(description="Generate synthetic M1 history")
    parser.add_argument(
        "--symbol", type=str, default="XAUUSD", choices=sorted(PROFILES),
        help="market profile to imitate (default: %(default)s)",
    )
    parser.add_argument("--bars", type=int, default=30000, help="number of M1 candles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-price", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()
    out = args.out or Path(f"history/{symbol}_M1_synthetic.csv")
    frame = generate(args.bars, args.seed, args.start_price, symbol)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8")
    print(
        f"Wrote {len(frame)} synthetic {symbol} candles to {out} "
        f"({frame['close'].min():.2f} - {frame['close'].max():.2f})"
    )
    print("NOTE: synthetic data. Backtest results on it are NOT evidence of an edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
