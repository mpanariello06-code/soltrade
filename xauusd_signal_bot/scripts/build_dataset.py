"""Turn raw MT5 candles into the causal feature matrix PPO trains on.

    python scripts/build_dataset.py --symbol XAUUSD

Reads ``data/raw/<SYMBOL>/M1/*.csv`` and writes
``data/processed/<SYMBOL>/M1/dataset_<fingerprint>.csv``.  Raw files are never
modified.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from _common import load_dataset, processed_path

from config import load_config
from src.markets import DEFAULT_MARKET, MARKET_ORDER, market_argument


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="Build the PPO feature dataset")
    parser.add_argument("--symbol", type=market_argument, default=DEFAULT_MARKET,
                        help=f"{', '.join(MARKET_ORDER)} (default: %(default)s)")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args(argv)

    frame, spec = load_dataset(config, args.symbol, start=args.start, end=args.end)
    times = frame["time"]
    print(f"Symbol       : {args.symbol}")
    print(f"Rows         : {len(frame):,}")
    print(f"Period       : {times.iloc[0]} -> {times.iloc[-1]}")
    print(f"Features     : {spec.size}")
    print(f"Fingerprint  : {spec.fingerprint()}")
    dropped = frame.attrs.get("warmup_dropped", 0)
    print(f"Warm-up rows dropped (never filled): {dropped}")
    print(f"Written to   : {processed_path(config, args.symbol)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
