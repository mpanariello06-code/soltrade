"""Download historical M1 candles from MetaTrader 5 into immutable raw files.

Usage::

    python scripts/download_mt5_history.py --symbol XAUUSD --timeframe M1 \
        --start 2015-01-01 --end 2026-08-25

    python scripts/download_mt5_history.py --symbol BTCUSD --start 2020-01-01

Writes ``data/raw/<SYMBOL>/M1/<year>.csv`` plus a ``data_quality_report.json``
describing every defect found.  **Nothing is repaired**: a missing candle is
reported, never invented, because a fabricated bar is indistinguishable from a
real one once it is on disk.

Requires a running MetaTrader 5 terminal, so this is Windows-only.  Everything
downstream - features, environment, training - reads the CSV files and runs
anywhere.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from src.logger import setup_logging  # noqa: E402
from src.markets import MARKET_ORDER, DEFAULT_MARKET, market_argument  # noqa: E402
from src.mt5_history import (  # noqa: E402
    DEFAULT_CHUNK_DAYS,
    MT5HistoryDownloader,
    analyse_quality,
    raw_dir,
    write_quality_report,
    write_yearly,
)


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="Download MT5 history for RL research")
    parser.add_argument(
        "--symbol", type=market_argument, default=DEFAULT_MARKET,
        help=f"market to download: {', '.join(MARKET_ORDER)} (default: %(default)s)",
    )
    parser.add_argument("--timeframe", type=str, default="M1",
                        help="M1 is what the RL pipeline uses (default: %(default)s)")
    parser.add_argument("--start", type=str, required=True, help="UTC date, e.g. 2015-01-01")
    parser.add_argument("--end", type=str, default=None,
                        help="UTC date, exclusive (default: now)")
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS,
                        help="days per MT5 request (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing yearly files instead of merging into them")
    args = parser.parse_args(argv)

    setup_logging(config.log_file, config.log_level)

    start = _utc(args.start)
    end = _utc(args.end) if args.end else datetime.now(timezone.utc)
    if end <= start:
        print("ERROR: --end must be after --start", file=sys.stderr)
        return 2

    downloader = MT5HistoryDownloader(config)
    if not downloader.connect():
        print(
            "ERROR: could not connect to MetaTrader 5.\n"
            "The terminal must be running, and this script is Windows-only.",
            file=sys.stderr,
        )
        return 2

    try:
        print(f"Downloading {args.symbol} {args.timeframe} "
              f"{start.date()} -> {end.date()} ...")
        frame = downloader.download(
            args.symbol, start, end,
            timeframe=args.timeframe, chunk_days=args.chunk_days,
        )
    finally:
        downloader.shutdown()

    if frame.empty:
        print("No candles were returned. Check the symbol and the period.",
              file=sys.stderr)
        return 1

    paths = write_yearly(frame, config, args.symbol, args.timeframe,
                         overwrite=args.overwrite)
    report = analyse_quality(
        frame, args.symbol, args.timeframe,
        requested_start=start.isoformat(), requested_end=end.isoformat(),
    )
    report_path = write_quality_report(report, config, args.symbol, args.timeframe)

    print()
    print(report.summary())
    print()
    print(f"Raw files : {raw_dir(config, args.symbol, args.timeframe)}  "
          f"({len(paths)} yearly file(s))")
    print(f"Report    : {report_path}")
    if not report.usable:
        print("\nWARNING: this series has structural defects. Fix the source "
              "before training - nothing here will repair it for you.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
