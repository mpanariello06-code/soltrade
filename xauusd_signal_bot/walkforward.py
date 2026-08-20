"""Walk-forward evaluation framework.

Splits a history into IN-SAMPLE / VALIDATION / OUT-OF-SAMPLE segments and runs
the *same* configuration over each, then reports the three side by side.

WHAT THIS IS AND IS NOT
-----------------------
This is a **robustness report**, not an optimiser.  It deliberately does not
search for parameters: V1 is a deterministic rule set, and the point of the
exercise is to see whether the same rules behave consistently across periods
they were not looked at during development.

If you do tune parameters, tune them on IN-SAMPLE only, sanity-check on
VALIDATION, and look at OUT-OF-SAMPLE exactly once.  Every extra look at the
out-of-sample segment turns it into in-sample data.

A strategy that scores well in-sample and falls apart out-of-sample is
overfitted, however good the in-sample numbers look.  Compare expectancy
(average R), profit factor and drawdown across segments - not win rate.

Usage::

    python walkforward.py --data history/XAUUSD_M5.csv
    python walkforward.py --data history/XAUUSD_M5.csv --split 0.5 0.25 0.25
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from backtest import Backtester, load_history, write_outputs
from config import load_config
from performance import Report, build_report
from src.logger import get_logger, setup_logging
from src.market_data import timeframe_minutes

LOGGER = get_logger("walkforward")

SEGMENT_NAMES = ("IN_SAMPLE", "VALIDATION", "OUT_OF_SAMPLE")


@dataclass
class Segment:
    """One walk-forward segment and its result."""

    name: str
    start: str
    end: str
    bars: int
    signals: int
    report: Optional[Report] = None


def warmup_bars(config) -> int:
    """M1 bars needed before the context timeframe is usable.

    A segment that starts cold cannot signal until the context frame has enough
    closed candles, so every segment is prefixed with this many M1 bars.
    """
    if not config.context_timeframe:
        return int(config.min_candles_required)
    ratio = timeframe_minutes(config.context_timeframe)
    return int(config.min_context_candles * ratio) + int(config.min_candles_required)


def split_history(
    m5: pd.DataFrame, fractions: Sequence[float], config
) -> List[Tuple[str, pd.DataFrame, int]]:
    """Split chronologically into named segments.

    Returns ``(name, frame, evaluated_bars)``.  Each segment after the first is
    prefixed with the warm-up bars it needs, so every segment can compute a
    200-period EMA on H1 from its own history.  The prefix is warm-up only: the
    backtester's own warm-up index means no signal is emitted inside it, so the
    segments do not overlap in the signals they produce.
    """
    total = len(m5)
    if total < 3:
        raise ValueError("not enough history to split")
    weights = [max(float(f), 0.0) for f in fractions]
    if sum(weights) <= 0:
        raise ValueError("split fractions must sum to more than zero")
    weights = [w / sum(weights) for w in weights]

    warmup = warmup_bars(config)
    segments: List[Tuple[str, pd.DataFrame, int]] = []
    cursor = 0
    for index, weight in enumerate(weights):
        length = total - cursor if index == len(weights) - 1 else int(total * weight)
        end = min(cursor + length, total)
        start = max(0, cursor - warmup) if index > 0 else 0
        segments.append(
            (SEGMENT_NAMES[index], m5.iloc[start:end].reset_index(drop=True), end - cursor)
        )
        cursor = end
    return segments


def run_walkforward(
    m5: pd.DataFrame,
    fractions: Sequence[float],
    out_dir: Optional[Path] = None,
    spread_points: Optional[float] = None,
) -> List[Segment]:
    """Run every segment and collect its report."""
    config = load_config()
    results: List[Segment] = []

    for name, frame, evaluated in split_history(m5, fractions, config):
        LOGGER.info("--- %s: %s bars (%s after warm-up) ---", name, len(frame), evaluated)
        segment = Segment(
            name=name,
            start=str(frame["time"].iloc[len(frame) - evaluated]) if evaluated else "-",
            end=str(frame["time"].iloc[-1]) if len(frame) else "-",
            bars=evaluated,
            signals=0,
        )
        try:
            result = Backtester(config, spread_points=spread_points).run(
                frame, log_evaluations=False, progress_every=0
            )
        except ValueError as exc:
            LOGGER.warning("%s skipped: %s", name, exc)
            results.append(segment)
            continue

        segment.signals = len(result.signals)
        segment.report = build_report(result.signals_frame(), result.outcomes_frame(), config)
        if out_dir is not None:
            write_outputs(result, out_dir, prefix=f"walkforward_{name.lower()}")
        results.append(segment)
    return results


def render_walkforward(segments: List[Segment]) -> str:
    """Render the side-by-side comparison table."""
    lines = [
        "=" * 78,
        "WALK-FORWARD COMPARISON",
        "=" * 78,
        "Consistency across segments matters more than any single segment's numbers.",
        "",
    ]
    header = f"{'segment':<14}{'bars':>7}{'signals':>9}{'closed':>8}{'netWin%':>7}{'avgNetR':>8}{'totNetR':>9}{'netPF':>7}{'maxDD':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for segment in segments:
        if segment.report is None or segment.report.overall.closed == 0:
            lines.append(f"{segment.name:<14}{segment.bars:>7}{segment.signals:>9}{'-':>8}{'-':>7}{'-':>8}{'-':>9}{'-':>7}{'-':>8}")
            continue
        stats = segment.report.overall
        profit_factor = (
            "inf" if stats.net_profit_factor == float("inf") else f"{stats.net_profit_factor:.2f}"
        )
        lines.append(
            f"{segment.name:<14}{segment.bars:>7}{segment.signals:>9}{stats.closed:>8}"
            f"{stats.net_win_rate:>7.1f}{stats.average_net_r:>8.3f}{stats.total_net_r:>9.2f}"
            f"{profit_factor:>7}{stats.max_drawdown_r:>8.2f}"
        )
    lines += ["", "Periods:"]
    for segment in segments:
        lines.append(f"  {segment.name:<14} {segment.start}  ->  {segment.end}")
    lines += [
        "",
        "Read this as: does NET expectancy hold up outside the data the rules",
        "were developed on?  A large drop from IN_SAMPLE to OUT_OF_SAMPLE is the",
        "signature of overfitting.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    config = load_config()
    parser = argparse.ArgumentParser(description="Walk-forward robustness report")
    parser.add_argument("--data", type=Path, required=True, help="M1 history CSV")
    parser.add_argument(
        "--split", type=float, nargs=3, default=(0.5, 0.25, 0.25),
        metavar=("IN", "VAL", "OOS"), help="segment fractions (default 0.5 0.25 0.25)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write per-segment CSVs here")
    parser.add_argument("--spread", type=float, default=None)
    args = parser.parse_args(argv)

    setup_logging(config.log_file, config.log_level)
    try:
        history = load_history(args.data)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    segments = run_walkforward(history, args.split, args.out, args.spread)
    print()
    print(render_walkforward(segments))
    return 0


if __name__ == "__main__":
    sys.exit(main())
