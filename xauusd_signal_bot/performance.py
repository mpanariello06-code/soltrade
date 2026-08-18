"""Paper-performance analyser for the XAUUSD signal engine.

Reads ``signals.csv`` and ``outcomes.csv`` and reports expectancy-focused
statistics.  Win rate alone is deliberately *not* the headline number: a 60%
strategy with strong R:R beats an 85% strategy that risks 3R to make 1R.

Usage::

    python performance.py
    python performance.py --signals data/signals.csv --outcomes data/outcomes.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from config import load_config
from src.utils import safe_div

def confidence_buckets(config=None) -> List[tuple]:
    """Build ``(low, high, label)`` buckets from the configured confidence bands.

    Derived from config rather than hard-coded so the breakdown table always
    matches the thresholds the engine actually used (see the calibration note
    in ``config.py``).
    """
    bands = list(getattr(config, "confidence_bands", ((80.0, "VERY_STRONG"), (72.0, "STRONG"), (66.0, "MODERATE"))))
    bands.sort(key=lambda item: item[0], reverse=True)
    buckets: List[tuple] = []
    upper = 100.1
    for minimum, label in bands:
        buckets.append((float(minimum), upper, f"{minimum:.0f}+ {label}"))
        upper = float(minimum)
    buckets.append((0.0, upper, f"<{upper:.0f}"))
    return buckets


@dataclass
class Stats:
    """Aggregate statistics for a set of closed signals."""

    label: str = "ALL"
    closed: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    total_r: float = 0.0
    average_r: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_r: float = 0.0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    tp1_rate: float = 0.0
    tp2_rate: float = 0.0
    tp3_rate: float = 0.0
    sl_rate: float = 0.0
    expired_rate: float = 0.0
    average_duration_min: float = 0.0

    def as_row(self) -> Dict[str, Any]:
        return {
            "group": self.label,
            "closed": self.closed,
            "win%": round(self.win_rate, 1),
            "avgR": round(self.average_r, 3),
            "totalR": round(self.total_r, 2),
            "PF": round(self.profit_factor, 2),
            "maxDD_R": round(self.max_drawdown_r, 2),
            "TP1%": round(self.tp1_rate, 1),
            "TP2%": round(self.tp2_rate, 1),
            "TP3%": round(self.tp3_rate, 1),
            "SL%": round(self.sl_rate, 1),
        }


@dataclass
class Report:
    """Full performance report."""

    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    active_signals: int = 0
    average_signals_per_day: float = 0.0
    first_signal: str = ""
    last_signal: str = ""
    overall: Stats = field(default_factory=Stats)
    by_direction: List[Stats] = field(default_factory=list)
    by_session: List[Stats] = field(default_factory=list)
    by_regime: List[Stats] = field(default_factory=list)
    by_confidence: List[Stats] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_frame(path: Path) -> pd.DataFrame:
    """Load a CSV into a DataFrame, returning an empty frame when missing."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return pd.DataFrame()


def merge_outcomes(signals: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Join outcomes onto their signals, keeping the signal's metadata."""
    if outcomes.empty:
        return pd.DataFrame()
    merged = outcomes.copy()
    if not signals.empty and "signal_id" in signals.columns:
        columns = [
            c for c in ("signal_id", "session", "regime", "confidence", "direction", "timeframe")
            if c in signals.columns
        ]
        meta = signals[columns].drop_duplicates(subset=["signal_id"])
        merged = merged.merge(meta, on="signal_id", how="left", suffixes=("", "_signal"))
        for column in ("session", "regime", "confidence", "direction"):
            fallback = f"{column}_signal"
            if fallback in merged.columns:
                merged[column] = merged[column].where(
                    merged[column].notna() & (merged[column].astype(str) != ""),
                    merged[fallback],
                )
                merged = merged.drop(columns=[fallback])
    merged["R_multiple"] = pd.to_numeric(merged.get("R_multiple"), errors="coerce")
    merged["tp_hits"] = pd.to_numeric(merged.get("tp_hits"), errors="coerce").fillna(0).astype(int)
    merged["confidence"] = pd.to_numeric(merged.get("confidence"), errors="coerce")
    merged["duration"] = pd.to_numeric(merged.get("duration"), errors="coerce")
    return merged.dropna(subset=["R_multiple"])


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def max_drawdown(r_series: Sequence[float]) -> float:
    """Largest peak-to-trough decline of the cumulative R curve (positive number)."""
    peak = 0.0
    equity = 0.0
    worst = 0.0
    for value in r_series:
        equity += float(value)
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return abs(worst)


def compute_stats(frame: pd.DataFrame, label: str = "ALL") -> Stats:
    """Compute expectancy statistics for one group of closed signals."""
    stats = Stats(label=label)
    if frame is None or frame.empty:
        return stats

    r_values = frame["R_multiple"].astype(float)
    stats.closed = int(len(frame))
    stats.wins = int((r_values > 0).sum())
    stats.losses = int((r_values < 0).sum())
    stats.breakeven = int((r_values == 0).sum())
    stats.total_r = float(r_values.sum())
    stats.average_r = float(r_values.mean())
    stats.win_rate = 100.0 * safe_div(stats.wins, stats.closed)
    stats.loss_rate = 100.0 * safe_div(stats.losses, stats.closed)

    gross_profit = float(r_values[r_values > 0].sum())
    gross_loss = abs(float(r_values[r_values < 0].sum()))
    stats.profit_factor = safe_div(gross_profit, gross_loss, float("inf") if gross_profit > 0 else 0.0)
    stats.max_drawdown_r = max_drawdown(r_values.tolist())

    hits = frame["tp_hits"].astype(int)
    stats.tp1_rate = 100.0 * safe_div(int((hits >= 1).sum()), stats.closed)
    stats.tp2_rate = 100.0 * safe_div(int((hits >= 2).sum()), stats.closed)
    stats.tp3_rate = 100.0 * safe_div(int((hits >= 3).sum()), stats.closed)

    if "result" in frame.columns:
        results = frame["result"].astype(str)
        stats.sl_rate = 100.0 * safe_div(int((results == "SL_HIT").sum()), stats.closed)
        stats.expired_rate = 100.0 * safe_div(int((results == "EXPIRED").sum()), stats.closed)
    if "duration" in frame.columns:
        stats.average_duration_min = float(pd.to_numeric(frame["duration"], errors="coerce").mean() or 0.0)
    return stats


def _group_stats(frame: pd.DataFrame, column: str) -> List[Stats]:
    """Statistics per distinct value of ``column``."""
    if frame.empty or column not in frame.columns:
        return []
    groups: List[Stats] = []
    for value, subset in frame.groupby(frame[column].astype(str), dropna=False):
        groups.append(compute_stats(subset, label=str(value) or "(unknown)"))
    return sorted(groups, key=lambda s: s.closed, reverse=True)


def _confidence_stats(frame: pd.DataFrame, config=None) -> List[Stats]:
    """Statistics per confidence band."""
    if frame.empty or "confidence" not in frame.columns:
        return []
    groups: List[Stats] = []
    for low, high, label in confidence_buckets(config):
        subset = frame[(frame["confidence"] >= low) & (frame["confidence"] < high)]
        if not subset.empty:
            groups.append(compute_stats(subset, label=label))
    return groups


def build_report(signals: pd.DataFrame, outcomes: pd.DataFrame, config=None) -> Report:
    """Assemble the full report from the two CSVs."""
    report = Report()
    if not signals.empty:
        report.total_signals = int(len(signals))
        directions = signals.get("direction", pd.Series(dtype=str)).astype(str)
        report.buy_signals = int((directions == "BUY").sum())
        report.sell_signals = int((directions == "SELL").sum())
        if "status" in signals.columns:
            statuses = signals["status"].astype(str)
            report.active_signals = int(statuses.isin(["ACTIVE", "TP1_HIT", "TP2_HIT"]).sum())
        if "timestamp" in signals.columns:
            times = pd.to_datetime(signals["timestamp"], errors="coerce", utc=True).dropna()
            if not times.empty:
                report.first_signal = str(times.min())
                report.last_signal = str(times.max())
                span_days = max((times.max() - times.min()).total_seconds() / 86400.0, 1.0)
                report.average_signals_per_day = report.total_signals / span_days

    merged = merge_outcomes(signals, outcomes)
    report.overall = compute_stats(merged, "ALL")
    report.by_direction = _group_stats(merged, "direction")
    report.by_session = _group_stats(merged, "session")
    report.by_regime = _group_stats(merged, "regime")
    report.by_confidence = _confidence_stats(merged, config)
    return report


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _table(rows: List[Dict[str, Any]]) -> str:
    """Render a list of dicts as a fixed-width table."""
    if not rows:
        return "  (no data)"
    headers = list(rows[0].keys())
    widths = {h: max(len(str(h)), *(len(str(row.get(h, ""))) for row in rows)) for h in headers}
    lines = ["  " + "  ".join(str(h).rjust(widths[h]) for h in headers)]
    lines.append("  " + "  ".join("-" * widths[h] for h in headers))
    for row in rows:
        lines.append("  " + "  ".join(str(row.get(h, "")).rjust(widths[h]) for h in headers))
    return "\n".join(lines)


def render_report(report: Report) -> str:
    """Format the report for the console."""
    overall = report.overall
    profit_factor = "inf" if overall.profit_factor == float("inf") else f"{overall.profit_factor:.2f}"
    lines = [
        "=" * 72,
        "XAUUSD SIGNAL ENGINE - PAPER PERFORMANCE",
        "=" * 72,
        f"Total signals      : {report.total_signals}",
        f"  BUY / SELL       : {report.buy_signals} / {report.sell_signals}",
        f"  still open       : {report.active_signals}",
        f"  closed & scored  : {overall.closed}",
        f"Average per day    : {report.average_signals_per_day:.2f}",
        f"Period             : {report.first_signal or '-'}  ->  {report.last_signal or '-'}",
        "",
        "--- EXPECTANCY " + "-" * 57,
        f"Win rate           : {overall.win_rate:.1f}%   ({overall.wins} wins)",
        f"Loss rate          : {overall.loss_rate:.1f}%   ({overall.losses} losses)",
        f"Breakeven          : {overall.breakeven}",
        f"Average R          : {overall.average_r:+.3f}R   <- expectancy per signal",
        f"Total R            : {overall.total_r:+.2f}R",
        f"Profit factor      : {profit_factor}",
        f"Max drawdown       : {overall.max_drawdown_r:.2f}R",
        f"Average duration   : {overall.average_duration_min:.0f} min",
        "",
        "--- TARGET HIT RATES " + "-" * 51,
        f"TP1 reached        : {overall.tp1_rate:.1f}%",
        f"TP2 reached        : {overall.tp2_rate:.1f}%",
        f"TP3 reached        : {overall.tp3_rate:.1f}%",
        f"Closed at SL       : {overall.sl_rate:.1f}%",
        f"Expired            : {overall.expired_rate:.1f}%",
        "",
        "--- BY DIRECTION " + "-" * 55,
        _table([s.as_row() for s in report.by_direction]),
        "",
        "--- BY SESSION " + "-" * 57,
        _table([s.as_row() for s in report.by_session]),
        "",
        "--- BY REGIME " + "-" * 58,
        _table([s.as_row() for s in report.by_regime]),
        "",
        "--- BY CONFIDENCE " + "-" * 54,
        _table([s.as_row() for s in report.by_confidence]),
        "",
        "=" * 72,
        "Paper results only. Past performance does not imply future results.",
        "=" * 72,
    ]
    return "\n".join(lines)


def analyse(signals_path: Path, outcomes_path: Path, config=None) -> Report:
    """Load both CSVs and build the report."""
    return build_report(load_frame(signals_path), load_frame(outcomes_path), config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    config = load_config()
    parser = argparse.ArgumentParser(description="XAUUSD signal engine performance report")
    parser.add_argument("--signals", type=Path, default=config.signals_csv)
    parser.add_argument("--outcomes", type=Path, default=config.outcomes_csv)
    args = parser.parse_args(argv)

    report = analyse(args.signals, args.outcomes, config)
    if report.total_signals == 0:
        print("No signals recorded yet - run main.py or backtest.py first.")
        return 0
    print(render_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
