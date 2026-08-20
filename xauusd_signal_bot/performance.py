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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config import load_config
from src.utils import safe_div

#: Fixed score bands (spec section 13).  The question these answer is whether a
#: higher score actually corresponds to a better outcome - which must be
#: measured, never assumed.
SCORE_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (40.0, 50.0, "40-49"),
    (50.0, 60.0, "50-59"),
    (60.0, 70.0, "60-69"),
    (70.0, 80.0, "70-79"),
    (80.0, 90.0, "80-89"),
    (90.0, 100.1, "90-100"),
)


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
    timeout_rate: float = 0.0
    average_duration_min: float = 0.0

    # -- after costs.  These are the numbers that matter for a scalp: a raw
    # -- edge smaller than the spread is not an edge.
    total_net_r: float = 0.0
    average_net_r: float = 0.0
    net_win_rate: float = 0.0
    average_cost_r: float = 0.0
    net_profit_factor: float = 0.0
    average_mfe_r: float = 0.0
    average_mae_r: float = 0.0
    median_minutes_to_tp1: float = 0.0

    def as_row(self) -> Dict[str, Any]:
        """Compact row for the console tables - NET first, deliberately."""
        return {
            "group": self.label,
            "n": self.closed,
            "netWin%": round(self.net_win_rate, 1),
            "avgNetR": round(self.average_net_r, 3),
            "totNetR": round(self.total_net_r, 2),
            "netPF": round(self.net_profit_factor, 2),
            "avgRawR": round(self.average_r, 3),
            "cost R": round(self.average_cost_r, 2),
            "TP1%": round(self.tp1_rate, 1),
            "SL%": round(self.sl_rate, 1),
            "TO%": round(self.timeout_rate, 1),
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
    by_score_band: List[Stats] = field(default_factory=list)
    by_result: List[Stats] = field(default_factory=list)

    def best_regime(self) -> str:
        """Regime with the highest average NET R (at least one closed signal)."""
        ranked = [s for s in self.by_regime if s.closed > 0]
        return max(ranked, key=lambda s: s.average_net_r).label if ranked else "-"

    def worst_regime(self) -> str:
        """Regime with the lowest average NET R."""
        ranked = [s for s in self.by_regime if s.closed > 0]
        return min(ranked, key=lambda s: s.average_net_r).label if ranked else "-"


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
            c for c in (
                "signal_id", "session", "regime", "confidence", "direction",
                "timeframe", "mode", "score", "threshold_used",
            )
            if c in signals.columns
        ]
        meta = signals[columns].drop_duplicates(subset=["signal_id"])
        merged = merged.merge(meta, on="signal_id", how="left", suffixes=("", "_signal"))
        for column in ("session", "regime", "confidence", "direction", "mode", "timeframe", "score"):
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

    # `score` is the newer column; fall back to `confidence` for rows written by
    # an earlier version so old data still groups into score bands.
    if "score" in merged.columns:
        merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
        merged["score"] = merged["score"].fillna(merged["confidence"])
    else:
        merged["score"] = merged["confidence"]
    if "mode" not in merged.columns:
        merged["mode"] = ""
    merged["mode"] = merged["mode"].fillna("").replace("", "SCALPING")
    if "timeframe" not in merged.columns:
        merged["timeframe"] = "M1"

    # NET R is the headline for a scalp.  Rows written by an older build have no
    # net column; fall back to raw so they still aggregate, and to a zero cost.
    if "net_r" in merged.columns:
        merged["net_r"] = pd.to_numeric(merged["net_r"], errors="coerce")
        merged["net_r"] = merged["net_r"].fillna(merged["R_multiple"])
    else:
        merged["net_r"] = merged["R_multiple"]
    if "cost_r" in merged.columns:
        merged["cost_r"] = pd.to_numeric(merged["cost_r"], errors="coerce").fillna(0.0)
    else:
        merged["cost_r"] = 0.0
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
    """Compute expectancy statistics, raw and after costs, for one group."""
    stats = Stats(label=label)
    if frame is None or frame.empty:
        return stats

    r_values = frame["R_multiple"].astype(float)
    net_values = frame["net_r"].astype(float)
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
    stats.max_drawdown_r = max_drawdown(net_values.tolist())

    # -- after costs -------------------------------------------------------- #
    stats.total_net_r = float(net_values.sum())
    stats.average_net_r = float(net_values.mean())
    stats.net_win_rate = 100.0 * safe_div(int((net_values > 0).sum()), stats.closed)
    net_profit = float(net_values[net_values > 0].sum())
    net_loss = abs(float(net_values[net_values < 0].sum()))
    stats.net_profit_factor = safe_div(
        net_profit, net_loss, float("inf") if net_profit > 0 else 0.0
    )
    if "cost_r" in frame.columns:
        stats.average_cost_r = float(pd.to_numeric(frame["cost_r"], errors="coerce").mean() or 0.0)

    hits = frame["tp_hits"].astype(int)
    stats.tp1_rate = 100.0 * safe_div(int((hits >= 1).sum()), stats.closed)
    stats.tp2_rate = 100.0 * safe_div(int((hits >= 2).sum()), stats.closed)
    stats.tp3_rate = 100.0 * safe_div(int((hits >= 3).sum()), stats.closed)

    if "result" in frame.columns:
        results = frame["result"].astype(str)
        stats.sl_rate = 100.0 * safe_div(int((results == "SL_HIT").sum()), stats.closed)
        stats.timeout_rate = 100.0 * safe_div(int((results == "TIMEOUT").sum()), stats.closed)
    if "duration" in frame.columns:
        stats.average_duration_min = float(
            pd.to_numeric(frame["duration"], errors="coerce").mean() or 0.0
        )
    for column, attribute in (("mfe_r", "average_mfe_r"), ("mae_r", "average_mae_r")):
        if column in frame.columns:
            setattr(
                stats, attribute,
                float(pd.to_numeric(frame[column], errors="coerce").mean() or 0.0),
            )
    if "minutes_to_tp1" in frame.columns:
        reached = pd.to_numeric(frame["minutes_to_tp1"], errors="coerce").dropna()
        stats.median_minutes_to_tp1 = float(reached.median()) if len(reached) else 0.0
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


def _score_band_stats(frame: pd.DataFrame) -> List[Stats]:
    """Statistics per fixed score band (spec section 13)."""
    if frame.empty or "score" not in frame.columns:
        return []
    groups: List[Stats] = []
    for low, high, label in SCORE_BANDS:
        subset = frame[(frame["score"] >= low) & (frame["score"] < high)]
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
    report.by_score_band = _score_band_stats(merged)
    report.by_result = _group_stats(merged, "result")
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
    """Format the report for the console.

    NET figures lead throughout.  On a scalp the spread is a large fraction of
    the target, so a raw-R headline would be actively misleading.
    """
    overall = report.overall
    net_pf = "inf" if overall.net_profit_factor == float("inf") else f"{overall.net_profit_factor:.2f}"
    raw_pf = "inf" if overall.profit_factor == float("inf") else f"{overall.profit_factor:.2f}"
    lines = [
        "=" * 76,
        "XAUUSD M1 SCALPER - PAPER PERFORMANCE",
        "=" * 76,
        f"Total signals      : {report.total_signals}",
        f"  BUY / SELL       : {report.buy_signals} / {report.sell_signals}",
        f"  still open       : {report.active_signals}",
        f"  closed & scored  : {overall.closed}",
        f"Average per day    : {report.average_signals_per_day:.2f}",
        f"Period             : {report.first_signal or '-'}  ->  {report.last_signal or '-'}",
        "",
        "--- AFTER COSTS (the numbers that matter) " + "-" * 33,
        f"Net win rate       : {overall.net_win_rate:.1f}%",
        f"Average NET R      : {overall.average_net_r:+.3f}R   <- expectancy per scalp",
        f"Total NET R        : {overall.total_net_r:+.2f}R",
        f"Net profit factor  : {net_pf}",
        f"Average cost       : {overall.average_cost_r:.2f}R per trade",
        f"Max drawdown (net) : {overall.max_drawdown_r:.2f}R",
        "",
        "--- BEFORE COSTS (for reference only) " + "-" * 37,
        f"Raw win rate       : {overall.win_rate:.1f}%   ({overall.wins} wins)",
        f"Average RAW R      : {overall.average_r:+.3f}R",
        f"Total RAW R        : {overall.total_r:+.2f}R",
        f"Raw profit factor  : {raw_pf}",
        "",
        "--- SCALP BEHAVIOUR " + "-" * 55,
        f"TP1 reached        : {overall.tp1_rate:.1f}%",
        f"TP2 reached        : {overall.tp2_rate:.1f}%",
        f"TP3 reached        : {overall.tp3_rate:.1f}%",
        f"Closed at SL       : {overall.sl_rate:.1f}%",
        f"Timed out          : {overall.timeout_rate:.1f}%",
        f"Average hold       : {overall.average_duration_min:.1f} min",
        f"Median mins to TP1 : {overall.median_minutes_to_tp1:.1f}",
        f"Average MFE / MAE  : {overall.average_mfe_r:+.2f}R / {overall.average_mae_r:+.2f}R",
        "",
        "--- BY SCORE BAND " + "-" * 56,
        "  Does a higher score actually mean a better outcome?  Measure, do not assume.",
        _table([s.as_row() for s in report.by_score_band]),
        "",
        "--- BY OUTCOME " + "-" * 59,
        _table([s.as_row() for s in report.by_result]),
        "",
        "--- BY DIRECTION " + "-" * 57,
        _table([s.as_row() for s in report.by_direction]),
        "",
        "--- BY SESSION " + "-" * 59,
        _table([s.as_row() for s in report.by_session]),
        "",
        "--- BY REGIME " + "-" * 60,
        _table([s.as_row() for s in report.by_regime]),
        "",
        "--- BY CONFIDENCE " + "-" * 56,
        _table([s.as_row() for s in report.by_confidence]),
        "",
        "=" * 76,
        "Paper results only. Past performance does not imply future results.",
        "=" * 76,
    ]
    return "\n".join(lines)


def analyse(signals_path: Path, outcomes_path: Path, config=None) -> Report:
    """Load both CSVs and build the report."""
    return build_report(load_frame(signals_path), load_frame(outcomes_path), config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    config = load_config()
    parser = argparse.ArgumentParser(description="XAUUSD M1 scalper performance report")
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
