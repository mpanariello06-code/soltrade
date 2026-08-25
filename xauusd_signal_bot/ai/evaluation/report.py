"""Write the ``reports/ppo/`` artefact set from a trade log.

WHY A SINGLE WRITER
-------------------
Walk-forward, holdout evaluation and backtests all produce the same shape of
answer, and each writing its own subtly different CSV is how two reports of the
same run end up disagreeing.  One function writes all five files, so every
report has the same columns and the same definitions.

Spec section 32 names the files; section 33 requires the regime breakdown, which
is written alongside because a single blended number hides an agent that works
in exactly one regime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .metrics import by_regime, equity_curve, summarise_trades

#: The artefact set.  Named exactly as the brief specifies.
REPORT_FILES = (
    "performance.csv", "equity.csv", "trades.csv", "drawdown.csv", "walk_forward.csv",
)


def drawdown_series(trades: pd.DataFrame) -> pd.DataFrame:
    """Drawdown over time, with the depth and length of each underwater stretch.

    Reported separately from the equity curve because the two answer different
    questions: equity says where you ended, drawdown says what you had to sit
    through to get there - and the second is what decides whether a strategy is
    actually runnable.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["exit_time", "equity_r", "peak_r",
                                     "drawdown_r", "underwater_trades"])
    frame = trades.copy()
    frame["net_r"] = pd.to_numeric(frame["net_r"], errors="coerce").fillna(0.0)
    frame["equity_r"] = frame["net_r"].cumsum()
    frame["peak_r"] = frame["equity_r"].cummax()
    frame["drawdown_r"] = frame["peak_r"] - frame["equity_r"]

    # How many consecutive trades this stretch has been below the peak.
    underwater, run = [], 0
    for value in frame["drawdown_r"]:
        run = run + 1 if value > 1e-9 else 0
        underwater.append(run)
    frame["underwater_trades"] = underwater

    columns = [c for c in ("exit_time", "equity_r", "peak_r", "drawdown_r",
                           "underwater_trades") if c in frame]
    return frame[columns]


def write_report(
    trades: pd.DataFrame,
    output_dir,
    label: str = "",
    walk_forward: Optional[pd.DataFrame] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write the full artefact set and return the paths written.

    An empty trade log still produces files: "the agent took no trades" is a
    result, and a missing report is indistinguishable from a run that never
    happened.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    trades = trades if trades is not None else pd.DataFrame()
    summary = summarise_trades(trades, label=label)
    if extra:
        summary.update(extra)

    performance = output_dir / "performance.csv"
    pd.DataFrame([summary]).to_csv(performance, index=False)
    written["performance"] = performance
    (output_dir / "performance.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    for name, frame in (
        ("trades", trades),
        ("equity", equity_curve(trades)),
        ("drawdown", drawdown_series(trades)),
        ("by_regime", by_regime(trades)),      # spec section 33
    ):
        path = output_dir / f"{name}.csv"
        (frame if frame is not None else pd.DataFrame()).to_csv(path, index=False)
        written[name] = path

    path = output_dir / "walk_forward.csv"
    (walk_forward if walk_forward is not None else pd.DataFrame()).to_csv(
        path, index=False
    )
    written["walk_forward"] = path
    return written


def render_summary(summary: Dict[str, Any], title: str = "PPO PERFORMANCE") -> str:
    """A console block, NET first because NET is what matters."""
    def value(key, default=0.0):
        return summary.get(key, default)

    profit_factor = value("profit_factor", 0.0)
    profit_factor = "inf" if profit_factor == float("inf") else f"{profit_factor:.3f}"

    lines = [
        "=" * 68,
        title,
        "=" * 68,
        f"Trades            : {value('trades', 0)}",
        f"Win rate          : {value('win_rate'):.2f}%",
        "",
        "--- AFTER COSTS (the numbers that matter) " + "-" * 25,
        f"NET R             : {value('net_r'):+.4f}",
        f"Average NET R     : {value('average_net_r'):+.4f}   <- expectancy per trade",
        f"Profit factor     : {profit_factor}",
        f"Max drawdown      : {value('max_drawdown_r'):.4f}R",
        "",
        "--- BEFORE COSTS (reference only) " + "-" * 33,
        f"Gross R           : {value('gross_r'):+.4f}",
        f"Cost R            : {value('cost_r'):.4f}",
        f"Cost share        : {value('cost_share'):.3f} of gross",
        "",
        "--- SCALP BEHAVIOUR " + "-" * 47,
        f"TP1 / TP2 / TP3   : {value('tp1_rate'):.1f}% / "
        f"{value('tp2_rate'):.1f}% / {value('tp3_rate'):.1f}%",
        f"SL / timeout      : {value('sl_rate'):.1f}% / {value('timeout_rate'):.1f}%",
        f"Holding (avg/med) : {value('average_holding'):.2f} / "
        f"{value('median_holding'):.2f} candles",
        f"Average spread    : {value('average_spread_points'):.2f} points",
    ]
    if "action_mix" in summary:
        mix = summary["action_mix"]
        lines += ["", f"Action mix        : HOLD {mix.get('HOLD', 0):.1f}%  "
                      f"BUY {mix.get('BUY', 0):.1f}%  SELL {mix.get('SELL', 0):.1f}%"]
    lines += ["=" * 68,
              "Paper/simulated results. Not evidence of a live edge."]
    return "\n".join(lines)


def compare(summaries: List[Dict[str, Any]]) -> str:
    """Side-by-side table, for RULE_ONLY vs PPO on the same period.

    The comparison only means anything when both ran over the same candles with
    the same cost model, which is why the backtester runs them that way.
    """
    if not summaries:
        return "nothing to compare"
    header = (f"{'strategy':<18}{'trades':>8}{'netR':>10}{'avgR':>9}"
              f"{'win%':>8}{'PF':>8}{'maxDD':>9}{'hold':>7}")
    lines = [header, "-" * len(header)]
    for summary in summaries:
        profit_factor = summary.get("profit_factor", 0.0)
        profit_factor = "inf" if profit_factor == float("inf") else f"{profit_factor:.2f}"
        lines.append(
            f"{str(summary.get('label', '?'))[:18]:<18}"
            f"{summary.get('trades', 0):>8}"
            f"{summary.get('net_r', 0.0):>10.2f}"
            f"{summary.get('average_net_r', 0.0):>9.3f}"
            f"{summary.get('win_rate', 0.0):>8.1f}"
            f"{profit_factor:>8}"
            f"{summary.get('max_drawdown_r', 0.0):>9.2f}"
            f"{summary.get('average_holding', 0.0):>7.1f}"
        )
    return "\n".join(lines)
