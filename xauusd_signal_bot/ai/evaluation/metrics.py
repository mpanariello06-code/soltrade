"""Trade-log metrics.  NET R leads; raw R is reference only.

Every number here is computed from a trades frame produced by the environment,
so PPO results and rule-engine results are measured the same way and can be
compared without translating between two vocabularies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def summarise_trades(trades: pd.DataFrame, label: str = "") -> Dict[str, Any]:
    """Full statistics for one trade log."""
    empty = {
        "label": label, "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
        "net_r": 0.0, "average_net_r": 0.0, "gross_r": 0.0, "cost_r": 0.0,
        "profit_factor": 0.0, "max_drawdown_r": 0.0,
        "average_holding": 0.0, "median_holding": 0.0,
        "tp1_rate": 0.0, "tp2_rate": 0.0, "tp3_rate": 0.0,
        "sl_rate": 0.0, "timeout_rate": 0.0,
        "average_spread_points": 0.0, "cost_share": 0.0,
    }
    if trades is None or trades.empty:
        return empty

    net = pd.to_numeric(trades["net_r"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(trades.get("gross_r", 0.0), errors="coerce").fillna(0.0)
    cost = pd.to_numeric(trades.get("cost_r", 0.0), errors="coerce").fillna(0.0)
    holding = pd.to_numeric(trades.get("holding_candles", 0), errors="coerce").fillna(0)
    result = trades.get("result", pd.Series(dtype=str)).astype(str)
    tp_hits = pd.to_numeric(trades.get("tp_hits", 0), errors="coerce").fillna(0)

    wins, losses = net[net > 0], net[net < 0]
    equity = net.cumsum()
    gross_loss = float(-losses.sum())
    count = len(trades)

    return {
        "label": label,
        "trades": int(count),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": round(100.0 * len(wins) / count, 2),
        "net_r": round(float(net.sum()), 4),
        "average_net_r": round(float(net.mean()), 4),
        "gross_r": round(float(gross.sum()), 4),
        "cost_r": round(float(cost.sum()), 4),
        "profit_factor": (
            round(float(wins.sum()) / gross_loss, 3) if gross_loss > 0 else float("inf")
        ),
        "max_drawdown_r": round(float((equity.cummax() - equity).max()), 4),
        "average_holding": round(float(holding.mean()), 2),
        "median_holding": round(float(holding.median()), 2),
        "tp1_rate": round(100.0 * float((tp_hits >= 1).mean()), 2),
        "tp2_rate": round(100.0 * float((tp_hits >= 2).mean()), 2),
        "tp3_rate": round(100.0 * float((result == "TP3_HIT").mean()), 2),
        "sl_rate": round(100.0 * float((result == "SL_HIT").mean()), 2),
        "timeout_rate": round(100.0 * float((result == "TIMEOUT").mean()), 2),
        "average_spread_points": round(
            float(pd.to_numeric(trades.get("spread_points", 0.0),
                                errors="coerce").fillna(0.0).mean()), 2
        ),
        # How much of the gross edge the costs ate.  The single most useful
        # number on a scalping strategy, and the easiest one to omit.
        "cost_share": round(
            float(cost.sum()) / abs(float(gross.sum())), 3
        ) if abs(float(gross.sum())) > 1e-9 else 0.0,
    }


def equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """Cumulative NET R and drawdown, trade by trade."""
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["exit_time", "net_r", "equity_r", "drawdown_r"])
    frame = trades.copy()
    frame["net_r"] = pd.to_numeric(frame["net_r"], errors="coerce").fillna(0.0)
    frame["equity_r"] = frame["net_r"].cumsum()
    frame["drawdown_r"] = frame["equity_r"].cummax() - frame["equity_r"]
    columns = [c for c in ("exit_time", "net_r", "equity_r", "drawdown_r") if c in frame]
    return frame[columns]


def by_regime(trades: pd.DataFrame, candles: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Break results down by volatility regime and by session.

    Spec section 33: an agent that works only in one regime has not been shown
    to work, and reporting a single blended number hides exactly that.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    frame["net_r"] = pd.to_numeric(frame["net_r"], errors="coerce").fillna(0.0)

    hour = frame["entry_time"].dt.hour
    frame["session"] = np.select(
        [hour.between(0, 7), hour.between(7, 12), hour.between(12, 16)],
        ["ASIAN", "LONDON", "LONDON_NY_OVERLAP"],
        default="NEW_YORK",
    )
    # Regime from the trade's own risk: a wide stop means a volatile minute.
    risk = pd.to_numeric(frame.get("initial_risk", 0.0), errors="coerce").fillna(0.0)
    if risk.notna().any() and risk.std() > 0:
        median = risk.median()
        frame["volatility"] = np.where(risk >= median, "HIGH_VOL", "LOW_VOL")
    else:
        frame["volatility"] = "UNKNOWN"

    rows = []
    for column in ("session", "volatility"):
        for name, group in frame.groupby(column):
            summary = summarise_trades(group, label=f"{column}:{name}")
            rows.append(summary)
    return pd.DataFrame(rows)
