"""Shadow mode: PPO watches, records and simulates - but never trades.

WHY SHADOW COMES BEFORE DEMO
----------------------------
A backtest tells you how a model would have done on data it was fitted near.
Shadow mode tells you how it behaves on data that did not exist when it was
trained, against live spreads, in real time - while risking nothing.  It is the
only honest bridge between the two, and it costs only patience.

THREE BOOKS, NEVER MERGED
-------------------------
1. rule paper results   ``outcomes.csv``
2. PPO simulated results ``ppo_decisions.csv`` + ``ppo_trades.csv``
3. real demo fills      ``executions.csv``

Merging any two of them destroys the comparison that justifies the whole
exercise, so they are written by different code to different files and are
never summed together.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ai.environment.scalping_env import BUY
from src.logger import get_logger
from src.utils import iso, now_utc

LOGGER = get_logger("ai.shadow")

#: One row per M1 decision - including HOLD, because knowing when the agent
#: declined is as informative as knowing when it acted.
DECISION_COLUMNS: Tuple[str, ...] = (
    "timestamp", "symbol", "timeframe", "model_version", "mode",
    "price", "spread_points", "atr",
    "ppo_action", "prob_hold", "prob_buy", "prob_sell", "confidence",
    "rule_signal", "rule_score", "selected_action",
    "entry", "sl", "tp1", "tp2", "tp3",
    "ok", "error",
)

#: One row per simulated PPO trade.  Same vocabulary as the environment's
#: trade log, so shadow results and backtest results are directly comparable.
TRADE_COLUMNS: Tuple[str, ...] = (
    "signal_id", "symbol", "model_version", "direction",
    "entry_time", "exit_time", "entry_price", "exit_price",
    "stop_loss", "tp1", "tp2", "tp3",
    "result", "tp_hits", "gross_r", "cost_r", "net_r",
    "holding_candles", "spread_points", "initial_risk",
)


def _append(path: Path, row: Dict[str, Any], columns: Tuple[str, ...]) -> None:
    """Append one row, writing the header on first use.  Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        with open(path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns),
                                    extrasaction="ignore", restval="")
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in columns})
    except OSError as exc:
        LOGGER.error("Could not append to %s: %s", path, exc)


@dataclass
class ShadowTrade:
    """A PPO trade being simulated against live candles."""

    signal_id: str
    symbol: str
    direction: int
    entry_time: str
    entry_price: float
    stop_loss: float
    targets: Tuple[float, float, float]
    initial_risk: float
    model_version: str = ""
    entry_index: int = 0
    remaining: float = 1.0
    tp_hits: int = 0
    realised_r: float = 0.0
    cost_r: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.direction == BUY


class ShadowRecorder:
    """Records PPO decisions and simulates the trades they imply, per market."""

    def __init__(self, config, model_version: str = "", env_config=None) -> None:
        self.config = config
        self.symbol = config.symbol
        self.model_version = model_version
        self.env_config = env_config
        directory = Path(config.data_dir) / "ai" / self.symbol
        self.decisions_path = directory / "ppo_decisions.csv"
        self.trades_path = directory / "ppo_trades.csv"
        self.open_trade: Optional[ShadowTrade] = None
        self.closed: List[Dict[str, Any]] = []
        self._candles = 0

    # -- recording ----------------------------------------------------------- #
    def record_decision(
        self, decision, candle: Dict[str, Any], mode: str,
        rule_signal: Optional[Any] = None, levels: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log one M1 decision, whatever it was."""
        probabilities = decision.probabilities or {}
        levels = levels or {}
        _append(self.decisions_path, {
            "timestamp": candle.get("time", iso(now_utc())),
            "symbol": self.symbol,
            "timeframe": "M1",
            "model_version": self.model_version,
            "mode": mode,
            "price": candle.get("close", ""),
            "spread_points": candle.get("spread", ""),
            "atr": candle.get("atr", ""),
            "ppo_action": decision.action_name,
            "prob_hold": probabilities.get("HOLD", ""),
            "prob_buy": probabilities.get("BUY", ""),
            "prob_sell": probabilities.get("SELL", ""),
            "confidence": round(decision.confidence, 4),
            "rule_signal": getattr(rule_signal, "direction", "") if rule_signal else "",
            "rule_score": getattr(rule_signal, "confidence", "") if rule_signal else "",
            # In shadow the SELECTED action is always the rule engine's; PPO's
            # is recorded beside it precisely so the two can be compared later.
            "selected_action": (
                getattr(rule_signal, "direction", "NONE") if rule_signal else "NONE"
            ),
            "entry": levels.get("entry", ""),
            "sl": levels.get("sl", ""),
            "tp1": levels.get("tp1", ""), "tp2": levels.get("tp2", ""),
            "tp3": levels.get("tp3", ""),
            "ok": int(bool(decision.ok)),
            "error": decision.error,
        }, DECISION_COLUMNS)

    # -- simulation ----------------------------------------------------------- #
    def observe_candle(self, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Advance any simulated trade against a newly closed candle."""
        self._candles += 1
        if self.open_trade is None:
            return None
        return self._advance(candle)

    def maybe_open(self, decision, candle: Dict[str, Any]) -> Optional[ShadowTrade]:
        """Start simulating a trade when PPO says BUY or SELL and we are flat."""
        if self.open_trade is not None or not decision.is_trade:
            return None
        atr = float(candle.get("atr", 0.0) or 0.0)
        close = float(candle.get("close", 0.0) or 0.0)
        if atr <= 0 or close <= 0:
            return None

        env = self.env_config
        sl_atr = float(getattr(env, "sl_atr", 0.70))
        tp_atr = tuple(getattr(env, "tp_atr", (0.45, 1.00, 1.70)))
        point_value = float(getattr(env, "point_value", 0.01))
        spread_points = float(candle.get("spread", 0.0) or 0.0)

        sign = 1.0 if decision.action == BUY else -1.0
        # Entry crosses the spread, exactly as the environment does.
        half = spread_points * point_value / 2.0
        slip = float(getattr(env, "slippage_points_entry", 2.0)) * point_value
        entry = close + sign * (half + slip)
        risk = sl_atr * atr

        self.open_trade = ShadowTrade(
            signal_id=f"ppo-{self.symbol}-{candle.get('time', '')}",
            symbol=self.symbol, direction=decision.action,
            entry_time=str(candle.get("time", iso(now_utc()))),
            entry_price=entry, stop_loss=entry - sign * risk,
            targets=tuple(entry + sign * m * atr for m in tp_atr),
            initial_risk=risk, model_version=self.model_version,
            entry_index=self._candles,
        )
        return self.open_trade

    def _advance(self, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Score the open simulated trade.  Pessimistic on ambiguity."""
        trade = self.open_trade
        assert trade is not None
        high = float(candle.get("high", 0.0) or 0.0)
        low = float(candle.get("low", 0.0) or 0.0)
        close = float(candle.get("close", 0.0) or 0.0)
        long = trade.is_long

        # Stop first: identical rule to the environment and to signal_tracker.
        if (low <= trade.stop_loss) if long else (high >= trade.stop_loss):
            return self._close(candle, trade.stop_loss, "SL_HIT")

        for level, target in enumerate(trade.targets):
            if trade.tp_hits > level:
                continue
            if (high >= target) if long else (low <= target):
                if level >= len(trade.targets) - 1:
                    return self._close(candle, target, "TP3_HIT")
                fractions = getattr(self.env_config, "tp_fractions", (0.33, 0.33, 0.34))
                fraction = min(fractions[level], trade.remaining)
                sign = 1.0 if long else -1.0
                trade.realised_r += (target - trade.entry_price) * sign / trade.initial_risk * fraction
                trade.cost_r += self._cost_r(candle, trade) * fraction
                trade.remaining = round(trade.remaining - fraction, 8)
                trade.tp_hits = level + 1

        held = self._candles - trade.entry_index
        if held >= int(getattr(self.env_config, "max_holding_candles", 15)):
            return self._close(candle, close, "TIMEOUT")
        return None

    def _cost_r(self, candle: Dict[str, Any], trade: ShadowTrade) -> float:
        env = self.env_config
        spread = float(candle.get("spread", 0.0) or 0.0)
        points = (
            spread
            + float(getattr(env, "slippage_points_entry", 2.0))
            + float(getattr(env, "slippage_points_exit", 2.0))
            + 2.0 * float(getattr(env, "commission_points_per_side", 0.0))
        )
        return points * float(getattr(env, "point_value", 0.01)) / trade.initial_risk

    def _close(self, candle: Dict[str, Any], exit_price: float,
               result: str) -> Dict[str, Any]:
        trade = self.open_trade
        assert trade is not None
        sign = 1.0 if trade.is_long else -1.0
        gross = (exit_price - trade.entry_price) * sign / trade.initial_risk * trade.remaining
        cost = self._cost_r(candle, trade) * trade.remaining if trade.remaining > 0 else 0.0

        total_gross = trade.realised_r + gross
        total_cost = trade.cost_r + cost
        row = {
            "signal_id": trade.signal_id, "symbol": trade.symbol,
            "model_version": trade.model_version,
            "direction": "BUY" if trade.is_long else "SELL",
            "entry_time": trade.entry_time,
            "exit_time": str(candle.get("time", iso(now_utc()))),
            "entry_price": round(trade.entry_price, 6),
            "exit_price": round(exit_price, 6),
            "stop_loss": round(trade.stop_loss, 6),
            "tp1": round(trade.targets[0], 6), "tp2": round(trade.targets[1], 6),
            "tp3": round(trade.targets[2], 6),
            "result": result, "tp_hits": trade.tp_hits,
            "gross_r": round(total_gross, 6), "cost_r": round(total_cost, 6),
            "net_r": round(total_gross - total_cost, 6),
            "holding_candles": self._candles - trade.entry_index,
            "spread_points": candle.get("spread", ""),
            "initial_risk": round(trade.initial_risk, 6),
        }
        _append(self.trades_path, row, TRADE_COLUMNS)
        self.closed.append(row)
        self.open_trade = None
        LOGGER.info("[%s] PPO shadow %s net %.3fR", self.symbol, result, row["net_r"])
        return row

    # -- reporting -------------------------------------------------------------- #
    def summary(self) -> Dict[str, Any]:
        """PPO's SIMULATED performance.  Never mixed with real demo fills."""
        from ai.evaluation.metrics import summarise_trades

        frame = pd.read_csv(self.trades_path) if self.trades_path.exists() else pd.DataFrame()
        summary = summarise_trades(frame, label=f"PPO shadow {self.symbol}")
        summary["model_version"] = self.model_version
        summary["simulated"] = True
        return summary
