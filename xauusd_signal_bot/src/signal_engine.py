"""The signal engine - orchestrates every component into a final decision.

Pipeline for one evaluation
---------------------------
1. validate the snapshot (closed candles, no gaps/dupes, fresh enough)
2. compute indicators on M5 / M15 / H1
3. run the nine analysis engines
4. aggregate into bullish/bearish scores (0-100)
5. classify the market regime -> adaptive threshold
6. run the filter chain on the dominant direction
7. build entry / SL / TP1-3 and evaluate R:R
8. emit a Signal, or a NO_SIGNAL evaluation carrying the rejection reason

CLOSED-CANDLE RULE
------------------
The engine only ever sees :class:`~src.market_data.MarketSnapshot` frames, and
those are built from closed candles only.  ``df.iloc[-1]`` is the *confirmed*
signal candle; there is no code path that can reach a forming candle.  This
rule overrides every other consideration in the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .indicators import compute_indicators
from .liquidity import analyze_liquidity
from .logger import get_logger
from .market_data import MarketSnapshot, validate_candles
from .momentum import analyze_momentum
from .price_action import analyze_price_action
from .regime import classify_regime
from .scoring import ScoreCard, compute_scorecard, confidence_band, reason_summary
from .structure import analyze_structure
from .support_resistance import analyze_support_resistance
from .targets import build_targets
from .trend import analyze_context, analyze_trend, htf_alignment
from .utils import (
    ComponentScore,
    as_utc,
    detect_session,
    fnum,
    iso,
    make_signal_id,
    now_utc,
    safe_div,
)
from .volatility import analyze_volatility, classify_volatility
from .timeframes import MODE_SCALPING, SIGNAL_TIMEFRAME, expected_hold_label
from .volume import analyze_volume
from .filters import (
    FilterInput,
    GateState,
    adaptive_threshold,
    is_near_signal,
    run_post_target_filters,
    run_pre_target_filters,
)

LOGGER = get_logger("engine")

#: Raw features preserved on every evaluated candle for future ML training
#: (spec section 45).  Order is fixed so ``evaluations.csv`` has a stable schema.
FEATURE_KEYS: Tuple[str, ...] = (
    "close", "atr", "atr_ratio", "adx", "plus_di", "minus_di", "rsi", "macd_hist",
    "stoch_k", "roc", "rel_volume", "body_ratio", "ema_fast_minus_slow",
    "close_minus_ema200", "bb_width", "structure_bos_bull", "structure_bos_bear",
    "structure_choch_bull", "structure_choch_bear", "consolidating",
    "swept_bull", "swept_bear", "sr_state",
    # --- M1 microstructure -------------------------------------------------- #
    # Everything a future model would need to answer "was the next few minutes'
    # move big enough to clear costs?" without re-deriving it from raw candles.
    "atr_pips", "spread_pips", "cost_pips", "atr_to_cost", "range_pips",
    "body_pips", "upper_wick_pips", "lower_wick_pips", "close_location",
    "displacement_atr", "velocity_pips_per_min", "acceleration",
    "micro_range_pips_5", "micro_range_pips_15", "dist_to_high_5_pips",
    "dist_to_low_5_pips", "minute_of_hour", "hour_of_day",
)

#: Column order of ``evaluations.csv``.  The first block matches the spec
#: exactly; the rest is diagnostic context plus the ML feature block.
EVALUATION_COLUMNS: Tuple[str, ...] = (
    "timestamp", "symbol", "timeframe", "bullish_score", "bearish_score",
    "trend_score", "htf_score", "momentum_score", "structure_score",
    "liquidity_score", "sr_score", "volume_score", "volatility_score",
    "price_action_score", "regime", "spread", "decision", "rejection_reason",
    "mode", "threshold_used", "near_signal", "session", "volatility_band",
    "htf_alignment", "signal_id",
) + tuple(f"f_{key}" for key in FEATURE_KEYS)

DECISION_BUY = "BUY"
DECISION_SELL = "SELL"
DECISION_NONE = "NO_SIGNAL"
#: Diagnostic only - a candidate that came close to the bar but did not qualify.
#: It is never sent as a trading signal.
DECISION_NEAR = "NEAR_SIGNAL"


@dataclass
class Signal:
    """A confirmed trading signal (spec section 32)."""

    signal_id: str
    symbol: str
    timeframe: str
    direction: str
    timestamp: datetime
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confidence: float
    bullish_score: float
    bearish_score: float
    regime: str
    risk_reward: float
    session: str
    reason_summary: str
    confidence_label: str = ""
    rr1: float = 0.0
    rr2: float = 0.0
    rr3: float = 0.0
    sl_mode: str = ""
    confirmations: Dict[str, bool] = field(default_factory=dict)
    mode: str = "SCALPING"
    threshold_used: float = 0.0

    # -- scalping geometry and costs, fixed at signal time ------------------ #
    spread_points: float = float("nan")
    cost_pips: float = 0.0
    cost_r: float = 0.0
    sl_pips: float = 0.0
    tp_pips: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    net_rr1: float = 0.0
    net_rr2: float = 0.0
    net_rr3: float = 0.0
    atr: float = 0.0
    expected_hold: str = "SHORT"

    def to_row(self) -> Dict[str, Any]:
        """Flat mapping for ``signals.csv``."""
        return {
            "signal_id": self.signal_id,
            "timestamp": iso(self.timestamp),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "entry": self.entry,
            "sl": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "confidence": self.confidence,
            "bullish_score": self.bullish_score,
            "bearish_score": self.bearish_score,
            "regime": self.regime,
            "status": "ACTIVE",
            "mode": self.mode,
            "threshold_used": round(self.threshold_used, 2),
            "score": self.confidence,
            "session": self.session,
            "spread_points": round(self.spread_points, 1) if pd.notna(self.spread_points) else "",
            "cost_pips": self.cost_pips,
            "cost_r": self.cost_r,
            "sl_pips": self.sl_pips,
            "tp1_pips": self.tp_pips[0],
            "tp2_pips": self.tp_pips[1],
            "tp3_pips": self.tp_pips[2],
            "atr": self.atr,
            "net_rr1": self.net_rr1,
            "net_rr2": self.net_rr2,
            "net_rr3": self.net_rr3,
            "expected_hold": self.expected_hold,
            "risk_reward": self.risk_reward,
            "rr1": self.rr1,
            "rr2": self.rr2,
            "rr3": self.rr3,
            "sl_mode": self.sl_mode,
            "confidence_label": self.confidence_label,
            "reason_summary": self.reason_summary,
        }


@dataclass
class Evaluation:
    """Result of evaluating one closed candle - logged whatever the outcome."""

    timestamp: datetime
    symbol: str
    timeframe: str
    decision: str = DECISION_NONE
    rejection_reason: str = ""
    card: Optional[ScoreCard] = None
    regime: str = ""
    volatility_band: str = ""
    session: str = ""
    spread_points: float = float("nan")
    threshold: float = 0.0
    htf_alignment: str = ""
    signal: Optional[Signal] = None
    features: Dict[str, float] = field(default_factory=dict)
    mode: str = "SCALPING"
    #: diagnostic flag - close to the threshold but rejected
    near_signal: bool = False

    @property
    def has_signal(self) -> bool:
        return self.signal is not None

    @property
    def best_score(self) -> float:
        """Score of the stronger direction, 0 when nothing was computed."""
        return self.card.dominant_score() if self.card else 0.0

    def to_row(self) -> Dict[str, Any]:
        """Flat mapping for ``evaluations.csv``.

        The first block matches the spec exactly; the trailing ``f_*`` columns
        are raw features preserved so a future ML filter can be trained on this
        file without re-deriving anything (spec sections 36 and 45).
        """
        card = self.card
        row: Dict[str, Any] = {
            "timestamp": iso(self.timestamp),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bullish_score": round(card.bullish_score, 2) if card else 0.0,
            "bearish_score": round(card.bearish_score, 2) if card else 0.0,
            "trend_score": 0.0,
            "htf_score": 0.0,
            "momentum_score": 0.0,
            "structure_score": 0.0,
            "liquidity_score": 0.0,
            "sr_score": 0.0,
            "volume_score": 0.0,
            "volatility_score": 0.0,
            "price_action_score": 0.0,
            "regime": self.regime,
            "spread": round(self.spread_points, 2) if pd.notna(self.spread_points) else "",
            "decision": self.decision,
            "rejection_reason": self.rejection_reason,
        }

        if card is not None:
            direction = DECISION_BUY if card.bullish_score >= card.bearish_score else DECISION_SELL
            mapping = {
                "trend_score": "trend",
                "htf_score": "htf",
                "momentum_score": "momentum",
                "structure_score": "structure",
                "liquidity_score": "liquidity",
                "sr_score": "support_resistance",
                "volume_score": "volume",
                "volatility_score": "volatility",
                "price_action_score": "price_action",
            }
            for column, component in mapping.items():
                row[column] = round(card.component_score(component, direction), 2)

        row.update(
            {
                "mode": self.mode,
                "threshold_used": round(self.threshold, 2),
                "near_signal": int(bool(self.near_signal)),
                "session": self.session,
                "volatility_band": self.volatility_band,
                "htf_alignment": self.htf_alignment,
                "signal_id": self.signal.signal_id if self.signal else "",
            }
        )
        # always emit the full feature block so the CSV schema never shifts
        row.update({f"f_{key}": self.features.get(key, "") for key in FEATURE_KEYS})
        return row


class SignalEngine:
    """Deterministic multi-confirmation signal engine.

    The same instance is used by ``main.py`` (live) and ``backtest.py``, so a
    backtested signal is produced by exactly the code that runs live.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.timeframe_minutes = 1   #: M1 - the only signal timeframe

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _ensure_indicators(df: pd.DataFrame, params) -> pd.DataFrame:
        """Attach indicators unless the frame already carries them.

        The backtester pre-computes indicators once over the whole history and
        slices it.  That is *not* lookahead: every indicator in
        :mod:`src.indicators` is causal, so the value at bar ``i`` is identical
        whether it was computed over ``0..i`` or over the full array.
        ``tests/test_indicators.py`` asserts this property.
        """
        if "ema_fast" in df.columns and "atr" in df.columns:
            return df
        return compute_indicators(df, params)

    def _collect_features(
        self, df: pd.DataFrame, components: Dict[str, ComponentScore], config, spread_points
    ) -> Dict[str, float]:
        """Raw readings preserved for future ML training.

        The eventual question a model would answer is *"given these M1
        conditions, how likely is a move large enough to clear costs within the
        holding window?"* - so the vector carries the microstructure and the
        cost context, not just the indicator values.
        """
        row = df.iloc[-1]
        pip = config.pip_value
        structure = components["structure"].details
        liquidity = components["liquidity"].details
        sr = components["support_resistance"].details

        atr_value = fnum(row["atr"])
        cost_price = config.round_trip_cost(spread_points)
        spread_price = config.effective_spread_points(spread_points) * config.point_value
        high, low = fnum(row["high"]), fnum(row["low"])
        candle_range = max(high - low, 1e-9)
        close = fnum(row["close"])
        candle_time = as_utc(pd.Timestamp(row["time"]).to_pydatetime())

        window5 = df.iloc[-5:]
        window15 = df.iloc[-15:]
        # Velocity over the last 5 minutes, in pips per minute; acceleration is
        # how much faster the last 3 minutes were than the 5 before them.
        velocity = safe_div(close - fnum(df["close"].iloc[-6]), 5.0 * pip) if len(df) > 6 else 0.0
        recent = safe_div(close - fnum(df["close"].iloc[-4]), 3.0 * pip) if len(df) > 4 else 0.0

        return {
            "close": round(close, 3),
            "atr": round(atr_value, 4),
            "atr_ratio": round(fnum(components["volatility"].details.get("atr_ratio"), 1.0), 3),
            "adx": round(fnum(row["adx"]), 2),
            "plus_di": round(fnum(row["plus_di"]), 2),
            "minus_di": round(fnum(row["minus_di"]), 2),
            "rsi": round(fnum(row["rsi"]), 2),
            "macd_hist": round(fnum(row["macd_hist"]), 5),
            "stoch_k": round(fnum(row["stoch_k"]), 2),
            "roc": round(fnum(row["roc"]), 4),
            "rel_volume": round(fnum(row["rel_volume"], 1.0), 3),
            "body_ratio": round(fnum(row["body_ratio"]), 3),
            "ema_fast_minus_slow": round(fnum(row["ema_fast"]) - fnum(row["ema_slow"]), 3),
            "close_minus_ema200": round(close - fnum(row["ema_trend"]), 3),
            "bb_width": round(fnum(row["bb_width"]), 5),
            "structure_bos_bull": int(bool(structure.get("bos_bull"))),
            "structure_bos_bear": int(bool(structure.get("bos_bear"))),
            "structure_choch_bull": int(bool(structure.get("choch_bull"))),
            "structure_choch_bear": int(bool(structure.get("choch_bear"))),
            "consolidating": int(bool(structure.get("consolidating"))),
            "swept_bull": int(bool(liquidity.get("swept_bull"))),
            "swept_bear": int(bool(liquidity.get("swept_bear"))),
            "sr_state": {
                "NEUTRAL": 0, "NEAR_SUPPORT": 1, "NEAR_RESISTANCE": 2,
                "BREAKOUT": 3, "BREAKDOWN": 4,
            }.get(str(sr.get("state", "NEUTRAL")), 0),
            # --- M1 microstructure --------------------------------------- #
            "atr_pips": round(config.pips(atr_value), 3),
            "spread_pips": round(config.pips(spread_price), 3),
            "cost_pips": round(config.pips(cost_price), 3),
            "atr_to_cost": round(safe_div(atr_value, cost_price), 3),
            "range_pips": round(config.pips(candle_range), 3),
            "body_pips": round(config.pips(abs(close - fnum(row["open"]))), 3),
            "upper_wick_pips": round(config.pips(fnum(row["upper_wick"])), 3),
            "lower_wick_pips": round(config.pips(fnum(row["lower_wick"])), 3),
            "close_location": round(safe_div(close - low, candle_range), 3),
            "displacement_atr": round(safe_div(abs(close - fnum(row["open"])), max(atr_value, 1e-9)), 3),
            "velocity_pips_per_min": round(velocity, 3),
            "acceleration": round(recent - velocity, 3),
            "micro_range_pips_5": round(
                config.pips(fnum(window5["high"].max()) - fnum(window5["low"].min())), 3
            ),
            "micro_range_pips_15": round(
                config.pips(fnum(window15["high"].max()) - fnum(window15["low"].min())), 3
            ),
            "dist_to_high_5_pips": round(config.pips(fnum(window5["high"].max()) - close), 3),
            "dist_to_low_5_pips": round(config.pips(close - fnum(window5["low"].min())), 3),
            "minute_of_hour": candle_time.minute,
            "hour_of_day": candle_time.hour,
        }

    def _validate(self, snapshot: MarketSnapshot, config=None) -> str:
        """Return an empty string when the snapshot is usable, else the reason."""
        cfg = config or self.config
        checks = [(snapshot.signal_df, SIGNAL_TIMEFRAME, cfg.min_candles_required)]
        if snapshot.context_df is not None and cfg.context_timeframe:
            checks.append(
                (snapshot.context_df, cfg.context_timeframe, cfg.min_context_candles)
            )
        for frame, label, minimum in checks:
            ok, reason = validate_candles(frame, label, minimum)
            if not ok:
                return f"{label}: {reason}"
        return ""

    # -- main entry point --------------------------------------------------- #
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        gate: Optional[GateState] = None,
        config=None,
    ) -> Evaluation:
        """Evaluate one closed M1 candle and return the full :class:`Evaluation`.

        ``config`` lets the caller pass a runtime-adjusted view of the
        configuration (threshold, holding period) without rebuilding the engine.
        """
        cfg = config or self.config
        gate = gate or GateState()
        candle_time = (
            snapshot.candle_time if not snapshot.signal_df.empty else now_utc()
        )

        evaluation = Evaluation(
            timestamp=candle_time,
            symbol=snapshot.symbol,
            timeframe=SIGNAL_TIMEFRAME,
            spread_points=snapshot.spread_points,
            mode=MODE_SCALPING,
        )

        reason = self._validate(snapshot, cfg)
        if reason:
            evaluation.rejection_reason = f"data validation failed - {reason}"
            return evaluation

        params = cfg.indicators
        m1 = self._ensure_indicators(snapshot.signal_df, params)
        context = (
            self._ensure_indicators(snapshot.context_df, params)
            if snapshot.context_df is not None
            else None
        )

        # -- 3. analysis engines ------------------------------------------- #
        components: Dict[str, ComponentScore] = {
            "trend": analyze_trend(m1, cfg),
            "htf": analyze_context(context, cfg),
            "momentum": analyze_momentum(m1, cfg),
            "structure": analyze_structure(m1, cfg),
            "liquidity": analyze_liquidity(m1, cfg),
            "support_resistance": analyze_support_resistance(m1, cfg),
            "volume": analyze_volume(m1, cfg),
            "volatility": analyze_volatility(m1, cfg),
            "price_action": analyze_price_action(m1, cfg),
        }

        # -- 4/5. scoring and regime ---------------------------------------- #
        card = compute_scorecard(components, cfg)
        regime, _regime_details = classify_regime(m1, components, cfg)
        volatility_band, _ratio = classify_volatility(m1, cfg)
        session = detect_session(candle_time, cfg.sessions, cfg.session_priority)

        evaluation.card = card
        evaluation.regime = regime
        evaluation.volatility_band = volatility_band
        evaluation.session = session
        evaluation.features = self._collect_features(
            m1, components, cfg, snapshot.spread_points
        )

        direction = card.direction
        if direction == "NONE":
            evaluation.rejection_reason = "no directional edge"
            evaluation.threshold = adaptive_threshold(regime, volatility_band, "NEUTRAL", cfg) or 0.0
            return self._finish_rejected(evaluation, cfg)

        alignment = htf_alignment(components["htf"], direction)
        evaluation.htf_alignment = alignment

        filter_input = FilterInput(
            df=m1,
            card=card,
            direction=direction,
            regime=regime,
            volatility_band=volatility_band,
            session=session,
            spread_points=snapshot.spread_points,
            candle_time=candle_time,
            htf_alignment=alignment,
            timeframe_minutes=1,
            gate=gate,
        )

        # -- 6. pre-target filters ------------------------------------------ #
        outcome = run_pre_target_filters(filter_input, cfg)
        evaluation.threshold = outcome.threshold
        if not outcome.passed:
            evaluation.rejection_reason = outcome.reason
            return self._finish_rejected(evaluation, cfg)

        # -- 7. targets, costs and R:R --------------------------------------- #
        targets, target_reason = build_targets(
            m1, cfg, direction, spread_points=snapshot.spread_points
        )
        if targets is None:
            evaluation.rejection_reason = target_reason or "could not build targets"
            return self._finish_rejected(evaluation, cfg)

        filter_input.targets = targets
        outcome = run_post_target_filters(filter_input, cfg, outcome.threshold)
        if not outcome.passed:
            evaluation.rejection_reason = outcome.reason
            return self._finish_rejected(evaluation, cfg)

        # -- 8. build the signal --------------------------------------------- #
        score = card.bullish_score if direction == DECISION_BUY else card.bearish_score
        atr_pips = cfg.pips(targets.atr)
        signal = Signal(
            signal_id=make_signal_id(snapshot.symbol, candle_time, direction),
            symbol=snapshot.symbol,
            timeframe=SIGNAL_TIMEFRAME,
            direction=direction,
            timestamp=candle_time,
            entry=targets.entry,
            stop_loss=targets.stop_loss,
            tp1=targets.tp1,
            tp2=targets.tp2,
            tp3=targets.tp3,
            confidence=round(score, 1),
            bullish_score=card.bullish_score,
            bearish_score=card.bearish_score,
            regime=regime,
            risk_reward=targets.rr2,
            session=session,
            reason_summary=reason_summary(card, direction),
            confidence_label=confidence_band(score, cfg),
            rr1=targets.rr1,
            rr2=targets.rr2,
            rr3=targets.rr3,
            sl_mode=targets.sl_mode,
            confirmations=card.confirmations(direction),
            mode=MODE_SCALPING,
            threshold_used=round(outcome.threshold, 2),
            spread_points=targets.spread_points,
            cost_pips=targets.cost_pips,
            cost_r=targets.cost_r,
            sl_pips=targets.sl_pips,
            tp_pips=targets.tp_pips,
            net_rr1=targets.net_rr1,
            net_rr2=targets.net_rr2,
            net_rr3=targets.net_rr3,
            atr=targets.atr,
            expected_hold=expected_hold_label(targets.tp_pips[2], atr_pips),
        )

        evaluation.decision = direction
        evaluation.signal = signal
        LOGGER.info(
            "SCALP %s %s @ %.2f | score %.1f | SL %.1fp TP %.1f/%.1f/%.1fp | "
            "netRR %.2f/%.2f/%.2f | cost %.1fp | %s",
            signal.direction, signal.symbol, signal.entry, signal.confidence,
            signal.sl_pips, *signal.tp_pips,
            signal.net_rr1, signal.net_rr2, signal.net_rr3,
            signal.cost_pips, signal.regime,
        )
        return evaluation

    # -- rejection bookkeeping ---------------------------------------------- #
    @staticmethod
    def _finish_rejected(evaluation: "Evaluation", config) -> "Evaluation":
        """Tag a rejected evaluation as NEAR_SIGNAL when it came close.

        Diagnostic only: the decision is still "no trade", but the row in
        ``evaluations.csv`` now says whether the threshold was the thing that
        stopped it, and by how little.
        """
        if evaluation.card is None:
            return evaluation
        if is_near_signal(evaluation.best_score, evaluation.threshold, config):
            evaluation.near_signal = True
            evaluation.decision = DECISION_NEAR
        return evaluation
