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
from .market_data import MarketSnapshot, timeframe_minutes, validate_candles
from .momentum import analyze_momentum
from .price_action import analyze_price_action
from .regime import classify_regime
from .scoring import ScoreCard, compute_scorecard, confidence_band, reason_summary
from .structure import analyze_structure
from .support_resistance import analyze_support_resistance
from .targets import build_targets
from .trend import analyze_htf, analyze_trend, htf_alignment
from .utils import (
    ComponentScore,
    detect_session,
    fnum,
    iso,
    make_signal_id,
    now_utc,
)
from .volatility import analyze_volatility, classify_volatility
from .timeframes import confirmation_label
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
)

#: Column order of ``evaluations.csv``.  The first block matches the spec
#: exactly; the rest is diagnostic context plus the ML feature block.
EVALUATION_COLUMNS: Tuple[str, ...] = (
    "timestamp", "symbol", "timeframe", "bullish_score", "bearish_score",
    "trend_score", "htf_score", "momentum_score", "structure_score",
    "liquidity_score", "sr_score", "volume_score", "volatility_score",
    "price_action_score", "regime", "spread", "decision", "rejection_reason",
    # operating context - what the engine was configured as at this moment
    "mode", "signal_timeframe", "confirmation_timeframes", "threshold_used",
    "near_signal", "session", "volatility_band", "htf_alignment", "signal_id",
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
    #: operating context, preserved so research and standard signals can be
    #: told apart when the CSVs are analysed later
    mode: str = "STANDARD"
    confirmation_timeframes: str = ""
    threshold_used: float = 0.0

    @property
    def is_research(self) -> bool:
        """True for a candidate collected under RESEARCH mode."""
        return self.mode == "RESEARCH"

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
            "signal_timeframe": self.timeframe,
            "confirmation_timeframes": self.confirmation_timeframes,
            "threshold_used": round(self.threshold_used, 2),
            "score": self.confidence,
            "session": self.session,
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
    mode: str = "STANDARD"
    confirmation_timeframes: str = ""
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
                "signal_timeframe": self.timeframe,
                "confirmation_timeframes": self.confirmation_timeframes,
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
        self.timeframe_minutes = timeframe_minutes(config.signal_timeframe)

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
        self, m5: pd.DataFrame, components: Dict[str, ComponentScore]
    ) -> Dict[str, float]:
        """Raw indicator readings preserved for future ML training."""
        row = m5.iloc[-1]
        structure = components["structure"].details
        liquidity = components["liquidity"].details
        sr = components["support_resistance"].details
        return {
            "close": round(fnum(row["close"]), 3),
            "atr": round(fnum(row["atr"]), 4),
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
            "close_minus_ema200": round(fnum(row["close"]) - fnum(row["ema_trend"]), 3),
            "bb_width": round(fnum(row["bb_width"]), 5),
            "structure_bos_bull": int(bool(structure.get("bos_bull"))),
            "structure_bos_bear": int(bool(structure.get("bos_bear"))),
            "structure_choch_bull": int(bool(structure.get("choch_bull"))),
            "structure_choch_bear": int(bool(structure.get("choch_bear"))),
            "consolidating": int(bool(structure.get("consolidating"))),
            "swept_bull": int(bool(liquidity.get("swept_bull"))),
            "swept_bear": int(bool(liquidity.get("swept_bear"))),
            "sr_state": {"NEUTRAL": 0, "NEAR_SUPPORT": 1, "NEAR_RESISTANCE": 2, "BREAKOUT": 3, "BREAKDOWN": 4}.get(
                str(sr.get("state", "NEUTRAL")), 0
            ),
        }

    def _validate(self, snapshot: MarketSnapshot, config=None) -> str:
        """Return an empty string when the snapshot is usable, else the reason."""
        cfg = config or self.config
        checks = [(snapshot.m5, cfg.signal_timeframe, cfg.min_candles_required)]
        # A confirmation timeframe is optional (H1 has one, H4 has none), but
        # whenever a frame *is* supplied it must be as sound as the signal one.
        for frame, label in (
            (snapshot.m15, cfg.intermediate_timeframe),
            (snapshot.h1, cfg.higher_timeframe),
        ):
            if frame is not None and label:
                checks.append((frame, label, cfg.min_htf_candles_required))
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
        """Evaluate one closed candle and return the full :class:`Evaluation`.

        ``config`` lets the caller pass a runtime-adjusted view of the
        configuration (mode, signal timeframe, threshold) without rebuilding the
        engine - see :func:`src.runtime_state.effective_config`.
        """
        cfg = config or self.config
        gate = gate or GateState()
        candle_time = snapshot.candle_time if not snapshot.m5.empty else now_utc()
        confirmation = snapshot.confirmation or confirmation_label(cfg.signal_timeframe)

        evaluation = Evaluation(
            timestamp=candle_time,
            symbol=snapshot.symbol,
            timeframe=cfg.signal_timeframe,
            spread_points=snapshot.spread_points,
            mode=str(getattr(cfg, "mode", "STANDARD")).upper(),
            confirmation_timeframes=confirmation,
        )

        reason = self._validate(snapshot, cfg)
        if reason:
            evaluation.rejection_reason = f"data validation failed - {reason}"
            return evaluation

        params = cfg.indicators
        m5 = self._ensure_indicators(snapshot.m5, params)
        m15 = self._ensure_indicators(snapshot.m15, params) if snapshot.m15 is not None else None
        h1 = self._ensure_indicators(snapshot.h1, params) if snapshot.h1 is not None else None

        # -- 3. analysis engines ------------------------------------------- #
        components: Dict[str, ComponentScore] = {
            "trend": analyze_trend(m5, cfg),
            "htf": analyze_htf(m15, h1, cfg),
            "momentum": analyze_momentum(m5, cfg),
            "structure": analyze_structure(m5, cfg),
            "liquidity": analyze_liquidity(m5, cfg),
            "support_resistance": analyze_support_resistance(m5, cfg),
            "volume": analyze_volume(m5, cfg),
            "volatility": analyze_volatility(m5, cfg),
            "price_action": analyze_price_action(m5, cfg),
        }

        # -- 4/5. scoring and regime ---------------------------------------- #
        card = compute_scorecard(components, cfg)
        regime, _regime_details = classify_regime(m5, components, cfg)
        volatility_band, _ratio = classify_volatility(m5, cfg)
        session = detect_session(candle_time, cfg.sessions, cfg.session_priority)

        evaluation.card = card
        evaluation.regime = regime
        evaluation.volatility_band = volatility_band
        evaluation.session = session
        evaluation.features = self._collect_features(m5, components)

        direction = card.direction
        if direction == "NONE":
            evaluation.rejection_reason = "no directional edge"
            evaluation.threshold = adaptive_threshold(regime, volatility_band, "NEUTRAL", cfg) or 0.0
            return self._finish_rejected(evaluation, cfg)

        alignment = htf_alignment(components["htf"], direction)
        evaluation.htf_alignment = alignment

        filter_input = FilterInput(
            df=m5,
            card=card,
            direction=direction,
            regime=regime,
            volatility_band=volatility_band,
            session=session,
            spread_points=snapshot.spread_points,
            candle_time=candle_time,
            htf_alignment=alignment,
            timeframe_minutes=timeframe_minutes(cfg.signal_timeframe),
            gate=gate,
        )

        # -- 6. pre-target filters ------------------------------------------ #
        outcome = run_pre_target_filters(filter_input, cfg)
        evaluation.threshold = outcome.threshold
        if not outcome.passed:
            evaluation.rejection_reason = outcome.reason
            return self._finish_rejected(evaluation, cfg)

        # -- 7. targets and R:R --------------------------------------------- #
        targets, target_reason = build_targets(m5, cfg, direction)
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
        signal = Signal(
            signal_id=make_signal_id(snapshot.symbol, candle_time, direction),
            symbol=snapshot.symbol,
            timeframe=cfg.signal_timeframe,
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
            mode=evaluation.mode,
            confirmation_timeframes=confirmation,
            threshold_used=round(outcome.threshold, 2),
        )

        evaluation.decision = direction
        evaluation.signal = signal
        LOGGER.info(
            "%s %s %s @ %.2f | score %.1f | regime %s | RR2 %.2f | %s",
            "RESEARCH SIGNAL" if signal.is_research else "SIGNAL",
            signal.direction,
            signal.symbol,
            signal.entry,
            signal.confidence,
            signal.regime,
            signal.rr2,
            signal.reason_summary,
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
