"""The single seam between the rule-based bot and the PPO research stack.

WHY A BRIDGE AND NOT DIRECT IMPORTS
-----------------------------------
``main.py`` must keep working when ``torch`` is not installed, when no model has
been trained, and when a checkpoint is corrupt.  Every import of ``ai/`` happens
inside this file, inside a try block, so all three degrade to RULE_ONLY instead
of to a traceback.

The dependency arrow points one way: ``ai/`` never imports the live bot's
runtime.  Deleting ``ai/`` entirely leaves a working scalper.

FAIL-SAFE
---------
Every failure here answers "do nothing".  A missing model, a bad observation, an
exception in the policy - all produce HOLD, never a trade.  The failure mode
being prevented is placing an order *because* the AI broke.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .logger import get_logger

LOGGER = get_logger("ppo_bridge")

#: Set false the moment an ``ai`` import fails, so we try once and not per candle.
_AI_AVAILABLE: Optional[bool] = None


def ai_available() -> bool:
    """Whether the RL stack can be imported at all.  Checked once."""
    global _AI_AVAILABLE
    if _AI_AVAILABLE is None:
        try:
            import ai.strategy_mode as _probe

            _AI_AVAILABLE = _probe is not None
        except Exception as exc:  # noqa: BLE001
            LOGGER.info("PPO stack unavailable (%s) - RULE_ONLY", exc)
            _AI_AVAILABLE = False
    return bool(_AI_AVAILABLE)


class PPOBridge:
    """Owns PPO's lifecycle for every market, and keeps it out of the loop."""

    def __init__(self, config) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._policies: Dict[str, Any] = {}
        self._recorders: Dict[str, Any] = {}
        self._last: Dict[str, Any] = {}
        self._mode = None
        self._mode_name = "RULE_ONLY"

        if not ai_available():
            return
        from ai.strategy_mode import load_strategy_mode

        self._mode = load_strategy_mode()
        self._mode_name = self._mode.value
        if self._mode.uses_ppo:
            LOGGER.info("Strategy mode: %s", self._mode_name)

    # -- state ------------------------------------------------------------- #
    @property
    def active(self) -> bool:
        """True only when PPO should be consulted at all."""
        return bool(self._mode is not None and self._mode.uses_ppo)

    def mode_name(self) -> str:
        return self._mode_name

    def set_mode(self, mode: str) -> str:
        """Switch strategy mode.  Returns the mode ACTUALLY in effect.

        A request that cannot be honoured - RL stack missing, no model for the
        active market - returns ``RULE_ONLY`` rather than pretending to succeed.
        """
        if not ai_available():
            LOGGER.warning("PPO requested but the RL stack is not installed")
            self._mode_name = "RULE_ONLY"
            return self._mode_name

        from ai.strategy_mode import StrategyMode, parse_strategy_mode

        wanted = parse_strategy_mode(mode)
        if wanted.uses_ppo:
            symbol = getattr(self.config, "symbol", "")
            if self._policy_for(symbol) is None:
                LOGGER.warning(
                    "%s requested but no PPO model could be loaded for %s - "
                    "staying on the rule engine", wanted.value, symbol,
                )
                self._mode = StrategyMode.RULE_ONLY
                self._mode_name = self._mode.value
                return self._mode_name

        self._mode = wanted
        self._mode_name = wanted.value
        LOGGER.info("Strategy mode -> %s", self._mode_name)
        return self._mode_name

    # -- model loading -------------------------------------------------------- #
    def _model_root(self, symbol: str) -> Path:
        return Path(self.config.data_dir).parent / "models"

    def _policy_for(self, symbol: str):
        """The policy for one market, loaded once.  ``None`` when unavailable."""
        with self._lock:
            if symbol in self._policies:
                return self._policies[symbol]
        policy = None
        try:
            from ai.model_registry import (
                STATUS_PRODUCTION, STATUS_SHADOW, STATUS_VALIDATED, ModelRegistry,
            )
            from ai.ppo.inference import PPOPolicy

            registry = ModelRegistry(self._model_root(symbol))
            # Prefer the most trusted model available, and never fall back to
            # EXPERIMENTAL: an untested model must not reach the live loop.
            record = None
            for status in (STATUS_PRODUCTION, STATUS_SHADOW, STATUS_VALIDATED):
                record = registry.latest(symbol, status)
                if record is not None:
                    break
            if record is None:
                LOGGER.info(
                    "No VALIDATED or better PPO model registered for %s", symbol
                )
            else:
                policy = PPOPolicy.load(
                    Path(record.path), symbol=symbol, version=record.version
                )
                if policy is not None:
                    policy.record = record  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Could not load a PPO policy for %s: %s", symbol, exc)
            policy = None

        with self._lock:
            self._policies[symbol] = policy
        return policy

    def _recorder_for(self, symbol: str, cfg):
        with self._lock:
            if symbol in self._recorders:
                return self._recorders[symbol]
        recorder = None
        try:
            from ai.environment.scalping_env import EnvConfig
            from ai.shadow import ShadowRecorder

            policy = self._policy_for(symbol)
            recorder = ShadowRecorder(
                cfg,
                model_version=getattr(policy, "version", "") if policy else "",
                env_config=EnvConfig.from_market_config(cfg),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Could not create the shadow recorder for %s: %s", symbol, exc)
        with self._lock:
            self._recorders[symbol] = recorder
        return recorder

    # -- the per-candle hook --------------------------------------------------- #
    def observe(self, symbol: str, cfg, snapshot, rule_signal=None):
        """Show PPO one closed candle.

        Returns a ``Signal`` ONLY in ``PPO_DEMO`` mode and only when the model
        asks to trade.  In every other case it returns ``None`` and has merely
        recorded what it would have done.
        """
        if not self.active:
            return None
        policy = self._policy_for(symbol)
        if policy is None:
            return None

        try:
            frame = self._features_for(cfg, snapshot)
            if frame is None or frame.empty:
                return None

            recorder = self._recorder_for(symbol, cfg)
            candle = self._candle_from(frame, snapshot)
            if recorder is not None:
                recorder.observe_candle(candle)

            decision = policy.decide(frame)
            with self._lock:
                self._last[symbol] = decision

            if recorder is not None:
                recorder.record_decision(decision, candle, self._mode_name, rule_signal)
                if not self._mode.ppo_decides:
                    recorder.maybe_open(decision, candle)
                    return None

            if not decision.ok or not decision.is_trade:
                return None
            return self._signal_from(decision, cfg, snapshot, candle)
        except Exception as exc:  # noqa: BLE001 - always fail to HOLD
            LOGGER.exception("[%s] PPO observation failed: %s", symbol, exc)
            return None

    def _features_for(self, cfg, snapshot):
        """Build the causal feature frame for the snapshot's closed candles."""
        from ai.features.feature_pipeline import build_dataset

        candles = snapshot.signal_df
        if candles is None or len(candles) < 120:
            return None
        policy = self._policy_for(cfg.symbol)
        spec = getattr(policy, "spec", None)
        frame, _spec = build_dataset(candles, spec=spec, drop_warmup=False)
        return frame

    @staticmethod
    def _candle_from(frame, snapshot) -> Dict[str, Any]:
        row = frame.iloc[-1]
        return {
            "time": str(row.get("time", "")),
            "open": float(row.get("open", 0.0)),
            "high": float(row.get("high", 0.0)),
            "low": float(row.get("low", 0.0)),
            "close": float(row.get("close", 0.0)),
            "spread": float(getattr(snapshot, "spread_points", 0.0) or 0.0),
            "atr": float(row.get("f_atr", 0.0) or 0.0),
        }

    def _signal_from(self, decision, cfg, snapshot, candle):
        """Turn a PPO action into a ``Signal`` the existing pipeline accepts.

        Geometry comes from the SAME ATR-relative rules the rule engine uses, so
        a PPO trade and a rule trade are shaped alike and their results are
        comparable.  The signal then enters ``demo_execution`` at exactly the
        point a rule signal does - no privileged route, no skipped gate.
        """
        from ai.environment.scalping_env import BUY
        from src.signal_engine import Signal
        from src.utils import now_utc

        atr = float(candle.get("atr", 0.0) or 0.0)
        close = float(candle.get("close", 0.0) or 0.0)
        if atr <= 0 or close <= 0:
            return None

        direction = "BUY" if decision.action == BUY else "SELL"
        sign = 1.0 if direction == "BUY" else -1.0
        risk = float(getattr(cfg, "sl_atr_multiplier", 0.70)) * atr
        multiples = tuple(getattr(cfg, "tp_atr_multiples", (0.45, 1.00, 1.70)))
        digits = int(getattr(cfg, "digits", 2))
        when = snapshot.candle_time if hasattr(snapshot, "candle_time") else now_utc()

        return Signal(
            signal_id=f"ppo-{cfg.symbol}-{direction}-{when:%Y%m%d%H%M%S}",
            symbol=cfg.symbol, timeframe="M1", direction=direction, timestamp=when,
            entry=round(close, digits),
            stop_loss=round(close - sign * risk, digits),
            tp1=round(close + sign * multiples[0] * atr, digits),
            tp2=round(close + sign * multiples[1] * atr, digits),
            tp3=round(close + sign * multiples[2] * atr, digits),
            confidence=round(100.0 * decision.confidence, 1),
            bullish_score=round(100.0 * decision.probabilities.get("BUY", 0.0), 1),
            bearish_score=round(100.0 * decision.probabilities.get("SELL", 0.0), 1),
            regime="PPO", risk_reward=multiples[1] / max(
                float(getattr(cfg, "sl_atr_multiplier", 0.70)), 1e-9),
            session=str(getattr(snapshot, "session", "")),
            reason_summary=f"PPO {decision.action_name} p={decision.confidence:.2f}",
            mode="PPO",
            threshold_used=0.0,
            spread_points=float(candle.get("spread", 0.0) or 0.0),
            estimated_slippage=float(getattr(cfg, "slippage_points_entry", 0.0))
            + float(getattr(cfg, "slippage_points_exit", 0.0)),
            atr=round(atr, 5),
        )

    # -- reporting -------------------------------------------------------------- #
    def state(self, symbol: str) -> Dict[str, Any]:
        """Everything the Telegram PPO panel needs.  Never raises."""
        state: Dict[str, Any] = {"mode": self._mode_name, "loaded": False}
        if not self.active and not self._policies:
            return state
        try:
            policy = self._policy_for(symbol)
            if policy is None:
                return state
            state.update(policy.health())
            state["loaded"] = True

            record = getattr(policy, "record", None)
            if record is not None:
                state["model_status"] = record.status
                state["train_period"] = f"{record.train_start} -> {record.train_end}"
                state["validation_period"] = (
                    f"{record.validation_start} -> {record.validation_end}"
                )
                state["test_period"] = f"{record.test_start} -> {record.test_end}"

            decision = self._last.get(symbol)
            if decision is not None:
                state["last_action"] = decision.action_name
                state["probabilities"] = decision.probabilities

            recorder = self._recorders.get(symbol)
            if recorder is not None:
                summary = recorder.summary()
                for key in ("trades", "net_r", "average_net_r", "win_rate",
                            "profit_factor", "max_drawdown_r", "average_holding"):
                    state[key] = summary.get(key, 0)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("PPO state unavailable: %s", exc)
        return state
