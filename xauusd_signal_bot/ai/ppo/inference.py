"""Serving a trained PPO policy in the live loop, fail-safe by construction.

THE ONE RULE
------------
**Any problem means HOLD.**  A missing model, an unreadable checkpoint, a
feature-contract mismatch, a NaN observation, an out-of-range action, an
exception inside the policy - every one of them produces "do nothing", never
"do something".

That is not defensive padding.  The failure mode this prevents is the system
placing a trade *because* the AI broke, which is the single worst behaviour a
system like this can have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ai.environment.scalping_env import ACTION_NAMES, BUY, HOLD, SELL
from ai.features.feature_pipeline import FeatureSpec
from src.logger import get_logger

LOGGER = get_logger("ai.inference")


@dataclass
class PPODecision:
    """One inference, whether or not it succeeded."""

    action: int = HOLD
    action_name: str = "HOLD"
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    ok: bool = True
    #: Populated when inference failed and HOLD was substituted.
    error: str = ""

    @property
    def is_trade(self) -> bool:
        return self.action in (BUY, SELL)


def _hold(error: str) -> PPODecision:
    """The fail-safe answer."""
    return PPODecision(
        action=HOLD, action_name="HOLD",
        probabilities={"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0},
        confidence=1.0, ok=False, error=error,
    )


class PPOPolicy:
    """A loaded model plus the contract it was trained against."""

    def __init__(self, model, spec: FeatureSpec, version: str = "",
                 symbol: str = "", position_features: int = 4) -> None:
        self.model = model
        self.spec = spec
        self.version = version
        self.symbol = symbol
        self.position_features = position_features
        self.failures = 0
        self.decisions = 0

    # -- loading ------------------------------------------------------------ #
    @classmethod
    def load(cls, model_dir, symbol: str = "", version: str = "",
             checkpoint: str = "best_model") -> Optional["PPOPolicy"]:
        """Load a policy, or return ``None``.

        ``None`` rather than an exception: a missing model at start-up must not
        stop the rule engine, which is the whole point of keeping it.
        """
        from pathlib import Path

        directory = Path(model_dir)
        weights = directory / f"{checkpoint}.zip"
        spec_path = directory / "feature_spec.json"
        if not weights.exists():
            LOGGER.warning("No PPO checkpoint at %s - staying on the rule engine", weights)
            return None
        if not spec_path.exists():
            LOGGER.error(
                "%s has no feature_spec.json. Refusing to load: without the "
                "feature contract the model could silently be fed the wrong "
                "columns.", directory,
            )
            return None
        try:
            from stable_baselines3 import PPO

            spec = FeatureSpec.load(spec_path)
            model = PPO.load(weights, device="cpu")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Could not load the PPO model at %s: %s", directory, exc)
            return None
        LOGGER.info(
            "PPO %s loaded for %s (%d features, fingerprint %s)",
            version or directory.name, symbol, spec.size, spec.fingerprint(),
        )
        return cls(model, spec, version=version or directory.name, symbol=symbol)

    # -- inference ---------------------------------------------------------- #
    def observation_from(
        self, frame: pd.DataFrame, position: Optional[Dict[str, float]] = None
    ) -> Optional[np.ndarray]:
        """Build the observation for the LAST row of ``frame``.

        Returns ``None`` when the contract cannot be met exactly - a partially
        satisfied feature set is worse than none, because the model would still
        produce a confident answer from the wrong inputs.
        """
        if frame is None or frame.empty:
            return None
        missing = [column for column in self.spec.columns if column not in frame.columns]
        if missing:
            LOGGER.error(
                "Observation is missing %d expected feature(s) (e.g. %s) - holding",
                len(missing), ", ".join(missing[:3]),
            )
            return None

        row = frame.iloc[-1]
        values = row[self.spec.columns].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            LOGGER.warning("Observation contains non-finite values - holding")
            return None
        values = np.clip(values, -self.spec.clip, self.spec.clip)

        if self.position_features:
            if position is None:
                extra = np.zeros(self.position_features, dtype=np.float64)
            else:
                extra = np.array([
                    float(position.get("direction", 0.0)),
                    float(np.clip(position.get("unrealised_r", 0.0), -5.0, 5.0)),
                    float(np.clip(position.get("age", 0.0), 0.0, 2.0)),
                    float(position.get("remaining", 0.0)),
                ], dtype=np.float64)
            values = np.concatenate([values, extra])
        return values.astype(np.float32)

    def decide(
        self, frame: pd.DataFrame, position: Optional[Dict[str, float]] = None,
        deterministic: bool = True,
    ) -> PPODecision:
        """Infer one action.  Never raises; every failure becomes HOLD."""
        self.decisions += 1
        try:
            observation = self.observation_from(frame, position)
            if observation is None:
                self.failures += 1
                return _hold("observation could not be built")

            expected = self.model.observation_space.shape[0]
            if observation.shape[0] != expected:
                self.failures += 1
                return _hold(
                    f"observation is {observation.shape[0]} wide, model expects {expected}"
                )

            probabilities = self._probabilities(observation)
            action, _state = self.model.predict(observation, deterministic=deterministic)
            action = int(np.asarray(action).ravel()[0])
            if action not in (HOLD, BUY, SELL):
                self.failures += 1
                return _hold(f"model returned an out-of-range action: {action}")

            name = ACTION_NAMES[action]
            return PPODecision(
                action=action, action_name=name,
                probabilities=probabilities,
                confidence=probabilities.get(name, 0.0),
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001 - inference must never break the loop
            self.failures += 1
            LOGGER.exception("PPO inference failed: %s", exc)
            return _hold(f"inference raised: {exc}")

    def _probabilities(self, observation: np.ndarray) -> Dict[str, float]:
        """Action probabilities, for the Telegram panel and the shadow log.

        Best-effort: if the policy shape does not expose them, an empty dict is
        returned rather than a fabricated distribution.
        """
        try:
            import torch

            tensor, _ = self.model.policy.obs_to_tensor(observation)
            with torch.no_grad():
                distribution = self.model.policy.get_distribution(tensor)
                probs = distribution.distribution.probs.cpu().numpy().ravel()
            if probs.size < 3:
                return {}
            return {
                "HOLD": round(float(probs[HOLD]), 4),
                "BUY": round(float(probs[BUY]), 4),
                "SELL": round(float(probs[SELL]), 4),
            }
        except Exception:  # noqa: BLE001
            return {}

    def health(self) -> Dict[str, Any]:
        """Decision and failure counts, for the status panel."""
        rate = (self.failures / self.decisions * 100.0) if self.decisions else 0.0
        return {
            "version": self.version,
            "symbol": self.symbol,
            "decisions": self.decisions,
            "failures": self.failures,
            "failure_rate": round(rate, 2),
            "features": self.spec.size,
            "fingerprint": self.spec.fingerprint(),
        }
