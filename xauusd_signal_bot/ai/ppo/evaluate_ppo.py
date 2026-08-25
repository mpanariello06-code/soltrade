"""Evaluate a trained policy over a period, and write the report artefacts.

Separated from ``train.py`` so a model can be re-measured without retraining -
on a different period, with different cost assumptions, or simply to regenerate
a report.  Training and measuring are different activities and the second must
not require the first.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ai.environment.scalping_env import EnvConfig, ScalpingEnv
from ai.evaluation.metrics import summarise_trades
from ai.features.feature_pipeline import FeatureSpec


def load_policy(model_dir, checkpoint: str = "best_model", device: str = "cpu"):
    """Load a checkpoint plus its feature contract.

    Raises rather than returning ``None``: an evaluation that silently measured
    nothing would be worse than one that failed.
    """
    from stable_baselines3 import PPO

    directory = Path(model_dir)
    weights = directory / f"{checkpoint}.zip"
    spec_path = directory / "feature_spec.json"
    if not weights.exists():
        raise FileNotFoundError(f"no checkpoint at {weights}")
    if not spec_path.exists():
        raise FileNotFoundError(
            f"{directory} has no feature_spec.json - refusing to evaluate a model "
            "whose observation contract is unknown"
        )
    return PPO.load(weights, device=device), FeatureSpec.load(spec_path)


def evaluate(
    model,
    frame: pd.DataFrame,
    spec: FeatureSpec,
    env_config: EnvConfig,
    deterministic: bool = True,
    advanced_actions: bool = False,
    label: str = "",
) -> Dict[str, Any]:
    """Run a policy once over ``frame`` and summarise what it did.

    ``random_start`` is disabled and the episode is unbounded, so the whole
    period is covered exactly once - a sampled policy over random windows gives
    a different answer each call and cannot be compared with anything.
    """
    config = EnvConfig(**{**asdict(env_config), "random_start": False,
                          "episode_length": 0})
    env = ScalpingEnv(frame, spec.columns, config, seed=0,
                      advanced_actions=advanced_actions)

    observation, _ = env.reset()
    total_reward = 0.0
    actions: list = []
    while True:
        action, _state = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        actions.append(int(pd.array([action]).to_numpy().ravel()[0])
                       if hasattr(action, "shape") and action.shape else int(action))
        if terminated or truncated:
            break

    trades = env.trades_frame()
    summary = summarise_trades(trades, label=label or "evaluation")
    summary["total_reward"] = round(total_reward, 4)
    summary["candles"] = int(len(frame))
    if actions:
        counts = pd.Series(actions).value_counts(normalize=True) * 100.0
        summary["action_mix"] = {
            "HOLD": round(float(counts.get(0, 0.0)), 2),
            "BUY": round(float(counts.get(1, 0.0)), 2),
            "SELL": round(float(counts.get(2, 0.0)), 2),
        }
    summary["trades_frame"] = trades
    return summary


def evaluate_checkpoint(
    model_dir,
    frame: pd.DataFrame,
    env_config: EnvConfig,
    checkpoint: str = "best_model",
    label: str = "",
    advanced_actions: bool = False,
) -> Dict[str, Any]:
    """Load and evaluate in one call, using the model's own feature contract."""
    model, spec = load_policy(model_dir, checkpoint)
    return evaluate(model, frame, spec, env_config, label=label,
                    advanced_actions=advanced_actions)
