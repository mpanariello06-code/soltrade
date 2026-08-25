"""PPO training: small MLP, CPU-friendly, reproducible, checkpointed.

DESIGN CHOICES AND WHY
----------------------
* **Small network (64x64 MLP).**  The first question is whether PPO can learn
  anything on this problem at all.  A Transformer would answer that question far
  more slowly and no more clearly, and would not train on the machine this has
  to run on.  The architecture is configurable for when there is evidence a
  bigger one is needed.
* **Chronological data only.**  No shuffling anywhere.  Shuffled market data
  lets a model learn from Thursday to trade Monday, and every metric that
  follows is fiction.
* **Validation selects the model; test is looked at once.**  Selecting on test
  is how you produce a backtest that never reproduces.
* **Everything is recorded** - seed, hyperparameters, feature fingerprint, date
  ranges - so a result can be reproduced or discarded on evidence.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ai.environment.scalping_env import EnvConfig, ScalpingEnv
from ai.features.feature_pipeline import FeatureSpec


def _require_sb3():
    """Import Stable-Baselines3 lazily with an actionable error."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        return PPO, Monitor, DummyVecEnv
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PPO training needs stable-baselines3 and torch:\n"
            "  pip install stable-baselines3 torch\n"
            "The rule-based bot does not need them and runs without them."
        ) from exc


def detect_device() -> str:
    """``cuda`` when a GPU is genuinely usable, else ``cpu``."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class PPOHyperParameters:
    """Every PPO knob, in one place, with CPU-sane defaults."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    #: Entropy bonus.  Slightly above the SB3 default: with HOLD always
    #: available, a scalping agent collapses to "never trade" very easily, and
    #: an agent that never trades has learned nothing.
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    #: Two hidden layers of 64. Small on purpose - see the module docstring.
    net_arch: Tuple[int, ...] = (64, 64)
    seed: int = 42
    device: str = "auto"

    def policy_kwargs(self) -> Dict[str, Any]:
        return {"net_arch": list(self.net_arch)}

    def sb3_kwargs(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["net_arch"] = list(self.net_arch)
        return data


@dataclass
class TrainingResult:
    """What one training run produced, and what it was produced from."""

    version: str = ""
    symbol: str = ""
    model_path: str = ""
    total_timesteps: int = 0
    train_rows: int = 0
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    env_config: Dict[str, Any] = field(default_factory=dict)
    feature_fingerprint: str = ""
    device: str = "cpu"
    created_at: str = ""
    duration_seconds: float = 0.0


def make_env(
    frame: pd.DataFrame, spec: FeatureSpec, env_config: EnvConfig,
    seed: int = 0, advanced_actions: bool = False, monitor: bool = True,
):
    """One vectorised environment over ``frame``."""
    PPO, Monitor, DummyVecEnv = _require_sb3()

    def build():
        env = ScalpingEnv(
            frame, spec.columns, env_config,
            seed=seed, advanced_actions=advanced_actions,
        )
        return Monitor(env) if monitor else env

    return DummyVecEnv([build])


def evaluate_policy_on(
    model, frame: pd.DataFrame, spec: FeatureSpec, env_config: EnvConfig,
    deterministic: bool = True, advanced_actions: bool = False,
) -> Dict[str, Any]:
    """Run a trained policy once, end to end, over ``frame``.

    Deterministic by default and with ``random_start`` disabled, so the whole
    period is evaluated exactly once - a sampled policy over random windows
    would give a different answer each call and could not be compared.
    """
    config = EnvConfig(**{**asdict(env_config), "random_start": False, "episode_length": 0})
    env = ScalpingEnv(frame, spec.columns, config, seed=0, advanced_actions=advanced_actions)
    observation, _ = env.reset()
    total_reward = 0.0
    actions: List[int] = []

    while True:
        action, _state = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        actions.append(int(np.asarray(action).ravel()[0]))
        if terminated or truncated:
            break

    summary = env.summary()
    summary["total_reward"] = round(total_reward, 4)
    counts = np.bincount(np.asarray(actions, dtype=int), minlength=3)
    total = max(int(counts.sum()), 1)
    summary["action_mix"] = {
        "HOLD": round(100.0 * counts[0] / total, 2),
        "BUY": round(100.0 * counts[1] / total, 2),
        "SELL": round(100.0 * counts[2] / total, 2),
    }
    summary["trades_frame"] = env.trades_frame()
    return summary


def train_ppo(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    spec: FeatureSpec,
    env_config: EnvConfig,
    output_dir: Path,
    hyper: Optional[PPOHyperParameters] = None,
    total_timesteps: int = 50_000,
    eval_every: int = 10_000,
    advanced_actions: bool = False,
    symbol: str = "",
    version: str = "",
    verbose: int = 0,
) -> TrainingResult:
    """Train one PPO model and keep the best-on-validation checkpoint.

    Checkpoints written (nothing is ever lost):

    * ``best_model.zip``  - highest validation NET R seen
    * ``latest.zip``      - most recent, for resuming
    * ``final_model.zip`` - the policy at the end of training
    """
    PPO, _Monitor, _DummyVecEnv = _require_sb3()
    hyper = hyper or PPOHyperParameters()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    device = detect_device() if hyper.device == "auto" else hyper.device
    vec_env = make_env(train_frame, spec, env_config, seed=hyper.seed,
                       advanced_actions=advanced_actions)

    model = PPO(
        "MlpPolicy", vec_env,
        policy_kwargs=hyper.policy_kwargs(),
        device=device, verbose=verbose,
        **hyper.sb3_kwargs(),
    )

    best_net_r = -float("inf")
    best_metrics: Dict[str, Any] = {}
    remaining = int(total_timesteps)
    chunk = max(int(eval_every), 1)
    steps_done = 0

    while remaining > 0:
        this_chunk = min(chunk, remaining)
        model.learn(total_timesteps=this_chunk, reset_num_timesteps=False,
                    progress_bar=False)
        remaining -= this_chunk
        steps_done += this_chunk

        model.save(output_dir / "latest")
        if not validation_frame.empty:
            metrics = evaluate_policy_on(
                model, validation_frame, spec, env_config,
                advanced_actions=advanced_actions,
            )
            metrics.pop("trades_frame", None)
            net_r = float(metrics.get("net_r", 0.0))
            # Selection is on VALIDATION only.  Test is never consulted here.
            if net_r > best_net_r:
                best_net_r = net_r
                best_metrics = dict(metrics)
                best_metrics["timesteps"] = steps_done
                model.save(output_dir / "best_model")

    model.save(output_dir / "final_model")
    if best_net_r == -float("inf"):
        # No validation data: the final model is the only candidate there is.
        model.save(output_dir / "best_model")
        best_metrics = {}

    train_metrics = evaluate_policy_on(
        model, train_frame, spec, env_config, advanced_actions=advanced_actions
    )
    train_metrics.pop("trades_frame", None)

    result = TrainingResult(
        version=version, symbol=symbol,
        model_path=str(output_dir / "best_model.zip"),
        total_timesteps=int(total_timesteps),
        train_rows=int(len(train_frame)),
        validation_metrics=best_metrics,
        train_metrics=train_metrics,
        hyperparameters=hyper.to_dict(),
        env_config=asdict(env_config),
        feature_fingerprint=spec.fingerprint(),
        device=device,
        created_at=started.isoformat(),
        duration_seconds=round(
            (datetime.now(timezone.utc) - started).total_seconds(), 1
        ),
    )

    spec.save(output_dir / "feature_spec.json")
    (output_dir / "training_result.json").write_text(
        json.dumps(asdict(result), indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps({
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
        }, indent=2),
        encoding="utf-8",
    )
    return result
