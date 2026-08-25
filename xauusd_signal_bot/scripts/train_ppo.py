"""Train one PPO model, chronologically, and register it as EXPERIMENTAL.

    python scripts/train_ppo.py --symbol XAUUSD --timesteps 200000

Splits are chronological and never shuffled.  Anything inside a sealed holdout
is excluded at the data layer, so it cannot reach the model by accident.  A new
model is registered EXPERIMENTAL - promotion is a separate, evidenced decision.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from _common import load_dataset, models_root

from ai.environment.scalping_env import EnvConfig
from ai.evaluation.holdout import load_seal, training_frame
from ai.features.feature_pipeline import split_chronologically
from ai.model_registry import ModelRecord, ModelRegistry
from ai.ppo.train import PPOHyperParameters, detect_device, train_ppo
from config import load_config
from src.markets import DEFAULT_MARKET, MARKET_ORDER, market_argument


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="Train a PPO scalping agent")
    parser.add_argument("--symbol", type=market_argument, default=DEFAULT_MARKET,
                        help=f"{', '.join(MARKET_ORDER)} (default: %(default)s)")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--train-end", type=str, required=True,
                        help="UTC date where training stops and validation starts")
    parser.add_argument("--validation-end", type=str, required=True,
                        help="UTC date where validation stops and test starts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--holding-candles", type=int, default=None,
                        help="override MAX_HOLDING_CANDLES for this experiment")
    parser.add_argument("--advanced-actions", action="store_true",
                        help="MultiDiscrete: direction x SL bucket x TP bucket")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)

    view = config.for_market(args.symbol)
    frame, spec = load_dataset(config, args.symbol)

    seal = load_seal(models_root(config), args.symbol)
    if seal is not None:
        before = len(frame)
        frame = training_frame(frame, seal)
        print(f"Sealed holdout {seal.start} -> {seal.end}: "
              f"{before - len(frame):,} rows excluded from training")

    train, validation, test = split_chronologically(
        frame, [args.train_end, args.validation_end]
    )
    if train.empty or validation.empty:
        print("ERROR: the chronological split left an empty train or validation set.")
        return 2

    print(f"train {len(train):,}  validation {len(validation):,}  test {len(test):,}")
    print(f"device: {detect_device()}")

    env_config = EnvConfig.from_market_config(view)
    if args.holding_candles:
        env_config.max_holding_candles = args.holding_candles

    registry = ModelRegistry(models_root(config))
    version = registry.next_version(args.symbol)
    output_dir = registry.model_dir(args.symbol, version)

    hyper = PPOHyperParameters(
        learning_rate=args.learning_rate, n_steps=args.n_steps,
        batch_size=args.batch_size, ent_coef=args.ent_coef,
        net_arch=tuple(args.net_arch), seed=args.seed,
    )
    result = train_ppo(
        train, validation, spec, env_config, output_dir, hyper=hyper,
        total_timesteps=args.timesteps,
        eval_every=max(args.timesteps // 5, 1),
        advanced_actions=args.advanced_actions,
        symbol=args.symbol, version=version, verbose=args.verbose,
    )

    times = frame["time"]
    registry.register(ModelRecord(
        version=version, symbol=args.symbol, path=str(output_dir),
        train_start=str(times.iloc[0]), train_end=args.train_end,
        validation_start=args.train_end, validation_end=args.validation_end,
        test_start=args.validation_end, test_end=str(times.iloc[-1]),
        feature_fingerprint=spec.fingerprint(), feature_count=spec.size,
        dataset_rows=int(len(frame)),
        hyperparameters=hyper.to_dict(),
        env_config=result.env_config, seed=args.seed,
        total_timesteps=args.timesteps,
        validation_metrics=result.validation_metrics,
        notes="EXPERIMENTAL until it clears the promotion gates on unseen data.",
    ))

    print()
    print(f"Model     : {version}  ({output_dir})")
    print(f"Trained in: {result.duration_seconds:.0f}s on {result.device}")
    print("Validation:", json.dumps(
        {k: result.validation_metrics.get(k) for k in
         ("trades", "net_r", "average_net_r", "win_rate", "action_mix")}, default=str))
    print()
    print("Registered EXPERIMENTAL. It will NOT be used live until promoted -")
    print("run scripts/walk_forward.py, then final_holdout_eval.py.")
    if int(result.validation_metrics.get("trades", 0) or 0) == 0:
        print()
        print("NOTE: the agent took no trades on validation. With realistic")
        print("costs that is often the correct answer, not a bug - compare")
        print("against a zero-cost control before concluding anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
