"""Sliding walk-forward validation: does it keep working, not did it work once.

    python scripts/walk_forward.py --symbol XAUUSD --train-days 180 \
        --validation-days 30 --test-days 30 --timesteps 50000

Each fold trains on data strictly before its validation window, which is
strictly before its test window.  No fold sees a later fold's data.  A sealed
holdout is excluded entirely.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from _common import load_dataset, models_root, reports_root

from ai.environment.scalping_env import EnvConfig
from ai.evaluation.holdout import load_seal, training_frame
from ai.evaluation.walk_forward import aggregate, run_walk_forward
from ai.ppo.train import PPOHyperParameters
from config import load_config
from src.markets import DEFAULT_MARKET, MARKET_ORDER, market_argument


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="PPO walk-forward validation")
    parser.add_argument("--symbol", type=market_argument, default=DEFAULT_MARKET,
                        help=f"{', '.join(MARKET_ORDER)} (default: %(default)s)")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--max-folds", type=int, default=0, help="0 = as many as fit")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holding-candles", type=int, default=None)
    parser.add_argument("--advanced-actions", action="store_true")
    args = parser.parse_args(argv)

    view = config.for_market(args.symbol)
    frame, spec = load_dataset(config, args.symbol)

    seal = load_seal(models_root(config), args.symbol)
    if seal is not None:
        frame = training_frame(frame, seal)
        print(f"Sealed holdout excluded: {seal.start} -> {seal.end}")

    env_config = EnvConfig.from_market_config(view)
    if args.holding_candles:
        env_config.max_holding_candles = args.holding_candles

    output_dir = reports_root(config) / args.symbol / "walk_forward"
    print(f"Running walk-forward on {len(frame):,} rows ...")
    results = run_walk_forward(
        frame, spec, env_config, output_dir,
        hyper=PPOHyperParameters(seed=args.seed),
        total_timesteps=args.timesteps,
        train_days=args.train_days, validation_days=args.validation_days,
        test_days=args.test_days, max_folds=args.max_folds,
        symbol=args.symbol, advanced_actions=args.advanced_actions,
    )
    if not results:
        print("No folds fit in the available data. Download more history, or "
              "shorten --train-days.")
        return 1

    print()
    header = f"{'fold':<6}{'test period':<26}{'trades':>8}{'netR':>10}{'avgR':>9}{'maxDD':>9}"
    print(header)
    print("-" * len(header))
    for result in results:
        if result.skipped:
            print(f"{result.fold:<6}{result.test_period:<26}  skipped: {result.skipped}")
            continue
        m = result.test_metrics
        print(f"{result.fold:<6}{result.test_period:<26}"
              f"{m.get('trades', 0):>8}{m.get('net_r', 0.0):>10.2f}"
              f"{m.get('average_net_r', 0.0):>9.3f}{m.get('max_drawdown_r', 0.0):>9.2f}")

    summary = aggregate(results)
    print()
    print("OUT-OF-SAMPLE AGGREGATE (test windows only):")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Written to: {output_dir}")
    print()
    print("Read the SPREAD across folds, not the total. One good fold beside")
    print("several poor ones is noise, not an edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
