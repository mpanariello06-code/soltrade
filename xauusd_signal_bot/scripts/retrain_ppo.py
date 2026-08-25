"""Train a challenger and promote it ONLY if it beats the gates on unseen data.

    python scripts/retrain_ppo.py --symbol XAUUSD --timesteps 200000

A finished training run is not evidence of anything.  The candidate is
registered EXPERIMENTAL, walk-forward evaluated, and compared against fixed
gates AND against the current production model.  Failing either leaves the
current model exactly where it was.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from _common import load_dataset, models_root, reports_root

from ai.environment.scalping_env import EnvConfig
from ai.evaluation.holdout import load_seal, training_frame
from ai.evaluation.walk_forward import aggregate, run_walk_forward
from ai.model_registry import (
    STATUS_PRODUCTION, STATUS_VALIDATED, ModelRecord, ModelRegistry, PromotionGates,
)
from ai.ppo.train import PPOHyperParameters
from config import load_config
from src.markets import DEFAULT_MARKET, MARKET_ORDER, market_argument


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="Retrain and conditionally promote")
    parser.add_argument("--symbol", type=market_argument, default=DEFAULT_MARKET,
                        help=f"{', '.join(MARKET_ORDER)} (default: %(default)s)")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--min-net-r", type=float, default=0.0)
    parser.add_argument("--max-drawdown-r", type=float, default=25.0)
    parser.add_argument("--min-profit-factor", type=float, default=1.0)
    parser.add_argument("--promote", action="store_true",
                        help="mark PRODUCTION if it passes (default: VALIDATED only)")
    args = parser.parse_args(argv)

    view = config.for_market(args.symbol)
    registry = ModelRegistry(models_root(config))
    incumbent = registry.production(args.symbol)

    frame, spec = load_dataset(config, args.symbol)
    seal = load_seal(models_root(config), args.symbol)
    if seal is not None:
        frame = training_frame(frame, seal)
        print(f"Sealed holdout excluded: {seal.start} -> {seal.end}")

    version = registry.next_version(args.symbol)
    print(f"Candidate: {version}  (incumbent: "
          f"{incumbent.version if incumbent else 'none'})")

    results = run_walk_forward(
        frame, spec, EnvConfig.from_market_config(view),
        reports_root(config) / args.symbol / f"retrain_{version}",
        hyper=PPOHyperParameters(seed=args.seed),
        total_timesteps=args.timesteps,
        train_days=args.train_days, validation_days=args.validation_days,
        test_days=args.test_days, max_folds=args.max_folds,
        symbol=args.symbol,
    )
    summary = aggregate(results)
    print()
    print("Candidate out-of-sample:", json.dumps(summary, indent=2))

    metrics = {
        "trades": summary.get("total_test_trades", 0),
        "net_r": summary.get("total_test_net_r", 0.0),
        "average_net_r": summary.get("mean_average_net_r", 0.0),
        "max_drawdown_r": max(
            (float(r.test_metrics.get("max_drawdown_r", 0.0) or 0.0) for r in results
             if r.test_metrics), default=0.0,
        ),
        "profit_factor": min(
            (float(r.test_metrics.get("profit_factor", 0.0) or 0.0) for r in results
             if r.test_metrics), default=0.0,
        ),
        "average_holding": summary.get("mean_average_net_r", 0.0),
    }

    last = next((r for r in reversed(results) if r.model_path), None)
    registry.register(ModelRecord(
        version=version, symbol=args.symbol,
        path=str(last.model_path).rsplit("/", 1)[0] if last else "",
        feature_fingerprint=spec.fingerprint(), feature_count=spec.size,
        dataset_rows=int(len(frame)), seed=args.seed,
        total_timesteps=args.timesteps, test_metrics=metrics,
        notes="candidate from retrain_ppo.py",
    ))

    gates = PromotionGates(
        min_trades=args.min_trades, min_net_r=args.min_net_r,
        max_drawdown_r=args.max_drawdown_r, min_profit_factor=args.min_profit_factor,
    )
    verdict = gates.evaluate(metrics)
    print()
    print("PROMOTION GATES:")
    for name, check in verdict["checks"].items():
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"  {mark}  {name:<22} value {check['value']}  limit {check['limit']}")

    if not verdict["passed"]:
        print()
        print(f"REJECTED: {', '.join(verdict['failures'])}")
        print(f"The current model ({incumbent.version if incumbent else 'none'}) "
              "is unchanged.")
        return 1

    # Beating fixed gates is necessary but not sufficient: it must also beat
    # whatever is already running, or there is no reason to swap.
    if incumbent is not None:
        current = float(incumbent.test_metrics.get("net_r", 0.0) or 0.0)
        if float(metrics["net_r"]) <= current:
            print()
            print(f"REJECTED: {metrics['net_r']:.2f}R does not beat the incumbent's "
                  f"{current:.2f}R.")
            print("Never replace a validated model with one that is not better.")
            return 1

    registry.set_status(args.symbol, version, STATUS_VALIDATED)
    print()
    print(f"{version} passed and is now VALIDATED.")
    if args.promote:
        registry.set_status(args.symbol, version, STATUS_PRODUCTION)
        print(f"{version} promoted to PRODUCTION (the previous one is RETIRED).")
    else:
        print("Run it in PPO_SHADOW first; promote with --promote when satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
