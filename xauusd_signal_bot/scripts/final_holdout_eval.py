"""Evaluate a FROZEN model on the sealed holdout.  Opening it is recorded.

    # reserve the period, once, before any tuning
    python scripts/final_holdout_eval.py --symbol XAUUSD --seal \
        --start 2025-01-01 --end 2026-01-01

    # much later, after the model is frozen
    python scripts/final_holdout_eval.py --symbol XAUUSD --version ppo_v003 --open

The seal cannot stop a determined person from peeking - nothing can - but it
makes an accidental peek impossible and a deliberate one permanently visible.
A holdout opened five times is not a holdout, and afterwards it is at least
obvious that it was.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from _common import load_dataset, models_root, reports_root

from ai.environment.scalping_env import EnvConfig
from ai.evaluation.holdout import create_seal, holdout_frame, load_seal, record_opening
from ai.evaluation.metrics import by_regime, equity_curve, summarise_trades
from ai.model_registry import ModelRegistry
from config import load_config
from src.markets import DEFAULT_MARKET, MARKET_ORDER, market_argument


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="Sealed final holdout evaluation")
    parser.add_argument("--symbol", type=market_argument, default=DEFAULT_MARKET,
                        help=f"{', '.join(MARKET_ORDER)} (default: %(default)s)")
    parser.add_argument("--seal", action="store_true", help="reserve a period and exit")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--version", type=str, default=None,
                        help="model version to evaluate (default: the latest)")
    parser.add_argument("--open", action="store_true",
                        help="BREAK THE SEAL and evaluate. Recorded permanently.")
    parser.add_argument("--reason", type=str, default="final evaluation")
    args = parser.parse_args(argv)

    root = models_root(config)

    if args.seal:
        if not args.start or not args.end:
            print("ERROR: --seal needs --start and --end", file=sys.stderr)
            return 2
        seal = create_seal(root, args.symbol, args.start, args.end)
        print(f"Sealed {args.symbol}: {seal.start} -> {seal.end}")
        print("Training and walk-forward now exclude this period automatically.")
        return 0

    seal = load_seal(root, args.symbol)
    if seal is None:
        print(f"No holdout is sealed for {args.symbol}. Create one first:\n"
              f"  python scripts/final_holdout_eval.py --symbol {args.symbol} "
              f"--seal --start ... --end ...")
        return 2

    print(f"Holdout: {seal.start} -> {seal.end}")
    print(f"Times opened so far: {seal.times_opened}")
    if not args.open:
        print()
        print("Not evaluating. Pass --open to break the seal.")
        print("Only do this ONCE, with the model frozen. Every opening is recorded.")
        return 0
    if not seal.intact:
        print()
        print(f"WARNING: this holdout has already been opened {seal.times_opened} "
              "time(s).")
        print("Results from a re-opened holdout are no longer out-of-sample.")

    registry = ModelRegistry(root)
    record = (
        registry.get(args.symbol, args.version) if args.version
        else registry.latest(args.symbol)
    )
    if record is None:
        print(f"No model registered for {args.symbol}.")
        return 2

    frame, spec = load_dataset(config, args.symbol)
    held = holdout_frame(frame, seal)
    if held.empty:
        print("The sealed period contains no candles in the dataset.")
        return 1

    from ai.ppo.train import evaluate_policy_on
    from stable_baselines3 import PPO

    model = PPO.load(Path(record.path) / "best_model", device="cpu")
    view = config.for_market(args.symbol)
    metrics = evaluate_policy_on(model, held, spec, EnvConfig.from_market_config(view))
    trades = metrics.pop("trades_frame", None)

    record_opening(root, args.symbol, record.version, args.reason)
    registry.update_metrics(args.symbol, record.version, test=metrics)

    output = reports_root(config) / args.symbol / "holdout"
    output.mkdir(parents=True, exist_ok=True)
    if trades is not None and not trades.empty:
        trades.to_csv(output / "trades.csv", index=False)
        equity_curve(trades).to_csv(output / "equity.csv", index=False)
        by_regime(trades).to_csv(output / "by_regime.csv", index=False)
        summary = summarise_trades(trades, label=f"{args.symbol} holdout")
    else:
        summary = metrics
    (output / "performance.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(f"=== SEALED HOLDOUT RESULT: {args.symbol} {record.version} ===")
    print(json.dumps(summary, indent=2, default=str))
    print()
    print(f"Written to: {output}")
    print()
    print("This number is what it is. Do not retune and re-open - a holdout")
    print("opened twice is a validation set with extra steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
