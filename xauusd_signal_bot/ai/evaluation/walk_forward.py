"""Sliding walk-forward validation.  Train, validate, test on unseen, slide.

WHY WALK-FORWARD AND NOT A SINGLE SPLIT
---------------------------------------
One train/test split answers "did this work once".  Walk-forward answers "does
this keep working", which is the only version of the question worth asking.  A
strategy that is excellent in 2019 and useless in 2020 will look fine on a
single split that happens to end in 2019.

THE RULE THAT MAKES IT HONEST
-----------------------------
Each fold trains on data strictly BEFORE its validation window, which is
strictly before its test window.  The model that is evaluated on a fold's test
period was selected using that fold's validation period and nothing later.  No
fold ever sees a later fold's data, so a good result cannot be an artefact of
knowing the future.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ai.environment.scalping_env import EnvConfig
from ai.features.feature_pipeline import FeatureSpec


@dataclass
class Fold:
    """One walk-forward window."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> Dict[str, str]:
        return {
            "fold": str(self.index),
            "train": f"{self.train_start.date()} -> {self.train_end.date()}",
            "validation": f"{self.train_end.date()} -> {self.validation_end.date()}",
            "test": f"{self.validation_end.date()} -> {self.test_end.date()}",
        }


def build_folds(
    frame: pd.DataFrame,
    train_days: int = 365 * 2,
    validation_days: int = 90,
    test_days: int = 90,
    step_days: Optional[int] = None,
    max_folds: int = 0,
) -> List[Fold]:
    """Slice a series into sliding folds.

    ``step_days`` defaults to ``test_days`` so consecutive test windows abut
    without overlapping - overlapping test windows would count the same period
    more than once and make the aggregate look steadier than it is.
    """
    if frame.empty:
        return []
    times = pd.to_datetime(frame["time"], utc=True)
    first, last = times.iloc[0], times.iloc[-1]
    step = pd.Timedelta(days=step_days or test_days)

    folds: List[Fold] = []
    train_start = first
    index = 1
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        validation_end = train_end + pd.Timedelta(days=validation_days)
        test_end = validation_end + pd.Timedelta(days=test_days)
        if test_end > last:
            break
        folds.append(Fold(index, train_start, train_end, validation_end, test_end))
        index += 1
        train_start = train_start + step
        if max_folds and len(folds) >= max_folds:
            break
    return folds


def slice_period(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    """Half-open ``[start, end)`` slice, chronological."""
    times = pd.to_datetime(frame["time"], utc=True)
    return frame[(times >= start) & (times < end)].reset_index(drop=True)


@dataclass
class FoldResult:
    """What one fold produced.  ``test_*`` was never used for selection."""

    fold: int = 0
    train_period: str = ""
    validation_period: str = ""
    test_period: str = ""
    train_rows: int = 0
    validation_rows: int = 0
    test_rows: int = 0
    model_path: str = ""
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    skipped: str = ""


def run_walk_forward(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    env_config: EnvConfig,
    output_dir: Path,
    hyper=None,
    total_timesteps: int = 50_000,
    train_days: int = 365 * 2,
    validation_days: int = 90,
    test_days: int = 90,
    max_folds: int = 0,
    symbol: str = "",
    advanced_actions: bool = False,
) -> List[FoldResult]:
    """Train and evaluate one model per fold; never reuse a later fold's data."""
    from ai.ppo.train import PPOHyperParameters, evaluate_policy_on, train_ppo

    hyper = hyper or PPOHyperParameters()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = build_folds(frame, train_days, validation_days, test_days,
                        max_folds=max_folds)
    results: List[FoldResult] = []

    for fold in folds:
        train = slice_period(frame, fold.train_start, fold.train_end)
        validation = slice_period(frame, fold.train_end, fold.validation_end)
        test = slice_period(frame, fold.validation_end, fold.test_end)

        described = fold.describe()
        result = FoldResult(
            fold=fold.index,
            train_period=described["train"],
            validation_period=described["validation"],
            test_period=described["test"],
            train_rows=len(train), validation_rows=len(validation), test_rows=len(test),
        )

        # A fold with no data in a window is reported, not silently skipped:
        # a walk-forward summary computed over folds that quietly vanished is
        # a different experiment from the one that was described.
        if train.empty or validation.empty or test.empty:
            result.skipped = "one or more windows contained no candles"
            results.append(result)
            continue

        fold_dir = output_dir / f"fold_{fold.index:02d}"
        training = train_ppo(
            train, validation, spec, env_config, fold_dir,
            hyper=hyper, total_timesteps=total_timesteps,
            eval_every=max(total_timesteps // 4, 1),
            advanced_actions=advanced_actions,
            symbol=symbol, version=f"fold_{fold.index:02d}",
        )
        result.model_path = training.model_path
        result.validation_metrics = training.validation_metrics

        from stable_baselines3 import PPO

        model = PPO.load(fold_dir / "best_model", device=training.device)
        test_run = evaluate_policy_on(
            model, test, spec, env_config, advanced_actions=advanced_actions
        )
        trades = test_run.pop("trades_frame", pd.DataFrame())
        result.test_metrics = test_run
        if not trades.empty:
            trades.to_csv(fold_dir / "test_trades.csv", index=False)
        results.append(result)

    write_results(results, output_dir)
    return results


def write_results(results: List[FoldResult], output_dir: Path) -> Dict[str, Path]:
    """Write ``walk_forward_results.csv`` and ``walk_forward_equity.csv``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for result in results:
        row: Dict[str, Any] = {
            "fold": result.fold,
            "train_period": result.train_period,
            "validation_period": result.validation_period,
            "test_period": result.test_period,
            "train_rows": result.train_rows,
            "validation_rows": result.validation_rows,
            "test_rows": result.test_rows,
            "skipped": result.skipped,
        }
        for prefix, metrics in (("val", result.validation_metrics),
                                ("test", result.test_metrics)):
            for key in ("trades", "net_r", "average_net_r", "win_rate",
                        "profit_factor", "max_drawdown_r", "average_holding"):
                row[f"{prefix}_{key}"] = metrics.get(key, "")
        rows.append(row)

    results_path = output_dir / "walk_forward_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)

    # Equity across folds, in fold order: the aggregate out-of-sample curve.
    equity, running = [], 0.0
    for result in results:
        net = float(result.test_metrics.get("net_r", 0.0) or 0.0)
        running += net
        equity.append({
            "fold": result.fold, "test_period": result.test_period,
            "test_net_r": net, "cumulative_net_r": round(running, 4),
        })
    equity_path = output_dir / "walk_forward_equity.csv"
    pd.DataFrame(equity).to_csv(equity_path, index=False)
    return {"results": results_path, "equity": equity_path}


def aggregate(results: List[FoldResult]) -> Dict[str, Any]:
    """Pool every fold's TEST metrics into one out-of-sample verdict.

    Consistency matters more than the total: a single spectacular fold beside
    several poor ones is noise, not an edge, so the spread across folds is
    reported alongside the sum.
    """
    scored = [r for r in results if not r.skipped and r.test_metrics]
    if not scored:
        return {"folds": 0, "note": "no fold produced test metrics"}

    net = [float(r.test_metrics.get("net_r", 0.0) or 0.0) for r in scored]
    trades = [int(r.test_metrics.get("trades", 0) or 0) for r in scored]
    averages = [float(r.test_metrics.get("average_net_r", 0.0) or 0.0) for r in scored]
    series = pd.Series(net)
    return {
        "folds": len(scored),
        "total_test_trades": int(sum(trades)),
        "total_test_net_r": round(float(series.sum()), 4),
        "mean_fold_net_r": round(float(series.mean()), 4),
        "std_fold_net_r": round(float(series.std(ddof=0)), 4),
        "positive_folds": int((series > 0).sum()),
        "negative_folds": int((series < 0).sum()),
        "mean_average_net_r": round(float(pd.Series(averages).mean()), 4),
        "worst_fold_net_r": round(float(series.min()), 4),
        "best_fold_net_r": round(float(series.max()), 4),
    }
