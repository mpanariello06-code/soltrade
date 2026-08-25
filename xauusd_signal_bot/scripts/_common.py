"""Shared plumbing for the RL scripts: paths, dataset loading, seals."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from ai.features.feature_pipeline import FeatureSpec, build_dataset  # noqa: E402
from src.mt5_history import load_raw  # noqa: E402


def models_root(config) -> Path:
    return PROJECT_ROOT / "models"


def reports_root(config) -> Path:
    return PROJECT_ROOT / "reports" / "ppo"


def processed_path(config, symbol: str) -> Path:
    return Path(config.data_dir) / "processed" / symbol / "M1"


def load_dataset(
    config, symbol: str, spec: Optional[FeatureSpec] = None,
    start: Optional[str] = None, end: Optional[str] = None,
    cache: bool = True,
) -> Tuple[pd.DataFrame, FeatureSpec]:
    """Raw candles -> feature matrix, cached under ``data/processed``.

    The cache is keyed by the feature fingerprint, so changing the feature set
    produces a new file rather than silently reusing a stale one - a mismatch
    that would be invisible and would poison every result after it.
    """
    raw = load_raw(config, symbol, "M1", start=start, end=end)
    if raw.empty:
        raise SystemExit(
            f"No raw data for {symbol}. Download it first:\n"
            f"  python scripts/download_mt5_history.py --symbol {symbol} "
            f"--start 2023-01-01"
        )
    frame, spec = build_dataset(raw, spec=spec)
    if cache and not frame.empty:
        directory = processed_path(config, symbol)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"dataset_{spec.fingerprint()}.csv"
        if not target.exists():
            frame.to_csv(target, index=False)
        spec.save(directory / f"spec_{spec.fingerprint()}.json")
    return frame, spec
