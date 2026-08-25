"""Checkpoint management: never lose training progress, never overwrite a model.

WHAT A CHECKPOINT DIRECTORY HOLDS
---------------------------------
``best_model.zip``      highest validation NET R seen during the run
``latest.zip``          most recent, so a killed run can be resumed
``final_model.zip``     the policy at the end of training
``feature_spec.json``   the observation contract it learned on
``training_result.json`` seed, hyperparameters, dates, metrics

The feature spec travels WITH the weights on purpose.  Serving a model a
different column set - or the same columns in a different order - is silent and
ruins every result after it, so the contract is never separable from the model.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CHECKPOINT_BEST = "best_model"
CHECKPOINT_LATEST = "latest"
CHECKPOINT_FINAL = "final_model"
CHECKPOINT_NAMES = (CHECKPOINT_BEST, CHECKPOINT_LATEST, CHECKPOINT_FINAL)


@dataclass(frozen=True)
class CheckpointInfo:
    """What a checkpoint directory actually contains."""

    directory: Path
    available: List[str]
    has_feature_spec: bool
    has_training_result: bool
    created_at: str = ""
    size_bytes: int = 0

    @property
    def usable(self) -> bool:
        """Whether this can be loaded for inference.

        Weights alone are NOT enough: without the feature spec the model would
        still answer confidently, from whatever columns it happened to be given.
        """
        return CHECKPOINT_BEST in self.available and self.has_feature_spec

    def describe(self) -> str:
        state = "usable" if self.usable else "INCOMPLETE"
        return (
            f"{self.directory.name}: {', '.join(self.available) or 'no weights'} "
            f"({state}, {self.size_bytes / 1_048_576:.1f} MB)"
        )


def inspect(directory) -> CheckpointInfo:
    """Report what is in a checkpoint directory without loading anything."""
    directory = Path(directory)
    if not directory.is_dir():
        return CheckpointInfo(directory, [], False, False)

    available = [n for n in CHECKPOINT_NAMES if (directory / f"{n}.zip").exists()]
    size = sum(p.stat().st_size for p in directory.glob("*") if p.is_file())
    created = ""
    result = directory / "training_result.json"
    if result.exists():
        try:
            created = str(json.loads(result.read_text(encoding="utf-8")).get("created_at", ""))
        except (OSError, json.JSONDecodeError):
            created = ""
    return CheckpointInfo(
        directory=directory, available=available,
        has_feature_spec=(directory / "feature_spec.json").exists(),
        has_training_result=result.exists(),
        created_at=created, size_bytes=size,
    )


def checkpoint_path(directory, name: str = CHECKPOINT_BEST) -> Optional[Path]:
    """Path to one checkpoint, or ``None`` when it is absent."""
    path = Path(directory) / f"{name}.zip"
    return path if path.exists() else None


def resume_from(directory) -> Optional[Path]:
    """The checkpoint to continue a killed run from.

    ``latest`` rather than ``best``: resuming from the best-scoring policy would
    silently discard however much training happened after it.
    """
    return checkpoint_path(directory, CHECKPOINT_LATEST)


def archive(directory, reason: str = "") -> Optional[Path]:
    """Move a checkpoint directory aside instead of deleting it.

    Nothing is ever removed: a model that produced a published number must stay
    reproducible, even a bad one.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = directory.with_name(f"{directory.name}.archived-{stamp}")
    shutil.move(str(directory), str(target))
    if reason:
        (target / "ARCHIVED.txt").write_text(f"{stamp} UTC\n{reason}\n", encoding="utf-8")
    return target


def list_checkpoints(root, symbol: str) -> List[CheckpointInfo]:
    """Every version directory for one market, oldest first."""
    base = Path(root) / "ppo" / symbol
    if not base.is_dir():
        return []
    return [inspect(p) for p in sorted(base.iterdir()) if p.is_dir()]


def verify(directory, expected_fingerprint: str = "") -> Dict[str, Any]:
    """Check a checkpoint is complete and matches an expected feature contract.

    Run this before promoting a model: a mismatch found here is a configuration
    problem, and the same mismatch found at inference time is a silent one.
    """
    info = inspect(directory)
    problems: List[str] = []
    if CHECKPOINT_BEST not in info.available:
        problems.append("best_model.zip is missing")
    if not info.has_feature_spec:
        problems.append("feature_spec.json is missing - the model cannot be served safely")

    fingerprint = ""
    if info.has_feature_spec:
        try:
            from ai.features.feature_pipeline import FeatureSpec

            fingerprint = FeatureSpec.load(Path(directory) / "feature_spec.json").fingerprint()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"feature_spec.json is unreadable: {exc}")

    if expected_fingerprint and fingerprint and fingerprint != expected_fingerprint:
        problems.append(
            f"feature fingerprint {fingerprint} does not match the dataset's "
            f"{expected_fingerprint} - the model would be served the wrong columns"
        )
    return {"ok": not problems, "problems": problems,
            "fingerprint": fingerprint, "info": info}
