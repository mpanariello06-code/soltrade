"""The sealed final holdout.

WHY A FILE ENFORCES THIS
------------------------
"Do not look at the holdout" is a rule that decays the moment results are
disappointing.  So the seal is mechanical: a JSON file records the reserved
period and every time it has been opened, and the evaluator refuses to run
unless the caller explicitly breaks the seal.

That cannot stop a determined person - nothing can - but it makes an accidental
peek impossible and a deliberate one **recorded**.  A holdout that has been
opened five times is not a holdout, and after this it is at least obvious that
it was.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class HoldoutSeal:
    """The reserved period, plus the audit trail of every time it was opened."""

    symbol: str
    start: str
    end: str
    created_at: str = ""
    #: One entry per evaluation.  Length > 1 means it is no longer sealed.
    openings: List[Dict[str, Any]] = field(default_factory=list)
    note: str = (
        "This period must never be used for feature selection, hyperparameter "
        "tuning, reward tuning, action-space design, threshold tuning or model "
        "selection. Every opening is recorded below."
    )

    @property
    def times_opened(self) -> int:
        return len(self.openings)

    @property
    def intact(self) -> bool:
        """True while the holdout has never been evaluated."""
        return self.times_opened == 0

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["times_opened"] = self.times_opened
        data["intact"] = self.intact
        return data


def seal_path(root: Path, symbol: str) -> Path:
    return Path(root) / f"holdout_seal_{symbol}.json"


def create_seal(root: Path, symbol: str, start: str, end: str) -> HoldoutSeal:
    """Reserve a period.  Refuses to silently redefine an existing seal.

    Moving the boundary after seeing results is the most effective way to
    manufacture a good holdout number, so it is made loud rather than easy.
    """
    path = seal_path(root, symbol)
    if path.exists():
        existing = load_seal(root, symbol)
        if existing and (existing.start != start or existing.end != end):
            raise ValueError(
                f"a holdout is already sealed for {symbol} "
                f"({existing.start} -> {existing.end}). Redefining it after the "
                "fact invalidates it; delete the seal deliberately if you mean to."
            )
        return existing  # type: ignore[return-value]

    seal = HoldoutSeal(
        symbol=symbol, start=start, end=end,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seal.to_dict(), indent=2), encoding="utf-8")
    return seal


def load_seal(root: Path, symbol: str) -> Optional[HoldoutSeal]:
    path = seal_path(root, symbol)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return HoldoutSeal(
        symbol=data.get("symbol", symbol),
        start=data.get("start", ""), end=data.get("end", ""),
        created_at=data.get("created_at", ""),
        openings=list(data.get("openings") or []),
    )


def record_opening(root: Path, symbol: str, model_version: str,
                   reason: str = "") -> HoldoutSeal:
    """Break the seal, permanently and visibly."""
    seal = load_seal(root, symbol)
    if seal is None:
        raise FileNotFoundError(
            f"no holdout is sealed for {symbol} - create one before evaluating"
        )
    seal.openings.append({
        "at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "reason": reason,
        "opening_number": seal.times_opened + 1,
    })
    seal_path(root, symbol).write_text(
        json.dumps(seal.to_dict(), indent=2), encoding="utf-8"
    )
    return seal


def holdout_frame(frame: pd.DataFrame, seal: HoldoutSeal) -> pd.DataFrame:
    """The sealed slice of a dataset."""
    times = pd.to_datetime(frame["time"], utc=True)
    start = pd.Timestamp(seal.start, tz="UTC")
    end = pd.Timestamp(seal.end, tz="UTC")
    return frame[(times >= start) & (times < end)].reset_index(drop=True)


def training_frame(frame: pd.DataFrame, seal: HoldoutSeal) -> pd.DataFrame:
    """Everything OUTSIDE the seal - what training is allowed to see.

    Used by the training scripts so the sealed period cannot reach a model by
    accident: exclusion is applied at the data layer, not left to discipline.
    """
    times = pd.to_datetime(frame["time"], utc=True)
    start = pd.Timestamp(seal.start, tz="UTC")
    return frame[times < start].reset_index(drop=True)
