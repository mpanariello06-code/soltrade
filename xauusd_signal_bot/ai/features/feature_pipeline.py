"""Raw candles in, a training-ready observation matrix out.

WHAT THIS GUARANTEES
--------------------
* Every column is causal (see :mod:`ai.features.m1_features`).
* Warm-up rows, where an indicator has not enough history, are **dropped**, not
  filled.  A back-filled RSI is a value the agent could not have had.
* The column list and its order are pinned in a :class:`FeatureSpec` and stored
  with every trained model, so a model can never be served a different feature
  set - or the same features in a different order - from the one it learned on.
  That failure is silent and catastrophic, so it is made impossible instead.
* Nothing here reads or writes ``data/raw``; raw files stay immutable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .m1_features import FEATURE_VERSION, build_m1_features, m1_feature_columns
from .mtf_features import (
    DEFAULT_CONTEXT_TIMEFRAMES,
    MTF_VERSION,
    attach_context,
    context_feature_columns,
)

#: Columns the environment needs that are NOT observations - prices it executes
#: against, and the timestamp.  Kept beside the features, never fed to the net.
MARKET_COLUMNS: Tuple[str, ...] = (
    "time", "open", "high", "low", "close", "tick_volume", "spread",
)


@dataclass
class FeatureSpec:
    """The exact observation contract a model was trained against."""

    columns: List[str] = field(default_factory=list)
    m1_version: str = FEATURE_VERSION
    mtf_version: str = MTF_VERSION
    context_timeframes: Tuple[str, ...] = DEFAULT_CONTEXT_TIMEFRAMES
    #: Per-column clip bound.  ATR-normalised features are mostly within a few
    #: units; a 500-sigma spike from a bad tick would otherwise dominate a batch.
    clip: float = 10.0

    @property
    def size(self) -> int:
        return len(self.columns)

    def fingerprint(self) -> str:
        """Stable hash of the contract, for the registry and for load-time checks."""
        payload = json.dumps(
            {
                "columns": self.columns,
                "m1": self.m1_version,
                "mtf": self.mtf_version,
                "context": list(self.context_timeframes),
                "clip": self.clip,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "m1_version": self.m1_version,
            "mtf_version": self.mtf_version,
            "context_timeframes": list(self.context_timeframes),
            "clip": self.clip,
            "size": self.size,
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSpec":
        return cls(
            columns=list(data.get("columns") or []),
            m1_version=str(data.get("m1_version", FEATURE_VERSION)),
            mtf_version=str(data.get("mtf_version", MTF_VERSION)),
            context_timeframes=tuple(data.get("context_timeframes") or DEFAULT_CONTEXT_TIMEFRAMES),
            clip=float(data.get("clip", 10.0)),
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def build_dataset(
    raw: pd.DataFrame,
    context_timeframes: Iterable[str] = DEFAULT_CONTEXT_TIMEFRAMES,
    spec: Optional[FeatureSpec] = None,
    drop_warmup: bool = True,
) -> Tuple[pd.DataFrame, FeatureSpec]:
    """Build the feature matrix for one market's raw M1 history.

    Passing an existing ``spec`` reuses that exact column list - which is how
    inference guarantees it presents the same contract the model trained on.
    """
    if raw.empty:
        return raw.copy(), spec or FeatureSpec(columns=[])

    context_timeframes = tuple(context_timeframes)
    frame = build_m1_features(raw)
    if context_timeframes:
        frame = attach_context(frame, context_timeframes)

    if spec is None:
        columns = m1_feature_columns(frame)
        columns += context_feature_columns(frame, context_timeframes)
        # `f_atr` is the normalisation ANCHOR, in raw price units.  Feeding it
        # to the network would smuggle the price level back in through the door
        # every other feature was designed to close.
        columns = [c for c in dict.fromkeys(columns) if c != "f_atr"]
        spec = FeatureSpec(columns=sorted(columns), context_timeframes=context_timeframes)

    missing = [column for column in spec.columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"feature spec expects columns absent from the data: {missing[:5]}"
        )

    if drop_warmup:
        # Warm-up rows are dropped, never filled: a filled indicator is a number
        # the agent could not have computed at that moment.
        before = len(frame)
        frame = frame.dropna(subset=spec.columns).reset_index(drop=True)
        dropped = before - len(frame)
        if dropped:
            frame.attrs["warmup_dropped"] = dropped

    # Infinities come from a zero ATR on a flat stretch; they are not signal.
    frame[spec.columns] = frame[spec.columns].replace([np.inf, -np.inf], 0.0)
    frame[spec.columns] = frame[spec.columns].clip(-spec.clip, spec.clip)
    return frame, spec


def observation_matrix(frame: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    """The float32 matrix the environment serves, in the spec's exact order."""
    if frame.empty:
        return np.zeros((0, spec.size), dtype=np.float32)
    values = frame[spec.columns].to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def split_chronologically(
    frame: pd.DataFrame, boundaries: Iterable[str]
) -> List[pd.DataFrame]:
    """Cut a frame at UTC date boundaries, in order.

    Market data is never shuffled and never randomly split: doing so lets a
    model learn from Tuesday to predict Monday, and the resulting metrics are
    meaningless.
    """
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    cuts = [pd.Timestamp(b, tz="UTC") for b in boundaries]
    if any(later <= earlier for earlier, later in zip(cuts, cuts[1:])):
        raise ValueError("split boundaries must be strictly increasing")

    pieces: List[pd.DataFrame] = []
    start = None
    for cut in cuts:
        mask = frame["time"] < cut if start is None else (
            (frame["time"] >= start) & (frame["time"] < cut)
        )
        pieces.append(frame[mask].reset_index(drop=True))
        start = cut
    pieces.append(frame[frame["time"] >= cuts[-1]].reset_index(drop=True))
    return pieces
