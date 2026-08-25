"""Versioned PPO model registry.  Old models are never overwritten.

WHY THIS EXISTS
---------------
A trading model that silently changes underneath you is untraceable: you cannot
tell whether last week's results came from the model running today.  So every
training run writes a NEW version directory, and the registry records what it
was trained on, what it scored, and whether anyone has decided to trust it.

Promotion is a decision, not a side effect of training finishing.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Lifecycle.  A model only moves forward on evidence, and only PRODUCTION may
#: place orders.
STATUS_EXPERIMENTAL = "EXPERIMENTAL"
STATUS_VALIDATED = "VALIDATED"
STATUS_SHADOW = "SHADOW"
STATUS_PRODUCTION = "PRODUCTION"
STATUS_RETIRED = "RETIRED"
STATUSES = (
    STATUS_EXPERIMENTAL, STATUS_VALIDATED, STATUS_SHADOW,
    STATUS_PRODUCTION, STATUS_RETIRED,
)


@dataclass
class ModelRecord:
    """Everything needed to reproduce and judge one trained model."""

    version: str
    symbol: str
    timeframe: str = "M1"
    status: str = STATUS_EXPERIMENTAL
    created_at: str = ""
    path: str = ""

    train_start: str = ""
    train_end: str = ""
    validation_start: str = ""
    validation_end: str = ""
    test_start: str = ""
    test_end: str = ""

    feature_fingerprint: str = ""
    feature_count: int = 0
    dataset_rows: int = 0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    env_config: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    total_timesteps: int = 0

    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRecord":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class ModelRegistry:
    """``models/model_registry.json`` plus the version directories it indexes."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "model_registry.json"
        self.root.mkdir(parents=True, exist_ok=True)

    # -- storage ------------------------------------------------------------ #
    def _read(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"models": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt index must not destroy the models it points at.
            backup = self.path.with_suffix(".json.corrupt")
            if self.path.exists():
                shutil.copy2(self.path, backup)
            return {"models": []}

    def _write(self, payload: Dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    # -- queries ------------------------------------------------------------ #
    def records(self, symbol: Optional[str] = None) -> List[ModelRecord]:
        models = [ModelRecord.from_dict(m) for m in self._read().get("models", [])]
        if symbol:
            models = [m for m in models if m.symbol == symbol]
        return sorted(models, key=lambda m: m.version)

    def get(self, symbol: str, version: str) -> Optional[ModelRecord]:
        for record in self.records(symbol):
            if record.version == version:
                return record
        return None

    def latest(self, symbol: str, status: Optional[str] = None) -> Optional[ModelRecord]:
        candidates = [
            r for r in self.records(symbol) if status is None or r.status == status
        ]
        return candidates[-1] if candidates else None

    def production(self, symbol: str) -> Optional[ModelRecord]:
        """The model allowed to trade, if any.  ``None`` is the safe answer."""
        return self.latest(symbol, STATUS_PRODUCTION)

    def next_version(self, symbol: str) -> str:
        """``ppo_v001``, ``ppo_v002``, ... - monotonic, never reused."""
        existing = [r.version for r in self.records(symbol)]
        numbers = []
        for version in existing:
            digits = "".join(c for c in version if c.isdigit())
            if digits:
                numbers.append(int(digits))
        return f"ppo_v{max(numbers, default=0) + 1:03d}"

    def model_dir(self, symbol: str, version: str) -> Path:
        return self.root / "ppo" / symbol / version

    # -- mutations ------------------------------------------------------- #
    def register(self, record: ModelRecord) -> ModelRecord:
        """Add a record.  Refuses to overwrite an existing version."""
        payload = self._read()
        if any(
            m.get("version") == record.version and m.get("symbol") == record.symbol
            for m in payload.get("models", [])
        ):
            raise ValueError(
                f"{record.symbol} {record.version} already exists - "
                "models are never overwritten"
            )
        if not record.created_at:
            record.created_at = datetime.now(timezone.utc).isoformat()
        if not record.path:
            record.path = str(self.model_dir(record.symbol, record.version))
        payload.setdefault("models", []).append(record.to_dict())
        self._write(payload)
        return record

    def set_status(self, symbol: str, version: str, status: str) -> ModelRecord:
        """Move a model through the lifecycle.

        Promoting to PRODUCTION retires the previous production model rather
        than leaving two, because "which one is live?" must have one answer.
        """
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r} (use {', '.join(STATUSES)})")
        payload = self._read()
        found: Optional[Dict[str, Any]] = None
        for entry in payload.get("models", []):
            if entry.get("symbol") == symbol and entry.get("version") == version:
                found = entry
                break
        if found is None:
            raise KeyError(f"{symbol} {version} is not registered")

        if status == STATUS_PRODUCTION:
            for entry in payload.get("models", []):
                if (
                    entry.get("symbol") == symbol
                    and entry.get("status") == STATUS_PRODUCTION
                    and entry is not found
                ):
                    entry["status"] = STATUS_RETIRED
        found["status"] = status
        self._write(payload)
        return ModelRecord.from_dict(found)

    def update_metrics(
        self, symbol: str, version: str,
        validation: Optional[Dict[str, Any]] = None,
        test: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = self._read()
        for entry in payload.get("models", []):
            if entry.get("symbol") == symbol and entry.get("version") == version:
                if validation is not None:
                    entry["validation_metrics"] = validation
                if test is not None:
                    entry["test_metrics"] = test
                self._write(payload)
                return
        raise KeyError(f"{symbol} {version} is not registered")


# --------------------------------------------------------------------------- #
# promotion gates
# --------------------------------------------------------------------------- #
@dataclass
class PromotionGates:
    """Thresholds a candidate must clear on UNSEEN data to be VALIDATED.

    Every value is configurable and none is claimed optimal.  The point is not
    the specific numbers - it is that the decision is made against fixed
    criteria stated in advance, rather than by looking at the result and
    deciding afterwards whether it looks good.
    """

    min_trades: int = 50
    min_net_r: float = 0.0
    min_average_net_r: float = 0.0
    max_drawdown_r: float = 25.0
    min_profit_factor: float = 1.0
    max_average_holding: float = 15.0

    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``{passed, failures, checks}`` for a metrics dict."""
        def number(key, default=0.0):
            try:
                return float(metrics.get(key, default))
            except (TypeError, ValueError):
                return default

        checks = {
            "min_trades": (number("trades"), self.min_trades,
                           number("trades") >= self.min_trades),
            "min_net_r": (number("net_r"), self.min_net_r,
                          number("net_r") >= self.min_net_r),
            "min_average_net_r": (number("average_net_r"), self.min_average_net_r,
                                  number("average_net_r") >= self.min_average_net_r),
            "max_drawdown_r": (number("max_drawdown_r"), self.max_drawdown_r,
                               number("max_drawdown_r") <= self.max_drawdown_r),
            "min_profit_factor": (number("profit_factor"), self.min_profit_factor,
                                  number("profit_factor") >= self.min_profit_factor),
            "max_average_holding": (number("average_holding"), self.max_average_holding,
                                    number("average_holding") <= self.max_average_holding),
        }
        failures = [name for name, (_, _, ok) in checks.items() if not ok]
        return {
            "passed": not failures,
            "failures": failures,
            "checks": {n: {"value": v, "limit": l, "ok": ok}
                       for n, (v, l, ok) in checks.items()},
        }
