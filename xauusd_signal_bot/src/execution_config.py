"""Execution mode, demo-account gating and the demo risk model.

WHY THIS IS A SEPARATE MODULE
-----------------------------
Everything that decides *whether an order may be sent at all* lives here, apart
from the code that actually sends one.  A reviewer can read this file alone and
convince themselves that:

* the default is :data:`SIGNAL_ONLY` and nothing executes without a deliberate
  opt-in on **two** independent switches;
* there is no mode that means "real money" - the enum has exactly two members,
  and :func:`assert_no_live_mode` fails loudly if one is ever added;
* the account-type check has no "assume demo" branch.

THE ONLY EXECUTION MODE IS DEMO
-------------------------------
There is deliberately no ``LIVE_AUTO``.  Adding one would require editing this
module, the mode enum, the account guard and the test that asserts the enum's
contents - which is the point.  This system is for measuring how much of a
paper edge survives real fills on a DEMO account.  It is not a trading system.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from .logger import get_logger

LOGGER = get_logger("execution.config")


class ExecutionMode(str, Enum):
    """How a generated signal is handled.

    ``SIGNAL_ONLY``  record and notify; no broker interaction whatsoever.
    ``DEMO_AUTO``    additionally submit the signal to a verified DEMO account.

    No third member may be added.  See :func:`assert_no_live_mode`.
    """

    SIGNAL_ONLY = "SIGNAL_ONLY"
    DEMO_AUTO = "DEMO_AUTO"

    @property
    def executes(self) -> bool:
        return self is ExecutionMode.DEMO_AUTO


#: The safe default.  A missing, empty or unrecognised setting lands here.
DEFAULT_EXECUTION_MODE = ExecutionMode.SIGNAL_ONLY

#: Words that must never name an execution mode.  Checked at import time.
_FORBIDDEN_MODE_WORDS = ("LIVE", "REAL", "PROD", "PRODUCTION")


def assert_no_live_mode() -> None:
    """Fail loudly if a live-trading mode is ever added to the enum.

    A guard rail with teeth: this runs at import and in the test suite, so a
    future edit that adds ``LIVE_AUTO`` breaks the build rather than quietly
    turning a research tool into a trading system.
    """
    for member in ExecutionMode:
        upper = member.value.upper()
        for word in _FORBIDDEN_MODE_WORDS:
            if word in upper:
                raise RuntimeError(
                    f"ExecutionMode.{member.name} looks like a live-trading mode. "
                    "This project is DEMO/paper only and must not gain one."
                )
    if len(list(ExecutionMode)) != 2:
        raise RuntimeError(
            "ExecutionMode must have exactly two members "
            f"(SIGNAL_ONLY, DEMO_AUTO); found {[m.name for m in ExecutionMode]}"
        )


assert_no_live_mode()


def parse_execution_mode(value: Optional[str]) -> ExecutionMode:
    """Read a mode from configuration, defaulting to ``SIGNAL_ONLY``.

    Anything unrecognised - including a typo, an empty string, or someone's
    hopeful ``LIVE_AUTO`` - resolves to the safe mode with a warning.  There is
    no spelling of this setting that turns on real-money trading.
    """
    text = str(value or "").strip().upper()
    if not text:
        return DEFAULT_EXECUTION_MODE
    try:
        return ExecutionMode(text)
    except ValueError:
        LOGGER.warning(
            "EXECUTION_MODE=%r is not a valid mode (%s) - using %s",
            value, ", ".join(m.value for m in ExecutionMode), DEFAULT_EXECUTION_MODE.value,
        )
        return DEFAULT_EXECUTION_MODE


# --------------------------------------------------------------------------- #
# account verification
# --------------------------------------------------------------------------- #
#: Sentinel returned (and logged) whenever an order is refused on safety grounds.
DEMO_EXECUTION_BLOCKED = "DEMO_EXECUTION_BLOCKED"


class DemoExecutionBlocked(RuntimeError):
    """Raised when execution is refused for a safety reason.

    Carries a human-readable ``reason`` that goes straight into the Telegram
    alert, because a blocked order the user cannot explain is a support ticket.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"{DEMO_EXECUTION_BLOCKED}: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class AccountInfo:
    """The subset of a broker account this layer cares about."""

    login: int = 0
    server: str = ""
    currency: str = ""
    balance: float = 0.0
    #: Raw trade-mode value as the platform reports it, normalised to a string.
    trade_mode: str = ""
    #: ``True`` only when the platform positively identified a DEMO account.
    is_demo: bool = False
    #: ``True`` when the platform reported something, false when it was silent.
    verified: bool = False

    def describe(self) -> str:
        kind = "DEMO" if self.is_demo else (self.trade_mode or "UNVERIFIED")
        return f"{kind} #{self.login} on {self.server or 'unknown server'}"


#: MT5 ``ACCOUNT_TRADE_MODE_*`` values.  0 = demo, 1 = contest, 2 = real.
#: Only 0 is accepted.  A contest account is not a demo account, and an unknown
#: value is never optimistically treated as one.
MT5_TRADE_MODE_DEMO = 0
MT5_TRADE_MODE_CONTEST = 1
MT5_TRADE_MODE_REAL = 2

_TRADE_MODE_NAMES = {
    MT5_TRADE_MODE_DEMO: "DEMO",
    MT5_TRADE_MODE_CONTEST: "CONTEST",
    MT5_TRADE_MODE_REAL: "LIVE",
}


def classify_account(raw) -> AccountInfo:
    """Turn a platform account object into an :class:`AccountInfo`.

    The whole point is the absence of an optimistic branch: ``is_demo`` is set
    only when the platform reported trade mode 0.  ``None``, a missing
    attribute, an unparseable value or a contest account all produce
    ``is_demo=False``, and callers refuse to trade on that.
    """
    if raw is None:
        return AccountInfo(verified=False, trade_mode="")

    def attribute(name, default=""):
        return getattr(raw, name, default)

    try:
        mode_value = int(attribute("trade_mode", -1))
    except (TypeError, ValueError):
        mode_value = -1

    mode_name = _TRADE_MODE_NAMES.get(mode_value, "")
    try:
        login = int(attribute("login", 0) or 0)
    except (TypeError, ValueError):
        login = 0
    try:
        balance = float(attribute("balance", 0.0) or 0.0)
    except (TypeError, ValueError):
        balance = 0.0

    return AccountInfo(
        login=login,
        server=str(attribute("server", "") or ""),
        currency=str(attribute("currency", "") or ""),
        balance=balance,
        trade_mode=mode_name or (str(mode_value) if mode_value >= 0 else ""),
        is_demo=mode_value == MT5_TRADE_MODE_DEMO,
        verified=mode_name != "",
    )


# --------------------------------------------------------------------------- #
# risk model
# --------------------------------------------------------------------------- #
@dataclass
class DemoRiskModel:
    """Position sizing for the demo account.

    Nothing here is a recommendation.  The lot size is derived from the
    signal's OWN stop distance so that a wider stop takes a smaller position -
    which is the only sizing rule that keeps risk per trade comparable across
    signals - and is then clamped hard.

    All of it is configurable, and none of it is claimed to be optimal.
    """

    #: Notional balance used for sizing.  Deliberately NOT read from the broker
    #: by default: a demo account's balance is arbitrary, and letting it drive
    #: size makes results incomparable between runs.
    account_balance: float = 10_000.0
    #: Fraction of ``account_balance`` risked if the stop is hit.  0.005 = 0.5%.
    risk_per_trade: float = 0.005
    minimum_lot: float = 0.01
    maximum_lot: float = 0.10
    #: Broker volume granularity.  Sizes are floored onto this grid.
    lot_step: float = 0.01
    #: Value of a one-point move per 1.00 lot, in account currency.
    #: Market-specific; see :class:`ExecutionSettings`.
    point_value_per_lot: float = 1.0

    def size_for(self, stop_distance_price: float, point_value: float) -> "SizingResult":
        """Lots to trade for a stop ``stop_distance_price`` away.

        ``point_value`` is the market's price-per-point, so the stop distance
        converts to points and then to money.  Returns a :class:`SizingResult`
        that records the requested size, the approved size and why they differ -
        because "the trade was rejected" is useless without the arithmetic.
        """
        risk_amount = float(self.account_balance) * float(self.risk_per_trade)

        if not math.isfinite(stop_distance_price) or stop_distance_price <= 0:
            return SizingResult(
                requested_lots=0.0, approved_lots=0.0, risk_amount=risk_amount,
                stop_distance_price=stop_distance_price,
                reason="stop distance is not a positive, finite number",
            )
        if point_value <= 0 or self.point_value_per_lot <= 0:
            return SizingResult(
                requested_lots=0.0, approved_lots=0.0, risk_amount=risk_amount,
                stop_distance_price=stop_distance_price,
                reason="market point value is not configured",
            )
        if risk_amount <= 0:
            return SizingResult(
                requested_lots=0.0, approved_lots=0.0, risk_amount=risk_amount,
                stop_distance_price=stop_distance_price,
                reason="risk per trade resolves to zero",
            )

        stop_points = stop_distance_price / point_value
        loss_per_lot = stop_points * self.point_value_per_lot
        if loss_per_lot <= 0:
            return SizingResult(
                requested_lots=0.0, approved_lots=0.0, risk_amount=risk_amount,
                stop_distance_price=stop_distance_price,
                reason="stop distance rounds to zero points",
            )

        requested = risk_amount / loss_per_lot
        approved = self._clamp(requested)
        reason = ""
        if approved <= 0:
            reason = (
                f"size {requested:.4f} floors to zero on a {self.lot_step} lot step"
            )
        elif approved < requested - 1e-12:
            reason = f"clamped to the {self.maximum_lot} maximum lot"
        return SizingResult(
            requested_lots=round(requested, 4),
            approved_lots=approved,
            risk_amount=round(risk_amount, 2),
            stop_distance_price=stop_distance_price,
            stop_points=round(stop_points, 1),
            reason=reason,
        )

    def _clamp(self, lots: float) -> float:
        """Floor onto the lot grid and clamp into the configured band.

        Flooring (not rounding) is deliberate: rounding up would risk more than
        the configured fraction, and on a micro-scalp that error is a large
        share of the trade.
        """
        step = max(float(self.lot_step), 1e-9)
        stepped = math.floor(float(lots) / step + 1e-9) * step
        stepped = round(stepped, 8)
        if stepped < self.minimum_lot:
            # Below the broker's minimum is not a small trade, it is no trade.
            return 0.0
        return round(min(stepped, float(self.maximum_lot)), 8)


@dataclass(frozen=True)
class SizingResult:
    """The full arithmetic behind one position-size decision."""

    requested_lots: float
    approved_lots: float
    risk_amount: float
    stop_distance_price: float
    stop_points: float = 0.0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.approved_lots > 0

    def describe(self) -> str:
        text = (
            f"requested {self.requested_lots:.4f} lots, approved "
            f"{self.approved_lots:.4f} lots (risk {self.risk_amount:.2f}, "
            f"stop {self.stop_distance_price:.5f} = {self.stop_points:.1f} points)"
        )
        return f"{text}; {self.reason}" if self.reason else text


# --------------------------------------------------------------------------- #
# execution settings
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionSettings:
    """Everything the execution layer reads, gathered in one place.

    The hard limits below are EXECUTION safety controls.  They are deliberately
    independent of the signal-generation filters that share some of their names:
    a signal may be perfectly valid and still be refused execution because the
    day's trade budget is spent.  Blocking a new trade never stops an existing
    position being managed.
    """

    mode: ExecutionMode = DEFAULT_EXECUTION_MODE
    #: Second, independent switch.  Both this and ``mode`` must be affirmative.
    demo_trading_enabled: bool = False

    #: A dedicated demo login must be configured; the data-feed credentials are
    #: deliberately NOT reused, so pointing the feed at a live account cannot
    #: silently arm execution.
    demo_login: int = 0
    demo_password: str = ""
    demo_server: str = ""

    # -- hard safety limits ------------------------------------------------- #
    max_open_positions: int = 2
    max_trades_per_day: int = 20
    #: Stop opening new trades once the day's realised demo P/L is this far
    #: down, in account currency.  A positive number meaning "loss of".
    max_daily_loss: float = 200.0
    max_order_lots: float = 0.10
    #: Refuse a signal whose stop or target is implausibly far away - a symptom
    #: of a bad quote or a misconfigured market rather than a real setup.
    max_sl_distance_pct: float = 0.02
    max_tp_distance_pct: float = 0.05
    #: Refuse to enter when the spread has blown out, in points.  Defaults to
    #: the market's own ``max_spread_points`` when left at zero.
    max_spread_points: float = 0.0

    # -- partial take-profit ladder ----------------------------------------- #
    #: Fractions of the position closed at TP1/TP2/TP3.  NOT claimed optimal -
    #: an even-ish split is simply the least opinionated starting point.
    tp_fractions: Tuple[float, float, float] = (0.33, 0.33, 0.34)

    # -- order mechanics ---------------------------------------------------- #
    #: Maximum deviation from the requested price, in points, the broker may
    #: fill within.  A scalp filled 20 points away is not the scalp we modelled.
    max_slippage_points: float = 20.0
    order_comment: str = "DEMO-SCALP"
    #: Magic number stamped on every order this layer creates, so reconciliation
    #: can tell our positions from anything else on the account.
    magic_number: int = 770_101

    risk: DemoRiskModel = field(default_factory=DemoRiskModel)

    # ------------------------------------------------------------------ #
    @property
    def executes(self) -> bool:
        """True only when BOTH independent switches are affirmative."""
        return self.mode.executes and bool(self.demo_trading_enabled)

    @property
    def has_demo_account(self) -> bool:
        """True when a dedicated demo account is configured."""
        return bool(self.demo_login and self.demo_password and self.demo_server)

    def blocking_reason(self) -> str:
        """Why execution is off, or ``""`` when it is armed.

        Used for both the Telegram panel and the pre-order guard, so what the
        user is told matches exactly what the code checks.
        """
        if not self.mode.executes:
            return f"EXECUTION_MODE is {self.mode.value}, not {ExecutionMode.DEMO_AUTO.value}"
        if not self.demo_trading_enabled:
            return "DEMO_TRADING_ENABLED is false"
        if not self.has_demo_account:
            missing = [
                name for name, value in (
                    ("DEMO_MT5_LOGIN", self.demo_login),
                    ("DEMO_MT5_PASSWORD", self.demo_password),
                    ("DEMO_MT5_SERVER", self.demo_server),
                )
                if not value
            ]
            return f"no dedicated demo account configured (missing {', '.join(missing)})"
        return ""

    def validate(self) -> None:
        """Raise ``ValueError`` on an internally inconsistent execution config."""
        total = sum(self.tp_fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"tp_fractions must sum to 1.0, got {total}")
        if any(fraction <= 0 for fraction in self.tp_fractions):
            raise ValueError("every tp_fraction must be positive")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be at least 1")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day must be at least 1")
        if self.risk.minimum_lot <= 0:
            raise ValueError("minimum_lot must be positive")
        if self.risk.maximum_lot < self.risk.minimum_lot:
            raise ValueError("maximum_lot must be >= minimum_lot")
        if self.max_order_lots < self.risk.minimum_lot:
            raise ValueError("max_order_lots is below the minimum tradeable lot")
        if not 0 < self.risk.risk_per_trade < 1:
            raise ValueError("risk_per_trade must be a fraction in (0, 1)")
        assert_no_live_mode()


def load_execution_settings(prefix: str = "") -> ExecutionSettings:
    """Build :class:`ExecutionSettings` from the environment.

    ``prefix`` exists for tests; production reads the bare names.
    """
    def text(name: str, default: str = "") -> str:
        return str(os.getenv(f"{prefix}{name}", default) or "").strip()

    def number(name: str, default: float) -> float:
        try:
            return float(text(name, str(default)))
        except ValueError:
            LOGGER.warning("%s is not a number - using %s", name, default)
            return default

    def whole(name: str, default: int) -> int:
        try:
            return int(float(text(name, str(default))))
        except ValueError:
            LOGGER.warning("%s is not a whole number - using %s", name, default)
            return default

    def flag(name: str, default: bool = False) -> bool:
        raw = text(name, "true" if default else "false").lower()
        return raw in ("1", "true", "yes", "on")

    fractions = (
        number("TP1_CLOSE_FRACTION", 0.33),
        number("TP2_CLOSE_FRACTION", 0.33),
        number("TP3_CLOSE_FRACTION", 0.34),
    )

    settings = ExecutionSettings(
        mode=parse_execution_mode(text("EXECUTION_MODE")),
        demo_trading_enabled=flag("DEMO_TRADING_ENABLED"),
        demo_login=whole("DEMO_MT5_LOGIN", 0),
        demo_password=text("DEMO_MT5_PASSWORD"),
        demo_server=text("DEMO_MT5_SERVER"),
        max_open_positions=whole("MAX_OPEN_POSITIONS", 2),
        max_trades_per_day=whole("MAX_DEMO_TRADES_PER_DAY", 20),
        max_daily_loss=number("MAX_DAILY_DEMO_LOSS", 200.0),
        max_order_lots=number("MAX_ORDER_LOTS", 0.10),
        max_sl_distance_pct=number("MAX_SL_DISTANCE_PCT", 0.02),
        max_tp_distance_pct=number("MAX_TP_DISTANCE_PCT", 0.05),
        max_spread_points=number("EXECUTION_MAX_SPREAD_POINTS", 0.0),
        tp_fractions=fractions,
        max_slippage_points=number("MAX_SLIPPAGE_POINTS", 20.0),
        magic_number=whole("DEMO_MAGIC_NUMBER", 770_101),
        risk=DemoRiskModel(
            account_balance=number("DEMO_ACCOUNT_BALANCE", 10_000.0),
            risk_per_trade=number("DEMO_RISK_PER_TRADE", 0.005),
            minimum_lot=number("MIN_DEMO_LOT", 0.01),
            maximum_lot=number("MAX_DEMO_LOT", 0.10),
            lot_step=number("DEMO_LOT_STEP", 0.01),
        ),
    )
    return settings
