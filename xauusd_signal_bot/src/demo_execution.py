"""Demo execution manager: validated signals in, managed demo positions out.

  Signal Engine -> Signal -> ExecutionManager -> DEMO BROKER -> Position
                                     |                              |
                                     +---- TP/SL/timeout management -+
                                                    |
                                        executions.csv + Telegram

WHAT THIS MODULE IS FOR
-----------------------
Measuring how much of a paper edge survives real fills.  The signal engine is
unchanged and remains the source of truth: this layer never decides *whether*
to trade, only whether an already-generated signal may be *executed* on a
verified demo account, and then manages the resulting position to its exit.

THE SIGNAL LAYER IS NOT TOUCHED
-------------------------------
Prices, stops and targets come from the signal object verbatim.  Nothing here
recalculates them, and paper outcomes continue to be recorded independently, so
signal performance and execution performance can be compared rather than one
overwriting the other (spec section 14).

SAFETY POSTURE
--------------
Every order passes ten gates in :meth:`ExecutionManager.execute_signal`, in
order, and the first failure aborts.  The account-type gate has no optimistic
branch: an account that cannot be *positively verified as demo* is refused.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from .demo_broker import BUY, SELL, BrokerPosition, OrderRequest, Quote
from .execution_config import (
    AccountInfo,
    DemoExecutionBlocked,
    ExecutionSettings,
    SizingResult,
)
from .logger import get_logger
from .markets import get_market
from .signal_tracker import append_csv, ensure_csv, read_csv_rows
from .utils import iso, now_utc, parse_iso

LOGGER = get_logger("execution")

# -- outcome labels, matching the paper tracker's vocabulary ---------------- #
RESULT_TP1 = "TP1_HIT"
RESULT_TP2 = "TP2_HIT"
RESULT_TP3 = "TP3_HIT"
RESULT_SL = "SL_HIT"
RESULT_TIMEOUT = "TIMEOUT"
RESULT_OPEN = "OPEN"
TERMINAL_RESULTS = (RESULT_TP3, RESULT_SL, RESULT_TIMEOUT)

#: ``executions.csv`` columns.  Signal data, execution data and outcome data are
#: kept in ONE row so a fill can always be traced back to the signal that caused
#: it, and so execution results can be compared against paper results per signal.
EXECUTION_COLUMNS: Tuple[str, ...] = (
    # -- signal ----------------------------------------------------------- #
    "signal_id", "symbol", "timeframe", "direction", "signal_timestamp",
    "signal_entry", "sl", "tp1", "tp2", "tp3",
    "signal_score", "threshold", "regime", "session",
    # -- execution -------------------------------------------------------- #
    "order_id", "broker_ticket", "broker_symbol",
    "requested_price", "actual_fill_price", "spread_at_entry", "slippage_points",
    "slippage_price", "position_size", "requested_size", "execution_timestamp",
    "account_login", "account_type",
    # -- outcome ---------------------------------------------------------- #
    "result", "exit_price", "exit_timestamp", "holding_seconds",
    "gross_profit", "estimated_cost", "net_profit", "R_multiple",
    "tp1_filled", "tp2_filled", "tp3_filled", "closed_volume", "notes",
)


# --------------------------------------------------------------------------- #
# guard outcomes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecutionDecision:
    """Why a signal was or was not executed.

    Returned rather than raised for ordinary refusals - a blocked trade is a
    normal, expected outcome that must be logged and shown, not an error.
    """

    executed: bool
    reason: str = ""
    blocked: bool = False
    trade: Optional["DemoTrade"] = None

    @property
    def ok(self) -> bool:
        return self.executed


@dataclass
class DemoTrade:
    """One demo position, from signal through fill to exit.

    Holds the signal's own levels verbatim.  ``remaining_volume`` shrinks as the
    partial-close ladder fires; the trade is terminal when it reaches zero or a
    stop/timeout closes it.
    """

    signal_id: str
    symbol: str
    broker_symbol: str
    direction: str
    timeframe: str = "M1"
    signal_timestamp: Optional[datetime] = None

    # -- levels, straight from the signal --------------------------------- #
    signal_entry: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    signal_score: float = 0.0
    threshold: float = 0.0
    regime: str = ""
    session: str = ""

    # -- execution --------------------------------------------------------- #
    ticket: int = 0
    order_id: int = 0
    requested_price: float = 0.0
    fill_price: float = 0.0
    spread_at_entry: float = float("nan")
    slippage_points: float = 0.0
    volume: float = 0.0
    requested_volume: float = 0.0
    remaining_volume: float = 0.0
    #: Price distance from the ACTUAL fill to the ORIGINAL stop, captured once
    #: at entry.  R is always measured against this: moving the stop to
    #: breakeven after TP1 must not retroactively divide by zero, nor make a
    #: trade look like it risked less than it did.
    initial_risk: float = 0.0
    opened_at: Optional[datetime] = None
    account_login: int = 0
    account_type: str = ""

    # -- outcome ----------------------------------------------------------- #
    result: str = RESULT_OPEN
    exit_price: float = 0.0
    exit_at: Optional[datetime] = None
    tp_hits: int = 0
    closed_volume: float = 0.0
    #: Realised money from the partial closes so far, in account currency.
    gross_profit: float = 0.0
    estimated_cost: float = 0.0
    notes: str = ""
    #: Set once the broker-side stop has been moved to breakeven, so it is
    #: never moved twice and never moved when the strategy does not ask for it.
    breakeven_applied: bool = False

    @property
    def is_open(self) -> bool:
        return self.result == RESULT_OPEN or self.remaining_volume > 1e-9

    @property
    def risk_per_unit(self) -> float:
        """The risk R is measured against: fill to the ORIGINAL stop.

        Measured from the fill rather than the signal's entry, so slippage
        shows up in R instead of vanishing (spec section 13).  Frozen at entry,
        so a later breakeven move cannot rewrite the denominator.
        """
        if self.initial_risk > 0:
            return self.initial_risk
        return abs(self.fill_price - self.stop_loss)

    def targets(self) -> Tuple[float, float, float]:
        return (self.tp1, self.tp2, self.tp3)

    def to_row(
        self, point_value: float = 0.01, money_per_point: float = 1.0
    ) -> Dict[str, Any]:
        """One ``executions.csv`` row: signal, execution and outcome together.

        R is computed from the money actually made against the money one unit
        of risk represented **at the real fill**, so slippage shows up in R
        rather than being quietly absorbed.
        """
        holding = 0.0
        if self.opened_at and self.exit_at:
            holding = round((self.exit_at - self.opened_at).total_seconds(), 1)
        risk = self.risk_per_unit
        net = self.gross_profit - self.estimated_cost
        r_multiple = ""
        if risk > 0 and self.volume > 0 and point_value > 0:
            risk_money = risk / point_value * money_per_point * self.volume
            r_multiple = round(net / risk_money, 4) if risk_money > 0 else ""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "signal_timestamp": iso(self.signal_timestamp),
            "signal_entry": self.signal_entry,
            "sl": self.stop_loss,
            "tp1": self.tp1, "tp2": self.tp2, "tp3": self.tp3,
            "signal_score": self.signal_score,
            "threshold": self.threshold,
            "regime": self.regime,
            "session": self.session,
            "order_id": self.order_id,
            "broker_ticket": self.ticket,
            "broker_symbol": self.broker_symbol,
            "requested_price": self.requested_price,
            "actual_fill_price": self.fill_price,
            "spread_at_entry": (
                "" if self.spread_at_entry != self.spread_at_entry else self.spread_at_entry
            ),
            "slippage_points": round(self.slippage_points, 2),
            "slippage_price": round(self.slippage_points * point_value, 6),
            "position_size": self.volume,
            "requested_size": self.requested_volume,
            "execution_timestamp": iso(self.opened_at),
            "account_login": self.account_login,
            "account_type": self.account_type,
            "result": self.result,
            "exit_price": self.exit_price,
            "exit_timestamp": iso(self.exit_at),
            "holding_seconds": holding,
            "gross_profit": round(self.gross_profit, 2),
            "estimated_cost": round(self.estimated_cost, 2),
            "net_profit": round(net, 2),
            "R_multiple": r_multiple,
            "tp1_filled": int(self.tp_hits >= 1),
            "tp2_filled": int(self.tp_hits >= 2),
            "tp3_filled": int(self.tp_hits >= 3),
            "closed_volume": round(self.closed_volume, 4),
            "notes": self.notes,
        }

    def state(self) -> Dict[str, Any]:
        """Compact form persisted to ``state.json`` for restart recovery."""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "broker_symbol": self.broker_symbol,
            "direction": self.direction,
            "ticket": self.ticket,
            "initial_risk": self.initial_risk,
            "signal_entry": self.signal_entry,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1, "tp2": self.tp2, "tp3": self.tp3,
            "fill_price": self.fill_price,
            "requested_price": self.requested_price,
            "slippage_points": self.slippage_points,
            "volume": self.volume,
            "remaining_volume": self.remaining_volume,
            "opened_at": iso(self.opened_at),
            "signal_timestamp": iso(self.signal_timestamp),
            "tp_hits": self.tp_hits,
            "gross_profit": self.gross_profit,
            "estimated_cost": self.estimated_cost,
            "closed_volume": self.closed_volume,
            "result": self.result,
            "signal_score": self.signal_score,
            "threshold": self.threshold,
            "regime": self.regime,
            "session": self.session,
            "breakeven_applied": self.breakeven_applied,
            "account_login": self.account_login,
            "account_type": self.account_type,
        }

    @classmethod
    def from_state(cls, data: Dict[str, Any]) -> "DemoTrade":
        trade = cls(
            signal_id=str(data.get("signal_id", "")),
            symbol=str(data.get("symbol", "")),
            broker_symbol=str(data.get("broker_symbol", "")),
            direction=str(data.get("direction", "")).upper(),
        )
        for key in (
            "signal_entry", "stop_loss", "tp1", "tp2", "tp3", "fill_price",
            "requested_price", "slippage_points", "volume", "remaining_volume",
            "gross_profit", "estimated_cost", "closed_volume", "signal_score",
            "threshold", "initial_risk",
        ):
            try:
                setattr(trade, key, float(data.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                setattr(trade, key, 0.0)
        trade.ticket = int(data.get("ticket", 0) or 0)
        trade.tp_hits = int(data.get("tp_hits", 0) or 0)
        trade.result = str(data.get("result", RESULT_OPEN) or RESULT_OPEN)
        trade.regime = str(data.get("regime", ""))
        trade.session = str(data.get("session", ""))
        trade.breakeven_applied = bool(data.get("breakeven_applied", False))
        trade.account_login = int(data.get("account_login", 0) or 0)
        trade.account_type = str(data.get("account_type", ""))
        trade.opened_at = parse_iso(str(data.get("opened_at", "")))
        trade.signal_timestamp = parse_iso(str(data.get("signal_timestamp", "")))
        return trade


# --------------------------------------------------------------------------- #
# manager
# --------------------------------------------------------------------------- #
class ExecutionManager:
    """Executes and manages demo positions for ONE market.

    One manager per market, exactly like the paper trackers, so a gold trade and
    a Bitcoin trade can never see each other's state, tickets or limits
    (spec section 21).  The broker connection is shared; nothing else is.
    """

    def __init__(
        self,
        config,
        settings: ExecutionSettings,
        broker,
        notifier=None,
        store=None,
    ) -> None:
        self.config = config
        self.symbol = config.symbol
        self.market = get_market(config.symbol)
        self.settings = settings
        self.broker = broker
        self.notifier = notifier
        self.store = store
        self._lock = threading.RLock()

        self.trades: Dict[str, DemoTrade] = {}
        #: Signal ids already acted upon - executed, refused or errored.  The
        #: persistence of this set is what stops a restart re-executing a
        #: signal that is already live (spec section 8).
        self.processed: set = set()
        self.account: AccountInfo = AccountInfo(verified=False)
        #: Set when reconciliation fails; blocks new orders until cleared.
        self.halted: bool = False
        self.halt_reason: str = ""
        self._day: date = now_utc().date()
        self._trades_today: int = 0
        self._realised_today: float = 0.0

        ensure_csv(config.executions_csv, EXECUTION_COLUMNS)
        self._load_state()

    # -- persistence -------------------------------------------------------- #
    def _load_state(self) -> None:
        """Restore processed ids and open trades from this market's state file."""
        if self.store is None:
            return
        section = dict(self.store.get("execution") or {})
        self.processed = set(section.get("processed_signal_ids") or [])
        for raw in section.get("open_trades") or []:
            try:
                trade = DemoTrade.from_state(raw)
            except Exception as exc:  # noqa: BLE001 - a bad row must not stop start-up
                LOGGER.warning("[%s] skipping unreadable stored trade: %s", self.symbol, exc)
                continue
            if trade.signal_id:
                self.trades[trade.signal_id] = trade
        saved_day = parse_iso(str(section.get("day") or ""))
        if saved_day is not None and saved_day.date() == self._day:
            self._trades_today = int(section.get("trades_today", 0) or 0)
            try:
                self._realised_today = float(section.get("realised_today", 0.0) or 0.0)
            except (TypeError, ValueError):
                self._realised_today = 0.0
        if self.trades:
            LOGGER.info(
                "[%s] restored %d open demo trade(s) from state", self.symbol, len(self.trades)
            )

    def _persist(self) -> None:
        if self.store is None:
            return
        with self._lock:
            self.store.update_section(
                "execution",
                {
                    # Bounded: the newest ids are the ones that can still be
                    # replayed, and an unbounded set would grow forever.
                    "processed_signal_ids": sorted(self.processed)[-500:],
                    "open_trades": [t.state() for t in self.trades.values() if t.is_open],
                    "day": iso(now_utc()),
                    "trades_today": self._trades_today,
                    "realised_today": round(self._realised_today, 2),
                },
            )

    # -- day rollover ------------------------------------------------------- #
    def _roll_day(self) -> None:
        today = now_utc().date()
        if today != self._day:
            self._day = today
            self._trades_today = 0
            self._realised_today = 0.0

    # -- public state ------------------------------------------------------- #
    def open_trades(self) -> List[DemoTrade]:
        with self._lock:
            return [t for t in self.trades.values() if t.is_open]

    def trades_today(self) -> int:
        self._roll_day()
        return self._trades_today

    def realised_today(self) -> float:
        self._roll_day()
        return round(self._realised_today, 2)

    def already_processed(self, signal_id: str) -> bool:
        with self._lock:
            return str(signal_id) in self.processed

    # -- gates -------------------------------------------------------------- #
    def verify_account(self) -> AccountInfo:
        """Ask the platform what kind of account this is.  Never assumes.

        Refreshed on every execution rather than cached at start-up: a terminal
        can be re-pointed at a different account while the process runs, and a
        stale "yes, demo" would be the worst possible thing to cache.
        """
        info = self.broker.account()
        self.account = info
        return info

    def _guard(self, signal, quote: Optional[Quote]) -> str:
        """Every reason this signal must not be executed, or ``""``.

        Ordered cheapest-first, and deliberately exhaustive: each branch names
        the specific thing that failed so the Telegram alert is actionable.
        """
        settings = self.settings

        blocking = settings.blocking_reason()
        if blocking:
            return blocking
        if self.halted:
            return f"execution halted: {self.halt_reason}"

        # -- account: the gate with no optimistic branch -------------------- #
        account = self.verify_account()
        if not account.verified:
            return "account type could not be verified"
        if not account.is_demo:
            return f"account is {account.trade_mode or 'not demo'} - refusing to trade"

        # -- market and symbol ---------------------------------------------- #
        broker_symbol = self.market.feed_symbol()
        if not self.broker.symbol_exists(broker_symbol):
            return f"symbol '{broker_symbol}' does not exist on this account"
        if quote is None:
            return f"no quote available for '{broker_symbol}'"

        # -- spread ---------------------------------------------------------- #
        limit = settings.max_spread_points or float(
            getattr(self.config, "max_spread_points", 0.0) or 0.0
        )
        spread = quote.spread_points
        if limit > 0 and spread == spread and spread > limit:
            return f"spread {spread:.0f} points exceeds the {limit:.0f} limit"

        # -- levels ---------------------------------------------------------- #
        level_problem = self._check_levels(signal, quote)
        if level_problem:
            return level_problem

        # -- limits ---------------------------------------------------------- #
        self._roll_day()
        if len(self.open_trades()) >= settings.max_open_positions:
            return f"already at the {settings.max_open_positions} open-position limit"
        if self._trades_today >= settings.max_trades_per_day:
            return f"already at the {settings.max_trades_per_day} trades-per-day limit"
        if settings.max_daily_loss > 0 and self._realised_today <= -abs(settings.max_daily_loss):
            return (
                f"daily loss limit reached ({self._realised_today:.2f} vs "
                f"-{abs(settings.max_daily_loss):.2f})"
            )
        return ""

    def _check_levels(self, signal, quote: Quote) -> str:
        """Sanity-check the signal's own stop and targets against the quote.

        This does NOT re-derive levels - it only refuses ones that cannot be
        traded: a stop on the wrong side of price, or a distance so large it
        indicates a bad quote rather than a real setup.
        """
        direction = str(signal.direction).upper()
        if direction not in (BUY, SELL):
            return f"unknown direction '{signal.direction}'"

        entry = quote.entry_price(direction)
        if entry <= 0:
            return "quote has no usable entry price"

        stop = float(signal.stop_loss)
        targets = [float(signal.tp1), float(signal.tp2), float(signal.tp3)]
        if stop <= 0 or any(t <= 0 for t in targets):
            return "signal carries a non-positive stop or target"

        if direction == BUY:
            if stop >= entry:
                return f"BUY stop {stop} is not below the entry {entry}"
            if any(t <= entry for t in targets):
                return "a BUY target is not above the entry"
            if not targets[0] < targets[1] < targets[2]:
                return "BUY targets are not in ascending order"
        else:
            if stop <= entry:
                return f"SELL stop {stop} is not above the entry {entry}"
            if any(t >= entry for t in targets):
                return "a SELL target is not below the entry"
            if not targets[0] > targets[1] > targets[2]:
                return "SELL targets are not in descending order"

        sl_fraction = abs(entry - stop) / entry
        if self.settings.max_sl_distance_pct > 0 and sl_fraction > self.settings.max_sl_distance_pct:
            return (
                f"stop is {sl_fraction:.2%} away, beyond the "
                f"{self.settings.max_sl_distance_pct:.2%} limit"
            )
        tp_fraction = abs(targets[2] - entry) / entry
        if self.settings.max_tp_distance_pct > 0 and tp_fraction > self.settings.max_tp_distance_pct:
            return (
                f"TP3 is {tp_fraction:.2%} away, beyond the "
                f"{self.settings.max_tp_distance_pct:.2%} limit"
            )
        return ""

    # -- sizing ------------------------------------------------------------- #
    def size_for(self, signal, entry_price: float) -> SizingResult:
        """Lots for this signal, from the signal's OWN stop distance."""
        risk = self.settings.risk
        risk.point_value_per_lot = float(
            getattr(self.config, "money_per_point_per_lot", risk.point_value_per_lot)
        )
        distance = abs(float(entry_price) - float(signal.stop_loss))
        result = risk.size_for(distance, float(self.config.point_value))
        if result.approved_lots > self.settings.max_order_lots:
            result = SizingResult(
                requested_lots=result.requested_lots,
                approved_lots=self.settings.max_order_lots,
                risk_amount=result.risk_amount,
                stop_distance_price=result.stop_distance_price,
                stop_points=result.stop_points,
                reason=f"clamped to MAX_ORDER_LOTS {self.settings.max_order_lots}",
            )
        return result

    # -- execution ---------------------------------------------------------- #
    def execute_signal(self, signal) -> ExecutionDecision:
        """Run every gate, then place and register one demo order.

        Returns a decision rather than raising for ordinary refusals; a refusal
        is a normal outcome that gets logged and surfaced.  The only exception
        is a live account, which raises :class:`DemoExecutionBlocked` so it can
        never be mistaken for a routine skip.
        """
        signal_id = str(getattr(signal, "signal_id", "") or "")
        if not signal_id:
            return ExecutionDecision(False, "signal has no id", blocked=True)

        with self._lock:
            # Duplicate prevention comes first: it must hold even when every
            # other gate would have refused anyway.
            if signal_id in self.processed:
                LOGGER.info("[%s] signal %s already processed - not re-executing",
                            self.symbol, signal_id)
                return ExecutionDecision(False, "signal already processed")

        broker_symbol = self.market.feed_symbol()
        quote = self.broker.quote(broker_symbol)

        reason = self._guard(signal, quote)
        if reason:
            # A live account is a category apart: mark the signal processed so
            # a retry loop cannot hammer it, and raise.
            if "refusing to trade" in reason or "could not be verified" in reason:
                self._mark_processed(signal_id)
                self._alert_blocked(reason, signal)
                raise DemoExecutionBlocked(reason)
            LOGGER.warning("[%s] not executing %s: %s", self.symbol, signal_id, reason)
            self._notify(f"⚠️ DEMO TRADE SKIPPED\n\n{self.symbol} {signal.direction}\n\n{reason}")
            return ExecutionDecision(False, reason, blocked=True)

        assert quote is not None  # guaranteed by _guard
        direction = str(signal.direction).upper()
        requested_price = quote.entry_price(direction)

        sizing = self.size_for(signal, requested_price)
        LOGGER.info("[%s] sizing %s: %s", self.symbol, signal_id, sizing.describe())
        if not sizing.ok:
            self._mark_processed(signal_id)
            message = f"position size rejected - {sizing.describe()}"
            self._notify(f"⚠️ DEMO TRADE SKIPPED\n\n{self.symbol} {direction}\n\n{message}")
            return ExecutionDecision(False, message, blocked=True)

        request = OrderRequest(
            symbol=self.symbol,
            broker_symbol=broker_symbol,
            direction=direction,
            volume=sizing.approved_lots,
            requested_price=requested_price,
            stop_loss=float(signal.stop_loss),
            # The broker holds the FINAL target; TP1/TP2 are partial closes this
            # manager performs, because MT5 has one TP per position.
            take_profit=float(signal.tp3),
            max_slippage_points=self.settings.max_slippage_points,
            comment=self.settings.order_comment,
            magic=self.settings.magic_number,
            signal_id=signal_id,
        )

        # Mark BEFORE sending: if the send raises or the reply is lost, the
        # signal must never be retried blindly (spec section 24).
        self._mark_processed(signal_id)
        result = self.broker.send_order(request)

        if result.indeterminate:
            self._halt(
                f"order outcome unknown for {signal_id} ({result.comment}) - "
                "reconcile before trading again"
            )
            return ExecutionDecision(False, "order outcome unknown", blocked=True)

        if not result.accepted:
            message = f"broker rejected the order: {result.comment or result.retcode}"
            LOGGER.error("[%s] %s", self.symbol, message)
            self._notify(f"🚨 DEMO ORDER REJECTED\n\n{self.symbol} {direction}\n\n{message}")
            return ExecutionDecision(False, message, blocked=True)

        trade = self._register(signal, request, result, quote)
        self._verify_stops(trade)
        self._notify(self._format_opened(trade))
        return ExecutionDecision(True, "", trade=trade)

    def _register(self, signal, request: OrderRequest, result, quote: Quote) -> DemoTrade:
        """Record an accepted fill, including what it actually cost."""
        filled = result.volume or request.volume
        slippage_price = result.fill_price - request.requested_price
        if request.direction == SELL:
            slippage_price = -slippage_price
        slippage_points = slippage_price / float(self.config.point_value)

        trade = DemoTrade(
            signal_id=request.signal_id,
            symbol=self.symbol,
            broker_symbol=request.broker_symbol,
            direction=request.direction,
            timeframe=str(getattr(signal, "timeframe", "M1")),
            signal_timestamp=getattr(signal, "timestamp", None),
            signal_entry=float(getattr(signal, "entry", 0.0)),
            stop_loss=float(signal.stop_loss),
            tp1=float(signal.tp1), tp2=float(signal.tp2), tp3=float(signal.tp3),
            signal_score=float(getattr(signal, "confidence", 0.0) or 0.0),
            threshold=float(getattr(signal, "threshold_used", 0.0) or 0.0),
            regime=str(getattr(signal, "regime", "")),
            session=str(getattr(signal, "session", "")),
            ticket=result.ticket,
            order_id=result.ticket,
            requested_price=request.requested_price,
            fill_price=result.fill_price or request.requested_price,
            spread_at_entry=quote.spread_points,
            slippage_points=slippage_points,
            volume=filled,
            requested_volume=request.volume,
            remaining_volume=filled,
            initial_risk=abs((result.fill_price or request.requested_price)
                             - float(signal.stop_loss)),
            opened_at=now_utc(),
            account_login=self.account.login,
            account_type=self.account.trade_mode,
        )
        if result.partial:
            trade.notes = (
                f"partial fill: {filled} of {request.volume} lots"
            )
            LOGGER.warning("[%s] %s", self.symbol, trade.notes)

        with self._lock:
            self.trades[trade.signal_id] = trade
            self._trades_today += 1
        self._persist()
        LOGGER.info(
            "[%s] DEMO %s %s ticket %s | requested %.5f filled %.5f (%.1f pts slip) | %.2f lots",
            self.symbol, trade.direction, trade.broker_symbol, trade.ticket,
            trade.requested_price, trade.fill_price, trade.slippage_points, trade.volume,
        )
        return trade

    def _verify_stops(self, trade: DemoTrade) -> None:
        """Confirm the broker actually holds an SL and a TP; attach if not.

        An order accepted without stops is the dangerous case: the position
        exists with no protection, so this re-reads the position rather than
        trusting the request.
        """
        positions = [p for p in self.broker.positions() if p.ticket == trade.ticket]
        if not positions:
            trade.notes = (trade.notes + "; " if trade.notes else "") + (
                "could not read back the position to verify stops"
            )
            LOGGER.error("[%s] %s", self.symbol, trade.notes)
            self._notify(
                f"🚨 DEMO STOPS UNVERIFIED\n\n{self.symbol} ticket {trade.ticket}\n\n"
                "The position could not be read back. Check the terminal."
            )
            return
        position = positions[0]
        missing_sl = not position.stop_loss
        missing_tp = not position.take_profit
        if missing_sl or missing_tp:
            LOGGER.warning(
                "[%s] ticket %s is missing %s - attaching",
                self.symbol, trade.ticket,
                " and ".join(n for n, m in (("SL", missing_sl), ("TP", missing_tp)) if m),
            )
            ok = self.broker.modify_position(
                trade.ticket,
                stop_loss=trade.stop_loss if missing_sl else None,
                take_profit=trade.tp3 if missing_tp else None,
            )
            if not ok:
                trade.notes = (trade.notes + "; " if trade.notes else "") + "stops not attached"
                self._notify(
                    f"🚨 DEMO STOPS MISSING\n\n{self.symbol} ticket {trade.ticket}\n\n"
                    "SL/TP could not be attached. Check the terminal."
                )

    def _mark_processed(self, signal_id: str) -> None:
        with self._lock:
            self.processed.add(str(signal_id))
        self._persist()

    # -- management --------------------------------------------------------- #
    def manage(self, quote: Optional[Quote] = None, when: Optional[datetime] = None) -> None:
        """Advance every open trade: partial targets, stop, timeout.

        Runs regardless of which market is selected in Telegram and regardless
        of whether the engine is paused - an open position must always be
        managed to its exit (spec sections 11, 12, 21).
        """
        open_trades = self.open_trades()
        if not open_trades:
            return
        if quote is None:
            quote = self.broker.quote(self.market.feed_symbol())
        if quote is None:
            LOGGER.debug("[%s] no quote - cannot manage %d trade(s)",
                         self.symbol, len(open_trades))
            return
        moment = when or now_utc()
        for trade in open_trades:
            try:
                self._manage_one(trade, quote, moment)
            except Exception as exc:  # noqa: BLE001 - one trade must not stop the rest
                LOGGER.exception("[%s] managing %s failed: %s", self.symbol, trade.signal_id, exc)

    def _manage_one(self, trade: DemoTrade, quote: Quote, moment: datetime) -> None:
        price = quote.exit_price(trade.direction)
        if price <= 0:
            return
        long = trade.direction == BUY

        # 1. stop first, and pessimistically: when a single observation could be
        #    read as either the stop or a target, the stop wins.  Anything else
        #    would flatter the result.
        stop_hit = price <= trade.stop_loss if long else price >= trade.stop_loss
        if stop_hit:
            self._close(trade, price, RESULT_SL, moment, volume=trade.remaining_volume)
            return

        # 2. timeout, using the strategy's OWN holding window
        if self._timed_out(trade, moment):
            self._close(trade, price, RESULT_TIMEOUT, moment, volume=trade.remaining_volume)
            return

        # 3. the partial-target ladder
        for index, target in enumerate(trade.targets()):
            if trade.tp_hits > index:
                continue
            reached = price >= target if long else price <= target
            if not reached:
                break
            self._take_partial(trade, price, index, moment)
            if not trade.is_open:
                return

    def _timed_out(self, trade: DemoTrade, moment: datetime) -> bool:
        """True once the position has outlived the configured holding window.

        Uses the existing scalping timeout: M1 candles, so the window is
        ``max_holding_candles`` minutes.  No new timeout concept is introduced.
        """
        if trade.opened_at is None:
            return False
        limit_minutes = int(getattr(self.config, "max_holding_candles", 15) or 15)
        elapsed = (moment - trade.opened_at).total_seconds()
        return elapsed >= limit_minutes * 60

    def _take_partial(
        self, trade: DemoTrade, price: float, index: int, moment: datetime
    ) -> None:
        """Close the configured fraction at TP``index+1``."""
        fractions = self.settings.tp_fractions
        is_final = index >= len(fractions) - 1
        volume = (
            trade.remaining_volume if is_final
            else self._grid(trade.volume * fractions[index])
        )
        volume = min(volume, trade.remaining_volume)
        if volume <= 0:
            # The position is too small to split; carry it to the final target
            # rather than emitting a zero-volume order.
            trade.tp_hits = index + 1
            return

        label = (RESULT_TP1, RESULT_TP2, RESULT_TP3)[min(index, 2)]
        self._close(trade, price, label, moment, volume=volume, partial=not is_final)

        if not is_final and trade.is_open:
            self._maybe_move_to_breakeven(trade, index)

    def _maybe_move_to_breakeven(self, trade: DemoTrade, index: int) -> None:
        """Move the stop to the fill after TP1 - only if the strategy does.

        ``move_sl_to_breakeven_after_tp1`` is the paper tracker's existing
        setting.  Execution follows it rather than deciding for itself, so demo
        results stay comparable with paper results; when it is off, nothing here
        introduces the behaviour (spec section 11).
        """
        if index != 0 or trade.breakeven_applied:
            return
        if not getattr(self.config, "move_sl_to_breakeven_after_tp1", False):
            return
        if self.broker.modify_position(trade.ticket, stop_loss=trade.fill_price):
            trade.stop_loss = trade.fill_price
            trade.breakeven_applied = True
            LOGGER.info("[%s] ticket %s stop moved to breakeven %.5f",
                        self.symbol, trade.ticket, trade.fill_price)

    def _grid(self, volume: float) -> float:
        """Floor a volume onto the broker's lot grid."""
        step = max(float(self.settings.risk.lot_step), 1e-9)
        return round(math.floor(float(volume) / step + 1e-9) * step, 8)

    def _close(
        self, trade: DemoTrade, price: float, result: str, moment: datetime,
        volume: float, partial: bool = False,
    ) -> None:
        """Close ``volume`` of ``trade`` and record the money it made or lost."""
        volume = min(float(volume), trade.remaining_volume)
        if volume <= 0:
            return
        reply = self.broker.close_position(trade.ticket, volume=volume, comment=result)
        if not reply.accepted:
            LOGGER.error(
                "[%s] could not close %s of ticket %s: %s",
                self.symbol, volume, trade.ticket, reply.comment,
            )
            trade.notes = (trade.notes + "; " if trade.notes else "") + (
                f"{result} close failed: {reply.comment}"
            )
            return

        exit_price = reply.fill_price or price
        direction = 1.0 if trade.direction == BUY else -1.0
        move = (exit_price - trade.fill_price) * direction
        money = move / float(self.config.point_value) * self._money_per_point() * volume

        trade.gross_profit += money
        trade.estimated_cost += self._cost_for(volume)
        trade.closed_volume = round(trade.closed_volume + volume, 8)
        trade.remaining_volume = round(trade.remaining_volume - volume, 8)
        trade.exit_price = exit_price
        trade.exit_at = moment
        if result in (RESULT_TP1, RESULT_TP2, RESULT_TP3):
            trade.tp_hits = max(trade.tp_hits, (RESULT_TP1, RESULT_TP2, RESULT_TP3).index(result) + 1)

        if partial and trade.remaining_volume > 1e-9:
            trade.result = RESULT_OPEN
            LOGGER.info(
                "[%s] %s on ticket %s: closed %.2f lots at %.5f, %.2f remaining",
                self.symbol, result, trade.ticket, volume, exit_price, trade.remaining_volume,
            )
            self._persist()
            self._notify(self._format_partial(trade, result, exit_price, volume))
            return

        trade.result = result
        trade.remaining_volume = 0.0
        with self._lock:
            self._realised_today += trade.gross_profit - trade.estimated_cost
            self.trades.pop(trade.signal_id, None)
        append_csv(
            self.config.executions_csv,
            trade.to_row(self.config.point_value, self._money_per_point()),
            EXECUTION_COLUMNS,
        )
        self._persist()
        LOGGER.info(
            "[%s] %s ticket %s | net %.2f | held %.0fs",
            self.symbol, result, trade.ticket,
            trade.gross_profit - trade.estimated_cost,
            (moment - trade.opened_at).total_seconds() if trade.opened_at else 0.0,
        )
        self._notify(self._format_closed(trade))

    def _money_per_point(self) -> float:
        return float(
            getattr(self.config, "money_per_point_per_lot",
                    self.settings.risk.point_value_per_lot)
        )

    def _cost_for(self, volume: float) -> float:
        """Modelled round-trip cost for ``volume`` lots.

        Uses the market's OWN cost assumptions - the same ones the paper layer
        charges - so demo net R and paper net R are computed on the same basis.
        """
        points = float(self.config.round_trip_cost(None)) / float(self.config.point_value)
        return points * self._money_per_point() * float(volume)

    # -- reconciliation ----------------------------------------------------- #
    def reconcile(self) -> bool:
        """Match stored trades against the broker's actual open positions.

        Called at start-up and after any reconnection.  Three outcomes matter:

        * a stored trade whose position is gone closed while we were away - it
          is recorded as closed rather than tracked forever;
        * a position with our magic number that we have no record of is an
          orphan - we adopt it so it is still managed rather than abandoned;
        * a failure to read positions at all halts execution, because opening
          new trades on an unknown book is how duplicates happen.
        """
        try:
            positions = self.broker.positions(magic=self.settings.magic_number)
        except Exception as exc:  # noqa: BLE001
            self._halt(f"could not read open positions: {exc}")
            return False

        broker_symbol = self.market.feed_symbol()
        mine = {p.ticket: p for p in positions if p.symbol == broker_symbol}
        stored = self.open_trades()

        for trade in stored:
            if trade.ticket in mine:
                self._adopt_levels(trade, mine.pop(trade.ticket))
                continue
            # Gone from the broker: it closed while we were not looking.  The
            # exact exit is unknown, so it is recorded honestly as such.
            trade.result = RESULT_TIMEOUT if trade.tp_hits == 0 else RESULT_TP1
            trade.notes = (trade.notes + "; " if trade.notes else "") + (
                "closed while the bot was offline; exit price not observed"
            )
            trade.exit_at = now_utc()
            trade.exit_price = trade.exit_price or trade.fill_price
            trade.remaining_volume = 0.0
            with self._lock:
                self.trades.pop(trade.signal_id, None)
            append_csv(
                self.config.executions_csv,
                trade.to_row(self.config.point_value, self._money_per_point()),
                EXECUTION_COLUMNS,
            )
            LOGGER.warning("[%s] ticket %s vanished while offline - recorded",
                           self.symbol, trade.ticket)

        for orphan in mine.values():
            LOGGER.warning(
                "[%s] adopting untracked demo position ticket %s (%.2f lots)",
                self.symbol, orphan.ticket, orphan.volume,
            )
            self.trades[f"adopted-{orphan.ticket}"] = self._adopted(orphan)

        self._persist()
        if self.halted:
            self.halted = False
            self.halt_reason = ""
            LOGGER.info("[%s] reconciliation succeeded - execution re-armed", self.symbol)
        return True

    def _adopt_levels(self, trade: DemoTrade, position: BrokerPosition) -> None:
        """Refresh a restored trade from what the broker actually holds."""
        trade.remaining_volume = position.volume
        if position.stop_loss:
            trade.stop_loss = position.stop_loss
        if position.take_profit:
            trade.tp3 = position.take_profit

    def _adopted(self, position: BrokerPosition) -> DemoTrade:
        """Wrap an orphaned broker position so it is at least managed."""
        return DemoTrade(
            signal_id=f"adopted-{position.ticket}",
            symbol=self.symbol,
            broker_symbol=position.symbol,
            direction=position.direction,
            ticket=position.ticket,
            order_id=position.ticket,
            fill_price=position.open_price,
            requested_price=position.open_price,
            stop_loss=position.stop_loss,
            tp1=position.take_profit or position.open_price,
            tp2=position.take_profit or position.open_price,
            tp3=position.take_profit or position.open_price,
            volume=position.volume,
            requested_volume=position.volume,
            remaining_volume=position.volume,
            initial_risk=abs(position.open_price - position.stop_loss)
            if position.stop_loss else 0.0,
            opened_at=position.open_time or now_utc(),
            notes="adopted at reconciliation; no originating signal",
        )

    def _halt(self, reason: str) -> None:
        """Stop opening new trades.  Existing ones keep being managed."""
        self.halted = True
        self.halt_reason = reason
        LOGGER.error("[%s] DEMO EXECUTION HALTED: %s", self.symbol, reason)
        self._notify(
            f"🚨 DEMO EXECUTION HALTED\n\n{self.symbol}\n\n{reason}\n\n"
            "No new demo orders will be placed until this is resolved."
        )

    # -- notifications ------------------------------------------------------ #
    def _notify(self, text: str) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.send_text(text)
        except Exception as exc:  # noqa: BLE001 - Telegram must never break trading
            LOGGER.debug("[%s] telegram notify failed: %s", self.symbol, exc)

    def _alert_blocked(self, reason: str, signal=None) -> None:
        direction = str(getattr(signal, "direction", "")) if signal else ""
        self._notify(
            "🚨 DEMO EXECUTION BLOCKED\n\n"
            f"{self.symbol} {direction}\n\n"
            f"{reason}\n\n"
            "No order was placed."
        )

    def _format_opened(self, trade: DemoTrade) -> str:
        digits = int(self.config.digits)
        return "\n".join([
            "🤖 DEMO TRADE OPENED",
            "",
            f"{trade.symbol} {trade.timeframe}",
            trade.direction,
            "",
            f"Signal Entry: {trade.signal_entry:.{digits}f}",
            f"Actual Fill:  {trade.fill_price:.{digits}f}",
            f"Slippage:     {trade.slippage_points:+.1f} points",
            "",
            f"SL:  {trade.stop_loss:.{digits}f}",
            f"TP1: {trade.tp1:.{digits}f}",
            f"TP2: {trade.tp2:.{digits}f}",
            f"TP3: {trade.tp3:.{digits}f}",
            "",
            f"Position Size: {trade.volume:.2f} lots",
            f"Order ID: {trade.ticket}",
            "",
            "Status:",
            "🟢 OPEN",
            "",
            "DEMO ACCOUNT - no real money.",
        ])

    def _format_partial(
        self, trade: DemoTrade, result: str, price: float, volume: float
    ) -> str:
        digits = int(self.config.digits)
        return "\n".join([
            f"🎯 DEMO {result.replace('_', ' ')}",
            "",
            f"{trade.symbol} {trade.direction}",
            f"Closed {volume:.2f} of {trade.volume:.2f} lots at {price:.{digits}f}",
            f"Remaining: {trade.remaining_volume:.2f} lots",
            f"Running net: {trade.gross_profit - trade.estimated_cost:+.2f}",
        ])

    def _format_closed(self, trade: DemoTrade) -> str:
        digits = int(self.config.digits)
        row = trade.to_row(self.config.point_value, self._money_per_point())
        icons = {
            RESULT_TP1: "✅ TP1", RESULT_TP2: "✅ TP2", RESULT_TP3: "✅ TP3",
            RESULT_SL: "❌ SL", RESULT_TIMEOUT: "⌛ TIMEOUT",
        }
        r_text = row["R_multiple"]
        return "\n".join([
            "━━━━━━━━━━━━━━━━━━",
            "📊 DEMO TRADE CLOSED",
            "━━━━━━━━━━━━━━━━━━",
            "",
            f"{trade.symbol} {trade.timeframe}",
            trade.direction,
            "",
            f"Entry: {trade.fill_price:.{digits}f}",
            f"Exit:  {trade.exit_price:.{digits}f}",
            "",
            "Result:",
            icons.get(trade.result, trade.result),
            "",
            "Holding Time:",
            f"{row['holding_seconds']:.0f} seconds",
            "",
            f"Gross: {trade.gross_profit:+.2f}",
            f"Costs: -{trade.estimated_cost:.2f}",
            f"Net:   {trade.gross_profit - trade.estimated_cost:+.2f}",
            "",
            f"R: {r_text if r_text != '' else 'n/a'}",
            f"Slippage: {trade.slippage_points:+.1f} points",
            "",
            "━━━━━━━━━━━━━━━━━━",
            "DEMO ACCOUNT - no real money.",
        ])

    # -- reporting ---------------------------------------------------------- #
    def describe_open(self) -> List[str]:
        """Human-readable lines for the OPEN TRADES panel."""
        digits = int(self.config.digits)
        quote = self.broker.quote(self.market.feed_symbol())
        lines: List[str] = []
        now = now_utc()
        for trade in self.open_trades():
            current = quote.exit_price(trade.direction) if quote else trade.fill_price
            age = (now - trade.opened_at).total_seconds() if trade.opened_at else 0.0
            lines += [
                f"{trade.symbol}",
                trade.direction,
                f"Entry: {trade.fill_price:.{digits}f}",
                f"Current: {current:.{digits}f}",
                f"SL: {trade.stop_loss:.{digits}f}",
                f"TP1: {trade.tp1:.{digits}f}",
                f"TP2: {trade.tp2:.{digits}f}",
                f"TP3: {trade.tp3:.{digits}f}",
                f"Size: {trade.remaining_volume:.2f} lots",
                f"Time Open: {age:.0f} sec",
                "",
            ]
        return lines


def load_executions(path) -> List[Dict[str, str]]:
    """Read an ``executions.csv``.  Returns ``[]`` when it does not exist."""
    return read_csv_rows(path)
