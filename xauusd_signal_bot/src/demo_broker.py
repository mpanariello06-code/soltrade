"""Broker port for demo execution, plus the MT5 adapter that implements it.

WHY A PORT
----------
The execution manager talks to this interface, never to MetaTrader5 directly.
That buys three things:

1. every test runs against a scripted fake, so the suite can never reach a real
   account (spec section 26);
2. the MT5-specific request shapes stay in one file;
3. the manager's safety logic is testable independently of a Windows-only
   package that is not importable on the machine this is developed on.

WHAT THIS LAYER DOES NOT DO
---------------------------
It makes no decisions.  It does not size positions, does not check limits and
does not decide whether an account may be traded.  It converts a validated
request into a platform call and reports what happened.  Every judgement lives
in :mod:`src.demo_execution`, so "can this send an order?" has one answer in one
place.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from .execution_config import AccountInfo, classify_account
from .logger import get_logger
from .utils import now_utc

LOGGER = get_logger("execution.broker")

BUY = "BUY"
SELL = "SELL"


# --------------------------------------------------------------------------- #
# data shapes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Quote:
    """A current two-sided price."""

    bid: float
    ask: float
    spread_points: float = float("nan")
    time: Optional[datetime] = None

    def entry_price(self, direction: str) -> float:
        """The side a market order of ``direction`` actually crosses.

        A BUY lifts the ASK and a SELL hits the BID.  Using the candle close for
        either would understate the cost of every scalp by half the spread,
        which at a three-pip target is not a rounding error (spec section 9).
        """
        return self.ask if str(direction).upper() == BUY else self.bid

    def exit_price(self, direction: str) -> float:
        """The side that closes a position of ``direction`` - the other one."""
        return self.bid if str(direction).upper() == BUY else self.ask


@dataclass(frozen=True)
class OrderRequest:
    """A fully validated intent to open a demo position.

    Reaching this type means every check in the execution manager has already
    passed; the broker adapter's job is only to transmit it.
    """

    symbol: str                 #: canonical market symbol
    broker_symbol: str          #: what the platform is actually asked for
    direction: str
    volume: float
    requested_price: float
    stop_loss: float
    take_profit: float          #: the FINAL target the broker holds (TP3)
    max_slippage_points: float
    comment: str = ""
    magic: int = 0
    signal_id: str = ""


@dataclass(frozen=True)
class OrderResult:
    """What the platform did with an :class:`OrderRequest`."""

    accepted: bool
    ticket: int = 0
    fill_price: float = 0.0
    volume: float = 0.0
    retcode: int = 0
    comment: str = ""
    #: True when the broker filled less than the requested volume.
    partial: bool = False
    requested_volume: float = 0.0
    #: Set when the adapter could not determine the outcome.  The manager must
    #: reconcile rather than retry - see spec section 24.
    indeterminate: bool = False

    @property
    def failed(self) -> bool:
        return not self.accepted


@dataclass(frozen=True)
class BrokerPosition:
    """An open position as the platform reports it."""

    ticket: int
    symbol: str                 #: the BROKER's symbol name
    direction: str
    volume: float
    open_price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    profit: float = 0.0
    magic: int = 0
    comment: str = ""
    open_time: Optional[datetime] = None


class DemoBroker(Protocol):
    """What the execution manager needs from a trading platform."""

    def connect(self) -> bool: ...
    def shutdown(self) -> None: ...
    def account(self) -> AccountInfo: ...
    def symbol_exists(self, broker_symbol: str) -> bool: ...
    def quote(self, broker_symbol: str) -> Optional[Quote]: ...
    def send_order(self, request: OrderRequest) -> OrderResult: ...
    def positions(self, magic: Optional[int] = None) -> List[BrokerPosition]: ...
    def close_position(
        self, ticket: int, volume: Optional[float] = None, comment: str = ""
    ) -> OrderResult: ...
    def modify_position(
        self, ticket: int, stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool: ...


# --------------------------------------------------------------------------- #
# MT5 adapter
# --------------------------------------------------------------------------- #
class MT5DemoBroker:
    """:class:`DemoBroker` backed by MetaTrader 5.

    Connects with the **dedicated demo credentials**, not the data feed's.
    Reusing the feed's login would mean that pointing the feed at a live
    account silently armed execution against it, which is precisely the failure
    this separation exists to prevent.
    """

    #: MT5 order-send return code meaning "done".
    RETCODE_DONE = 10009
    RETCODE_DONE_PARTIAL = 10010

    def __init__(self, settings, mt5_module=None) -> None:
        self.settings = settings
        self._mt5 = mt5_module
        self._lock = threading.RLock()
        self.connected = False

    # -- module access ------------------------------------------------------ #
    @property
    def mt5(self):
        """The MetaTrader5 module, imported lazily.

        Lazy because the package is Windows-only: importing at module scope
        would make this file unimportable - and therefore untestable - anywhere
        else.
        """
        if self._mt5 is None:
            from . import market_data

            self._mt5 = market_data.mt5
        return self._mt5

    # -- lifecycle ---------------------------------------------------------- #
    def connect(self) -> bool:
        """Initialise a terminal session against the DEMO credentials."""
        mt5 = self.mt5
        if mt5 is None:
            LOGGER.error("MetaTrader5 is unavailable - demo execution cannot connect")
            self.connected = False
            return False
        if not self.settings.has_demo_account:
            LOGGER.error("No dedicated demo account configured - refusing to connect")
            self.connected = False
            return False
        with self._lock:
            try:
                ok = mt5.initialize(
                    login=int(self.settings.demo_login),
                    password=self.settings.demo_password,
                    server=self.settings.demo_server,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Demo broker initialise failed: %s", exc)
                self.connected = False
                return False
        self.connected = bool(ok)
        if not ok:
            LOGGER.error("Demo broker initialise returned false: %s", self._last_error())
        return self.connected

    def shutdown(self) -> None:
        mt5 = self.mt5
        if mt5 is None:
            return
        with self._lock:
            try:
                mt5.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.connected = False

    def _last_error(self) -> str:
        try:
            return str(self.mt5.last_error())
        except Exception:  # noqa: BLE001
            return "unknown"

    # -- queries ------------------------------------------------------------ #
    def account(self) -> AccountInfo:
        """The connected account, classified.

        Any failure yields an *unverified* account, never an optimistic one.
        """
        mt5 = self.mt5
        if mt5 is None:
            return AccountInfo(verified=False)
        try:
            with self._lock:
                raw = mt5.account_info()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("account_info() failed: %s", exc)
            return AccountInfo(verified=False)
        return classify_account(raw)

    def symbol_exists(self, broker_symbol: str) -> bool:
        mt5 = self.mt5
        if mt5 is None:
            return False
        try:
            with self._lock:
                return mt5.symbol_info(broker_symbol) is not None
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("symbol_info(%s) failed: %s", broker_symbol, exc)
            return False

    def quote(self, broker_symbol: str) -> Optional[Quote]:
        mt5 = self.mt5
        if mt5 is None:
            return None
        try:
            with self._lock:
                tick = mt5.symbol_info_tick(broker_symbol)
                info = mt5.symbol_info(broker_symbol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("quote(%s) failed: %s", broker_symbol, exc)
            return None
        if tick is None:
            return None
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if bid <= 0 or ask <= 0:
            return None
        spread = float(getattr(info, "spread", 0.0) or 0.0) if info is not None else float("nan")
        return Quote(bid=bid, ask=ask, spread_points=spread, time=now_utc())

    def positions(self, magic: Optional[int] = None) -> List[BrokerPosition]:
        mt5 = self.mt5
        if mt5 is None:
            return []
        try:
            with self._lock:
                raw = mt5.positions_get()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("positions_get() failed: %s", exc)
            return []
        found: List[BrokerPosition] = []
        for item in raw or ():
            position_magic = int(getattr(item, "magic", 0) or 0)
            if magic is not None and position_magic != magic:
                continue
            found.append(
                BrokerPosition(
                    ticket=int(getattr(item, "ticket", 0) or 0),
                    symbol=str(getattr(item, "symbol", "") or ""),
                    direction=BUY if int(getattr(item, "type", 0) or 0) == 0 else SELL,
                    volume=float(getattr(item, "volume", 0.0) or 0.0),
                    open_price=float(getattr(item, "price_open", 0.0) or 0.0),
                    stop_loss=float(getattr(item, "sl", 0.0) or 0.0),
                    take_profit=float(getattr(item, "tp", 0.0) or 0.0),
                    profit=float(getattr(item, "profit", 0.0) or 0.0),
                    magic=position_magic,
                    comment=str(getattr(item, "comment", "") or ""),
                )
            )
        return found

    # -- orders ------------------------------------------------------------- #
    def send_order(self, request: OrderRequest) -> OrderResult:
        """Send a market order.  Never decides whether one *should* be sent."""
        mt5 = self.mt5
        if mt5 is None:
            return OrderResult(accepted=False, comment="MetaTrader5 unavailable")
        is_buy = str(request.direction).upper() == BUY
        payload = {
            "action": getattr(mt5, "TRADE_ACTION_DEAL", 1),
            "symbol": request.broker_symbol,
            "volume": float(request.volume),
            "type": getattr(mt5, "ORDER_TYPE_BUY", 0) if is_buy
            else getattr(mt5, "ORDER_TYPE_SELL", 1),
            "price": float(request.requested_price),
            "sl": float(request.stop_loss),
            "tp": float(request.take_profit),
            "deviation": int(request.max_slippage_points),
            "magic": int(request.magic),
            "comment": request.comment[:31],
            "type_time": getattr(mt5, "ORDER_TIME_GTC", 0),
            "type_filling": getattr(mt5, "ORDER_FILLING_IOC", 1),
        }
        return self._transmit(payload, requested_volume=float(request.volume))

    def close_position(
        self, ticket: int, volume: Optional[float] = None, comment: str = ""
    ) -> OrderResult:
        """Close all or part of a position by ticket."""
        mt5 = self.mt5
        if mt5 is None:
            return OrderResult(accepted=False, comment="MetaTrader5 unavailable")
        existing = [p for p in self.positions() if p.ticket == int(ticket)]
        if not existing:
            return OrderResult(accepted=False, comment="position not found")
        position = existing[0]
        closing = float(volume if volume is not None else position.volume)
        closing = min(closing, position.volume)
        quote = self.quote(position.symbol)
        if quote is None:
            return OrderResult(accepted=False, comment="no quote to close against")
        is_buy = position.direction == BUY
        payload = {
            "action": getattr(mt5, "TRADE_ACTION_DEAL", 1),
            "symbol": position.symbol,
            "volume": closing,
            # closing a BUY means SELLing at the bid, and vice versa
            "type": getattr(mt5, "ORDER_TYPE_SELL", 1) if is_buy
            else getattr(mt5, "ORDER_TYPE_BUY", 0),
            "position": int(ticket),
            "price": quote.bid if is_buy else quote.ask,
            "deviation": int(self.settings.max_slippage_points),
            "magic": int(self.settings.magic_number),
            "comment": (comment or "close")[:31],
            "type_time": getattr(mt5, "ORDER_TIME_GTC", 0),
            "type_filling": getattr(mt5, "ORDER_FILLING_IOC", 1),
        }
        return self._transmit(payload, requested_volume=closing)

    def modify_position(
        self, ticket: int, stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Attach or move the broker-side SL/TP on an open position."""
        mt5 = self.mt5
        if mt5 is None:
            return False
        existing = [p for p in self.positions() if p.ticket == int(ticket)]
        if not existing:
            return False
        position = existing[0]
        payload = {
            "action": getattr(mt5, "TRADE_ACTION_SLTP", 2),
            "symbol": position.symbol,
            "position": int(ticket),
            "sl": float(position.stop_loss if stop_loss is None else stop_loss),
            "tp": float(position.take_profit if take_profit is None else take_profit),
        }
        result = self._transmit(payload, requested_volume=position.volume)
        return result.accepted

    def _transmit(self, payload: Dict[str, Any], requested_volume: float) -> OrderResult:
        """Send one request and classify the reply.

        A raised exception is reported as *indeterminate* rather than failed:
        the request may well have reached the server, so the caller must
        reconcile against actual positions instead of retrying blindly.
        """
        mt5 = self.mt5
        try:
            with self._lock:
                raw = mt5.order_send(payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("order_send raised: %s", exc)
            return OrderResult(
                accepted=False, comment=f"order_send raised: {exc}",
                indeterminate=True, requested_volume=requested_volume,
            )

        if raw is None:
            # No reply is not the same as a rejection - the order may exist.
            return OrderResult(
                accepted=False, comment=f"no reply ({self._last_error()})",
                indeterminate=True, requested_volume=requested_volume,
            )

        retcode = int(getattr(raw, "retcode", 0) or 0)
        filled = float(getattr(raw, "volume", 0.0) or 0.0)
        accepted = retcode in (self.RETCODE_DONE, self.RETCODE_DONE_PARTIAL)
        return OrderResult(
            accepted=accepted,
            ticket=int(getattr(raw, "order", 0) or getattr(raw, "deal", 0) or 0),
            fill_price=float(getattr(raw, "price", 0.0) or 0.0),
            volume=filled,
            retcode=retcode,
            comment=str(getattr(raw, "comment", "") or ""),
            partial=accepted and 0 < filled < requested_volume - 1e-9,
            requested_volume=requested_volume,
        )


# --------------------------------------------------------------------------- #
# test double
# --------------------------------------------------------------------------- #
@dataclass
class FakeDemoBroker:
    """Scripted :class:`DemoBroker` for tests and dry runs.

    Lives beside the real adapter deliberately: the two must satisfy the same
    interface, and keeping them together makes a drift between them obvious.
    Nothing here touches a network or a terminal.
    """

    account_info: AccountInfo = field(
        default_factory=lambda: AccountInfo(
            login=5_000_001, server="Demo-Server", currency="USD",
            balance=10_000.0, trade_mode="DEMO", is_demo=True, verified=True,
        )
    )
    quotes: Dict[str, Quote] = field(default_factory=dict)
    known_symbols: Optional[set] = None
    open_positions: List[BrokerPosition] = field(default_factory=list)
    #: Queue of results to return from ``send_order`` before falling back to a
    #: normal fill.  Lets a test script a rejection, a requote, a partial fill.
    scripted_results: List[OrderResult] = field(default_factory=list)
    sent: List[OrderRequest] = field(default_factory=list)
    closed: List[Dict[str, Any]] = field(default_factory=list)
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    connected: bool = False
    fail_connect: bool = False
    #: Points added to the requested price on fill, to simulate slippage.
    slippage_points: float = 0.0
    point_value: float = 0.01
    _next_ticket: int = 900_001

    def connect(self) -> bool:
        self.connected = not self.fail_connect
        return self.connected

    def shutdown(self) -> None:
        self.connected = False

    def account(self) -> AccountInfo:
        return self.account_info

    def symbol_exists(self, broker_symbol: str) -> bool:
        if self.known_symbols is None:
            return broker_symbol in self.quotes
        return broker_symbol in self.known_symbols

    def quote(self, broker_symbol: str) -> Optional[Quote]:
        return self.quotes.get(broker_symbol)

    def positions(self, magic: Optional[int] = None) -> List[BrokerPosition]:
        if magic is None:
            return list(self.open_positions)
        return [p for p in self.open_positions if p.magic == magic]

    def send_order(self, request: OrderRequest) -> OrderResult:
        self.sent.append(request)
        if self.scripted_results:
            result = self.scripted_results.pop(0)
            if result.accepted:
                self._record(request, result.fill_price or request.requested_price,
                             result.volume or request.volume, result.ticket)
            return result

        direction = str(request.direction).upper()
        drift = self.slippage_points * self.point_value
        fill = request.requested_price + (drift if direction == BUY else -drift)
        ticket = self._next_ticket
        self._next_ticket += 1
        self._record(request, fill, request.volume, ticket)
        return OrderResult(
            accepted=True, ticket=ticket, fill_price=fill,
            volume=request.volume, retcode=MT5DemoBroker.RETCODE_DONE,
            requested_volume=request.volume,
        )

    def _record(self, request: OrderRequest, fill: float, volume: float, ticket: int) -> None:
        self.open_positions.append(
            BrokerPosition(
                ticket=ticket, symbol=request.broker_symbol, direction=request.direction,
                volume=volume, open_price=fill, stop_loss=request.stop_loss,
                take_profit=request.take_profit, magic=request.magic,
                comment=request.comment, open_time=now_utc(),
            )
        )

    def close_position(
        self, ticket: int, volume: Optional[float] = None, comment: str = ""
    ) -> OrderResult:
        for index, position in enumerate(self.open_positions):
            if position.ticket != int(ticket):
                continue
            closing = float(volume if volume is not None else position.volume)
            closing = min(closing, position.volume)
            quote = self.quotes.get(position.symbol)
            price = quote.exit_price(position.direction) if quote else position.open_price
            self.closed.append(
                {"ticket": ticket, "volume": closing, "price": price, "comment": comment}
            )
            remaining = round(position.volume - closing, 8)
            if remaining > 1e-9:
                self.open_positions[index] = BrokerPosition(
                    **{**position.__dict__, "volume": remaining}
                )
            else:
                self.open_positions.pop(index)
            return OrderResult(
                accepted=True, ticket=int(ticket), fill_price=price,
                volume=closing, retcode=MT5DemoBroker.RETCODE_DONE,
                requested_volume=closing,
            )
        return OrderResult(accepted=False, comment="position not found")

    def modify_position(
        self, ticket: int, stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        for index, position in enumerate(self.open_positions):
            if position.ticket != int(ticket):
                continue
            self.modifications.append(
                {"ticket": ticket, "sl": stop_loss, "tp": take_profit}
            )
            self.open_positions[index] = BrokerPosition(
                **{
                    **position.__dict__,
                    "stop_loss": position.stop_loss if stop_loss is None else stop_loss,
                    "take_profit": position.take_profit if take_profit is None else take_profit,
                }
            )
            return True
        return False
