"""Telegram notifications.

Implemented against the Bot HTTP API with ``requests`` rather than
``python-telegram-bot``: the only thing this project needs is "post a text
message", and the main loop is synchronous, so pulling in an async framework
and its event loop would add moving parts without adding capability.

Every method is failure-tolerant - a Telegram outage must never stop signal
generation or CSV logging.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from .logger import get_logger
from .markets import get_market, is_supported

LOGGER = get_logger("telegram")

API_BASE = "https://api.telegram.org/bot{token}/{method}"
REQUEST_TIMEOUT = 15
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (2, 4, 8)

DIVIDER = "━━━━━━━━━━━━━━━━━━"

#: outcome event -> message prefix
OUTCOME_ICONS: Dict[str, str] = {
    "TP1_HIT": "🎯 TP1 HIT",
    "TP2_HIT": "🎯 TP2 HIT",
    "TP3_HIT": "🎯 TP3 HIT",
    "SL_HIT": "❌ SL HIT",
    "INVALIDATED": "⚠️ SIGNAL INVALIDATED",
    "TIMEOUT": "⌛ SCALP TIMED OUT",
}

class TelegramNotifier:
    """Minimal, resilient Telegram sender."""

    def __init__(self, config) -> None:
        self.config = config
        self.token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self.connected = False
        self._session = requests.Session()

    @property
    def enabled(self) -> bool:
        """True when Telegram is configured and switched on."""
        return bool(self.config.telegram_enabled and self.token and self.chat_id)

    # -- transport --------------------------------------------------------- #
    def _call(
        self,
        method: str,
        payload: Dict[str, Any],
        timeout: Optional[int] = None,
        attempts: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """POST to the Bot API with retries.  Returns the JSON result or ``None``."""
        url = API_BASE.format(token=self.token, method=method)
        timeout = REQUEST_TIMEOUT if timeout is None else timeout
        max_attempts = MAX_ATTEMPTS if attempts is None else max(1, attempts)
        for attempt in range(max_attempts):
            try:
                response = self._session.post(url, json=payload, timeout=timeout)
                if response.status_code == 200:
                    return response.json()
                # 429 carries a retry-after hint; anything else we simply retry.
                LOGGER.warning(
                    "Telegram %s returned HTTP %s: %s",
                    method,
                    response.status_code,
                    response.text[:200],
                )
            except requests.RequestException as exc:
                LOGGER.warning("Telegram %s failed (attempt %s): %s", method, attempt + 1, exc)
            if attempt < max_attempts - 1:
                time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
        LOGGER.error("Telegram %s failed after %s attempts", method, max_attempts)
        return None

    def test_connection(self) -> bool:
        """Verify the bot token; also logs the bot username."""
        if not self.enabled:
            LOGGER.warning("Telegram disabled or not configured - notifications will be skipped")
            self.connected = False
            return False
        result = self._call("getMe", {})
        if result and result.get("ok"):
            username = result.get("result", {}).get("username", "unknown")
            LOGGER.info("Telegram connected as @%s", username)
            self.connected = True
            return True
        self.connected = False
        return False

    def send_message(
        self, text: str, keyboard: Optional[List[List[Dict[str, str]]]] = None
    ) -> Optional[int]:
        """Send a plain-text message, optionally with an inline keyboard.

        Returns the sent ``message_id`` (truthy) on success, or ``None``.
        """
        if not self.enabled:
            LOGGER.debug("Telegram disabled - message suppressed:\n%s", text)
            return None
        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if keyboard:
            payload["reply_markup"] = {"inline_keyboard": keyboard}
        result = self._call("sendMessage", payload)
        if result and result.get("ok"):
            return int(result.get("result", {}).get("message_id", 0)) or None
        return None

    def edit_message(
        self,
        message_id: int,
        text: str,
        keyboard: Optional[List[List[Dict[str, str]]]] = None,
    ) -> bool:
        """Replace the text/keyboard of an existing message.

        Editing in place is what makes the control panel feel like a panel
        rather than a growing wall of duplicate messages.
        """
        if not self.enabled or not message_id:
            return False
        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "message_id": int(message_id),
            "text": text,
            "disable_web_page_preview": True,
        }
        if keyboard:
            payload["reply_markup"] = {"inline_keyboard": keyboard}
        result = self._call("editMessageText", payload, attempts=1)
        if result and result.get("ok"):
            return True
        # "message is not modified" is a success for our purposes
        description = str((result or {}).get("description", ""))
        return "not modified" in description

    def answer_callback(self, callback_id: str, text: str = "") -> bool:
        """Acknowledge a button press so Telegram stops showing the spinner."""
        if not self.enabled or not callback_id:
            return False
        payload: Dict[str, Any] = {"callback_query_id": callback_id}
        if text:
            payload["text"] = text[:200]
        result = self._call("answerCallbackQuery", payload, attempts=1)
        return bool(result and result.get("ok"))

    def get_updates(self, offset: int, timeout: int = 25) -> List[Dict[str, Any]]:
        """Long-poll for new updates.  Returns ``[]`` on any failure."""
        if not self.enabled:
            return []
        payload = {
            "offset": offset,
            "timeout": int(timeout),
            "allowed_updates": ["callback_query", "message"],
        }
        # one attempt only: a failed long-poll is simply retried by the caller's
        # loop, and stacking retries here would multiply the wait.
        result = self._call(
            "getUpdates", payload, timeout=int(timeout) + REQUEST_TIMEOUT, attempts=1
        )
        if result and result.get("ok"):
            return list(result.get("result", []))
        return []

    # -- formatting --------------------------------------------------------- #
    PAPER_DISCLAIMER = "🔬 PAPER TEST ONLY"

    def _view(self, symbol: str):
        """Config folded onto ``symbol``'s market.

        A card must be rendered with the digits, pip unit and icon of the
        market the SIGNAL belongs to - not whichever market happens to be
        selected in the control panel when the message is sent.
        """
        if symbol and is_supported(symbol):
            return self.config.for_market(symbol)
        return self.config

    @staticmethod
    def _icon(symbol: str) -> str:
        return get_market(symbol).icon if symbol and is_supported(symbol) else "⚡"

    def format_signal(self, signal) -> str:
        """Render the M1 scalp card.

        Both RAW and NET reward are shown.  Quoting raw R alone on a trade whose
        target is a few pips would be actively misleading - the spread is often
        a large fraction of the move.
        """
        view = self._view(signal.symbol)
        digits = view.digits
        unit = view.pip_name
        spread_price = signal.spread_points * view.point_value
        lines = [
            DIVIDER,
            f"{self._icon(signal.symbol)} {signal.symbol} M1 SCALP",
            DIVIDER,
            "",
            f"Direction: {signal.direction}",
            "",
            f"Score: {signal.confidence:.0f}/100",
            "",
            f"Entry: {signal.entry:.{digits}f}",
            "",
            f"TP1: {signal.tp1:.{digits}f}   ({signal.tp_pips[0]:.1f}{unit})",
            f"TP2: {signal.tp2:.{digits}f}   ({signal.tp_pips[1]:.1f}{unit})",
            f"TP3: {signal.tp3:.{digits}f}   ({signal.tp_pips[2]:.1f}{unit})",
            "",
            f"SL: {signal.stop_loss:.{digits}f}   ({signal.sl_pips:.1f}{unit})",
            "",
            "Expected holding period:",
            signal.expected_hold,
            "",
            "Spread:",
            f"{spread_price:.2f}  ({signal.spread_points:.0f} points)",
            "",
            "Risk/Reward:",
            f"TP1 {signal.rr1:.2f}R",
            f"TP2 {signal.rr2:.2f}R",
            f"TP3 {signal.rr3:.2f}R",
            "",
            f"After costs ({signal.cost_pips:.1f}{unit} = {signal.cost_r:.2f}R):",
            f"TP1 {signal.net_rr1:.2f}R",
            f"TP2 {signal.net_rr2:.2f}R",
            f"TP3 {signal.net_rr3:.2f}R",
            "",
            "Reason:",
            signal.reason_summary,
            "",
            DIVIDER,
            self.PAPER_DISCLAIMER,
        ]
        return "\n".join(lines)

    def format_near_signal(self, evaluation) -> str:
        """Render the optional NEAR_SIGNAL diagnostic."""
        card = evaluation.card
        bullish = card.bullish_score if card else 0.0
        bearish = card.bearish_score if card else 0.0
        direction = "BUY" if bullish >= bearish else "SELL"
        best = max(bullish, bearish)
        return "\n".join(
            [
                DIVIDER,
                "👀 NEAR SIGNAL",
                DIVIDER,
                "",
                f"{self._icon(evaluation.symbol)} {evaluation.symbol} M1"
                f"  ({direction} side)",
                "",
                f"Threshold: {evaluation.threshold:.0f}",
                f"Bullish: {bullish:.0f}",
                f"Bearish: {bearish:.0f}",
                f"Short by: {max(evaluation.threshold - best, 0.0):.1f}",
                "",
                f"Regime: {evaluation.regime.replace('_', ' ')}",
                f"Blocked by: {evaluation.rejection_reason or '-'}",
                "",
                DIVIDER,
                "Diagnostic only - NOT a signal.",
            ]
        )

    def format_outcome(
        self,
        signal_row: Dict[str, Any],
        event: str,
        price: float,
        r_multiple: Optional[float] = None,
        net_r: Optional[float] = None,
    ) -> str:
        """Render a TP/SL/timeout update for an existing signal."""
        symbol = str(signal_row.get("symbol", ""))
        digits = self._view(symbol).digits
        header = OUTCOME_ICONS.get(event, event)
        lines = [
            DIVIDER,
            f"{header}",
            DIVIDER,
            f"{self._icon(symbol)} {symbol} M1 {signal_row.get('direction', '')}"
            f" @ {float(signal_row.get('entry', 0.0)):.{digits}f}",
            f"Level: {price:.{digits}f}",
        ]
        if r_multiple is not None:
            lines.append(f"Raw: {r_multiple:+.2f}R")
        if net_r is not None:
            lines.append(f"Net after costs: {net_r:+.2f}R")
        lines.append(f"Signal: {signal_row.get('signal_id', '')}")
        lines.append(DIVIDER)
        return "\n".join(lines)

    def send_signal(self, signal) -> bool:
        """Send a new-signal card."""
        return bool(self.send_message(self.format_signal(signal)))

    def send_near_signal(self, evaluation) -> bool:
        """Send a near-signal diagnostic (only when the user enabled them)."""
        return bool(self.send_message(self.format_near_signal(evaluation)))

    def send_outcome(
        self,
        signal_row: Dict[str, Any],
        event: str,
        price: float,
        r_multiple: Optional[float] = None,
        net_r: Optional[float] = None,
    ) -> bool:
        """Send an outcome update."""
        return bool(
            self.send_message(self.format_outcome(signal_row, event, price, r_multiple, net_r))
        )

    def send_text(self, text: str) -> bool:
        """Send an arbitrary status message (startup/shutdown notices)."""
        return bool(self.send_message(text))
