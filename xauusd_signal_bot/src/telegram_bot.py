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
from typing import Any, Dict, Optional

import requests

from .logger import get_logger

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
    "EXPIRED": "⌛ SIGNAL EXPIRED",
}

#: component key -> label shown in the CONFIRMATIONS block
CONFIRMATION_LABELS = (
    ("trend", "Trend"),
    ("htf", "HTF"),
    ("momentum", "Momentum"),
    ("structure", "Structure"),
    ("liquidity", "Liquidity"),
    ("support_resistance", "S/R"),
    ("volume", "Volume"),
    ("volatility", "Volatility"),
    ("price_action", "Price Action"),
)


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
    def _call(self, method: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """POST to the Bot API with retries.  Returns the JSON result or ``None``."""
        url = API_BASE.format(token=self.token, method=method)
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
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
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
        LOGGER.error("Telegram %s failed after %s attempts", method, MAX_ATTEMPTS)
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

    def send_message(self, text: str) -> bool:
        """Send a plain-text message.  Returns ``True`` on success."""
        if not self.enabled:
            LOGGER.debug("Telegram disabled - message suppressed:\n%s", text)
            return False
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        result = self._call("sendMessage", payload)
        return bool(result and result.get("ok"))

    # -- formatting --------------------------------------------------------- #
    def format_signal(self, signal) -> str:
        """Render the signal card (spec section 33)."""
        icon = "🟢" if signal.direction == "BUY" else "🔴"
        digits = self.config.digits
        lines = [
            DIVIDER,
            f"{icon} {signal.symbol} {signal.direction}",
            DIVIDER,
            "",
            f"⭐ Confidence: {signal.confidence:.0f}/100",
            f"📊 Timeframe: {signal.timeframe}",
            f"📈 Regime: {signal.regime.replace('_', ' ')}",
            f"🕐 Session: {signal.session.replace('_', ' ')}",
            "",
            f"Entry: {signal.entry:.{digits}f}",
            f"SL: {signal.stop_loss:.{digits}f}",
            "",
            f"TP1: {signal.tp1:.{digits}f}",
            f"TP2: {signal.tp2:.{digits}f}",
            f"TP3: {signal.tp3:.{digits}f}",
            "",
            "R:R",
            f"TP1: {signal.rr1:.1f}R",
            f"TP2: {signal.rr2:.1f}R",
            f"TP3: {signal.rr3:.1f}R",
            "",
            "CONFIRMATIONS",
        ]
        for key, label in CONFIRMATION_LABELS:
            mark = "✅" if signal.confirmations.get(key) else "▫️"
            lines.append(f"{mark} {label}")
        lines += ["", DIVIDER, "Signal only - not financial advice."]
        return "\n".join(lines)

    def format_outcome(
        self, signal_row: Dict[str, Any], event: str, price: float, r_multiple: Optional[float] = None
    ) -> str:
        """Render a TP/SL/expiry update for an existing signal."""
        digits = self.config.digits
        header = OUTCOME_ICONS.get(event, event)
        lines = [
            DIVIDER,
            f"{header}",
            DIVIDER,
            f"{signal_row.get('symbol', '')} {signal_row.get('direction', '')}"
            f" @ {float(signal_row.get('entry', 0.0)):.{digits}f}",
            f"Level: {price:.{digits}f}",
        ]
        if r_multiple is not None:
            lines.append(f"Result: {r_multiple:+.2f}R")
        lines.append(f"Signal: {signal_row.get('signal_id', '')}")
        lines.append(DIVIDER)
        return "\n".join(lines)

    def send_signal(self, signal) -> bool:
        """Send a new-signal card."""
        return self.send_message(self.format_signal(signal))

    def send_outcome(
        self, signal_row: Dict[str, Any], event: str, price: float, r_multiple: Optional[float] = None
    ) -> bool:
        """Send an outcome update."""
        return self.send_message(self.format_outcome(signal_row, event, price, r_multiple))

    def send_text(self, text: str) -> bool:
        """Send an arbitrary status message (startup/shutdown notices)."""
        return self.send_message(text)
