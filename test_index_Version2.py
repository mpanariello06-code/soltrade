import os
import signal
import logging
import unittest
from unittest.mock import Mock, call, patch

import requests
from index_Version2 import TradingBot, register_shutdown_handlers, SafeStreamHandler


class TradingBotTelegramTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(
            os.environ,
            {
                "PRIVATE_KEY": "test-private-key",
                "SOLANA_TRACKER_API_KEY": "test-api-key",
                "TELEGRAM_BOT_TOKEN": "telegram-token",
                "TELEGRAM_CHAT_ID": "123456",
            },
            clear=False,
        )
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def build_bot(self):
        bot = TradingBot()
        self.addCleanup(self.close_bot, bot)
        return bot

    def close_bot(self, bot):
        for handler in bot.logger.handlers[:]:
            handler.close()
            bot.logger.removeHandler(handler)
        for path in ("trading-bot.log", "trading-bot-error.log"):
            if os.path.exists(path):
                os.remove(path)

    def test_validate_config_rejects_partial_telegram_config(self):
        with patch.dict(os.environ, {"TELEGRAM_CHAT_ID": ""}, clear=False):
            with self.assertRaisesRegex(
                ValueError,
                r"TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must either both be set or both be empty "
                r"\(TELEGRAM_BOT_TOKEN is set, TELEGRAM_CHAT_ID is empty\)",
            ):
                self.build_bot()

    @patch("index_Version2.requests.post")
    def test_send_telegram_notification_posts_message(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"ok": True}
        mock_post.return_value = response

        bot = self.build_bot()
        bot._send_telegram_notification_sync("hello")

        mock_post.assert_called_once_with(
            "https://api.telegram.org/bottelegram-token/sendMessage",
            json={
                "chat_id": "123456",
                "text": "hello",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )

    @patch("index_Version2.requests.post")
    def test_send_telegram_notification_retries_with_form_payload(self, mock_post):
        failed_response = Mock()
        failed_response.raise_for_status.side_effect = requests.HTTPError("bad request")

        successful_response = Mock()
        successful_response.raise_for_status.return_value = None
        successful_response.json.return_value = {"ok": True}

        mock_post.side_effect = [failed_response, successful_response]

        bot = self.build_bot()
        bot._send_telegram_notification_sync("hello")

        self.assertEqual(mock_post.call_count, 2)
        mock_post.assert_has_calls(
            [
                call(
                    "https://api.telegram.org/bottelegram-token/sendMessage",
                    json={
                        "chat_id": "123456",
                        "text": "hello",
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                ),
                call(
                    "https://api.telegram.org/bottelegram-token/sendMessage",
                    data={
                        "chat_id": "123456",
                        "text": "hello",
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                ),
            ]
        )

    @patch("index_Version2.requests.post")
    def test_send_telegram_notification_skips_when_disabled(self, mock_post):
        with patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""},
            clear=False,
        ):
            bot = self.build_bot()

        bot._send_telegram_notification_sync("hello")

        mock_post.assert_not_called()

    @patch("index_Version2.requests.post")
    def test_send_telegram_notification_async_wrapper(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"ok": True}
        mock_post.return_value = response

        bot = self.build_bot()

        import asyncio

        asyncio.run(bot.send_telegram_notification("hello from async"))

        mock_post.assert_called_once()

    def test_register_shutdown_handlers_registers_supported_signals(self):
        loop = Mock()
        callback = Mock()

        self.assertTrue(register_shutdown_handlers(loop, callback))
        self.assertEqual(
            loop.add_signal_handler.call_args_list,
            [call(signal.SIGINT, callback), call(signal.SIGTERM, callback)],
        )

    def test_register_shutdown_handlers_ignores_unsupported_loops(self):
        loop = Mock()
        loop.add_signal_handler.side_effect = NotImplementedError

        self.assertFalse(register_shutdown_handlers(loop, Mock()))

    def test_register_shutdown_handlers_keeps_supported_signals(self):
        loop = Mock()
        callback = Mock()

        def add_signal_handler(sig, cb):
            if sig == signal.SIGTERM:
                raise NotImplementedError

        loop.add_signal_handler.side_effect = add_signal_handler

        self.assertTrue(register_shutdown_handlers(loop, callback))
        self.assertEqual(
            loop.add_signal_handler.call_args_list,
            [call(signal.SIGINT, callback), call(signal.SIGTERM, callback)],
        )

    def test_safe_stream_handler_replaces_unencodable_characters(self):
        class Cp1252LikeStream:
            encoding = "cp1252"

            def __init__(self):
                self.writes = []

            def write(self, text):
                if "🚀" in text:
                    emoji_index = text.index("🚀")
                    raise UnicodeEncodeError("charmap", text, emoji_index, emoji_index + 1, "character maps to <undefined>")
                self.writes.append(text)
                return len(text)

            def flush(self):
                return None

        stream = Cp1252LikeStream()
        handler = SafeStreamHandler(stream=stream)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("safe-stream-handler-test")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        self.addCleanup(handler.close)

        logger.info("🚀 Initializing HTTP Trading Bot...")

        self.assertEqual("".join(stream.writes), "? Initializing HTTP Trading Bot...\n")
