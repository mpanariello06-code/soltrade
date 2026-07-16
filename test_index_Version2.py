import os
import unittest
from unittest.mock import Mock, patch

from index_Version2 import TradingBot


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
