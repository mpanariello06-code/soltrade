# Soltrade (Python, HTTP-only)

Python version of the Solana trading bot using HTTP polling only (no WebSocket mode).

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements_Version2.txt
```

2. Create your environment file:
```bash
cp config.env.example config.env
```

3. Add your required Solana settings to `config.env`:
   - `PRIVATE_KEY`
   - `SOLANA_TRACKER_API_KEY`
   - `RPC_URL`

4. Optional: enable Telegram notifications.

   ### Create the bot
   1. Open Telegram and start a chat with `@BotFather`.
   2. Send `/newbot`.
   3. Choose a bot name.
   4. Choose a unique bot username ending in `bot`.
   5. Copy the bot token BotFather gives you.

   ### Get your chat ID
   1. Open your new bot in Telegram.
   2. Press **Start** or send any message to the bot.
   3. In a browser, open this URL after replacing `{YOUR_BOT_TOKEN}` with your real token:
      `https://api.telegram.org/bot{YOUR_BOT_TOKEN}/getUpdates`
   4. If the response is empty, send a fresh message to the bot and refresh the page.
   5. Find the `chat` object in the response and copy the `id` value.

   ### Add Telegram settings to `config.env`
   ```env
   TELEGRAM_BOT_TOKEN=your-bot-token
   TELEGRAM_CHAT_ID=your-chat-id
   ```

   The bot will send notifications for startup, buys, sells, and shutdown.

5. Review the rest of the trading settings in `config.env`, then run:
```bash
python index_Version2.py
```

## Notes
- WebSocket trading is removed.
- Project is Python-only.
- Position state is persisted in:
  - `positions.json`
  - `sold_positions.json`
- Logs:
  - `trading-bot.log`
  - `trading-bot-error.log`