import os
import json
import time
import signal
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set

import requests
from dotenv import load_dotenv


load_dotenv()


def sleep_ms(ms: int):
    return asyncio.sleep(ms / 1000)


def format_usd(value: float) -> str:
    return f"${value:,.6f}" if value < 1 else f"${value:,.2f}"


def format_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


@dataclass
class Config:
    amount: float
    delay: int
    monitor_interval: int
    slippage: int
    priority_fee: float
    use_jito: bool
    rpc_url: str

    min_liquidity: float
    max_liquidity: float
    min_market_cap: float
    max_market_cap: float
    min_risk_score: int
    max_risk_score: int
    min_holders: int
    require_social_data: bool

    max_negative_pnl: float
    max_positive_pnl: float

    markets: List[str]
    max_positions: int
    max_retries: int
    debug: bool
    telegram_bot_token: str
    telegram_chat_id: str


class TradingBot:
    SOL_ADDRESS = "So11111111111111111111111111111111111111112"

    def __init__(self):
        self.config = self.load_config()
        self.validate_config()

        self.private_key = os.getenv("PRIVATE_KEY", "")
        self.api_key = os.getenv("SOLANA_TRACKER_API_KEY", "")

        self.positions_file = "positions.json"
        self.sold_positions_file = "sold_positions.json"

        self.positions: Dict[str, Dict[str, Any]] = {}
        self.sold_positions: List[Dict[str, Any]] = []

        self.seen_tokens: Set[str] = set()
        self.buying_tokens: Set[str] = set()
        self.selling_positions: Set[str] = set()

        self.stats = {
            "total_buys": 0,
            "total_sells": 0,
            "successful_buys": 0,
            "successful_sells": 0,
            "total_pnl": 0.0,
            "start_time": time.time(),
        }

        self.shutdown_requested = False
        self.shutdown_lock = asyncio.Lock()
        self.setup_logger()

    def setup_logger(self):
        self.logger = logging.getLogger("soltrade")
        self.logger.setLevel(logging.DEBUG if self.config.debug else logging.INFO)
        self.logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if self.config.debug else logging.INFO)
        ch.setFormatter(formatter)

        fh = logging.FileHandler("trading-bot.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        eh = logging.FileHandler("trading-bot-error.log")
        eh.setLevel(logging.ERROR)
        eh.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        self.logger.addHandler(eh)

    def load_config(self) -> Config:
        return Config(
            amount=float(os.getenv("AMOUNT", "0.01")),
            delay=int(os.getenv("DELAY", "5000")),
            monitor_interval=int(os.getenv("MONITOR_INTERVAL", "30000")),
            slippage=int(os.getenv("SLIPPAGE", "15")),
            priority_fee=float(os.getenv("PRIORITY_FEE", "0.00001")),
            use_jito=os.getenv("JITO", "false").lower() == "true",
            rpc_url=os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com"),
            min_liquidity=float(os.getenv("MIN_LIQUIDITY", "1000")),
            max_liquidity=float(os.getenv("MAX_LIQUIDITY", "100000")),
            min_market_cap=float(os.getenv("MIN_MARKET_CAP", "10000")),
            max_market_cap=float(os.getenv("MAX_MARKET_CAP", "1000000")),
            min_risk_score=int(os.getenv("MIN_RISK_SCORE", "0")),
            max_risk_score=int(os.getenv("MAX_RISK_SCORE", "7")),
            min_holders=int(os.getenv("MIN_HOLDERS", "10")),
            require_social_data=os.getenv("REQUIRE_SOCIAL_DATA", "false").lower() == "true",
            max_negative_pnl=float(os.getenv("MAX_NEGATIVE_PNL", "-50")),
            max_positive_pnl=float(os.getenv("MAX_POSITIVE_PNL", "100")),
            markets=[m.strip() for m in os.getenv("MARKETS", "raydium,orca,pumpfun,moonshot,raydium-cpmm").split(",") if m.strip()],
            max_positions=int(os.getenv("MAX_POSITIONS", "10")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        )

    def validate_config(self):
        required = ["PRIVATE_KEY", "SOLANA_TRACKER_API_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        if self.config.amount <= 0:
            raise ValueError("AMOUNT must be > 0")
        if self.config.max_negative_pnl > 0:
            raise ValueError("MAX_NEGATIVE_PNL must be <= 0")
        if self.config.max_positive_pnl < 0:
            raise ValueError("MAX_POSITIVE_PNL must be >= 0")
        if bool(self.config.telegram_bot_token) != bool(self.config.telegram_chat_id):
            token_state = "set" if self.config.telegram_bot_token else "empty"
            chat_state = "set" if self.config.telegram_chat_id else "empty"
            raise ValueError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must either both be set or both be empty "
                f"(TELEGRAM_BOT_TOKEN is {token_state}, TELEGRAM_CHAT_ID is {chat_state})"
            )

    async def initialize(self):
        self.logger.info("🚀 Initializing HTTP Trading Bot...")
        await self.load_positions()
        await self.load_sold_positions()
        self.display_config()
        self.logger.info("✅ Initialization complete")
        await self.send_telegram_notification(
            "🚀 Soltrade bot started\n"
            f"Amount: {self.config.amount} SOL\n"
            f"Markets: {', '.join(self.config.markets)}\n"
            f"Max Positions: {self.config.max_positions}\n"
            f"Stop Loss: {self.config.max_negative_pnl}%\n"
            f"Take Profit: {self.config.max_positive_pnl}%"
        )

    def display_config(self):
        self.logger.info("📋 Configuration")
        self.logger.info(f"  - Trade Amount: {self.config.amount} SOL")
        self.logger.info(f"  - Markets: {', '.join(self.config.markets)}")
        self.logger.info(f"  - Liquidity: {format_usd(self.config.min_liquidity)} - {format_usd(self.config.max_liquidity)}")
        self.logger.info(f"  - Market Cap: {format_usd(self.config.min_market_cap)} - {format_usd(self.config.max_market_cap)}")
        self.logger.info(f"  - Risk Score: {self.config.min_risk_score} - {self.config.max_risk_score}")
        self.logger.info(f"  - Stop Loss: {self.config.max_negative_pnl}%")
        self.logger.info(f"  - Take Profit: {self.config.max_positive_pnl}%")
        self.logger.info(f"  - Max Positions: {self.config.max_positions}")
        self.logger.info(f"  - Telegram Notifications: {'Enabled' if self.telegram_enabled() else 'Disabled'}")

    def _headers(self):
        return {"x-api-key": self.api_key, "accept": "application/json"}

    def telegram_enabled(self) -> bool:
        return bool(self.config.telegram_bot_token and self.config.telegram_chat_id)

    def _send_telegram_notification_sync(self, message: str):
        if not self.telegram_enabled():
            return

        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": message,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data.get("ok", False):
                self.logger.warning("Telegram notification was rejected by the Telegram API")
        except Exception as e:
            self.logger.error(f"Telegram notification failed: {e}")

    async def send_telegram_notification(self, message: str):
        await asyncio.to_thread(self._send_telegram_notification_sync, message)

    def fetch_latest_tokens(self) -> List[Dict[str, Any]]:
        url = "https://data.solanatracker.io/tokens/latest"
        try:
            r = requests.get(url, headers=self._headers(), timeout=15)
            if r.status_code == 429:
                self.logger.warning("Rate limited while fetching latest tokens")
                return []
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            self.logger.error(f"Error fetching latest tokens: {e}")
            return []

    def fetch_token_info(self, token_mint: str) -> Optional[Dict[str, Any]]:
        url = f"https://data.solanatracker.io/tokens/{token_mint}"
        try:
            r = requests.get(url, headers=self._headers(), timeout=15)
            if r.status_code == 429:
                self.logger.warning(f"Rate limited for token {token_mint}")
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.logger.error(f"Error fetching token {token_mint}: {e}")
            return None

    def filter_tokens(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(self.positions) >= self.config.max_positions:
            return []

        out = []
        for token in tokens:
            try:
                pools = token.get("pools") or []
                if not pools:
                    continue
                pool = pools[0]

                liquidity = float((((pool.get("liquidity") or {}).get("usd")) or 0))
                market_cap = float((((pool.get("marketCap") or {}).get("usd")) or 0))
                risk_score = int(((token.get("risk") or {}).get("score")) or 10)
                holders = int(token.get("holders") or 0)

                social = token.get("token") or {}
                has_social = bool(social.get("twitter") or social.get("telegram") or social.get("website"))

                market = pool.get("market")
                mint = (token.get("token") or {}).get("mint")
                if not mint:
                    continue

                passes = (
                    self.config.min_liquidity <= liquidity <= self.config.max_liquidity
                    and self.config.min_market_cap <= market_cap <= self.config.max_market_cap
                    and self.config.min_risk_score <= risk_score <= self.config.max_risk_score
                    and holders >= self.config.min_holders
                    and (not self.config.require_social_data or has_social)
                    and market in self.config.markets
                    and mint not in self.seen_tokens
                    and mint not in self.buying_tokens
                    and mint not in self.positions
                )

                if passes:
                    out.append(token)
            except Exception as e:
                self.logger.error(f"Filter error: {e}")
        return out

    async def perform_buy(self, token: Dict[str, Any]):
        mint = token["token"]["mint"]
        symbol = token["token"].get("symbol", "UNKNOWN")
        self.stats["total_buys"] += 1

        # Placeholder for actual swap execution
        self.logger.info(f"🟢 [BUY] Simulated buy for {symbol} ({mint})")

        price = float(((token["pools"][0].get("price") or {}).get("usd")) or 0)
        position = {
            "symbol": symbol,
            "name": token["token"].get("name", symbol),
            "entryPrice": price,
            "amount": 1.0,
            "investment": self.config.amount,
            "openTime": int(time.time() * 1000),
            "market": token["pools"][0].get("market"),
            "riskScore": ((token.get("risk") or {}).get("score") or 0),
        }

        self.positions[mint] = position
        self.seen_tokens.add(mint)
        self.buying_tokens.discard(mint)
        self.stats["successful_buys"] += 1
        await self.save_positions()
        await self.send_telegram_notification(
            "🟢 Buy executed\n"
            f"Token: {symbol}\n"
            f"Mint: {mint}\n"
            f"Entry Price: {format_usd(price)}\n"
            f"Investment: {self.config.amount} SOL\n"
            f"Market: {position.get('market', 'unknown')}"
        )

    async def perform_sell(self, mint: str, token_data: Dict[str, Any]):
        position = self.positions.get(mint)
        if not position:
            return
        symbol = position.get("symbol", mint)
        self.stats["total_sells"] += 1

        current_price = float((((token_data.get("pools") or [{}])[0].get("price") or {}).get("usd") or 0))
        entry_price = float(position.get("entryPrice", 0))
        amount = float(position.get("amount", 0))

        pnl = (current_price - entry_price) * amount
        pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0

        sold = {
            **position,
            "exitPrice": current_price,
            "pnl": pnl,
            "pnlPercentage": pnl_pct,
            "closeTime": int(time.time() * 1000),
        }
        self.sold_positions.append(sold)
        self.stats["total_pnl"] += pnl
        self.stats["successful_sells"] += 1

        self.positions.pop(mint, None)
        self.selling_positions.discard(mint)

        self.logger.info(f"🔴 [SELL] {symbol} PnL: {format_usd(pnl)} ({format_pct(pnl_pct)})")

        await self.save_positions()
        await self.save_sold_positions()
        await self.send_telegram_notification(
            "🔴 Sell executed\n"
            f"Token: {symbol}\n"
            f"Mint: {mint}\n"
            f"Exit Price: {format_usd(current_price)}\n"
            f"PnL: {format_usd(pnl)} ({format_pct(pnl_pct)})"
        )

    async def buy_loop(self):
        self.logger.info("👀 HTTP buy loop started")
        while not self.shutdown_requested:
            try:
                if len(self.positions) < self.config.max_positions:
                    tokens = self.fetch_latest_tokens()
                    candidates = self.filter_tokens(tokens)
                    for token in candidates:
                        if len(self.positions) >= self.config.max_positions:
                            break
                        mint = token["token"]["mint"]
                        if mint not in self.buying_tokens and mint not in self.positions:
                            self.buying_tokens.add(mint)
                            await self.perform_buy(token)
                            await sleep_ms(1000)
            except Exception as e:
                self.logger.error(f"Buy loop error: {e}")
            await sleep_ms(self.config.delay)

    async def monitor_loop(self):
        self.logger.info("📊 Position monitor started")
        while not self.shutdown_requested:
            try:
                for mint, position in list(self.positions.items()):
                    if mint in self.selling_positions:
                        continue
                    token_data = self.fetch_token_info(mint)
                    if not token_data or not token_data.get("pools"):
                        continue
                    current = float((((token_data["pools"][0].get("price") or {}).get("usd")) or 0))
                    entry = float(position.get("entryPrice", 0))
                    pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0.0

                    if pnl_pct <= self.config.max_negative_pnl or pnl_pct >= self.config.max_positive_pnl:
                        self.selling_positions.add(mint)
                        await self.perform_sell(mint, token_data)

                self.display_stats()
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
            await sleep_ms(self.config.monitor_interval)

    def display_stats(self):
        runtime_min = int((time.time() - self.stats["start_time"]) / 60)
        win_count = len([x for x in self.sold_positions if x.get("pnl", 0) > 0])
        win_rate = (win_count / self.stats["successful_sells"] * 100) if self.stats["successful_sells"] else 0
        self.logger.info(
            f"📈 Stats ({runtime_min}m): PnL={format_usd(self.stats['total_pnl'])} "
            f"Buys={self.stats['successful_buys']}/{self.stats['total_buys']} "
            f"Sells={self.stats['successful_sells']}/{self.stats['total_sells']} "
            f"WinRate={win_rate:.1f}% Positions={len(self.positions)}/{self.config.max_positions}"
        )

    async def load_positions(self):
        try:
            with open(self.positions_file, "r", encoding="utf-8") as f:
                self.positions = json.load(f)
                self.seen_tokens = set(self.positions.keys())
            self.logger.info(f"📂 Loaded {len(self.positions)} active positions")
        except FileNotFoundError:
            self.positions = {}
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")

    async def save_positions(self):
        try:
            with open(self.positions_file, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving positions: {e}")

    async def load_sold_positions(self):
        try:
            with open(self.sold_positions_file, "r", encoding="utf-8") as f:
                self.sold_positions = json.load(f)
            self.stats["total_pnl"] = sum(float(x.get("pnl", 0)) for x in self.sold_positions)
            self.logger.info(f"📂 Loaded {len(self.sold_positions)} sold positions")
        except FileNotFoundError:
            self.sold_positions = []
        except Exception as e:
            self.logger.error(f"Error loading sold positions: {e}")

    async def save_sold_positions(self):
        try:
            with open(self.sold_positions_file, "w", encoding="utf-8") as f:
                json.dump(self.sold_positions, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving sold positions: {e}")

    async def shutdown(self):
        async with self.shutdown_lock:
            if self.shutdown_requested:
                return
            self.shutdown_requested = True
        self.logger.info("🛑 Shutting down...")
        await self.send_telegram_notification(
            "🛑 Soltrade bot shutting down\n"
            f"Open Positions: {len(self.positions)}\n"
            f"Realized PnL: {format_usd(self.stats['total_pnl'])}"
        )
        await self.save_positions()
        await self.save_sold_positions()
        self.display_stats()

    async def run(self):
        await self.initialize()
        await asyncio.gather(self.buy_loop(), self.monitor_loop())


async def main():
    bot = TradingBot()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.shutdown()))

    try:
        await bot.run()
    except Exception as e:
        bot.logger.error(f"Fatal error: {e}")
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())