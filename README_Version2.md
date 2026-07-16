# Soltrade (Python, HTTP-only)

Python version of the Solana trading bot using HTTP polling only (no WebSocket mode).

## Setup

1. Create venv and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# then edit .env
```

3. Run:
```bash
python index.py
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