# Running the AI — a short operating manual

How to run PPO for trading research and how to monitor it. For every command in
order see **PPO_USER_GUIDE.md**; for how it works see **AI_RL_DESIGN.md**.

---

## What this is

Your rule-based scalper, plus an **optional** PPO reinforcement-learning agent
that learns whether a short M1 bracket trade is worth taking **after costs**.

**The rule bot is untouched and is still the default.** If PPO is not installed,
not trained, or broken, everything behaves exactly as it did before.

---

## The three modes

| Mode | What happens | Risk |
|---|---|---|
| **RULE_ONLY** *(default)* | Your existing scalper. PPO not loaded. | none |
| **PPO_SHADOW** | Rule engine still decides. PPO watches, records and simulates. | **none — PPO cannot place an order** |
| **PPO_DEMO** | PPO decides. Signals go through every existing safety gate to your **demo** account. | demo money only |

There is **no live mode**, and one cannot be enabled by configuration.

---

## Quick start

```bash
# 1. install (optional - the rule bot does not need it)
pip install gymnasium stable-baselines3 torch

# 2. get data (Windows, MT5 running). Start with one year.
python scripts/download_mt5_history.py --symbol XAUUSD --start 2024-01-01

# 3. build features
python scripts/build_dataset.py --symbol XAUUSD

# 4. seal the holdout BEFORE any tuning
python scripts/final_holdout_eval.py --symbol XAUUSD --seal \
    --start 2025-07-01 --end 2025-12-31

# 5. train
python scripts/train_ppo.py --symbol XAUUSD \
    --train-end 2025-01-01 --validation-end 2025-06-01 --timesteps 100000

# 6. does it keep working?
python scripts/walk_forward.py --symbol XAUUSD --train-days 180 \
    --validation-days 30 --test-days 30 --timesteps 50000
```

Then run it in shadow for weeks before considering demo.

---

## Monitoring

### Telegram — **🤖 PPO**

`📊 PPO STATUS` is the screen to watch:

```
🤖 PPO STATUS

Symbol: XAUUSDs
Timeframe: M1
Mode: PPO_SHADOW
Model: ppo_v001

Latest action: BUY
BUY probability:  70%
SELL probability: 10%
HOLD probability: 20%

PPO paper trades: 42
NET R: +3.50R
Win rate: 55.0%

PPO is NOT placing orders.
These are simulated results only.
```

`📈 PPO PERFORMANCE` shows PPO **beside** the rule engine — never summed.
`🧠 PPO MODEL` shows the version, status and what it was trained on.

### Files

| What | Where |
|---|---|
| Every PPO decision (including HOLD) | `data/ai/<SYMBOL>/ppo_decisions.csv` |
| Simulated PPO trades | `data/ai/<SYMBOL>/ppo_trades.csv` |
| Rule paper results | `data/<market>/outcomes.csv` |
| Real demo fills | `data/<market>/executions.csv` |
| Walk-forward | `reports/ppo/<SYMBOL>/walk_forward/` |
| Model registry | `models/model_registry.json` |

**Those first four are separate books and are never merged.** That comparison is
the entire point of running shadow mode.

### Logs

```bash
tail -f data/system_log.txt | grep -i ppo
```

---

## What to watch for

| Sign | Meaning |
|---|---|
| `PPO paper trades: 0` after days | The agent is refusing to trade. Often **correct** with realistic costs — check the zero-cost control in the user guide §12 before changing anything. |
| `⚠️ Inference failures: N` | Those candles **held**. A high rate means the model is effectively switched off while looking switched on. |
| Walk-forward: 1 good fold, 4 bad | Noise, not an edge. Read the spread, not the total. |
| PPO NET R ≫ rule NET R in shadow | Promising — but shadow is still simulated. Demo fills are the next test, not proof. |
| `No PPO model is loaded` | Nothing is trained or promoted yet. The rule engine is running normally. |

---

## Safety

1. **`RULE_ONLY` is the default.** PPO must be turned on deliberately.
2. **Shadow cannot trade.** It has no broker connection at all.
3. **`PPO_DEMO` needs three separate switches** plus a verified demo account.
4. **PPO bypasses nothing.** Its signals pass the same account, symbol, spread,
   level, limit and sizing gates a rule signal does.
5. **Any failure = HOLD.** Missing model, bad features, NaN, exception — the
   system never trades because the AI broke.
6. **Only VALIDATED or better models load live.**
7. **Models are never overwritten.** `ppo_v001`, `ppo_v002`, …
8. **The holdout records every opening.**

**Panic button:** Telegram → 🤖 PPO → **⏸ PPO PAUSE**. Back to `RULE_ONLY`
immediately. Or delete `ai/` — the bot keeps working.

---

## Honest status

**PPO has not been shown to be profitable, on either market.**

The first end-to-end run on synthetic M1 with realistic costs produced an agent
that **always holds**. A zero-cost control on the identical pipeline traded 474
times at +0.108R — so the learning loop works, and holding is the correct
economic answer when the spread is comparable to the stop distance.

Your rule engine reached the same conclusion independently (−0.449R after
costs). Two different methods agreeing that M1 scalping is hard at retail
spreads is **information**, not failure.

The most valuable thing you can do next is run this on **real MT5 data with
recorded spreads**. Everything above is plumbing until then.

**Do not fix a disappointing result by lowering the cost assumptions.** That
does not make the strategy work — it makes the measurement stop being true.
