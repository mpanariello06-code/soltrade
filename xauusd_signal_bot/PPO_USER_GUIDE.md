# PPO User Guide — every command, in order

Ten tasks, start to finish. Design rationale is in **AI_RL_DESIGN.md**;
a short operating manual is in **README_AI.md**.

> **Nothing here can place a live trade.** PPO starts in shadow mode, and even
> `PPO_DEMO` only reaches a **verified demo account** through the same safety
> gates a rule signal uses. There is no live mode.

---

## 0. Install (once)

The RL stack is **optional**. Your existing bot does not need it.

```bash
pip install gymnasium stable-baselines3 torch
```

CPU-only is fine and is what this is designed for. To check:

```bash
python -c "from ai.ppo.train import detect_device; print(detect_device())"
```

---

## 1. Download MT5 history

Windows only — it needs your running terminal.

```bash
python scripts/download_mt5_history.py --symbol XAUUSD --timeframe M1 \
    --start 2015-01-01 --end 2026-08-25
```

Writes `data/raw/XAUUSDs/M1/<year>.csv` plus `data_quality_report.json`.

**Start small.** One year first, confirm the pipeline runs end to end, then
scale up:

```bash
python scripts/download_mt5_history.py --symbol XAUUSD --start 2024-01-01
```

Read the report. Gaps are normal (the market closes). Duplicates, disordered
rows or bad OHLC mean `USABLE: NO` — fix the source, because **nothing here will
repair it for you**: a fabricated candle is indistinguishable from a real one
once it is on disk.

---

## 2. Build the dataset

```bash
python scripts/build_dataset.py --symbol XAUUSD
```

Turns raw candles into 65 causal features. Raw files are never modified. Note
the **fingerprint** — it identifies this exact feature contract.

---

## 3. Seal the holdout — do this BEFORE any tuning

```bash
python scripts/final_holdout_eval.py --symbol XAUUSD --seal \
    --start 2026-01-01 --end 2026-08-25
```

Training and walk-forward now exclude that period automatically. Sealing it
afterwards is worthless, and moving it later is refused.

---

## 4. Train

```bash
python scripts/train_ppo.py --symbol XAUUSD \
    --train-end 2025-01-01 --validation-end 2025-07-01 \
    --timesteps 200000
```

~28 s per 20k timesteps on 4 CPU cores, so 200k ≈ 5 minutes.

Useful flags: `--seed`, `--learning-rate`, `--ent-coef`, `--net-arch 128 128`,
`--holding-candles 5`, `--advanced-actions`.

Registered **EXPERIMENTAL**. It will not be used live until promoted.

> **If it reports 0 trades**, that is often correct, not a bug — with realistic
> costs, holding may genuinely be the best policy. Confirm the loop works by
> running a zero-cost control (see §10) before concluding anything.

---

## 5. Walk-forward

The real test: *does it keep working*, not *did it work once*.

```bash
python scripts/walk_forward.py --symbol XAUUSD \
    --train-days 365 --validation-days 60 --test-days 60 --timesteps 50000
```

Writes `reports/ppo/XAUUSDs/walk_forward/walk_forward_results.csv` and
`walk_forward_equity.csv`.

**Read the spread across folds, not the total.** One good fold beside several
poor ones is noise. `positive_folds` vs `negative_folds` and `std_fold_net_r`
matter more than `total_test_net_r`.

---

## 6. Compare holding windows — on VALIDATION only

```bash
for h in 3 5 10 15; do
  python scripts/train_ppo.py --symbol XAUUSD --holding-candles $h \
      --train-end 2025-01-01 --validation-end 2025-07-01 --timesteps 100000
done
```

Pick on validation. **Never on the holdout** — that is what sealing is for.

---

## 7. Final holdout — once, with the model frozen

```bash
python scripts/final_holdout_eval.py --symbol XAUUSD --version ppo_v003 --open
```

Without `--open` it only reports the seal's status. Every opening is recorded
permanently. **Do not retune and re-open** — a holdout opened twice is a
validation set with extra steps.

---

## 8. Shadow mode

In `.env`:

```dotenv
STRATEGY_MODE=PPO_SHADOW
```

Or from Telegram: **🤖 PPO → ○ SHADOW**.

The rule engine keeps deciding. PPO watches every M1 candle, records to
`data/ai/XAUUSDs/ppo_decisions.csv`, and simulates its trades to
`ppo_trades.csv`. **It cannot place an order in this mode.**

Run it for weeks. This is the only honest bridge between a backtest and real
money, and it costs nothing but patience.

---

## 9. Telegram

**🤖 PPO** on the main panel:

| Button | Shows |
|---|---|
| 📊 PPO STATUS | mode, model, latest action, BUY/SELL/HOLD probabilities, simulated NET R, win rate, inference failures |
| ○ RULE ONLY / ○ SHADOW | switch mode |
| ○ PPO DEMO | asks to confirm first |
| 📈 PPO PERFORMANCE | PPO simulated **beside** rule paper — never summed |
| 🧠 PPO MODEL | version, status, features, fingerprint, training dates |
| ⏸ PPO PAUSE | back to `RULE_ONLY` immediately |

---

## 10. PPO demo mode — only after shadow testing

**Three things are required, and any one missing means no order:**

```dotenv
STRATEGY_MODE=PPO_DEMO
EXECUTION_MODE=DEMO_AUTO
DEMO_TRADING_ENABLED=true
DEMO_MT5_LOGIN=...
DEMO_MT5_PASSWORD=...
DEMO_MT5_SERVER=...
```

Then Telegram: **🤖 PPO → ○ PPO DEMO → ✅ ENABLE**.

PPO signals pass **every** existing gate: demo-account verification, symbol,
spread, level coherence, position and daily limits, sizing. PPO gets no
privileged route to the broker.

The model must also be `VALIDATED` or better — an `EXPERIMENTAL` model is never
loaded live.

---

## 11. Retrain and promote

```bash
python scripts/retrain_ppo.py --symbol XAUUSD --timesteps 200000 \
    --min-trades 50 --min-net-r 0.0 --min-profit-factor 1.0
```

Trains a challenger, walk-forward evaluates it, and promotes **only** if it
clears the gates **and** beats the incumbent. Otherwise the current model is
left exactly where it is. Add `--promote` to go straight to PRODUCTION;
without it the candidate stops at VALIDATED so you can shadow it first.

---

## 12. Sanity check: is the loop learning?

If the agent never trades, prove the machinery works before touching anything:

```python
from ai.environment.scalping_env import EnvConfig
zero_cost = EnvConfig(assumed_spread_points=0.0, use_data_spread=False,
                      slippage_points_entry=0.0, slippage_points_exit=0.0,
                      trade_penalty=0.0)
```

Train on that. If it now trades, the loop is fine and HOLD was the correct
economic answer. If it still refuses, something is broken.

**Do not fix a disappointing result by lowering the cost assumptions.** That
does not make the strategy work; it makes the measurement stop being true.

---

## BTCUSD

Every command takes `--symbol BTCUSD`. Separate data, separate models, separate
registry entries, separate holdout seals.

**Do not assume one model works on both.** They have different volatility,
different cost structures and different session behaviour.
