# PPO Reinforcement Learning — Design

The technical reference. For step-by-step commands see **PPO_USER_GUIDE.md**;
for a short operating manual see **README_AI.md**.

---

## 1. What the agent is actually learning

Not "will price go up". The question is:

> Is a short bracket trade taken **here** expected to return positive **NET R
> after costs**?

Those are different questions, and at a three-pip target the difference *is* the
edge: a move can be correctly predicted and still lose money. The reward is
therefore NET R — never dollars, never trade count.

## 2. Architecture

```
MT5 ──► data/raw/          immutable, per year, never rewritten
         │
         ▼
    ai/features/           causal only, ATR/price normalised
         │
         ▼
   ai/environment/         Gymnasium bracket sim, real costs
         │
         ▼
      ai/ppo/              PPO (SB3), small MLP, CPU
         │
         ▼
   ai/evaluation/          walk-forward, sealed holdout, metrics
         │
         ▼
 src/ppo_bridge.py ──► main.py     the ONE seam
```

`ai/` never imports the live bot's runtime. `src/ppo_bridge.py` holds every
`ai/` import inside a `try`, so a missing `torch`, a missing model or a corrupt
checkpoint degrades to `RULE_ONLY`. **Deleting `ai/` leaves a working bot.**

## 3. Strategy modes

| Mode | Rule engine | PPO | Orders |
|---|---|---|---|
| `RULE_ONLY` *(default)* | decides | not loaded | rule signals |
| `PPO_SHADOW` | decides | observes, records, simulates | rule signals |
| `PPO_DEMO` | still records paper | decides | PPO signals, **same gates** |

Deliberately separate from `ExecutionMode`, which asserts it has exactly two
members so a live mode cannot be added. They compose:

```
StrategyMode.PPO_DEMO  AND  EXECUTION_MODE=DEMO_AUTO
                       AND  DEMO_TRADING_ENABLED=true
                       AND  a verified demo account
```

Anything weaker means no order.

## 4. Observation space

**69 values**: 65 causal features + 4 position channels.

### M1 block (42)

| Group | Features |
|---|---|
| Candle shape | `range/ATR`, `body/ATR`, upper and lower wick `/ATR`, body-to-range ratio, direction, close location |
| Returns | 1/3/5/15/30-candle returns `/ATR`, momentum, acceleration |
| Volatility | 15- and 60-candle log-return std, and their ratio (regime) |
| Range position | distance from 5/15/60-candle highs and lows `/ATR`, position 0–1 |
| Oscillators | RSI (0–1), MACD line/signal/histogram `/ATR` |
| Trend | EMA9−EMA21 `/ATR`, price vs EMA21, EMA21 slope |
| Streak | signed consecutive same-direction closes, capped |
| Liquidity/cost | tick volume vs its 60-candle median, spread, spread ratio |
| Time | cyclical sin/cos of time-of-day and day-of-week |

### M5 + M15 context (24 total, 12 each)

Range, body, 1- and 3-bucket returns, RSI, EMA spread, price vs EMA, EMA slope,
distance from 20-bucket extremes, position, ATR as a fraction of price.

### Position channels (4)

Direction (±1), unrealised R, age as a fraction of the holding window, fraction
of the position still open.

### Why normalised

Every feature is a multiple of ATR or a fraction of price. Gold at 2,300 and
Bitcoin at 60,000 produce comparable inputs, and a model cannot learn the price
*level* instead of the *behaviour*. `f_atr` itself is computed but **excluded**
from the observation — it is in raw price units and would smuggle the level back
in through the door every other feature was designed to close.

## 5. Causality

At candle *t* the agent may see only data from candles ≤ *t*.

* Rolling windows look backwards only — no `center=True`, no negative `shift`.
* Warm-up rows are **dropped, never filled**. A back-filled RSI is a value the
  agent could not have computed.
* **MTF is the dangerous part.** Naïvely joining M1 to the M5 bucket it belongs
  to hands the agent, at 12:01, a bucket that does not close until 12:04 — three
  minutes of future. Instead each bucket is stamped with its **close** time and
  merged with `direction="backward"`, so at 12:01 the newest visible bucket is
  11:55–12:00.

**Enforced, not asserted:** `tests/test_ai_features.py` computes features over
`[0..n]`, appends later candles, recomputes, and fails if any historical value
moves. A leaking feature does not crash — it produces an excellent backtest that
never reproduces live, which is worse, because it is believed.

## 6. Action space

**Discrete(3)** by default: `0 HOLD`, `1 BUY`, `2 SELL`.

**MultiDiscrete([3,3,3])** optionally — direction × SL bucket × TP bucket, 27
combinations. Small on purpose: a larger action space is a slower thing to
learn, not a richer one.

| Bucket | SL (×ATR) | TP scale |
|---|---|---|
| 0 | 0.50 tight | 0.75 conservative |
| 1 | 0.70 normal | 1.00 normal |
| 2 | 0.90 wide | 1.40 extended |

## 7. Reward function

```
on close:      NET R  =  gross R − cost R          (for the portion closed)
each candle:   − holding_penalty_per_candle
on entry:      − trade_penalty
on new low:    − drawdown_penalty × drawdown
HOLD:          hold_reward (default 0.0)
finally:       clip to ±reward_clip
```

| Term | Default | Why |
|---|---|---|
| `trade_penalty` | 0.02 | The agent must believe a setup beats doing nothing. Discourages churn without dictating a trade count. |
| `holding_penalty_per_candle` | 0.002 | A scalp that drifts is not free. |
| `drawdown_penalty` | 0.05 | Two paths to the same total R are not equally good. |
| `hold_reward` | **0.0** | Paying an agent to do nothing is the easiest policy to learn and teaches nothing. |
| `reward_clip` | 5.0 | One freak candle must not dominate a batch's gradient. |

`gross R` is the price move divided by the risk taken at entry. `cost R` is the
round-trip cost divided by the same risk — a 2-pip cost is trivial against a
50-pip stop and ruinous against a 3-pip one, and only the ratio says which
situation the agent is in.

**Partials are paid when they fire**, not re-counted at the close. (The first
implementation returned the trade total at close on top of the partials already
paid, making a laddered winner look better than the same move taken in one
piece. Fixed, with a test.)

## 8. Execution realism

This is what decides whether any of the numbers mean anything.

| | Rule |
|---|---|
| Entry | BUY lifts the **ask**, SELL hits the **bid**, plus entry slippage. The candle close is never the fill. |
| Costs | `spread + entry slip + exit slip + 2 × commission`, from the **live bot's own** `round_trip_cost()`. PPO gets no better fills than the rule engine's backtester. |
| Spread | Per-candle from MT5 when present; a configurable assumption otherwise. |
| **Ambiguity** | **If one M1 candle trades through both the stop and a target, the STOP is assumed.** Identical to `signal_tracker.py`. Two copies of a pessimism rule eventually disagree and the optimistic one wins, so the rule is shared. |
| Timeout | Existing `MAX_HOLDING_CANDLES`. No new timeout concept. |
| Breakeven | Follows `MOVE_SL_TO_BREAKEVEN_AFTER_TP1` — used if the strategy uses it, not introduced if it does not. |
| Episode end | An open position is **closed at market**, not abandoned — otherwise every loser at an episode edge silently disappears. |

## 9. PPO model

Small MLP `[64, 64]`. The first question is whether PPO can learn anything here
at all; a Transformer answers that more slowly, no more clearly, and would not
train on a 4-core CPU. Configurable via `--net-arch` when there is evidence a
larger one is needed.

Defaults: `lr 3e-4`, `n_steps 2048`, `batch 64`, `gamma 0.99`,
`gae_lambda 0.95`, `clip 0.2`, `ent_coef 0.01`, `vf_coef 0.5`,
`max_grad_norm 0.5`, `seed 42`. Device auto-detects CUDA, falls back to CPU.

`ent_coef` is slightly above the SB3 default: with HOLD always available a
scalping agent collapses to "never trade" very easily, and an agent that never
trades has learned nothing.

**Measured: 20,000 timesteps in 28 s on 4 CPU cores.**

## 10. Training, walk-forward, holdout

Chronological only. **Never shuffled** — a shuffled split lets a model learn
from Thursday to trade Monday, and every metric after that is fiction.

Walk-forward slides train → validation → test. Validation selects the model;
test is looked at once per fold. Test windows abut but never overlap, so no
period is counted twice.

The **sealed holdout** is mechanical: a JSON file records the reserved period
and every opening. Training and walk-forward exclude it at the data layer, so it
cannot reach a model by accident. It cannot stop a determined person — nothing
can — but it makes an accidental peek impossible and a deliberate one
**recorded**. A holdout opened five times is not a holdout, and afterwards it is
at least obvious that it was.

## 11. Model versioning

`models/ppo/<SYMBOL>/ppo_vNNN/` — model, `feature_spec.json`,
`training_result.json`, `environment.json`, and `best_model` / `latest` /
`final_model` checkpoints. **Never overwritten**; re-registering a version is
refused.

Lifecycle: `EXPERIMENTAL → VALIDATED → SHADOW → PRODUCTION → RETIRED`.
Promoting to PRODUCTION retires the previous one, because "which is live?" must
have one answer. Only VALIDATED or better is ever loaded live.

The **feature fingerprint** — a hash of the column list *and its order* — is
stored with every model and checked at load. Serving a model different columns
from those it trained on is silent and catastrophic, so it is made impossible.

## 12. Fail-safe

Every failure answers HOLD: missing model, unreadable checkpoint, feature
mismatch, wrong observation width, NaN, out-of-range action, any exception. The
failure mode being prevented is the system placing a trade *because* the AI
broke.

## 13. Three books, never merged

| Book | File |
|---|---|
| Rule paper | `data/<market>/outcomes.csv` |
| PPO simulated | `data/ai/<SYMBOL>/ppo_trades.csv` |
| Real demo fills | `data/<market>/executions.csv` |

Merging any two destroys the comparison that justifies the whole exercise. They
are written by different code to different files and are never summed.

## 14. Honest status

**PPO has not been shown to be profitable, on either market.**

The first end-to-end experiment on synthetic M1 with realistic costs produced an
agent that **always holds**. A zero-cost control on the identical pipeline trades
474 times at +0.108R, so the learning loop works — HOLD is the correct economic
answer when the spread is comparable to the stop distance.

That is the same conclusion the rule engine reached independently (−0.449R
after costs). Two different methods agreeing that this is hard is information,
not failure.
