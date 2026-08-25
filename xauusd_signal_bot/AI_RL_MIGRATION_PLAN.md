# AI / RL Migration Plan — PPO for the M1 Scalper

**Phase 1 (audit) and Phase 2 (plan).** Written before any file was modified.
The existing rule-based bot is untouched by this document.

---

## 1. Current architecture

18,064 lines across 33 modules, 377 passing tests. Four layers, cleanly split:

```
MT5 (read-only)          src/market_data.py
      ↓
MarketSnapshot           closed M1 candles + M5 context + live spread
      ↓
Analysis engines         trend, momentum, structure, liquidity, support_resistance,
   (nine, all pure)      volume, volatility, price_action, regime
      ↓
scoring.py               weighted 0-100 bull/bear scores
      ↓
filters.py               threshold, conflict, fakeout, spread, session, cooldown
      ↓
targets.py               ATR-relative SL/TP1/TP2/TP3 + round-trip cost model
      ↓
Signal                   → signal_tracker.py  → signals.csv / outcomes.csv  (PAPER)
                         → demo_execution.py  → executions.csv             (DEMO, opt-in)
                         → telegram_bot.py    → cards and alerts
```

Cross-cutting: `config.py` (`for_market()` folds a `MarketConfig` onto the base
config — the single market seam), `runtime_state.py` (global state + one
`MarketRuntime` per market), `telegram_control.py` (inline panel).

### Facts that constrain the RL design

| Fact | Consequence for PPO |
|---|---|
| `ExecutionMode` asserts **exactly two members** (`assert_no_live_mode`) | PPO modes must NOT extend it. Strategy ("who decides") and execution ("may we send orders") are orthogonal — see §5. |
| `resample_candles()` returns **complete buckets only** | Reusable verbatim for causal M5/M15 features. Already the right semantics. |
| Ambiguous candle → **SL first** (`signal_tracker.py:271`) | The RL environment must use the identical rule. Not re-derived — extracted and shared. |
| `Config.for_market()` overlays ~35 fields incl. all paths | PPO datasets, models and reports get per-market paths for free. |
| Costs live in `config.round_trip_cost()` | The environment charges the **same** function. PPO cannot get better fills than the backtester. |
| 377 tests, incl. anti-lookahead and market isolation | The regression baseline. Any PPO change that breaks one is rejected. |
| MetaTrader5 is Windows-only, imported lazily | The downloader must import lazily too, and every RL test must run without it. |

---

## 2. Existing scalping logic (what PPO is competing with)

Nine weighted components → 0-100 → threshold 68 (regime-adjusted) → filter chain
→ ATR-relative targets (0.45/1.00/1.70 × ATR) floored by cost → 15-candle timeout
→ outcome tracked with RAW and NET R separately.

**Measured result on synthetic M1: −0.449R average NET (gold), −0.344R (BTC).**
Negative after costs. That is the baseline PPO must beat, and it is a *low* bar
— which is exactly why the holdout must stay sealed.

---

## 3. What remains, untouched

* Every analysis engine, `scoring.py`, `filters.py`, `targets.py`, `signal_engine.py`
* `signal_tracker.py`, paper outcome tracking, `outcomes.csv`
* `demo_execution.py` and all ten of its safety gates
* `markets.py`, `config.py`, `runtime_state.py`, market isolation
* The whole existing Telegram panel
* All 377 tests — the regression gate for every phase

**RULE_ONLY remains the default and stays fully usable with PPO absent,
uninstalled, or crashed.**

---

## 4. What is replaced

**Nothing is replaced.** PPO is added *alongside* as a second opinion. The rule
engine is never removed, never modified, and never has PPO spliced into its
decision path. This is deliberate: with the rule engine at −0.449R, there is no
evidence PPO will do better, and destroying the working baseline to find out
would be indefensible.

Only two existing files gain PPO awareness, both additively:
`main.py` (a hook to feed the shadow recorder) and `telegram_control.py`
(a 🤖 PPO submenu).

---

## 5. What PPO controls

Strategy mode is a **new, independent** enum:

| Mode | Rule engine | PPO | Orders |
|---|---|---|---|
| `RULE_ONLY` (default) | decides | not loaded | rule signals only |
| `PPO_SHADOW` | decides | observes and records | rule signals only |
| `PPO_DEMO` | still records paper | decides | PPO signals, through the **same** ten safety gates |

`PPO_DEMO` additionally requires `EXECUTION_MODE=DEMO_AUTO` + `DEMO_TRADING_ENABLED`
+ verified demo account. **PPO cannot bypass a single existing safety gate** —
it produces a `Signal`, and that signal enters `demo_execution.py` at exactly the
point a rule signal does.

Fail-safe: any exception, missing model, NaN observation, out-of-range action or
malformed output → log, count, and fall back to `RULE_ONLY` for that candle.
Never "execute because the AI errored".

---

## 6. New modules

```
ai/
├── features/    m1_features.py, mtf_features.py, feature_pipeline.py
├── environment/ scalping_env.py         (Gymnasium, market-agnostic)
├── ppo/         train.py, evaluate.py, inference.py, checkpoints.py
├── evaluation/  walk_forward.py, holdout.py, metrics.py
├── shadow.py    decision recorder
├── strategy_mode.py
└── model_registry.py

scripts/ download_mt5_history.py, build_dataset.py, train_ppo.py,
         walk_forward.py, final_holdout_eval.py, retrain_ppo.py
src/     mt5_history.py   (MT5 chunked download + quality report)
```

`ai/` is a **separate top-level package**. The rule engine does not import from
it; the dependency arrow points one way only, so deleting `ai/` leaves a working
bot.

---

## 7. Data pipeline

```
MT5 →  data/raw/<SYMBOL>/M1/<year>.csv        immutable, never rewritten
       + data_quality_report.json
   →  ai/features (causal only)
   →  data/processed/<SYMBOL>/M1/dataset_<version>.parquet|csv
```

Raw preserves `time, open, high, low, close, tick_volume, spread, real_volume`.
Feature building **reads** raw and never writes to it. Missing candles are
reported, never fabricated.

**Causality contract:** at candle *t* the observation may use only candles
`≤ t`. MTF features use `resample_candles()` (complete buckets) and are then
shifted so the M5 bucket containing *t* is never visible until it closes.
Enforced by a future-append stability test (§7 of the brief): compute features
on `[0..n]`, append `[n+1..m]`, recompute, and assert rows `[0..n]` are
bit-identical within tolerance.

---

## 8. Training pipeline

Chronological only, never shuffled. Small MLP (`[64, 64]`) for CPU. Reward is
**NET R after the existing cost model**, never dollars, never trade count.
Walk-forward: train → validate (model selection) → test (untouched until the
fold ends) → slide.

## 9. Live / shadow pipeline

Same features, same environment semantics, same cost model. Shadow records every
M1 decision to `data/ai/<SYMBOL>/ppo_decisions.csv`, simulates the trade it would
have taken, and reports PPO paper NET R **separately** from rule paper results
and from demo execution results. Three books, never merged.

## 10. Safety mechanisms

1. `RULE_ONLY` default; PPO opt-in per mode, `PPO_DEMO` manually enabled only.
2. Fail-safe on any PPO error → `RULE_ONLY` for that candle.
3. PPO output validated: finite, in-range action, coherent SL/TP sides.
4. All ten existing demo gates apply unchanged to PPO signals.
5. `assert_no_live_mode()` still holds — no live mode gains entry via PPO.
6. Sealed holdout, enforced by a file the evaluator refuses to read early.
7. Model promotion gated; an unvalidated model never auto-replaces a validated one.

## 11. Testing strategy

Every phase ends with the full 377-test suite plus that phase's new tests.
New areas: causality and the future-append test, resampling boundaries, env
reset/step/action validation, SL/TP/timeout/pessimism, reward arithmetic,
walk-forward split ordering, holdout sealing, registry versioning, shadow
isolation, fail-safe fallback, XAUUSDs/BTCUSDs separation, Telegram routing.

**Dependency posture:** `gymnasium`, `stable-baselines3` and `torch` are optional.
Every `ai/` module imports them lazily; tests that need them `skip` when absent
so the existing suite still runs on a machine without them.

---

## Phase order and status

| Phase | Status |
|---|---|
| 1 Audit · 2 Plan | ✅ this document |
| 3 MT5 collector · 4 Validation | next |
| 5 Features · 6 Environment · 7 PPO · 8 Smoke train | then |
| 9 Walk-forward · 10 Holdout | then |
| 11 Shadow · 12 Telegram · 13 PPO demo | then |
| 14 Collection · 15 Retraining | last |
