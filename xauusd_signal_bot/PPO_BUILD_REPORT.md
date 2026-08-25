# PPO Build Report

Phases 1–13 of the RL migration. The rule-based scalper is intact and remains
the default.

---

## FILES CREATED

**Research package (`ai/`)** — 3,247 lines

| File | Purpose |
|---|---|
| `ai/features/m1_features.py` | 42 causal M1 features, ATR/price normalised |
| `ai/features/mtf_features.py` | M5/M15 context merged on bucket CLOSE time |
| `ai/features/feature_pipeline.py` | `FeatureSpec` contract, dataset build, chronological splits |
| `ai/environment/scalping_env.py` | Gymnasium bracket environment |
| `ai/ppo/train.py` | PPO training loop and checkpointing |
| `ai/ppo/evaluate_ppo.py` | Re-measure a model without retraining |
| `ai/ppo/checkpoints.py` | Checkpoint inspection, resume, archive, verify |
| `ai/ppo/inference.py` | Fail-safe serving |
| `ai/evaluation/walk_forward.py` | Sliding folds, non-overlapping tests |
| `ai/evaluation/holdout.py` | Sealed holdout with an audit trail |
| `ai/evaluation/metrics.py` | NET-first metrics, equity, regime breakdown |
| `ai/evaluation/report.py` | The `reports/ppo/` artefact set, one writer |
| `ai/model_registry.py` | Versioning, lifecycle, promotion gates |
| `ai/shadow.py` | Decision recorder and trade simulator |
| `ai/strategy_mode.py` | `RULE_ONLY` / `PPO_SHADOW` / `PPO_DEMO` |

**Live-bot side (`src/`, `scripts/`)**

`src/mt5_history.py` (chunked download + quality report), `src/ppo_bridge.py`
(the single seam), and seven scripts: `download_mt5_history`, `build_dataset`,
`train_ppo`, `walk_forward`, `final_holdout_eval`, `retrain_ppo`, `_common`.

**Tests** — 130 new across `test_ai_features.py` (13), `test_ai_environment.py`
(23), `test_ai_ppo.py` (58), `test_ai_integration.py` (36).

**Docs** — `AI_RL_MIGRATION_PLAN.md`, `AI_RL_DESIGN.md`, `PPO_USER_GUIDE.md`,
`README_AI.md`, this report.

## FILES MODIFIED

| File | Change |
|---|---|
| `main.py` | PPO bridge, `_observe_ppo` hook, three Telegram hooks |
| `backtest.py` | `--strategy RULE_ONLY\|PPO_BACKTEST\|PPO_SHADOW`, `--model` |
| `src/telegram_control.py` | 🤖 PPO submenu and four renderers |
| `tests/test_telegram_control.py` | one keyboard-layout assertion |
| `requirements.txt`, `.gitignore` | optional deps, RL artefacts |

## FILES DELETED

**None.**

`git diff` over `signal_engine.py`, `scoring.py`, `filters.py`, `targets.py`,
`indicators.py` and all nine analysis engines is **empty**.

---

## TEST RESULTS

**507 passed** (377 existing + 130 new), ~135 s. `pyflakes` clean.
Zero regressions.

Coverage of note: the future-append leakage test; MTF bucket visibility;
pessimistic ambiguity; reward-equals-NET-R; every inference failure path;
registry versioning and promotion; holdout sealing and audit; shadow isolation;
XAUUSDs/BTCUSDs separation throughout.

---

## DATA PIPELINE

```
MT5 → data/raw/<SYMBOL>/M1/<year>.csv   (immutable) + data_quality_report.json
    → ai/features (causal)
    → data/processed/<SYMBOL>/M1/dataset_<fingerprint>.csv
```

Duplicates removed, order restored. **Gaps, bad OHLC and abnormal spreads are
reported, never repaired** — a fabricated candle is indistinguishable from a
real one once on disk. Re-downloading merges rather than truncating.

## MT5 DATA

`copy_rates_range` in 30-day chunks; server timestamps converted to UTC.
`time, open, high, low, close, tick_volume, spread, real_volume` all preserved.
Windows-only; everything downstream runs anywhere.

## FEATURES

65 columns: 42 M1 + 24 M5/M15 context (minus the excluded ATR anchor).
Everything is a multiple of ATR or a fraction of price, so gold at 2,300 and
Bitcoin at 60,000 give comparable inputs. Warm-up rows dropped, never filled.

## OBSERVATION SPACE

`Box(69)` — 65 features + 4 position channels (direction, unrealised R, age,
fraction remaining).

## ACTION SPACE

`Discrete(3)`: HOLD / BUY / SELL. Optional `MultiDiscrete([3,3,3])`:
direction × SL bucket × TP bucket.

## REWARD FUNCTION

NET R after costs, per closed portion; minus `trade_penalty` on entry,
`holding_penalty_per_candle` while open, `drawdown_penalty × drawdown` on a new
low; clipped to ±5. `hold_reward = 0.0` — paying an agent to do nothing is the
easiest policy to learn and teaches nothing.

## PPO ARCHITECTURE

SB3 `MlpPolicy`, `[64, 64]`. Seeded and reproducible. CPU auto-detected.
**20,000 timesteps in 28 s on 4 cores.**

## TRAINING

Chronological, never shuffled. Validation selects; test is seen once.
Checkpoints: `best_model`, `latest`, `final_model`, plus `feature_spec.json`
and `training_result.json`.

## WALK-FORWARD

Sliding folds; test windows abut but never overlap. Writes
`walk_forward_results.csv` and `walk_forward_equity.csv`. `aggregate()` reports
the **spread** across folds, not just the total.

## HOLDOUT

A JSON seal records the period and **every opening**. Training and walk-forward
exclude it at the data layer. Moving a seal after the fact is refused.

## SHADOW MODE

Records every M1 decision including HOLD; simulates the implied trade with the
same pessimism and the same costs. **No broker connection exists in this mode.**

## DEMO MODE

`PPO_DEMO` + `EXECUTION_MODE=DEMO_AUTO` + `DEMO_TRADING_ENABLED` + a verified
demo account. PPO produces a `Signal` that enters `demo_execution.py` at exactly
the point a rule signal does — **no privileged route, no skipped gate**.

## BACKTESTING  (spec section 31)

`backtest.py --strategy` selects `RULE_ONLY` (default, unchanged),
`PPO_BACKTEST` (PPO alone) or `PPO_SHADOW` (both over the same candles, printed
side by side). `EnvConfig.from_market_config()` copies the live bot's own
spread, slippage, commission, holding window and target geometry, and `--spread`
applies to both sides — **PPO is never given better fills than the engine it is
compared against**, and a test asserts the cost figures agree.

## REPORTS  (spec sections 32, 33)

`ai/evaluation/report.py` is the single writer for `reports/ppo/`:
`performance.csv` (+ `.json`), `equity.csv`, `trades.csv`, `drawdown.csv`
(depth *and* consecutive underwater trades), `walk_forward.csv`, and
`by_regime.csv` — session and volatility breakdown, because a blended number
hides an agent that works in exactly one regime.

## TELEGRAM

🤖 PPO → STATUS / RULE ONLY / SHADOW / PPO DEMO (confirmed) / PERFORMANCE /
MODEL / PAUSE. Status shows probabilities, simulated NET R and inference
failures. Performance shows PPO **beside** the rule engine, never summed.

## MODEL VERSIONING

`models/ppo/<SYMBOL>/ppo_vNNN/`, never overwritten. Lifecycle
`EXPERIMENTAL → VALIDATED → SHADOW → PRODUCTION → RETIRED`. Only VALIDATED or
better loads live. Feature fingerprint checked at load.

---

## BUGS FOUND AND FIXED

1. **Reward double-counting.** `_close_position` returned the trade *total*
   while TP1/TP2 partials had already been paid, making a laddered winner look
   better than the same move taken in one piece. Reward now covers only the
   portion closed. Caught by a test asserting reward == NET R.
2. **TP3 closes left `tp_hits` at 2** instead of 3.
3. **`isolate()` did not repoint `executions_csv`**, so an earlier test run had
   been appending to the real data directory.

---

## FIRST EXPERIMENT — reported honestly

Synthetic M1, 20k candles, 12-point spread, chronological split, 20k timesteps:

| Run | Trades | Avg NET R | Action mix |
|---|---|---|---|
| Realistic costs | **0** | — | 100% HOLD |
| Zero-cost control | 474 | +0.108R | 29% HOLD / 64% BUY / 8% SELL |

The learning loop works. With realistic costs the agent **correctly refuses to
trade**: at a 12-point spread against a ~0.23 stop, round-trip cost is ≈1.0R per
trade, so a stop-out costs ≈−2R.

Your rule engine reached the same conclusion independently (−0.449R after
costs). **Two different methods agreeing that M1 scalping is hard at retail
spreads is information, not failure.**

---

## KNOWN LIMITATIONS

* **No profitability has been demonstrated**, on either market, by either method.
* **All results so far are on synthetic data.** They test the plumbing, nothing
  more. Real MT5 history with recorded spreads is the next real step.
* **The MT5 downloader cannot be integration-tested here** — Windows-only. Its
  logic is unit-tested against synthetic rates; its live request shapes are not.
* **No tick-level fidelity.** M1 OHLC cannot resolve intra-candle order, so
  ambiguity is always scored as a stop. Live tick replay would be more accurate
  and less pessimistic.
* **Shadow simulation is not execution.** It models fills; it does not get them.
* **The agent may simply learn to hold.** That is a legitimate answer, not a bug,
  and must not be "fixed" by lowering cost assumptions.
* **BTCUSDs has no model and no data yet.** The pipeline supports it; nothing has
  been run.
* Walk-forward retrains per fold, so a long history is slow. Start small.

---

## NEXT STEPS

1. **Download real XAUUSDs M1 history with recorded spreads.** Everything else is
   premature until the cost model uses measured spreads rather than an
   assumption. Start with one year.
2. **Re-run the first experiment on it.** If the agent still holds under real
   spreads, that is a finding about the strategy, not the model.
3. **Seal a holdout before any tuning**, then walk-forward. Read the spread
   across folds, not the total.
4. **Compare holding windows (3/5/10/15) on validation only.** The 15-candle
   default was inherited from the rule engine and is not known to be right.
5. **Run shadow for weeks** before considering `PPO_DEMO`. It is the only honest
   bridge between a backtest and real fills, and it costs only patience.
6. Only then consider a larger network, tick-level simulation, or BTCUSDs.

**Phases 14–15** (continuous collection, automated retraining) have their
plumbing in place — `ppo_decisions.csv` accumulates in shadow, and
`retrain_ppo.py` gates promotion — but neither should run until there is real
data to run them on.
