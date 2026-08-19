# Build Report — Research Mode & Telegram Control Upgrade

Upgrade of the existing `xauusd_signal_bot` project. **The project was not
rebuilt.** The scoring architecture, the nine analysis engines, their weights
and the anti-lookahead machinery are unchanged; the work added an operating-mode
layer, live Telegram control, multi-timeframe support and richer data capture
on top of them.

* **Tests:** 252 passing (was 151), ~90 s
* **Static analysis:** `pyflakes` clean across all modules
* **Order execution:** still none — verified by grep, see [Verification](#verification-performed)

---

## 1. What changed, and why

### The one architectural seam

Rather than thread a settings object through every engine, all live settings are
folded into a **copy of the config** by `src/runtime_state.py::config_view()`.
Every analysis engine still reads plain config attributes (`signal_timeframe`,
`base_threshold`, …) exactly as before — they simply now reflect whatever the
Telegram panel last set. This is why a fairly large feature set landed without
touching `trend.py`, `momentum.py`, `structure.py`, `liquidity.py`,
`support_resistance.py`, `volume.py`, `volatility.py`, `price_action.py` or
`regime.py` at all.

The same function backs the backtester, so an offline run reproduces exactly
what live mode would do for a given mode/timeframe.

### Threshold model

The old model stored one **absolute** threshold per regime. That could not
express "mode × timeframe × regime". It is now compositional:

```
effective threshold = base(mode, signal timeframe)
                    + regime offset
                    + counter-trend extra
                    → clamped to [min_threshold, max_threshold]
```

The regime offsets were derived from the previous absolute values, so
**STANDARD/M5 behaviour is numerically identical to before** (68 / 72 / 77 for
strong-trend / weak-trend / range).

### Scoring: weight renormalisation

H4 has no confirmation timeframe, so the HTF component (15 points) cannot apply.
Scoring it as a flat zero would silently cap every H4 score at 85 and make the
thresholds meaningless. `ComponentScore` gained an `applicable` flag; a component
that is *present and explicitly inapplicable* releases its weight, which is
redistributed proportionally across the rest.

This is deliberately narrow: a component merely **absent** from the mapping does
*not* trigger redistribution, so passing a partial component set (as unit tests
do) behaves exactly as before.

---

## 2. Files

### New

| File | Purpose |
|---|---|
| `src/timeframes.py` | Supported timeframes, confirmation hierarchy, micro-timeframe map, mode constants |
| `src/runtime_state.py` | `JsonStateStore` (single-writer state.json), `RuntimeState` (live settings), `config_view` / `effective_config` |
| `src/telegram_control.py` | Inline-button panel, callback routing, long-poll thread |
| `tests/test_research_mode.py` | 47 tests — modes, thresholds, near-signal, research data capture, score bands |
| `tests/test_telegram_control.py` | 44 tests — panel, buttons, run state, safety, duplicate prevention |
| `BUILD_REPORT.md` | This document |

### Modified

| File | Change |
|---|---|
| `config.py` | Mode/timeframe threshold matrix, regime **offsets**, min/max limits, near-signal settings, Telegram control settings, editable-settings whitelist, `threshold_for()` / `clamp_threshold()` |
| `src/market_data.py` | Timeframe-driven `build_snapshot`, optional confirmation frames, candle cache with per-timeframe TTL, `latest_closed_candle_time` probe, `_MT5_LOCK` around every terminal call |
| `src/scoring.py` | Weight renormalisation for inapplicable components; `weight_scale` / `excluded_components` on the scorecard |
| `src/trend.py` | `analyze_htf` handles two / one / zero confirmation timeframes |
| `src/filters.py` | Compositional `adaptive_threshold`, new `is_near_signal` |
| `src/signal_engine.py` | Per-call config, `NEAR_SIGNAL` decision, mode/timeframe/threshold on `Signal` and `Evaluation`, extended `EVALUATION_COLUMNS` |
| `src/signal_tracker.py` | Shared state store, per-timeframe `update()` and cooldown scoping, `active_timeframes()`, new CSV columns |
| `src/telegram_bot.py` | Inline keyboards, `edit_message`, `answer_callback`, `get_updates`, research + near-signal formats |
| `src/utils.py` | `ComponentScore.applicable` |
| `main.py` | Runtime state, control-panel wiring, PAUSE/STOP, per-timeframe tracking, cheap new-candle probe, `analyze_now()` |
| `performance.py` | By mode, by timeframe, by score band; `best_regime` / `worst_regime` / `stats_for_mode` |
| `backtest.py` | `--mode`, `--timeframe`, `--source-timeframe`; resamples to the active hierarchy; optional confirmation frames |
| `tests/conftest.py` | Shared `FakeNotifier`, `FakeMarket`, `isolated_config` |
| `tests/test_*.py` | Updated for the new threshold model and CSV schemas |
| `README.md`, `.env.example` | Documented the new features |

---

## 3. New Telegram controls

Send `/panel` (or `/start`, `/menu`) to summon it. Buttons are the interface;
those commands exist only to bring the panel up.

| Button | Callback | Effect |
|---|---|---|
| ▶️ START / ⏸ PAUSE / ⏹ STOP | `run:*` | PAUSE keeps MT5 connected and keeps tracking open signals but generates none; STOP exits the loop cleanly |
| 🔬 / 📊 / 🛡 | `mode:*` | Switch mode; the threshold follows |
| M1…H4 | `tf:*` | Switch signal timeframe; the confirmation hierarchy follows and the candle cache is cleared |
| 🎯 THRESHOLD | `menu:threshold`, `thr:±1`, `thr:±5`, `thr:reset` | Adjust the base threshold, clamped to 40–95 |
| 📊 ANALYSIS | `view:analysis` | On-demand evaluation of the latest **closed** candle. Read-only |
| 📈 PERFORMANCE | `view:performance`, `perf:{timeframe,score,regime,mode}` | Straight from the CSVs |
| ⚙️ SETTINGS | `menu:settings`, `set:{cooldown,rr,session,near}` | Cycle each editable value |
| 🔄 REFRESH | `panel:refresh` | Redraw |

The active mode and timeframe are marked `●` in the keyboard. The panel is
edited **in place** rather than re-posted.

**Safety.** Updates from any chat other than `TELEGRAM_CHAT_ID` are dropped. Only
settings listed in `Config.telegram_editable_settings` have handlers —
credentials, tokens, symbol and file paths are unreachable and are never
rendered into a message (asserted by test). On startup the poller discards
updates queued while the engine was down, so a restart never replays stale
presses.

---

## 4. Supported timeframes

| Signal TF | Intermediate | Higher | Micro (TP/SL ordering) | STANDARD | RESEARCH | CONSERVATIVE |
|---|---|---|---|---|---|---|
| M1 | M5 | M15 | — | 80 | 55 | 88 |
| M5 | M15 | H1 | M1 | 72 | 50 | 80 |
| M15 | M30 | H1 | M1 | 70 | 50 | 78 |
| M30 | H1 | H4 | M5 | 68 | 50 | 76 |
| H1 | H4 | — | M5 | 65 | 50 | 73 |
| H4 | — | — | M15 | 65 | 50 | 73 |

CONSERVATIVE is derived as STANDARD + 8, which puts M5 at the specified 80.

**None of these values are claimed to be optimal or profitable.** They are
starting points chosen so faster (noisier) timeframes demand more confirmation.

---

## 5. Threshold behaviour

```
effective = base(mode, timeframe) + regime offset + counter-trend extra
            clamped to [40, 95]
```

Regime offsets: strong trend −4, weak trend / breakout 0, low volatility +3,
range / high volatility +5. Extreme volatility blocks signalling entirely
(configurable). Counter-trend setups add +5.

Threshold overrides set from Telegram are stored **per mode and per timeframe**,
so raising the RESEARCH/M5 bar does not affect STANDARD/M5 or RESEARCH/M15.
`RESET` drops the override and returns to the configured default.

---

## 6. Data collection for future ML

No ML is implemented. The dataset needed to train one later is now complete:

* `evaluations.csv` — **one row per evaluated candle, signal or not**, now
  including `mode`, `signal_timeframe`, `confirmation_timeframes`,
  `threshold_used`, `near_signal`, the nine component sub-scores, regime,
  session, spread, decision, rejection reason, and 23 raw `f_*` features.
  In the verification run: 400 evaluations, of which 328 `NO_SIGNAL`, 66
  `NEAR_SIGNAL` and 8 signals — losers and non-events are kept, not just winners.
* `signals.csv` — adds `mode`, `signal_timeframe`, `confirmation_timeframes`,
  `threshold_used`, `score`.
* `outcomes.csv` — adds `mode`, `timeframe`, `score`, `threshold_used`, so
  outcomes group by mode/timeframe/score band without a join.

Research signals are tracked to TP1/TP2/TP3/SL/expiry exactly like standard
ones, as paper simulations.

---

## 7. Verification performed

All checks run on this machine. Live MT5 mode needs Windows, so the live loop was
exercised through the real `SignalRunner` and `TelegramController` with a fake
market feed and a fake Telegram transport.

| # | Check | Result |
|---|---|---|
| 1 | Full test suite | **252 passed**, 0 failed, 0 skipped |
| 2 | `pyflakes` on every module and test | clean |
| 3 | Research backtest, 6,361 M5 bars | 105 signals vs 49 in STANDARD — more candidates, same engine |
| 4 | Timeframe backtests | M5 ✓, M15 ✓ (38 signals), H1 ✓ (77), H4 ✓ (28) |
| 5 | M1 from M5 history | rejected with a clear message, as it must be |
| 6 | H4 weight renormalisation | HTF excluded, `weight_scale` 1.176, score still reaches 100 |
| 7 | Telegram buttons | every callback exercised: mode, timeframe, run state, threshold, settings, analysis, performance |
| 8 | Pause / resume | 20 polls while paused → **0** evaluations; 5 polls after resume → **5** |
| 9 | Threshold clamping | pinned at 95 and 40 under repeated presses |
| 10 | Duplicate prevention | 8 further polls on the same candle → **0** new rows; no duplicate timestamps, signal ids or outcome ids |
| 11 | Restart safety | mode/timeframe/status/threshold restored; first tick after restart evaluated **0** candles |
| 12 | ANALYSIS side effects | 0 evaluations written, 0 signals recorded, marker unmoved |
| 13 | CSV schemas | signals / outcomes / evaluations headers all match their column tuples |
| 14 | Order execution | `grep -riE "order_send\|order_check\|positions_get\|TRADE_ACTION"` → only the doc comment saying there is none |
| 15 | Credentials | no hard-coded secrets; panel rendering asserted not to leak password, token or login |
| 16 | Lookahead review | all guarantees re-verified after the refactor (below) |

### Lookahead review after the refactor

* Forming candle still dropped in `get_candles`; `MarketSnapshot` still has no
  field that could carry one.
* Swing pivots still hidden until their confirmation bar closes
  (`confirmed_at <= as_of`).
* Backtester still cuts confirmation frames by **close time**, now for a
  variable number of them.
* Cooldown context still ignores signals dated after the evaluated bar, and is
  now additionally scoped by timeframe.
* Outcome tracking still starts on the candle *after* the signal candle.
* **Candle cache:** the signal-timeframe frame is fetched fresh whenever a new
  candle is being evaluated (`use_cache=False`); the cache is only consulted for
  confirmation frames on idle polls and for the read-only ANALYSIS button.

---

## 8. Bugs found and fixed during the work

1. **Fake-market spread made several live-loop tests pass vacuously.** The test
   double returned a hard-coded 20-point spread; on the low-volatility test
   fixture that exceeds `MAX_SPREAD_ATR_RATIO`, so *every* candle was rejected
   before scoring and the pre-existing
   `test_live_loop_records_signals_and_never_duplicates_them` was asserting over
   an empty list. Fixed by defaulting the fake spread to "unknown" (as the
   backtester does). Several tests became meaningful as a result.

2. **Weight renormalisation initially over-triggered.** The first version
   renormalised over whatever components were present, so a partial component
   dict silently rescaled the score (a trend-only card jumped from 20 to 85).
   Narrowed to components that are present *and* explicitly inapplicable.

3. **Cooldown was not timeframe-aware.** A cooldown counts candles, and an M5
   candle is not an H4 candle. Before the fix, an old H4 signal could mute a
   fresh M5 one. `build_gate_state` now scopes the cooldown by timeframe while
   keeping the daily and concurrency caps global.

4. **Outcome tracking would have used the wrong candles after a timeframe
   switch.** Signals raised on a previous timeframe were being advanced with the
   *new* timeframe's candles, mis-counting expiry and mis-reading stops.
   `SignalTracker.update()` now takes a `timeframe` argument and `main.py`
   fetches each open signal's own timeframe.

5. **A 3-candle probe evicted the full cached frame.** `latest_closed_candle_time`
   stored its tiny result under the same cache key; added `cache_result=False`.

6. **Two writers to `state.json`.** The tracker and the runtime state each held
   their own copy of the document and would have clobbered each other's keys.
   Both now share a single lock-guarded `JsonStateStore`.

7. **Stale button presses replayed on startup.** Telegram queues updates while a
   bot is offline; without draining them, a restart would immediately re-apply
   whatever was last pressed (including STOP). The poller now discards the
   backlog.

---

## 9. Results observed (and why they are not evidence)

From the 6,361-bar RESEARCH backtest on **synthetic** data, the score-band table:

```
group  closed   win%   avgR  totalR    PF
40-49       3  100.0  0.613    1.84   inf
50-59      27   48.1  0.115    3.12  1.24
60-69      40   52.5  0.197    7.88  1.42
70-79      28   67.9  0.317    8.87  2.13
80-89       5   60.0  0.005    0.02  1.01
```

This is exactly the table the score-band analysis exists to produce, and it
already shows the assumption "higher score ⇒ better outcome" **failing** at the
top: the 80-89 band underperforms 70-79. With 5 signals in that band it is
noise, not a finding — which is the point. The numbers come from a
regime-switching random walk and say nothing about real gold.

The end-to-end verification run reported "Win Rate: 100.0%" over 8 signals on an
upward-drifting synthetic series. That is an artifact of the fixture and must not
be read as performance.

**No profitability claim is made for any mode, timeframe or threshold.**

---

## 10. Known limitations

* **Live mode is Windows-only** (`MetaTrader5` has no Linux/macOS build). The
  control panel, engine and backtester were verified here with fake feeds; the
  Telegram HTTP calls themselves were exercised against a fake transport, not
  against the real Bot API.
* **M1 signal timeframe cannot be backtested from M5 history** — it needs an M1
  data file (`--source-timeframe M1`). The error message says so.
* **H1/H4 backtests need a lot of history**: H1 needs ~10,500 M5 candles of
  warm-up (220 closed H4 candles), H4 needs ~12,500.
* **The candle cache can serve confirmation frames up to ~2 minutes stale** on
  idle polls and for the ANALYSIS button. The signal timeframe is always fresh
  when a candle is actually evaluated.
* **Settings buttons cycle through fixed lists** rather than accepting free
  numeric entry, because text input is not the interface here.
* **PAUSE still fetches data** each poll so open signals keep being tracked. It
  reduces MT5 traffic but does not eliminate it.
* **Score bands below 40 are not reported** — the minimum threshold is 40, so
  nothing below it can become a signal.
* **Research mode raises signal frequency substantially** (~2× on the test data,
  4.8/day vs 2.3/day). With `MAX_SIGNALS_PER_DAY` at 8 this can bind on busy
  days; raise it if you want the full research stream.
* Everything in the original build report's limitations still applies: tick
  volume rather than real volume, no spread/slippage modelling in backtests,
  pessimistic intrabar assumptions, fixed UTC session windows, no news awareness.

---

## 11. What was deliberately not done

* **No ML.** V1 stays deterministic and transparent; only the dataset was
  prepared.
* **No new indicators**, no scoring-weight changes, no re-tuning for historical
  win rate. RESEARCH mode moves *only* the threshold — if it changed the scoring,
  research candidates could not be compared with standard ones.
* **No web dashboard.**
