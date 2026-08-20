# Build Report — M1 Micro-Scalping Simplification

Third iteration of `xauusd_signal_bot`. **The project was not rebuilt.** The
nine analysis engines, the scoring architecture, the CSV/state layer and the
anti-lookahead machinery were kept and adapted; the multi-mode and
multi-timeframe layers were removed, and a micro-scalping target model, a cost
model and detailed outcome tracking were added.

* **Tests:** 216 passing (was 252 — the mode/timeframe suites were deleted, not
  weakened; scalping-specific ones replaced them), ~100 s
* **Static analysis:** `pyflakes` clean across every module
* **Order execution:** still none anywhere
* **Headline finding:** after costs the strategy is **negative** on the test
  data — see [§9](#9-measured-results-and-what-they-mean)

---

## 1. What was removed

| Removed | Why |
|---|---|
| RESEARCH / STANDARD / CONSERVATIVE modes | Replaced by a single `SCALPING` mode |
| `mode_timeframe_thresholds` matrix | Replaced by one `SCALP_THRESHOLD` |
| M5 / M15 / M30 / H1 / H4 as signal timeframes | M1 only, enforced by `config.validate()` |
| The confirmation *hierarchy* (`M1→M5+M15`, `M5→M15+H1`, …) | Replaced by one optional M5 context timeframe |
| Telegram mode buttons and timeframe buttons | Nothing left to switch |
| `set_mode` / `set_timeframe` / `config_view` | Runtime state no longer carries either |
| `--mode` / `--timeframe` / `--source-timeframe` backtest flags | M1-only |
| Per-timeframe cooldown scoping, per-timeframe outcome tracking | Only one timeframe exists |

The `htf` component **key** was kept (weight, scorecard key, CSV column) so
stored data stays readable; it now means "M5 context" and the engine is
`analyze_context`.

### Why M15/H1/H4 context went too

The spec removed higher-timeframe *selection*. I also dropped higher-timeframe
*confirmation* beyond M5, and that is a judgement call worth flagging: a
one-hour trend has almost no bearing on a position held for three minutes, and
keeping it made the engine reject good scalps for disagreeing with a timeframe
that would not resolve inside the holding window. M5 context is retained at a
reduced weight (15 → 6) and can be switched off entirely.

---

## 2. What was added

### Cost model (`config.round_trip_cost`)

```
spread + entry slippage + exit slippage + commission (both sides)
```

Defaults: 20-point spread + 2+2 points slippage = **2.4 pips per round trip**.
When the live spread is unknown (backtests) `ASSUMED_SPREAD_POINTS` is charged,
so a backtest never trades for free.

Everything downstream reports **RAW R and NET R**. `cost_r` is fixed at signal
time and carried into the outcome, so NET R never depends on the spread at
close.

### Micro-scalping target model (`src/targets.py`, rewritten)

Distances come from the live M1 ATR, then three floors apply:

1. `tp_atr_multiples × ATR` — market conditions
2. `min_tp_pips` — the tick grid
3. TP1 only: `MIN_TP1_COST_MULTIPLE × cost` — worth taking at all

If the cost floor lifts TP1, **the whole ladder lifts proportionally**. Merely
re-spacing TP2/TP3 by a minimum gap would collapse them onto TP1 and silently
turn a 1:2 setup into a 1:1 one; the geometry has to say honestly that a wider
spread demands a bigger move.

Rejections: TP1 that cannot clear costs, and a cost-adjusted TP3 beyond
`MAX_TP3_PIPS` (a "scalp" needing 20 pips is not a scalp).

### Timeout

`MAX_HOLDING_CANDLES` (default 15 minutes). A scalp that has not resolved is
marked to market and recorded as `TIMEOUT`. `STATUS_EXPIRED` was renamed
throughout. Nothing sits active indefinitely.

### Detailed outcome record

`outcomes.csv` gained: `raw_r`, `net_r`, `cost_r`, `spread_points`,
`minutes_to_tp1/2/3`, `minutes_to_sl`, `bars_to_*`, `mfe_price`, `mae_price`,
`mfe_pips`, `mae_pips`, `timeout`, `ambiguous_bars`.

### Tick-based ambiguity resolution

`MarketData.refresh_tick_buffer` keeps a rolling, incrementally-fetched tick
buffer; the tracker replays it to decide whether the target or the stop came
first inside a candle that traded through both. Without ticks the pessimistic
assumption applies. At scalping distances this is a large systematic penalty,
which is why the live path bothers.

### M1 microstructure features

`evaluations.csv` grew from 23 to **41** `f_*` columns, adding `atr_pips`,
`spread_pips`, `cost_pips`, `atr_to_cost`, `range_pips`, `body_pips`,
`upper/lower_wick_pips`, `close_location`, `displacement_atr`,
`velocity_pips_per_min`, `acceleration`, `micro_range_pips_5/15`,
`dist_to_high/low_5_pips`, `minute_of_hour`, `hour_of_day`.

---

## 3. Re-weighting for M1 (documented, not optimised)

| Component | Was | Now | Reason |
|---|---|---|---|
| Momentum | 15 | **22** | Short-term momentum and acceleration drive a 3-minute move |
| Price action | 5 | **18** | Displacement and candle shape are the M1 signal |
| Liquidity | 10 | **14** | Sweeps of the immediate highs/lows |
| Structure | 15 | **12** | Micro BOS/CHoCH still useful |
| Support/Resistance | 10 | 10 | The levels the next few pips must clear |
| Trend | 20 | **10** | A slow trend matters far less at this horizon |
| Context (`htf`) | 15 | **6** | M5 only, secondary |
| Volatility | 5 | 5 | Is the move big relative to noise |
| Volume | 5 | **3** | Weakest signal on M1 |

Indicator periods shortened: EMA 9/21/50/200 → **5/13/34/100**, RSI 14 → 9,
MACD 12/26/9 → 6/13/5, structure fractal 2-bar → **1-bar**, ATR history 100 →
120 minutes.

**None of this was tuned against results.** It was chosen by reasoning about the
holding period, and the report below shows it does not rescue the economics.

---

## 4. Files

**New:** `tests/test_scalping.py` (47 tests).
**Deleted:** `tests/test_research_mode.py`, `tests/test_targets.py` (superseded).

**Rewritten:** `src/targets.py`, `src/timeframes.py`, `main.py` docstring/loop,
`backtest.py`, large parts of `src/telegram_control.py` and `performance.py`.

**Modified:** `config.py` (weights, indicator periods, threshold, cost model,
targets, timeout, limits, validation), `src/market_data.py` (M1+context
snapshot, tick buffer), `src/signal_tracker.py` (timeout, milestone timing, net
R, schemas), `src/signal_engine.py` (M1 evaluation, microstructure features),
`src/trend.py` (`analyze_htf` → `analyze_context`), `src/filters.py` (spread and
net R:R gates), `src/runtime_state.py` (mode/timeframe removed), `src/telegram_bot.py`
(scalp card), `src/scoring.py`, `walkforward.py`, `make_synthetic_history.py`,
`README.md`, `.env.example`, `tests/conftest.py`, remaining test modules.

---

## 5. Telegram

Main panel is exactly the specified layout:

```
⚡ XAUUSD SCALPER
Status / Mode: SCALPING / Timeframe: M1 / Threshold
Signals today / Open signals / Paper Net R
[START] [PAUSE]
[CURRENT ANALYSIS]
[PERFORMANCE]
[SETTINGS]
[REFRESH]
```

STOP moved into Settings so it cannot be hit while reaching for PAUSE.
Performance views are BY SCORE / BY REGIME / BY SESSION / BY OUTCOME, all
leading with NET R. The signal card matches the specified scalp format and adds
an "After costs" block — quoting raw R alone on a few-pip target would be
misleading.

Safety is unchanged: foreign chats ignored, only whitelisted settings reachable,
credentials never rendered (asserted by test), startup backlog discarded.

---

## 6. Tests

216 passing. New coverage for: M1/SCALPING invariants (including that
`set_mode`/`set_timeframe` are gone), cost-model arithmetic and fallbacks, pip
conversion, ATR-scaled targets, all three target floors, proportional ladder
lifting, exact-rounded-price R:R, net-below-raw, wider-spread-worse-net,
cost-based rejection, scalp-range rejection, stop noise floor and ATR ceiling,
spread-vs-holding-window gate, net R:R gate rejecting a raw winner, timeout at
the limit / mark-to-market / never-indefinitely / configurable, milestone
timing, excursions in R and price, full outcome row, cost_r derivation,
pessimistic ambiguity, tick-resolved target-first and stop-first, ticks outside
the bar ignored, and ticks that are themselves ambiguous.

---

## 7. Bugs found and fixed

1. **Target geometry was structurally incoherent.** I first set TP2 at 0.90×ATR
   with the stop also at 0.90×ATR, so R:R was ~1.0 by construction while
   `MIN_TP2_RR` demanded 1.2 — the gate rejected essentially everything. Fixed
   by separating the multiples (TP 0.45/1.00/1.70, SL 0.70) and adding a
   `validate()` check that TP2 must exceed the stop.

2. **The stop ceiling let HYBRID break the ladder.** The structure branch takes
   the *wider* distance, which on M1 routinely put the stop at the old 1.80×ATR
   cap while targets stayed put — R:R below 1 again. Capped at 0.80×ATR, with a
   second `validate()` check that the widest allowed stop still permits
   `MIN_TP2_RR`. Without these two fixes the engine emitted **zero** signals.

3. **The spread filter was calibrated for M5.** It compared the spread with a
   single candle's ATR; on M1 those are nearly the same size, so it rejected
   every candle. Replaced with spread vs the move available over the holding
   window (ATR × √candles).

4. **`FakeMarket` ignored `candles_signal`**, feeding the engine 4,601 candles
   where live feeds 900. A longer window changes which reference levels exist
   (a previous *day* only appears once the window spans one), so the double was
   not reproducing live behaviour.

5. **`FakeNotifier.send_outcome` had a stale signature** after `net_r` was
   added, which silently routed every outcome alert into the tracker's
   exception handler. Caught because the handler logged rather than swallowing —
   worth noting the handler did its job.

6. **`MAX_SIGNALS_PER_DAY=8` was an M5-era cap** and bound long before the
   engine's own filters on M1 (1,440 candles/day). Raised to 30; it exists to
   keep Telegram usable, not as quality control.

---

## 8. Verification performed

Live MT5 mode needs Windows, so the loop was exercised through the real
`SignalRunner` and `TelegramController` with fake market and Telegram feeds.

| # | Check | Result |
|---|---|---|
| 1 | Full test suite | **216 passed**, 0 failed |
| 2 | `pyflakes`, all modules | clean |
| 3 | M1-only enforcement | `signal_timeframe != "M1"` raises; no mode/TF setters remain |
| 4 | Backtest, 29k M1 bars, spread 6 | 348 signals, 17/day |
| 5 | Backtest, spread 20 | targets lift to 3.6/6.5/10.9 pips; 64% time out |
| 6 | Cost gating | rejections dominated by R:R and scalp-range at wide spreads |
| 7 | Telegram panel | all six controls exercised; layout matches spec |
| 8 | Pause / resume | 20 polls paused → **0** evaluations; resumed → 5 in 5 |
| 9 | Threshold controls | ±1/±5/reset, clamped at 40 and 95 |
| 10 | Duplicate prevention | 8 polls on one candle → **0** new rows; unique signal/outcome ids; no duplicate timestamps |
| 11 | Restart safety | threshold/status/hold restored; first tick after restart evaluated 0 candles |
| 12 | ANALYSIS side effects | 0 evaluations written, 0 signals, marker unmoved |
| 13 | CSV schemas | signals / outcomes / evaluations headers all match |
| 14 | Order execution | grep for `order_send`/`order_check`/`positions_get`/`TRADE_ACTION` → only the doc comment saying there is none |
| 15 | Lookahead review | all guarantees re-verified; ticks filtered to the bar's own minute |

---

## 9. Measured results, and what they mean

30,000 synthetic M1 candles (~20 trading days). **Synthetic data — this proves
the machinery measures what it claims, nothing more.**

| | spread 6 pts | spread 20 pts |
|---|---|---|
| Round-trip cost | 1.0 pip | 2.4 pips |
| Signals | 348 | 42 (shorter slice) |
| Median TP ladder | 1.6 / 3.6 / 6.0 pips | 3.6 / 6.5 / 10.9 pips |
| Median stop | 2.9 pips | 4.0 pips |
| Median TP2 | 1.26R raw → **0.94R net** | 1.62R raw → **1.02R net** |
| **Raw** win rate | 58.0% | 52.4% |
| **Raw** expectancy | −0.090R | **+0.033R** |
| **Net** win rate | 24.4% | 23.8% |
| **Net** expectancy | **−0.518R** | **−0.567R** |
| Outcomes | 87% SL, 12% TP3 | 64% TIMEOUT, 36% SL |
| Median hold | 2 min | 15 min |

Three things worth reading carefully:

* **At a 20-point spread the strategy has a positive raw expectancy and still
  loses 0.567R per trade.** That single row is the argument for the cost model.
* **The mechanism is the breakeven stop.** At the tighter spread 58% of scalps
  touch TP1, but moving the stop to breakeven then converts most of them into
  ~0R while losers pay −1R plus costs. Small-target scalping dies on that
  asymmetry, not on signal quality.
* **The 1–3 pip concept mostly does not survive costs.** At a 20-point spread
  the cost floor lifts TP1 to 3.6 pips — four times the original concept — and
  the holding window then stretches until 64% time out. Only on a raw/ECN
  spread does a 1.6-pip first target exist at all.

**No profitability is claimed for any setting.** The negative result is the
research output, and the system is built to keep producing that number honestly
rather than to make it look better.

---

## 10. Known limitations

* Live mode is Windows-only (`MetaTrader5` has no Linux/macOS build).
* Backtests have no ticks, so every ambiguous candle is scored as a stop — live
  and backtested outcomes are **not directly comparable**.
* Slippage is an assumption (2+2 points), plausible in liquid hours and
  optimistic around news.
* **Cost is deducted once per trade, not per partial.** With three partials the
  true cost is higher, so NET R here is mildly optimistic.
* `assumed_spread_points` is a single number; real spreads vary by hour and
  widen exactly when setups look attractive.
* Tick fetching adds MT5 load; disable with `USE_TICKS_FOR_AMBIGUOUS_CANDLES=false`.
* No news/economic-calendar awareness.
* Session windows are fixed UTC and do not follow DST.
* Backtest speed ~50 bars/s; a month of M1 is ~43,000 bars.
* The M5 context is fetched on every poll (cached up to 2 minutes); with context
  disabled the system is pure M1.

---

## 11. What was deliberately not done

* **No ML** — only the dataset was prepared. The label (`net_r`) is already
  cost-adjusted, which is the part that matters.
* **No threshold tuning against the evaluation data.** `SCALP_THRESHOLD=68` was
  set from the score distribution (99th percentile ≈ 71), not from P&L.
* **No re-tuning to make the results positive.** Two geometry bugs were fixed
  because the ladder was mathematically incoherent, not because signals were
  scarce; the economics were left to say what they say.
* **No new indicators** were added to generate more signals.
