# XAUUSD Signal Engine (V1)

A standalone, deterministic **signal-only** system for XAUUSD (gold).

```
MT5 market data
      ↓
technical + market-structure analysis
      ↓
multi-confirmation signal engine (9 components → 0-100 score)
      ↓
strict filtering (threshold, conflict, fakeout, session, spread, cooldown)
      ↓
dynamic entry / SL / TP1-3 + R:R validation
      ↓
Telegram notification
      ↓
CSV logging
      ↓
automatic outcome tracking (TP/SL/expiry)
```

> **This system never places, modifies or closes a trade.**
> MetaTrader 5 is used as a *read-only market-data feed*. No order-execution
> function is imported or called anywhere in the codebase. It is a paper-signal
> and research tool.

> **No profitability claim is made.** See [Honest assessment](#honest-assessment).

---

## Table of contents

1. [Quick start (Windows)](#quick-start-windows)
2. [Project layout](#project-layout)
3. [Operating modes](#operating-modes)
4. [Telegram control panel](#telegram-control-panel)
5. [Timeframes](#timeframes)
6. [How the signal engine works](#how-the-signal-engine-works)
7. [Scoring](#scoring)
8. [Filtering](#filtering)
9. [Near-signal diagnostic](#near-signal-diagnostic)
10. [Entry, stop loss and take profits](#entry-stop-loss-and-take-profits)
11. [Telegram messages](#telegram-messages)
12. [Where data is stored](#where-data-is-stored)
13. [Outcome tracking and the R model](#outcome-tracking-and-the-r-model)
14. [Backtesting](#backtesting)
15. [Walk-forward testing](#walk-forward-testing)
16. [Paper testing and reading the results](#paper-testing-and-reading-the-results)
17. [Calibration](#calibration)
18. [Timezones](#timezones)
19. [Anti-lookahead guarantees](#anti-lookahead-guarantees)
20. [Tests](#tests)
21. [Honest assessment](#honest-assessment)
22. [Known limitations](#known-limitations)
23. [Recommended V2 improvements](#recommended-v2-improvements)

---

## Quick start (Windows)

**1. Install Python 3.11 or newer**

Download from <https://www.python.org/downloads/>. Tick **"Add python.exe to PATH"**
during installation. Verify:

```bat
python --version
```

**2. Install dependencies**

```bat
cd xauusd_signal_bot
pip install -r requirements.txt
```

**3. Install and log in to MetaTrader 5**

Install the MT5 desktop terminal from your broker, log into the account, and
confirm that **XAUUSD** appears in Market Watch (right-click → Show All if it is
hidden). Then in MT5: *Tools → Options → Expert Advisors* and make sure
**"Allow algorithmic trading"** is ticked — the Python API needs it to connect
even though this project only reads data.

Leave the terminal **running and logged in** while the engine runs.

**4. Configure `.env`**

```bat
copy .env.example .env
notepad .env
```

Fill in `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, and set
`MT5_SERVER_UTC_OFFSET_HOURS` (see [Timezones](#timezones)). If your broker names
gold something other than `XAUUSD` (e.g. `XAUUSD.m`, `GOLD`), set `SYMBOL` to the
exact name shown in Market Watch.

**5. Get a Telegram bot token and chat id**

* Open Telegram, message **@BotFather**, send `/newbot`, follow the prompts.
  BotFather replies with a token like `123456789:AAF...` → `TELEGRAM_BOT_TOKEN`.
* Message **@userinfobot** (or **@RawDataBot**); it replies with your numeric
  `Id` → `TELEGRAM_CHAT_ID`.
* **Send your new bot any message first** (e.g. `/start`). A bot cannot message a
  user who has never messaged it.
* For a group: add the bot to the group, post a message, then read the chat id
  from `https://api.telegram.org/bot<TOKEN>/getUpdates` (group ids are negative).

**6. Run**

Double-click **`run.bat`**, or:

```bat
python main.py
```

You should see:

```
==============================================
XAUUSD SIGNAL ENGINE
Status: RUNNING
==============================================
MT5:      CONNECTED
Telegram: CONNECTED
...
MODE: PAPER SIGNAL ONLY - no orders are ever sent.
```

**7. Verify Telegram**

On a successful start the bot sends *"✅ XAUUSD signal engine started"*. If it
does not arrive, check `data/system_log.txt` — the most common causes are a wrong
chat id, or not having messaged the bot first.

Stop with `Ctrl+C`.

### Running on macOS / Linux

The `MetaTrader5` package is **Windows-only**, so live mode needs Windows (or
Windows in a VM). Everything else — indicators, `backtest.py`, `walkforward.py`,
`performance.py` and the whole test suite — runs anywhere:

```bash
pip install pandas numpy requests python-dotenv pytest
python -m pytest tests/ -q
python backtest.py --data history/XAUUSD_M5.csv
```

---

## Project layout

```
xauusd_signal_bot/
├── main.py                  live paper-signal runner
├── backtest.py              historical replay through the same engine
├── walkforward.py           in-sample / validation / out-of-sample report
├── performance.py           expectancy statistics from the CSVs
├── make_synthetic_history.py  generates test data (NOT real prices)
├── config.py                every tunable, loaded from .env
├── requirements.txt
├── .env.example
├── run.bat                  Windows launcher
├── data/                    created on first run
│   ├── signals.csv          one row per signal, status updated in place
│   ├── evaluations.csv      one row per evaluated candle (signal or not)
│   ├── outcomes.csv         one row per closed signal
│   ├── system_log.txt       rotating log
│   └── state.json           last processed candle, last signal
├── src/
│   ├── market_data.py       MT5 access (READ-ONLY), validation, resampling
│   ├── indicators.py        EMA/RSI/MACD/ATR/ADX/Stoch/BB/swings
│   ├── trend.py             trend engine (0-20) + HTF engine (0-15)
│   ├── momentum.py          momentum engine (0-15)
│   ├── structure.py         swings, BOS, CHoCH (0-15)
│   ├── liquidity.py         reference levels, sweeps (0-10)
│   ├── support_resistance.py  zones, breakouts (0-10)
│   ├── volume.py            tick-volume confirmation (0-5)
│   ├── volatility.py        ATR regime (0-5, direction-neutral)
│   ├── price_action.py      candle confirmation (0-5)
│   ├── regime.py            market-regime classifier
│   ├── timeframes.py        signal timeframes + confirmation hierarchy
│   ├── runtime_state.py     live-editable settings, persisted to state.json
│   ├── telegram_control.py  inline-button control panel
│   ├── scoring.py           weighted aggregation → 0-100
│   ├── filters.py           thresholds and rejection rules
│   ├── targets.py           entry / SL / TP1-3 / R:R
│   ├── signal_engine.py     orchestration
│   ├── signal_tracker.py    CSV persistence + outcome tracking
│   ├── telegram_bot.py      notifications
│   ├── logger.py            logging setup
│   └── utils.py             helpers
└── tests/                   252 unit tests
```

### Two deliberate deviations from a plain dependency list

* **No `pandas-ta`.** The project needs about a dozen classic indicators.
  `pandas-ta` is currently unmaintained against NumPy 2.x / pandas 3.x and pulls
  in a large dependency tree. `src/indicators.py` implements them directly in
  pandas/NumPy — fewer moving parts, and every calculation is unit-tested
  against a reference implementation.
* **No `python-telegram-bot`.** The only thing needed is "post a text message",
  and the main loop is synchronous. `src/telegram_bot.py` calls the Bot HTTP API
  with `requests`, avoiding an async framework and its event loop.

---

## Operating modes

Three modes, all sharing **exactly the same scoring engine**.  They differ only
in how much confirmation a candidate must show before it is reported.

| Mode | Icon | Nominal threshold | Purpose |
|---|---|---|---|
| `RESEARCH` | 🔬 | 50 | Surface many candidates for observation, paper testing and future ML training |
| `STANDARD` | 📊 | 72 | The default |
| `CONSERVATIVE` | 🛡 | 80 | Demands more confirmation than STANDARD |

> **RESEARCH mode is not a "more profitable" mode.** It lowers the bar so that
> more candidate setups become visible and get recorded. Every research message
> is labelled `🔬 RESEARCH SIGNAL` and carries an explicit disclaimer. Nothing
> about a research candidate is validated.

The scoring weights, the engines and the filters are **unchanged** between
modes. Only the threshold moves. That is deliberate: if research candidates
were scored differently they could not be compared with standard ones, which
would defeat the purpose of collecting them.

Thresholds are per mode **and** per timeframe (see below), configurable in
`config.py` and adjustable live from Telegram.

## Telegram control panel

Send `/panel` (or `/start`) to your bot to summon the panel. Buttons are the
interface; text commands only exist to bring it up.

```
━━━━━━━━━━━━━━━━━━
🤖 XAUUSD SIGNAL ENGINE
━━━━━━━━━━━━━━━━━━

Status: 🟢 RUNNING
Mode: 🔬 RESEARCH
Signal TF: M5
Confirmation: M15+H1

Threshold: 50
Cooldown: 10 candles
Min R:R (TP2): 1.5

Signals today: 4
MT5: CONNECTED

[▶️ START] [⏸ PAUSE] [⏹ STOP]
[● 🔬 RESEARCH]
[○ 📊 STANDARD]
[○ 🛡 CONSERVATIVE]
[ M1] [●M5] [ M15]
[ M30] [ H1] [ H4]
[🎯 THRESHOLD (50)]
[📊 ANALYSIS]
[📈 PERFORMANCE]
[⚙️ SETTINGS] [🔄 REFRESH]
```

Every press takes effect in the running process - **nothing requires a
restart** - and is persisted to `data/state.json`.

| Control | Effect |
|---|---|
| START / PAUSE / STOP | PAUSE keeps MT5 connected and keeps tracking open signals, but generates no new ones. STOP shuts the engine down cleanly. |
| Mode buttons | Switch RESEARCH / STANDARD / CONSERVATIVE; the threshold follows |
| Timeframe buttons | Switch the signal timeframe; the confirmation hierarchy follows |
| 🎯 THRESHOLD | `-5 / -1 / +1 / +5 / RESET`, clamped to 40-95 |
| 📊 ANALYSIS | Latest evaluation on demand, from closed candles only. Read-only: it records nothing and cannot emit or suppress a signal |
| 📈 PERFORMANCE | Summary from the CSVs, with BY TIMEFRAME / BY SCORE / BY REGIME / BY MODE views |
| ⚙️ SETTINGS | Mode, timeframe, threshold, cooldown, minimum R:R, session filter, near-signal alerts |

**Safety.** Updates from any chat other than `TELEGRAM_CHAT_ID` are ignored.
Only the settings in `Config.telegram_editable_settings` are reachable -
credentials, the bot token, the symbol and all file paths have no handler and
are never rendered into a message. On startup the poller **discards updates
queued while the engine was down**, so a restart never replays stale presses.

## Timeframes

Six signal timeframes are supported. Changing one moves the whole confirmation
hierarchy - confirming an M1 setup against H1 is a very different statement
from confirming an M5 setup against H1.

| Signal TF | Intermediate confirmation | Higher confirmation | Default STANDARD threshold | RESEARCH |
|---|---|---|---|---|
| M1 | M5 | M15 | 80 | 55 |
| M5 | M15 | H1 | 72 | 50 |
| M15 | M30 | H1 | 70 | 50 |
| M30 | H1 | H4 | 68 | 50 |
| H1 | H4 | — | 65 | 50 |
| H4 | — | — | 65 | 50 |

Faster timeframes carry a higher bar because their signals are noisier. These
are **starting points, not optimised values**.

When only one confirmation timeframe exists (H1), the HTF component is computed
from that timeframe alone. When none exists (H4), the HTF component is marked
*not applicable* and its 15 points are redistributed across the other eight
components, so the score stays on a true 0-100 scale instead of being silently
capped at 85.

Switching timeframe clears the candle cache, and each timeframe keeps its **own**
last-processed-candle marker in `state.json`, so switching away and back can
never re-evaluate a candle that was already done. Signals raised on a timeframe
you have since left are still tracked to completion, using candles of their own
timeframe.

## Near-signal diagnostic

A rejected candidate whose best score lands within `NEAR_SIGNAL_MARGIN` (default
10) points **below** the active threshold is recorded as `NEAR_SIGNAL` in
`evaluations.csv`.

```
Threshold: 72
Bullish: 66
-> NEAR SIGNAL
```

It is never sent as a trading signal. The point is to answer a question the
score distribution alone cannot: *is the threshold slightly too strict for this
market?* If most candles sit just under the bar, the threshold is the binding
constraint; if they sit far below, it is not.

Telegram alerts for near-signals are **off by default** and can be toggled in
Settings.


---

## How the signal engine works

The engine evaluates **once per newly closed M5 candle** — never on a forming
candle, and never twice for the same candle. `main.py` polls MT5 every
`POLL_SECONDS`, compares the newest closed candle's open time with
`last_processed_candle` in `data/state.json`, and only proceeds when it changes.

For each evaluation:

1. **Validate** — enough history, ordered timestamps, no duplicates, no NaN or
   impossible OHLC, data not stale. Any failure → `NO SIGNAL`, reason logged.
2. **Compute indicators** on M5, M15 and H1.
3. **Run the nine analysis engines**, each producing an independent
   bullish and bearish sub-score.
4. **Aggregate** into `bullish_score` and `bearish_score`, both 0-100.
5. **Classify the market regime**, which selects the score threshold.
6. **Filter** the stronger direction through the rejection chain.
7. **Build entry / SL / TP1-3** and check R:R.
8. **Emit** a signal, or record a `NO_SIGNAL` evaluation with its reason.

Every evaluation is written to `data/evaluations.csv` **whether or not a signal
was produced** — that file is the audit trail and the future ML training set.

### The nine components

| Component | Weight | What it looks at |
|---|---|---|
| Trend | 20 | EMA 9/21/50/200 alignment, price vs EMAs, EMA slopes, ADX, DI+/DI− |
| HTF confirmation | 15 | M15 + H1 trend, structure and momentum blended (H1 weighted highest) |
| Momentum | 15 | RSI level and slope, MACD state and acceleration, stochastic, ROC, RSI divergence |
| Structure | 15 | Confirmed swings, HH/HL vs LH/LL, break of structure, change of character, consolidation |
| Liquidity | 10 | Previous day/session extremes, equal highs/lows, sweeps and reclaims |
| Support/Resistance | 10 | Clustered zones, breakouts, reactions, room to the next opposing zone |
| Volume | 5 | Relative tick volume expansion, directional volume bias |
| Volatility | 5 | ATR vs its own long-run average, Bollinger width (direction-neutral) |
| Price action | 5 | Displacement, engulfing, wick rejection, close location |

**Avoiding double-counting.** The engines are kept disjoint in their inputs: the
momentum engine never looks at EMA direction (that is the trend engine's job),
structure uses only confirmed swing pivots, volatility is direction-neutral, and
so on. The HTF component *does* re-run trend/structure/momentum, but on M15 and
H1 — that is confirmation across timeframes, not a second count of the same bars.

**Momentum is never a signal on its own.** There is deliberately no
"RSI below 30 ⇒ buy" rule. An oversold reading only contributes through the
RSI-slope and divergence terms, which require the reading to actually be turning.

---

## Scoring

Each engine scores on its own natural scale, then `src/scoring.py` normalises to
`[0, 1]` and multiplies by the configured weight:

```
component_points = (component_score / component_max) * weight
bullish_score    = Σ component_points   (0-100)
bearish_score    = Σ component_points   (0-100)
```

Weights live in `config.Weights` and must sum to 100 — `Config.validate()`
enforces this at startup, and a unit test asserts that all components at maximum
produce exactly 100.

Confidence bands (`config.confidence_bands`) label the resulting score. See
[Calibration](#calibration) for why the shipped band values are what they are.

---

## Filtering

Filters run in order; the **first** one to fire rejects the candidate and its
reason is written to `evaluations.csv`.

| # | Filter | Rule |
|---|---|---|
| 1 | Session | Candle's UTC session must be in `ALLOWED_SESSIONS` |
| 2 | Spread | Reject above `MAX_SPREAD_POINTS`, or above `MAX_SPREAD_ATR_RATIO` × ATR |
| 3 | Volatility | `EXTREME` volatility blocks signals (or raises the bar, configurable) |
| 4 | Threshold | Dominant score ≥ the regime's adaptive threshold |
| 5 | Separation | `dominant − opposite ≥ MIN_SCORE_SEPARATION` |
| 6 | Fakeout | Five distinct unconfirmed-breakout / HTF-disagreement checks |
| 7 | Cooldown | `COOLDOWN_CANDLES` any direction, `SAME_DIRECTION_COOLDOWN_CANDLES` repeated |
| 8 | Limits | `MAX_SIGNALS_PER_DAY`, `MAX_CONCURRENT_ACTIVE_SIGNALS` |
| 9 | R:R | TP2 reward ≥ `MIN_TP2_RR` |

### Adaptive threshold

The regime chooses how much confirmation is demanded:

| Regime | Threshold |
|---|---|
| `STRONG_BULL_TREND` / `STRONG_BEAR_TREND` | 68 |
| `WEAK_TREND` / `BREAKOUT` | 72 (base) |
| `LOW_VOLATILITY` | 75 |
| `RANGE` / `HIGH_VOLATILITY` | 77 |
| Extreme volatility | **no signal** |

A **counter-trend** setup (against the H1 direction) adds
`COUNTER_TREND_EXTRA_SCORE` on top.

### Bull/bear conflict filter

The spec's example: bull 84 / bear 79 is a *conflicted* market, not a buy. A
signal needs both `score ≥ threshold` **and**
`dominant − opposite ≥ MIN_SCORE_SEPARATION`.

### Fakeout filter

Rejects when: a breakout has neither volume nor a decisive body behind it; the
candle closed back inside a range it broke on the previous bar; the higher
timeframe actively disagrees; participation is thin (relative volume < 0.5); or
the signal candle itself was rejected in our direction by a long opposing wick.

---

## Entry, stop loss and take profits

**Entry** = the close of the confirmed signal candle.

**Stop loss** — three modes via `SL_MODE`:

* `ATR` — `ATR × SL_ATR_MULTIPLIER`
* `STRUCTURE` — beyond the last confirmed swing, plus `SL_STRUCTURE_BUFFER_ATR × ATR`
* `HYBRID` *(default)* — the **wider** of the two

HYBRID is the default because taking the wider distance keeps the stop from
sitting exactly on the obvious swing where stop orders cluster, while the clamp
to `[0.8, 3.0] × ATR` stops a distant swing producing absurd risk.

**Take profits** — `R = |entry − stop|`, then TP1/TP2/TP3 at **1.0R / 1.8R /
2.8R**. If a *major* opposing zone (previous day extreme, or a repeatedly
rejected area) sits between entry and a target, that target is pulled back to
just in front of it and flagged obstructed; it is never pulled closer than
0.6/1.2/1.8R. The ladder is then re-ordered so TP1 < TP2 < TP3 always holds.

Only *major* zones may truncate a target — otherwise every minor pivot on a noisy
chart would clip TP2 and the R:R filter would reject everything.

**R:R** is computed from the **rounded, published prices**, so the numbers in the
Telegram message are the numbers that were validated.

---

## Telegram messages

Two message types, both plain text (no Markdown, so gold prices and emoji cannot
break the formatting).

```
━━━━━━━━━━━━━━━━━━
🟢 XAUUSD BUY
━━━━━━━━━━━━━━━━━━

⭐ Confidence: 87/100
📊 Timeframe: M5
📈 Regime: STRONG BULL TREND
🕐 Session: LONDON NEW YORK OVERLAP

Entry: 3342.50
SL: 3339.80

TP1: 3345.20
TP2: 3347.90
TP3: 3351.00

R:R
TP1: 1.0R
TP2: 1.8R
TP3: 2.8R

CONFIRMATIONS
✅ Trend
✅ HTF
✅ Momentum
✅ Structure
✅ Liquidity
✅ S/R
✅ Volume
✅ Volatility
▫️ Price Action

━━━━━━━━━━━━━━━━━━
Signal only - not financial advice.
```

Outcome alerts: `🎯 TP1/TP2/TP3 HIT`, `❌ SL HIT`, `⚠️ SIGNAL INVALIDATED`,
`⌛ SIGNAL EXPIRED`.

**No duplicate alerts.** An alert fires only when a signal's status *changes*
against the value persisted in `signals.csv`. Because status is on disk, a
restart replays nothing. Sends are retried three times with backoff and never
raise — a Telegram outage cannot stop signal generation or CSV logging.

---

## Where data is stored

Plain files under `data/`, created automatically with headers, appended safely,
UTF-8, and rewritten atomically (temp file + replace) so a crash cannot truncate
them. No database.

| File | Contents |
|---|---|
| `signals.csv` | One row per signal; `status` updated in place. Carries `mode`, `signal_timeframe`, `confirmation_timeframes`, `threshold_used` and `score` so research and standard candidates can be separated later |
| `evaluations.csv` | **Every** evaluated candle: all nine sub-scores, regime, spread, decision (`BUY`/`SELL`/`NO_SIGNAL`/`NEAR_SIGNAL`), rejection reason, mode, timeframe, threshold used, near-signal flag, plus 23 raw `f_*` feature columns |
| `outcomes.csv` | One row per closed signal: result, exit level, R multiple, duration, MFE/MAE, plus `mode`, `timeframe`, `score` and `threshold_used` carried through from the signal |
| `system_log.txt` | Rotating log (5 MB × 3) |
| `state.json` | `last_processed_candles` (per timeframe), `last_signal_id`, `last_signal_time`, and a `runtime` section holding the live Telegram-controlled settings |

`signals.csv` is deliberately small (a handful of rows per day) so it can be held
in memory and rewritten on status changes. `evaluations.csv` is the large one and
is **append-only — the live loop never reads it back**.

If the schema of a CSV ever changes between versions, the old file is renamed to
`<name>.csv.bak-<timestamp>` rather than being appended to under a stale header,
which would silently misalign every column.

### Future ML compatibility (not implemented in V1)

`evaluations.csv` is designed as a training set: each row is the complete feature
vector the engine saw, and `outcomes.csv` joins to it via `signal_id` to supply
the label. A future V2 could train

```
features → P(setup succeeds) → probability filter → final signal
```

V1 deliberately contains **no** machine learning: it is transparent and
deterministic, and the same inputs always give the same outputs.

---

## Outcome tracking and the R model

A signal is modelled as **three equal partials** taken at TP1/TP2/TP3. After TP1
is reached the stop moves to breakeven (`MOVE_SL_TO_BREAKEVEN_AFTER_TP1`). The R
multiple is the size-weighted sum of realised parts plus the remaining size
marked out at the exit:

| Path | R |
|---|---|
| Stopped out before TP1 | **−1.00** |
| TP1, then stopped at breakeven | **+0.33** |
| TP1, TP2, then breakeven | **+0.93** |
| All three targets | **+1.87** |
| Expired after `SIGNAL_EXPIRY_CANDLES` | marked out at that candle's close |

Two rules keep this honest:

* **A signal cannot trade against its own candle.** Tracking starts on the
  candle *after* the signal candle, because the entry is that candle's close.
* **Ambiguous candles are resolved pessimistically.** When one M5 candle touches
  both the next target and the stop, OHLC cannot say which came first. Live, the
  M1 feed is replayed to resolve the order; when that is unavailable (always, in
  a backtest) the system assumes **the stop was hit first**.
* **A breakeven stop only activates on the next candle.** Activating it inside
  the same bar that reached TP1 would stop out every winner at +0.33R, since the
  bar's low may well have occurred before the target was tagged.

`tp_hits` in `outcomes.csv` records how many targets were reached, so TP1/TP2/TP3
hit rates stay meaningful even when the final `result` is `SL_HIT` at breakeven.

---

## Backtesting

Feed a CSV of M5 candles; M15 and H1 are derived by resampling, so one file is
enough.

```bash
python backtest.py --data history/XAUUSD_M5.csv
python backtest.py --data history/XAUUSD_M5.csv --start 2024-01-01 --end 2024-06-30
python backtest.py --data history/XAUUSD_M5.csv --spread 25   # simulate a fixed spread
python backtest.py --data history/XAUUSD_M5.csv --mode RESEARCH
python backtest.py --data history/XAUUSD_M5.csv --timeframe M15   # hierarchy follows
```

Required columns: `time, open, high, low, close, tick_volume`
(`date`/`datetime`/`timestamp` and `volume`/`vol` are accepted as aliases).
Export from MT5 with *Tools → History Center*, or *View → Symbols → Bars*.

**You need roughly 2,900 M5 candles of warm-up** (about 10 trading days) before
the first signal can be produced, because the H1 view needs 220 closed H1 candles
for its 200-period EMA. The backtester reports the first bar it actually
evaluated.

Outputs `backtest_signals.csv`, `backtest_outcomes.csv` and
`backtest_evaluations.csv`, then prints the performance report.

Expect roughly **30-40 bars/second**; a full year of M5 data takes about half an
hour and prints a progress line with an ETA.

---

## Walk-forward testing

```bash
python walkforward.py --data history/XAUUSD_M5.csv
python walkforward.py --data history/XAUUSD_M5.csv --split 0.5 0.25 0.25
```

Splits the history chronologically into IN_SAMPLE / VALIDATION / OUT_OF_SAMPLE
(each prefixed with its own warm-up bars) and runs the **same** configuration
over each.

This is a **robustness report, not an optimiser** — V1 is a fixed rule set. If
you do tune parameters: tune on IN_SAMPLE, sanity-check on VALIDATION, and look
at OUT_OF_SAMPLE **exactly once**. Every extra look turns it into in-sample data.

Compare *expectancy*, *profit factor* and *drawdown* across segments — not win
rate. A large drop from IN_SAMPLE to OUT_OF_SAMPLE is the signature of
overfitting.

---

## Paper testing and reading the results

```bash
python performance.py
python performance.py --signals data/signals.csv --outcomes data/outcomes.csv
```

Reports overall expectancy plus breakdowns by direction, session, regime and
confidence band.

**How to read it:**

* **Average R is the headline number.** It is the expectancy per signal. Positive
  and stable beats a high win rate.
* **Profit factor** = gross R won ÷ gross R lost. Below 1.0 loses money.
* **Max drawdown (in R)** tells you the worst peak-to-trough run — this is what
  determines whether a strategy is survivable, not the average.
* **Win rate is the least informative statistic here.** With a 1.0/1.8/2.8R
  ladder, 45% winners with good R can beat 70% winners with poor R.
* **Sample size dominates.** Fewer than ~100 closed signals tells you almost
  nothing; treat 30 signals as noise, not evidence.
* Check the breakdowns for concentration: if all the profit comes from one
  session or one regime, that is a fragility, not an edge.

Run at least a few weeks of live paper signals before drawing any conclusion.
Backtests cannot model slippage, requotes, or news-driven gaps.

---

## Calibration

**Read this before changing the thresholds.**

The score is an additive weighted sum of nine components, several of which are
*event driven* — the liquidity engine only scores when a sweep actually happens,
structure only scores a CHoCH when character actually changes. Those events
rarely all coincide, so in practice the raw score distribution on XAUUSD M5 peaks
in the low 80s rather than running to 100: the median bar scores around 47, the
99th percentile around 76.

The shipped thresholds keep the **structure** of the nominal 90/82/75 confidence
bands (very strong / strong / moderate, regime-adaptive) but use the values the
engine's actual distribution supports (80/72/66, base threshold 72). These were
derived **from the score distribution alone, never from backtest profitability**.

To re-derive them for your own broker's data:

```bash
BASE_THRESHOLD=1 python backtest.py --data history/XAUUSD_M5.csv
python - <<'EOF'
import pandas as pd
d = pd.read_csv("data/backtest_evaluations.csv")
best = d[["bullish_score", "bearish_score"]].max(axis=1)
print(best.describe())
print(best.quantile([0.5, 0.9, 0.95, 0.99, 0.995]))
EOF
```

Set `BASE_THRESHOLD` near the 95th-99th percentile depending on how selective you
want to be, and scale the regime thresholds around it in the same proportions.
**Do not** tune thresholds by watching the P&L go up — that is how you overfit.

---

## Timezones

**Everything inside this project is UTC.** MT5 returns candle timestamps in
*broker server time*, which for most gold brokers is UTC+2 (winter) / UTC+3
(summer). `MT5_SERVER_UTC_OFFSET_HOURS` tells the system how far the broker clock
is ahead of UTC so timestamps can be normalised.

The default is `0` and `ALLOWED_SESSIONS` defaults to `ALL_SESSIONS`, so an unset
offset cannot silently mute every signal. **If you enable a session filter you
must set the offset correctly.** Session windows (UTC, end exclusive):

| Session | UTC window |
|---|---|
| `ASIAN` | 00:00 - 08:00 |
| `LONDON` | 07:00 - 16:00 |
| `NEW_YORK` | 12:00 - 21:00 |
| `LONDON_NEW_YORK_OVERLAP` | 12:00 - 16:00 |

Windows overlap; a single label is chosen by priority (overlap → London → New
York → Asian). The detected session is written to every row of
`evaluations.csv`, so you can check the offset is right by confirming that the
busiest hours land in `LONDON` and `LONDON_NEW_YORK_OVERLAP`.

Note these are fixed UTC windows and do **not** shift with DST — during northern
summer the real London/New York sessions sit an hour earlier than these labels.

---

## Anti-lookahead guarantees

This is the part that matters most in a signal system, so it is enforced
structurally rather than by convention, and asserted by tests.

1. **Closed candles only.** `MarketData.get_candles` discards the forming candle
   before returning. `MarketSnapshot` has no field that could carry one. The
   engine only ever sees a snapshot.
2. **Swings never repaint.** A swing high at bar *i* needs `swing_right` bars
   after it, so it is invisible until bar *i + swing_right* closes.
   `confirmed_swings()` discards any pivot whose confirmation bar has not closed,
   and every structure/liquidity/S-R calculation goes through it.
   `test_confirmed_swings_do_not_repaint` asserts a pivot known at bar N is still
   present and unchanged at bar N+k.
3. **Higher timeframes are cut by close time.** In the backtester an H1 candle
   becomes visible only once its *close* time is at or before the M5 candle's
   close time. `test_backtester_never_shows_an_unclosed_higher_timeframe_candle`
   asserts it.
4. **Indicators are causal.** Every function in `src/indicators.py` computes bar
   *i* from bars `0..i` only. That is what lets the backtester pre-compute
   indicators once over the whole history and slice — an optimisation, not a
   leak. `test_indicators_are_causal` asserts prefix-computed values equal
   full-array values at the same bar, to 1e-9.
5. **The cooldown context ignores the future.** `build_gate_state` discards any
   signal dated after the bar being evaluated.
6. **Outcome tracking starts on the next candle**, since the entry is the signal
   candle's close.
7. **One evaluation per candle.** `state.json` records the last processed candle
   open time; a candle can never be evaluated twice, so a signal cannot be
   duplicated.

---

## Tests

```bash
python -m pytest tests/ -q
```

252 tests covering indicator correctness (against reference implementations),
causality and no-repaint, score aggregation and weight reconfiguration,
confidence bands, adaptive thresholds, bull/bear separation, spread, session,
cooldown and duplicate prevention, fakeout rules, SL modes and clamping, TP
ladder ordering and pull-back, R:R arithmetic, the full signal lifecycle,
outcome scoring, CSV creation/append/schema-change handling, `state.json`
round-trip and corruption tolerance, Telegram formatting (no network), and the
backtester's no-lookahead guarantees.

Runtime is about 90 seconds — the end-to-end backtest tests dominate it.

---

## Honest assessment

**No claim is made that this system is profitable.** It has not been tested on
real XAUUSD history. The repository ships `make_synthetic_history.py`, which
generates a regime-switching random walk used only to exercise the code paths:

```bash
python make_synthetic_history.py
python backtest.py --data history/XAUUSD_M5_synthetic.csv
```

Results on synthetic data are meaningless as evidence of edge.

The walk-forward run on that synthetic data is instructive precisely because it
looks *bad*:

```
segment          bars  signals  closed   win%    avgR   totalR     PF   maxDD
IN_SAMPLE        4500       19      19   63.2   0.540    10.27   2.47    2.67
VALIDATION       2250       13      13   76.9   0.422     5.49   3.48    1.00
OUT_OF_SAMPLE    2250       19      18   44.4  -0.219    -3.95   0.58    5.64
```

Positive in-sample, positive in validation, **negative out-of-sample**, on ~50
signals total. That is exactly the pattern you must learn to distrust, and it is
what a small sample on near-random data should look like. Do not read the
in-sample numbers as encouraging.

Before trusting this system with anything: run it on **real** multi-year XAUUSD
M5 data, then paper-trade the live signals for weeks, and judge it on expectancy
and drawdown over a few hundred signals.

---

## Known limitations

* **Windows-only live mode** — the `MetaTrader5` package has no Linux/macOS build.
* **Tick volume, not real volume** — MT5 forex/CFD feeds report tick counts, a
  proxy for activity, not traded contracts.
* **Backtests cannot model spread, slippage or requotes.** `--spread` applies a
  constant, which is optimistic: real spreads widen exactly when the volatility
  and news filters matter most.
* **Intrabar order is unknown in backtests.** Without an M1 feed, any candle that
  touches both target and stop is scored as a stop. Live tracking replays M1 and
  is more accurate, so **live and backtested outcomes are not strictly
  comparable**.
* **Session windows are fixed UTC and do not follow DST.**
* **Weekend gaps** — gold gaps over the weekend; a stop can be jumped, which this
  model scores as a clean stop-out at the stop price. Real fills would be worse.
* **The score scale is compressed** (see [Calibration](#calibration)); the
  confidence number is a relative ranking, not a probability.
* **Single symbol, single timeframe.** No portfolio logic, no correlation
  awareness, no position sizing — it is a signal system, not a trading system.
* **Backtest speed** — about 30-40 bars/second; multi-year runs take a while.
* **No news or economic-calendar awareness.** NFP and CPI releases are exactly
  when gold behaves worst for technical setups, and the engine cannot see them.

---

## Recommended V2 improvements

1. **Validate on real data first.** Multi-year XAUUSD M5 history, walk-forward,
   then extended paper trading. Everything below is premature until this is done.
2. **ML probability filter.** `evaluations.csv` + `outcomes.csv` already form a
   training set. A gradient-boosted model estimating P(TP1 before SL) would sit
   *after* the deterministic engine as a veto, never replacing it — keeping the
   system explainable.
3. **Economic-calendar filter.** Suppress signals in a window around
   high-impact USD releases.
4. **Spread and slippage modelling in the backtester**, using recorded live
   spreads by hour of day rather than a constant.
5. **M1-based backtesting** to resolve intrabar order, closing the gap between
   backtested and live outcomes.
6. **Dynamic partial sizing** — the current fixed thirds are a modelling choice,
   not an optimised one.
7. **Multi-symbol support** with correlation awareness (XAUUSD, DXY, US10Y).
8. **A small dashboard** (the spec correctly excludes it from V1) reading the
   CSVs — equity curve, live status, rejection-reason histogram.
9. **Rejection-reason analytics.** The distribution in `evaluations.csv` shows
   which filter dominates; if one rejects 95% of candidates, either it is
   miscalibrated or the others are redundant.
10. **Regime-specific parameters** — the thresholds already adapt, but SL/TP
    geometry does not. Trends and ranges want different ladders.

---

*This software produces trading signals for research and paper-testing. It is not
financial advice, and it does not execute trades. Trading leveraged instruments
carries substantial risk of loss.*
