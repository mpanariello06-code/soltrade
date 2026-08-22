# XAUUSD + BTCUSD M1 Micro-Scalping Research System

A standalone, deterministic **signal-only** system running one M1 micro-scalping
engine over two markets: **XAUUSD** and **BTCUSD**.

```
MT5 M1 market data (per market)
      ↓
market configuration (XAUUSD | BTCUSD) folded onto the shared engine
      ↓
M1 microstructure analysis (momentum, displacement, sweeps, immediate S/R)
      ↓
multi-confirmation scoring (9 components → 0-100)
      ↓
strict filtering (threshold, conflict, fakeout, spread, session, cooldown)
      ↓
micro-scalping targets sized from live ATR + a full cost model
      ↓
Telegram notification
      ↓
CSV logging
      ↓
outcome tracking with timeout, milestone timing, RAW **and NET** R
```

> **This system never places, modifies or closes a trade.**
> MetaTrader 5 is used as a *read-only market-data feed*. No order-execution
> function is imported or called anywhere. It is a paper-signal and research
> tool.

> **No profitability claim is made — and the measured results are negative.**
> See [Honest assessment](#honest-assessment). Reading only the RAW numbers
> would be actively misleading at this timescale.

**One timeframe (M1). One mode (SCALPING). Two markets.** The multi-timeframe
selector and the RESEARCH/STANDARD/CONSERVATIVE modes were removed.

> **BTCUSD's parameters are INITIAL RESEARCH PARAMETERS.**
> They were chosen from the arithmetic of price scale and typical cost, not
> from a backtest. They are not optimised, not validated and **not claimed to
> be profitable**. See [Markets](#markets).

## Table of contents

1. [Quick start (Windows)](#quick-start-windows)
2. [Markets](#markets)
3. [Project layout](#project-layout)
4. [What "scalping" means here](#what-scalping-means-here)
5. [The cost model](#the-cost-model)
6. [Targets](#targets)
7. [Telegram control panel](#telegram-control-panel)
8. [How the signal engine works](#how-the-signal-engine-works)
9. [Scoring](#scoring)
10. [Filtering](#filtering)
11. [Near-signal diagnostic](#near-signal-diagnostic)
12. [Telegram messages](#telegram-messages)
13. [Where data is stored](#where-data-is-stored)
14. [Outcome tracking, timeout and the R model](#outcome-tracking-timeout-and-the-r-model)
15. [Backtesting](#backtesting)
16. [Walk-forward testing](#walk-forward-testing)
17. [Paper testing and reading the results](#paper-testing-and-reading-the-results)
18. [Calibration](#calibration)
19. [Timezones](#timezones)
20. [Anti-lookahead guarantees](#anti-lookahead-guarantees)
21. [Tests](#tests)
22. [Honest assessment](#honest-assessment)
23. [Known limitations](#known-limitations)
24. [Recommended next steps](#recommended-next-steps)

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

## Markets

Two markets, **one engine**. `ScalpingEngine → MarketConfig → XAUUSD | BTCUSD`.
Nothing in `src/` is duplicated per market: `src/markets.py` holds one
`MarketConfig` per instrument, and `Config.for_market(symbol)` folds it onto the
base configuration to produce the view the engine runs on.

```
config.Config                       base settings from .env
        │
        ├── for_market("XAUUSD") ──► view: gold's scale, costs, targets, paths
        └── for_market("BTCUSD") ──► view: BTC's  scale, costs, targets, paths
                                             │
                       effective_config() ───┴─► + that market's live
                                                  Telegram settings
```

### What differs, and why

| | XAUUSD | BTCUSD |
|---|---|---|
| Icon | 🥇 | ₿ |
| Trading hours | Sun 22:00 – Fri 21:00 UTC | 24/7 |
| Session filter | applied | **bypassed** (label still recorded) |
| Quote digits | 2 | 2 |
| 1 point | 0.01 | 0.01 |
| 1 "pip" | 0.10 (`p`) | 1.00 (`$`) |
| Base threshold | 68 | 68 *(initial, for comparability)* |
| TP ladder | 0.45 / 1.00 / 1.70 × ATR | **same** |
| TP floors | 1.0 / 1.8 / 3.0 pips | 0.020% / 0.036% / 0.060% of price |
| TP3 ceiling | 12 pips | 0.240% of price |
| Stop | 0.70 × ATR, clamped 0.50–0.80 | **same** |
| Assumed spread | 20 points (2 pips) | 1 000 points ($10) |
| Assumed slippage | 2 + 2 points | 200 + 200 points ($2/side) |
| Cost model name | `XAUUSD_RETAIL` | `BTC_RETAIL_CFD` |
| Spread rejection | > 35 points | > 4 000 points ($40) |
| Cooldown | 10 candles | 10 candles *(initial)* |
| Timeout | 15 candles | 15 candles *(initial)* |
| Data directory | `data/xauusd/` | `data/btcusd/` |

**The target model transfers; the floors do not.** Every distance is already a
multiple of the live M1 ATR, so it self-scales: a market that moves more per
minute automatically gets a wider target. What does *not* transfer is the
absolute floor underneath it. "At least 1.8 pips" is a sensible tick-grid floor
on gold at \$2 300 and a meaningless one on Bitcoin at \$60 000, so BTCUSD's
floors are expressed as a **fraction of price** instead. Gold's percentage
floors are set to zero, which leaves its pip floors in sole control and its
behaviour byte-for-byte unchanged.

### BTCUSD parameters are INITIAL RESEARCH PARAMETERS

Read this before drawing any conclusion from a BTCUSD run:

* They were derived from **arithmetic, not from a backtest** — price scale,
  plausible retail CFD spread, and the requirement that a target clear its own
  cost. No optimisation was performed, on any data.
* The spread and slippage figures are **assumptions about a venue we have not
  measured**. A retail crypto CFD, a spot exchange and a perpetual-futures
  venue have materially different cost structures. `src/markets.py` also ships
  `BTCUSD_EXCHANGE_COSTS`, a commented alternative for a maker/taker exchange.
* The threshold starts at gold's 68 **only so the two markets are comparable**
  on the first run. It is not a calibrated value for Bitcoin.
* Nothing here is claimed to be profitable. **Re-derive every number from your
  own venue's data** before trusting it.

### Retuning a market

Each market's values live in `src/markets.py`, beside the reasoning for them.
For per-install changes, `.env` overrides are namespaced by symbol, so no
variable can move both markets at once:

```dotenv
BTCUSD_THRESHOLD=72
BTCUSD_ASSUMED_SPREAD_POINTS=1500
XAUUSD_COOLDOWN_CANDLES=15
# brokers that rename an instrument:
BTCUSD_BROKER_SYMBOL=BTCUSD.x
```

The legacy unprefixed names (`SCALP_THRESHOLD`, `ASSUMED_SPREAD_POINTS`, …)
still work and apply to **XAUUSD only** — Bitcoin never inherits them.

### Adding a third market

Append one `MarketConfig` and call `register_market()`. No engine, tracker,
report or Telegram change is required; `tests/test_markets.py` asserts this.

---

## Project layout

```
xauusd_signal_bot/
├── main.py                  live paper-signal runner
├── backtest.py              M1 historical replay through the same engine
├── walkforward.py           in-sample / validation / out-of-sample report
├── performance.py           expectancy statistics from the CSVs
├── make_synthetic_history.py  generates test data (NOT real prices)
├── config.py                every tunable, loaded from .env
├── requirements.txt
├── .env.example
├── run.bat                  Windows launcher
├── data/                    created on first run
│   ├── state.json           GLOBAL: run status, active market
│   ├── system_log.txt       rotating log
│   ├── xauusd/              gold's data - never mixed with Bitcoin's
│   │   ├── signals.csv      one row per signal, status updated in place
│   │   ├── evaluations.csv  one row per evaluated candle (signal or not)
│   │   ├── outcomes.csv     one row per closed signal
│   │   └── state.json       gold's settings + last processed M1 candle
│   └── btcusd/              same five files, Bitcoin's own
├── src/
│   ├── market_data.py       MT5 access (READ-ONLY), validation, ticks, resampling
│   ├── indicators.py        EMA/RSI/MACD/ATR/ADX/Stoch/BB/swings
│   ├── trend.py             M1 trend engine + M5 context engine
│   ├── momentum.py          momentum engine (0-15)
│   ├── structure.py         swings, BOS, CHoCH (0-15)
│   ├── liquidity.py         reference levels, sweeps (0-10)
│   ├── support_resistance.py  zones, breakouts (0-10)
│   ├── volume.py            tick-volume confirmation (0-5)
│   ├── volatility.py        ATR regime (0-5, direction-neutral)
│   ├── price_action.py      candle confirmation (0-5)
│   ├── regime.py            market-regime classifier
│   ├── markets.py           MarketConfig per instrument (XAUUSD, BTCUSD)
│   ├── timeframes.py        M1 / SCALPING constants
│   ├── runtime_state.py     global state + one MarketRuntime per market
│   ├── telegram_control.py  inline-button control panel
│   ├── scoring.py           weighted aggregation → 0-100
│   ├── filters.py           thresholds and rejection rules
│   ├── targets.py           micro-scalping targets + the cost model
│   ├── signal_engine.py     orchestration
│   ├── signal_tracker.py    CSV persistence + outcome tracking (timeout, net R)
│   ├── telegram_bot.py      notifications
│   ├── logger.py            logging setup
│   └── utils.py             helpers
└── tests/                   216 unit tests
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

## What "scalping" means here

The system looks for very short-term XAUUSD moves — minutes, not hours. The
research concept is a first target of roughly **1–3 pips** (0.10–0.30 in price;
gold is quoted to 2 decimals and a pip is the first decimal, so 1 pip = 10
points).

> **That target is a starting concept, not a claim.** Whether a 1–3 pip move is
> worth taking depends entirely on what it costs to take it, which is why the
> cost model below is the most important part of this build.

Everything is sized from live conditions:

| | |
|---|---|
| Signal timeframe | **M1** — fixed |
| Context timeframe | M5 by default, optional (`CONTEXT_TIMEFRAME=""` to disable) |
| Holding period | `MAX_HOLDING_CANDLES` minutes, default **15**, then TIMEOUT |
| Targets | multiples of the **live M1 ATR**, floored by cost |
| Stop | ATR / structure / hybrid, capped so the ladder's R:R stays coherent |

M15/H1/H4 context was dropped deliberately: a one-hour trend says very little
about a position held for three minutes, and carrying it made the engine reject
good scalps for disagreeing with a timeframe that would not resolve inside the
holding window.

## The cost model

At a few pips of target, the spread is not a rounding error — it is often the
whole trade. Round-trip cost is

```
spread + entry slippage + exit slippage + commission (both sides)
```

With the defaults (20-point spread, 2+2 points slippage) that is **2.4 pips per
round trip**. A 2-pip target does not survive it.

Consequences, all enforced in code:

* **TP1 is floored** at `MIN_TP1_COST_MULTIPLE` × cost (default 1.5×). If the
  ATR-based target is smaller, it is raised — and the whole ladder is raised
  proportionally, so a wider spread demands a *bigger* move rather than a
  closer target.
* **The stop is floored** at `MIN_SL_COST_MULTIPLE` × spread (default 2×), so
  ordinary quote noise cannot take the trade out.
* **A setup is rejected** when TP1 still cannot clear costs, or when the
  cost-adjusted TP3 would exceed `MAX_TP3_PIPS` — a "scalp" that needs a
  20-pip move is not a scalp.
* **Every reward figure is reported twice**: `rr1/rr2/rr3` (RAW, price movement
  only) and `net_rr1/net_rr2/net_rr3` (NET, after cost). `MIN_NET_TP2_RR`
  rejects setups that pass the raw gate and still lose after costs.

`cost_r` — the cost expressed in R — is fixed at signal time and carried through
to the outcome, so NET R never depends on what the spread happens to be when
the trade closes.

## Targets

```
TP_i distance = tp_atr_multiples[i] × ATR      (market conditions)
              ⌊ floored at min_tp_pips[i]      (tick grid)
              ⌊ TP1 floored at 1.5 × cost      (worth taking)
              → whole ladder lifted proportionally if TP1 was raised
              → pulled back in front of major opposing structure
              → re-ordered, rounded to the instrument's 2 decimals
```

Defaults: `tp_atr_multiples = (0.45, 1.00, 1.70)`, stop `0.70 × ATR` clamped to
`[0.50, 0.80] × ATR`.

The stop ceiling matters: HYBRID takes the *wider* of the ATR and structure
distances, and on M1 the structure branch can otherwise put the stop 2–3 ATR
away while the targets stay put — leaving the ladder's reward/risk below 1 by
construction. `config.validate()` refuses a configuration where the widest
allowed stop makes TP2 worth less than `MIN_TP2_RR`.

**R:R is computed from the exact rounded prices that are published**, so the
numbers in the Telegram message are the numbers that were validated.

## Telegram control panel

Send `/panel` (or `/start`) to your bot. Buttons are the interface.

```
━━━━━━━━━━━━━━━━━━
⚡ M1 SCALPER
━━━━━━━━━━━━━━━━━━

Market: 🥇 XAUUSD
Timeframe: M1
Mode: SCALPING
Status: 🟢 RUNNING
Threshold: 68

Signals today: 4
Open signals: 1
Paper Net R: -2.30

Max hold: 15 min
MT5: CONNECTED

Also tracking: ₿ BTCUSD 1 open

━━━━━━━━━━━━━━━━━━
🔬 PAPER TEST ONLY - no orders are placed.

[● 🥇 XAUUSD] [○ ₿ BTCUSD]
[📊 CURRENT ANALYSIS]
[📈 PERFORMANCE]
[⏸ PAUSE]
[⚙️ SETTINGS]
[🔄 REFRESH]
```

The market row is first, the selected market is marked `●`, and the active
market is named on the panel, the analysis view, the settings menu and the
performance header — so it is never ambiguous which market a button will act on.
Every press takes effect in the running process and is persisted.

| Control | Effect |
|---|---|
| 🥇 XAUUSD / ₿ BTCUSD | Selects which market generates signals. **A selection, not a reset** — see below. |
| START / PAUSE | One toggle showing the action that is available. PAUSE keeps MT5 connected and keeps tracking open scalps on **both** markets, but generates none. STOP lives in Settings so it cannot be hit by accident. |
| 📊 CURRENT ANALYSIS | The latest evaluation for the **selected** market, on demand, from closed M1 candles only. Read-only: records nothing, cannot emit or suppress a signal. |
| 📈 PERFORMANCE | Per market by default, never merged. `[● 🥇 XAUUSD] [○ ₿ BTCUSD] [○ 📊 COMBINED]`, then BY SCORE / BY REGIME / BY HOUR / BY SESSION / BY OUTCOME. NET R leads. |
| ⚙️ SETTINGS | Headed `⚙️ XAUUSD SETTINGS` or `⚙️ BTCUSD SETTINGS`. Threshold, max holding period, cooldown, minimum R:R, session filter, near-signal alerts, STOP. **Applies to the active market only.** |

### Switching markets is safe

Switching changes *which market is evaluated* and nothing else. It does not
reset, clear or overwrite anything belonging to the market being left:

* its `signals.csv`, `evaluations.csv`, `outcomes.csv` and `state.json` are
  untouched;
* its threshold, cooldown, holding time and session settings are untouched;
* its processed-candle marker is untouched, so nothing is re-evaluated or
  skipped when you switch back;
* **its open paper trades keep being tracked, against its own candles**, on
  every subsequent cycle. An open XAUUSD scalp still reaches its TP, its SL or
  its timeout after you switch to BTCUSD, and vice versa.

`tests/test_markets.py` asserts each of these, including a live-loop test that
records the price range of every frame handed to each market's tracker and fails
if one ever receives the other's.

`COMBINED` performance is offered but opt-in and labelled: an R on gold and an R
on Bitcoin share a unit but come from different cost and volatility regimes.

**Safety.** Updates from any chat other than `TELEGRAM_CHAT_ID` are ignored.
Only the settings in `Config.telegram_editable_settings` are reachable —
credentials, the bot token, broker symbols and all file paths have no handler
and are never rendered into a message. On startup the poller discards updates
queued while the engine was down, so a restart never replays stale presses (the
active market *is* restored; the button press that set it is not replayed).
No button on any panel can place, modify or close an order — there is no such
code path in the project.

---

## How the signal engine works

The engine evaluates **once per newly closed M1 candle** — never on a forming
candle, and never twice for the same candle. `main.py` polls MT5 every
`POLL_SECONDS`, probes cheaply for a new close, and only then does the full
fetch.

For each evaluation:

1. **Validate** — enough history, ordered timestamps, no duplicates, no NaN or
   impossible OHLC, data not stale (an M1 candle over 3 minutes old means the
   feed has stalled).
2. **Compute indicators** on M1, and on the M5 context if enabled.
3. **Run the nine analysis engines**, each producing independent bullish and
   bearish sub-scores.
4. **Aggregate** into `bullish_score` and `bearish_score`, both 0-100.
5. **Classify the regime**, which adjusts the threshold.
6. **Filter** the stronger direction through the rejection chain.
7. **Build targets and cost** — and reject if the move cannot pay for itself.
8. **Emit** a scalp, or record a `NO_SIGNAL` / `NEAR_SIGNAL` evaluation.

Every evaluation is written to `data/evaluations.csv` **whether or not a signal
was produced** — that file is the audit trail and the future ML training set.

### The nine components, re-weighted for M1

| Component | Weight | What it looks at |
|---|---|---|
| Momentum | 22 | RSI/MACD/stochastic/ROC — short-term momentum and its acceleration |
| Price action | 18 | Displacement, engulfing, wick rejection, close location |
| Liquidity | 14 | Sweeps of the immediate highs/lows, reclaims, failed breaks |
| Structure | 12 | Micro swings, break of structure, change of character |
| Support/Resistance | 10 | The levels the next few pips must clear |
| Trend | 10 | M1 EMA alignment and slope, ADX/DI (was 20 on the M5 build) |
| Context | 6 | M5 bias only — secondary at this horizon (was 15) |
| Volatility | 5 | Is the expected move large relative to noise |
| Volume | 3 | Tick-volume confirmation — the weakest signal on M1 |

**These weights are not claimed to be optimal.** They were chosen by reasoning
about a 15-minute holding period — momentum and candle shape matter, a
multi-hour trend does not — and never by optimising against backtest results.

Indicator periods were shortened to match: EMA 5/13/34/100, RSI 9, MACD 6/13/5,
ATR 14 (i.e. 14 minutes), and structure pivots use a 1-bar fractal.

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

When the context timeframe is switched off (`CONTEXT_TIMEFRAME=""`) the context
component is marked *not applicable* and its 6 points are redistributed across
the other eight, so the score stays on a true 0-100 scale rather than being
silently capped at 94.

Confidence bands (`config.confidence_bands`) label the resulting score. See
[Calibration](#calibration) for why the shipped values are what they are.

---

## Filtering

Filters run in order; the **first** to fire rejects the candidate and its reason
is written to `evaluations.csv`.

| # | Filter | Rule |
|---|---|---|
| 1 | Session | Candle's UTC session must be in `ALLOWED_SESSIONS` |
| 2 | Spread | Absolute cap, **and** spread vs the move price can plausibly make in the holding window (ATR × √holding candles) |
| 3 | Volatility | `EXTREME` volatility blocks signals (configurable) |
| 4 | Threshold | Dominant score ≥ threshold + regime offset + counter-trend extra |
| 5 | Separation | `dominant − opposite ≥ MIN_SCORE_SEPARATION` |
| 6 | Fakeout | Unconfirmed breakouts, close back inside the range, context disagreement, thin participation, candle rejected in our direction |
| 7 | Cooldown | `COOLDOWN_CANDLES` minutes any direction, longer for a repeat |
| 8 | Limits | `MAX_SIGNALS_PER_DAY`, `MAX_CONCURRENT_ACTIVE_SIGNALS` |
| 9 | Cost & R:R | TP1 must clear costs, TP3 must fit the scalp range, TP2 must pass **both** the raw and the net R:R gates |

The spread test deserves a note. On M5 the old rule compared the spread with a
single candle's ATR; on M1 those two are nearly the same size, so the rule was
meaningless. It now compares the spread with the move available over the whole
holding window, which is the quantity that actually decides whether a scalp can
pay for itself.

### Adaptive threshold

`SCALP_THRESHOLD` (default 68) plus a regime offset: strong trend −4, weak
trend / breakout 0, low volatility +3, range / high volatility +5. Extreme
volatility blocks signalling. A counter-context setup adds +5. The result is
clamped to `[40, 95]`.

## Telegram messages

```
━━━━━━━━━━━━━━━━━━
⚡ XAUUSD M1 SCALP
━━━━━━━━━━━━━━━━━━

Direction: BUY

Score: 71/100

Entry: 2345.67

TP1: 2345.91   (2.4p)
TP2: 2346.20   (5.3p)
TP3: 2346.58   (9.1p)

SL: 2345.42   (2.5p)

Expected holding period:
SHORT

Spread:
0.12  (12 points)

Risk/Reward:
TP1 0.96R
TP2 2.12R
TP3 3.64R

After costs (1.6p = 0.64R):
TP1 0.32R
TP2 1.48R
TP3 3.00R

Reason:
momentum=18.4; price_action=14.1; liquidity=9.8

━━━━━━━━━━━━━━━━━━
🔬 PAPER TEST ONLY
```

Outcome alerts: `🎯 TP1/TP2/TP3 HIT`, `❌ SL HIT`, `⚠️ SIGNAL INVALIDATED`,
`⌛ SCALP TIMED OUT` — each reporting **raw and net** R.

**No duplicate alerts.** An alert fires only when a signal's status *changes*
against the value persisted in `signals.csv`. Because status is on disk, a
restart replays nothing. Sends are retried with backoff and never raise.

## Where data is stored

Plain files under `data/`, created automatically with headers, appended safely,
UTF-8, and rewritten atomically (temp file + replace) so a crash cannot truncate
them. No database.

**Every market gets its own directory.** No CSV ever holds rows from more than
one market, and every row is self-identifying (`symbol`, `timeframe`,
`timestamp`) so a file can still be checked, joined or merged deliberately:

```
data/
├── state.json          GLOBAL - run status, active_market
├── system_log.txt
├── xauusd/  signals.csv  evaluations.csv  outcomes.csv  state.json
└── btcusd/  signals.csv  evaluations.csv  outcomes.csv  state.json
```

Each `state.json` has exactly **one writer**. The global file is written only by
`RuntimeState`; each market file is written only by that market's
`MarketRuntime`, whose store the market's `SignalTracker` shares rather than
opening a second handle. That single-writer rule is what stops two components
clobbering each other's keys, and it is covered by a test.

| File (per market) | Contents |
|---|---|
| `signals.csv` | One row per scalp; `status` updated in place. Carries `symbol`, `timeframe`, `timestamp`, direction, entry, TP1-3, SL, score, bullish/bearish score, threshold used, regime, session, and the full geometry and cost at signal time: `spread_points`, `estimated_slippage`, `cost_pips`, `cost_r`, `sl_pips`, `tp1/2/3_pips`, `atr`, `rr1-3`, `net_rr1-3`, `expected_hold` |
| `evaluations.csv` | **Every** evaluated M1 candle: all nine sub-scores, regime, spread, decision (`BUY`/`SELL`/`NO_SIGNAL`/`NEAR_SIGNAL`), rejection reason, threshold used, near-signal flag, plus **41** raw `f_*` feature columns including the M1 microstructure block (`atr_pips`, `spread_pips`, `cost_pips`, `atr_to_cost`, `velocity_pips_per_min`, `acceleration`, `micro_range_pips_5/15`, `dist_to_high/low_5_pips`, `close_location`, `minute_of_hour`, `hour_of_day`) |
| `outcomes.csv` | One row per closed scalp: `symbol`, result (incl. `TIMEOUT`), exit level, **`raw_r` and `net_r`**, `cost_r`, `spread_points`, `estimated_slippage`, `minutes_to_tp1/2/3`, `minutes_to_sl`, `bars_to_*`, `mfe_price`/`mae_price`, `mfe_pips`/`mae_pips`, `timeout`, `ambiguous_bars`, holding time |
| `<market>/state.json` | `last_processed_candles` (that market's M1 marker), `last_signal_id`, `last_signal_time`, and a `runtime` section holding that market's live Telegram-controlled settings |
| `state.json` *(global)* | Run status (`RUNNING`/`PAUSED`/`STOPPED`), `active_market`, near-signal alert flag |
| `system_log.txt` | Rotating log (5 MB × 3), shared |

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
M1 features → P(move clears costs within the holding window) → filter → signal
```

That is the exact question a future model should answer: *given these M1
conditions, how likely is a move large enough to clear the spread, slippage and
commission before the holding window expires?* The feature vector and the
labelled outcome (`net_r`, `minutes_to_tp1`, `timeout`) are both already stored.

V1 deliberately contains **no** machine learning: it is transparent and
deterministic, and the same inputs always give the same outputs.

Because each market's features and labels are stored separately but in an
identical schema, either shape of future model is possible without migrating
anything: **one model per market** (train on `data/btcusd/`) or **one shared
model with the market as a feature** (concatenate both directories — every row
already carries its `symbol`).

---

## Outcome tracking, timeout and the R model

A signal is modelled as **three equal partials** at TP1/TP2/TP3. After TP1 the
stop moves to breakeven (`MOVE_SL_TO_BREAKEVEN_AFTER_TP1`).

**Every scalp closes.** If it has not reached a target or the stop within
`MAX_HOLDING_CANDLES` minutes it is marked to market and recorded as `TIMEOUT`.
Nothing sits "active" indefinitely.

| Path | RAW R | NET R (cost 0.6R) |
|---|---|---|
| Stopped before TP1 | −1.00 | −1.60 |
| TP1, then breakeven | +0.33 | −0.27 |
| TP1, TP2, then breakeven | +0.93 | +0.33 |
| All three targets | +1.87 | +1.27 |
| Timed out flat | ~0.00 | −0.60 |

Note the second row: a trade that *reaches its first target* still loses money.
That is the arithmetic the cost model exists to make visible.

Because scalps resolve in minutes, the record carries **when**, not only
whether: `minutes_to_tp1/2/3`, `minutes_to_sl`, `bars_to_*`, `mfe_r`/`mae_r`,
`mfe_pips`/`mae_pips`, `timeout`, and `ambiguous_bars`.

### Rules that keep it honest

* **A signal cannot trade against its own candle.** Tracking starts on the
  candle *after* the signal candle, because the entry is that candle's close.
* **Ambiguous candles.** When one M1 candle trades through both the next target
  and the stop, OHLC cannot say which came first. Live, the raw **tick** feed is
  replayed to resolve the order; when ticks are unavailable — always, in a
  backtest — the system assumes **the stop was hit first**. At scalping
  distances this is a large systematic penalty, which is why the live path
  bothers with ticks and why live and backtested outcomes are not strictly
  comparable.
* **A breakeven stop activates only on the next candle**, since the bar's low
  may well have occurred before the target was tagged.

## Backtesting

Feed a CSV of **M1** candles; the M5 context is resampled from the same file.

```bash
python backtest.py --data history/XAUUSD_M1.csv                   # default: XAUUSD
python backtest.py --symbol BTCUSD --data history/BTCUSD_M1.csv
python backtest.py --data history/XAUUSD_M1.csv --spread 6        # raw/ECN account
python backtest.py --data history/XAUUSD_M1.csv --start 2024-01-01 --end 2024-01-31
```

`--symbol` selects which `MarketConfig` is folded onto the engine — **the same
engine runs both markets**. Output goes to that market's own directory
(`data/xauusd/backtest_*.csv`, `data/btcusd/backtest_*.csv`), so two runs can
never contaminate each other's results. The header of every run states the
market, its cost model and, for BTCUSD, the research-parameters warning.

Required columns: `time, open, high, low, close, tick_volume`
(`date`/`datetime`/`timestamp` and `volume`/`vol` are accepted as aliases).

A backtest has no live quote, so **every trade is charged
`ASSUMED_SPREAD_POINTS`** unless `--spread` overrides it. It never trades for
free. It also has no tick feed, so every ambiguous candle is scored
pessimistically — the backtest is the conservative one.

Warm-up is about 1,000 M1 candles (the M5 context needs 150 closed candles).
Expect roughly 50 bars/second; a day of M1 data is ~1,440 bars.

`python make_synthetic_history.py [--symbol BTCUSD]` generates an M1 file for
smoke-testing the plumbing. It is **not** real data — and the BTCUSD profile in
particular is a plausible-looking guess at crypto M1 behaviour, not a
calibration. **No result computed on it is evidence of an edge on either
market.**

## Walk-forward testing

```bash
python walkforward.py --data history/XAUUSD_M1.csv
python walkforward.py --symbol BTCUSD --data history/BTCUSD_M1.csv
python walkforward.py --data history/XAUUSD_M1.csv --split 0.5 0.25 0.25
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
```

The report leads with the **after-costs** block, because on a scalp the raw
numbers are close to meaningless:

* **Average NET R is the headline.** It is the expectancy per scalp after
  spread, slippage and commission. Positive and stable is the only thing that
  matters.
* **Compare it with average RAW R.** The gap is what you are paying to trade. If
  the raw number is positive and the net one is negative, the strategy has no
  edge — it has a spread bill.
* **Net win rate ≠ raw win rate.** A trade that reaches TP1 and then stops at
  breakeven is a raw "win" and a net loss.
* **Timeout rate** tells you whether the holding window fits the setup. A high
  rate means scalps are being cut mid-move; a near-zero rate with a high SL rate
  means the stop is too tight for the noise.
* **`minutes_to_tp1`** is the most useful single diagnostic: if the median is
  close to the holding limit, the window is too short.
* **Score bands** answer whether a higher score actually produced a better
  outcome. Do not assume it did — measure.
* **Sample size dominates.** Fewer than ~200 closed scalps tells you very
  little.

## Calibration

**Read this before changing the threshold.**

The score is an additive weighted sum of nine components, several of them
event-driven (a liquidity sweep either happened or it did not). Those events
rarely coincide, so the distribution does not run to 100 — on the synthetic M1
data the median candle scores about 45 and the 99th percentile about 71.

`SCALP_THRESHOLD` defaults to **68**: high enough to produce candidates without
emitting one every other candle. On M1 there are 1,440 candles a day, so a
threshold a few points too low buries the useful setups in noise.

To re-derive it from your own broker's data:

```bash
SCALP_THRESHOLD=40 python backtest.py --data history/XAUUSD_M1.csv
python - <<'EOF'
import pandas as pd
d = pd.read_csv("data/backtest_evaluations.csv")
best = d[["bullish_score", "bearish_score"]].max(axis=1)
print(best.describe())
print(best.quantile([0.5, 0.9, 0.95, 0.99, 0.995]))
EOF
```

Set the threshold near the 95th–99th percentile depending on how selective you
want to be. **Do not tune it by watching the P&L go up** — that is how you
overfit, and the evaluation set is the same data.

The weights and target multiples carry the same warning: they were chosen from
reasoning about the holding period, not from results.

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
3. **The context timeframe is cut by close time.** In the backtester an M5
   candle becomes visible only once its *close* time is at or before the M1
   candle's close time. `test_backtester_never_shows_an_unclosed_context_candle`
   asserts it.
4. **Indicators are causal.** Every function in `src/indicators.py` computes bar
   *i* from bars `0..i` only. That is what lets the backtester pre-compute
   indicators once over the whole history and slice — an optimisation, not a
   leak. `test_indicators_are_causal` asserts prefix-computed values equal
   full-array values at the same bar, to 1e-9.
5. **The cooldown context ignores the future.** `build_gate_state` discards any
   signal dated after the bar being evaluated.
6. **Outcome tracking starts on the next candle**, since the entry is the signal
   candle's close. Tick data used to resolve ambiguity is filtered to the bar's
   own minute, so a later tick cannot decide an earlier candle.
7. **One evaluation per candle.** `state.json` records the last processed M1
   candle open time; a candle can never be evaluated twice, so a signal cannot
   be duplicated.

---

## Tests

```bash
python -m pytest tests/ -q
```

262 tests covering indicator correctness and causality, no-repaint swings,
score aggregation and weight renormalisation, the adaptive threshold, bull/bear
separation, the spread and **net** R:R gates, the cost model, target geometry
(ATR scaling, pip floors, percentage floors, cost floors, proportional lifting,
ordering, exact rounded pricing), timeout behaviour, milestone timing and
excursions, pessimistic and tick-resolved ambiguous candles, the full signal
lifecycle, CSV creation/append/schema-change handling, `state.json` round-trip
and corruption tolerance, Telegram formatting and every panel control, duplicate
prevention, restart safety, and the backtester's no-lookahead guarantees.

`tests/test_markets.py` (46 tests) covers the two-market behaviour specifically:

* the registry, symbol normalisation, and adding a third market without an
  engine change;
* that XAUUSD's shipped parameters are still the single-market build's values;
* that BTCUSD copies none of gold's absolute distances, and that its percentage
  floors scale with price while gold's pip floors do not;
* that the two markets produce different target distances and different costs
  from the same ATR ladder;
* that BTCUSD is never blocked by the session filter while its session label is
  still recorded, and that gold's session filter still works;
* **state isolation** — a threshold, cooldown, holding-time, R:R, session or
  processed-candle change on one market provably does not touch the other, and
  survives a restart per market;
* **file isolation** — separate directories, disjoint signal ids, and a gold
  signal that never appears in Bitcoin's files;
* **cooldown isolation** — a gold signal does not put Bitcoin on cooldown;
* **tracking isolation** — a live-loop test that records the price range of
  every frame handed to each market's tracker and fails if one ever receives the
  other's;
* **switch safety** — switching markets three times leaves the other market's
  full state, marker and open trades identical, and an open gold trade keeps
  being tracked after switching to Bitcoin;
* Telegram routing, per-market settings, threshold bounds on both markets,
  per-market analysis, per-market (not merged) performance, and that no button
  on any panel maps to an order.

Runtime is about 100 seconds.

## Honest assessment

**The measured results are negative after costs.** This is not a hedge — it is
the finding.

Backtested on 30,000 synthetic M1 candles (~20 trading days), charging spread
plus 2+2 points of slippage:

| | spread 6 pts (raw/ECN) | spread 20 pts (retail) |
|---|---|---|
| Round-trip cost | 1.0 pip | 2.4 pips |
| Signals | 348 | 42 (shorter slice) |
| Median TP ladder | 1.6 / 3.6 / 6.0 pips | 3.6 / 6.5 / 10.9 pips |
| Median stop | 2.9 pips | 4.0 pips |
| Median TP2 R:R | 1.26R raw → **0.94R net** | 1.62R raw → **1.02R net** |
| **Raw** win rate | 58.0% | 52.4% |
| **Raw** expectancy | −0.090R | +0.033R |
| **Net** win rate | 24.4% | 23.8% |
| **Net** expectancy | **−0.518R** | **−0.567R** |
| Median hold | 2 min | 15 min (64% timed out) |

Read the two win-rate rows together. At a 20-point spread the strategy wins
52% of the time and makes a *positive raw* expectancy — and still loses
0.567R per trade. The gap is the spread.

The mechanism is visible in the outcome mix: at the tighter spread 58% of
scalps touch TP1, but the breakeven stop then converts most of them into ~0R
while the losers pay the full −1R *plus* costs. That asymmetry is what kills
small-target scalping, and it is why the report leads with NET.

Two further caveats:

* **This is synthetic data.** A regime-switching random walk has no
  microstructure, no order flow and no news. Results on it say nothing about
  real gold — they only prove the machinery measures what it claims to.
* **The 1–3 pip concept mostly does not survive contact with the cost model.**
  At a 20-point spread the cost floor lifts TP1 to 3.6 pips, four times the
  original concept. Only on a raw/ECN spread does a 1.6-pip first target
  survive at all.

Before trusting any of this: run it on **real** multi-month XAUUSD M1 data with
your broker's actual spreads, then paper-trade the live signals for weeks, and
judge on NET expectancy over a few hundred scalps.

---

## Known limitations

* **Live mode is Windows-only** — the `MetaTrader5` package has no Linux/macOS
  build. Everything else runs anywhere.
* **Backtests cannot use ticks**, so every ambiguous candle is scored as a stop.
  Live tracking replays ticks and will differ — **live and backtested outcomes
  are not directly comparable**.
* **Slippage is an assumption, not a measurement.** Gold's default (2+2 points)
  is plausible for liquid hours and optimistic around news. **Bitcoin's
  (200+200 points = $2/side) has not been measured at all** — it is an initial
  research figure.
* **BTCUSD has not been backtested on real data, or calibrated on any data.**
  Its threshold, cooldown, holding window, spread and slippage are starting
  points, not findings. Treat any BTCUSD number this system produces as a test
  of the plumbing until you have re-derived them from your own venue.
* **Only the selected market generates signals.** That is deliberate (it keeps
  alert volume and MT5 load bounded), but it means the two markets are not
  evaluated in parallel by default. `EVALUATE_ALL_MARKETS=true` lifts this for
  research runs; open trades on *both* markets are always tracked regardless.
* **R is not comparable across markets.** Gold R and Bitcoin R share a unit but
  come from different cost and volatility regimes. The COMBINED performance view
  exists, is opt-in, and says so on the panel.
* **The cost model deducts cost once per trade**, not per partial. With three
  partials the true cost is somewhat higher, so NET R here is mildly optimistic.
* **Tick fetching adds MT5 load.** The buffer is incremental (about a minute of
  ticks per poll) but a slow terminal may lag; set
  `USE_TICKS_FOR_AMBIGUOUS_CANDLES=false` to fall back to pessimistic scoring.
* **No news or economic-calendar awareness.** Scalping through an NFP release is
  exactly where the spread and slippage assumptions break down.
* **Session windows are fixed UTC** and do not follow DST.
* **`assumed_spread_points` is a single number.** Real spreads vary by hour and
  widen precisely when volatility makes setups look attractive.
* **Weekend gaps** can jump the stop; the model scores that as a clean stop-out.
* Everything remains **paper only** — no order execution exists anywhere.

## Recommended next steps

1. **Get real M1 data with recorded spreads.** Everything below is premature
   until the numbers above are re-measured on real ticks. Record the live spread
   per candle rather than assuming one.
2. **Re-examine the breakeven rule.** It is the largest single driver of the raw
   → net collapse. Test holding the original stop, or moving it only after TP2.
3. **Re-examine the holding window.** 64% of scalps timed out at the wider
   spread. `minutes_to_tp1` in the outcomes file tells you what the window
   should be — but tune it on one period and verify on another.
4. **Per-partial cost accounting**, so NET R stops being mildly optimistic.
5. **An ML probability filter.** `evaluations.csv` + `outcomes.csv` already form
   the training set, and the label (`net_r`) is already cost-adjusted. It would
   sit *after* the deterministic engine as a veto, never replacing it.
6. **Spread-aware scheduling** — only signal in the hours where the recorded
   spread historically supports the target size.
7. **Economic-calendar filter** around high-impact USD releases.

---

*This software produces trading signals for research and paper-testing. It is
not financial advice, and it does not execute trades. Trading leveraged
instruments carries substantial risk of loss.*
