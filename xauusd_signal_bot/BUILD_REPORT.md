# Build Report — DEMO Auto-Execution Layer

Fifth iteration of `xauusd_signal_bot`. **The project was not rebuilt, and no
signal-generation logic changed.** A DEMO-only execution layer was added
downstream of the existing engine so that the gap between paper expectancy and
real fills can be measured.

* **Tests:** 362 passing (280 before + 82 new execution tests), ~110 s
* **Static analysis:** `pyflakes` clean across every module
* **Live trading:** none, and none can be configured — see [§C2](#c2-there-is-no-live-mode)
* **Default mode:** `SIGNAL_ONLY`. Execution is opt-in on two independent switches.

---

## C1. Architecture

```
Signal Engine → Signal → ExecutionManager → DEMO BROKER → Position
                              │                               │
                              └──── TP / SL / timeout ────────┘
                                          │
                              executions.csv + Telegram
```

Three new modules, none of which the signal path imports:

| Module | Responsibility |
|---|---|
| `src/execution_config.py` | Modes, the demo-account guard, the risk model. Everything deciding *whether* an order may be sent. |
| `src/demo_broker.py` | A `DemoBroker` port, the MT5 adapter, and a scripted fake. Transmits; decides nothing. |
| `src/demo_execution.py` | The manager: gates, sizing, fills, the partial ladder, stop/timeout, reconciliation, `executions.csv`. |

The split between the first and third is deliberate. A reviewer can read
`execution_config.py` alone and satisfy themselves that the default is safe and
that no live mode exists, without reading any trading code.

`signal_engine.py` and every analysis engine are untouched and contain no
reference to a broker, an order or the execution layer; a test asserts it.
Execution consumes a *completed* signal and uses its symbol, direction, entry,
SL, TP1–TP3, score, timeframe and timestamp verbatim.

One `ExecutionManager` per market, created in `MarketSlot` beside the existing
paper tracker, so gold and Bitcoin share the broker connection and nothing else.

## C2. There is no live mode

`ExecutionMode` has exactly two members. `assert_no_live_mode()` runs at import
and in the suite, and raises if any member's name contains `LIVE`, `REAL` or
`PROD`, or if the member count is not two. Adding `LIVE_AUTO` therefore breaks
the build rather than quietly turning a research tool into a trading system.

`parse_execution_mode()` maps anything unrecognised — including `LIVE_AUTO` —
to `SIGNAL_ONLY` with a warning. There is no spelling of `EXECUTION_MODE` that
enables real trading.

## C3. Demo-only safety

Execution requires **three** independent things, and any one missing means no
orders are sent:

1. `EXECUTION_MODE=DEMO_AUTO`
2. `DEMO_TRADING_ENABLED=true`
3. dedicated `DEMO_MT5_*` credentials

The demo credentials are deliberately **not** the `MT5_*` data-feed ones.
Reusing them would mean that re-pointing the data feed at a live account
silently armed execution against it.

Ten gates run in order before any transmission; the first failure aborts:

| # | Gate |
|---|---|
| 1 | signal has an id and has not been processed before |
| 2 | both switches + demo credentials |
| 3 | **account positively verified as DEMO** |
| 4 | broker symbol exists on the account |
| 5 | a usable quote is available |
| 6 | spread within the limit |
| 7 | SL/TP coherent, correctly sided, not absurdly distant |
| 8 | open-position, daily-trade and daily-loss limits |
| 9 | position size resolves to a tradeable lot |
| 10 | order accepted, and stops read back and verified |

**The account gate has no optimistic branch.** `classify_account()` sets
`is_demo=True` only for MT5 trade mode `0`. A live account, a *contest* account
(mode 1), a missing account object and a failed `account_info()` call all yield
`is_demo=False`, and all are refused with `DEMO_EXECUTION_BLOCKED` and a
Telegram alert. The check runs before **every** order, never cached, because a
terminal can be re-pointed mid-session and a stale "yes, demo" is the worst
possible thing to cache. A test asserts that changing the account between two
signals stops the second order.

Refusals return a decision object rather than raising — a blocked trade is a
normal, loggable outcome — except an unverified or live account, which raises so
it can never be mistaken for a routine skip.

## C4. Position sizing

Never hard-coded. Derived from the **signal's own stop distance**, so a wider
stop takes a smaller position and risk per trade stays comparable:

```
risk_amount = DEMO_ACCOUNT_BALANCE × DEMO_RISK_PER_TRADE
lots        = risk_amount / (stop_points × money_per_point_per_lot)
```

Then **floored** onto the lot grid — never rounded up, which would risk more
than configured, an error that is a large share of a micro-scalp — and clamped
into `[MIN_DEMO_LOT, MAX_DEMO_LOT]` and under `MAX_ORDER_LOTS`. A size below the
broker minimum is a rejection, not a minimum-size trade. `SizingResult` carries
the requested size, approved size, stop distance, risk amount and the reason
they differ, and it is logged on every decision.

The balance used is **notional**, not the broker's. A demo balance is arbitrary,
and letting it drive size makes runs incomparable.

## C5. Entry, SL/TP and management

* **A BUY lifts the ask, a SELL hits the bid.** The candle close is never used
  as an execution price. `signal_entry`, `requested_price` and
  `actual_fill_price` are three separate recorded fields, with signed slippage
  derived from the last two.
* **The signal's levels are used verbatim.** MT5 holds one TP per position, so
  the broker holds TP3 as a hard target and the manager runs a configurable
  partial ladder (default 33/33/34, not claimed optimal) for TP1 and TP2.
* **Stops are read back after the fill**, not assumed from the request, and
  attached if missing. An accepted order with no protection is the dangerous
  case.
* Positions are managed on **every poll**, not only on candle close, because a
  stop can be reached mid-candle.
* Where one observation could be read as either the stop or a target, **the stop
  wins**. Anything else would flatter the result.
* **Breakeven follows `MOVE_SL_TO_BREAKEVEN_AFTER_TP1`** rather than deciding
  for itself — used if the strategy uses it, not introduced if it does not.
* **Timeout reuses `MAX_HOLDING_CANDLES`.** No new timeout concept.

One bug worth recording: the first implementation measured R against the
*current* stop, so the breakeven move after TP1 set the denominator to zero and
R vanished from every winning trade. `initial_risk` is now captured at entry and
frozen; a test asserts R survives a breakeven move.

## C6. Failure handling

* **A signal is marked processed BEFORE the order is transmitted.** If the send
  raises or the reply is lost, it is never blindly retried.
* An **indeterminate** reply (exception, or `None` from `order_send`) is
  distinguished from a rejection: the order may exist, so execution **halts**
  and asks for reconciliation rather than retrying.
* Rejections, invalid volume, invalid stops, unavailable symbol, missing quote
  and partial fills are each logged with the broker's own reason and surfaced to
  Telegram. A partial fill is recorded at the size actually filled.
* Halting stops **new** orders only; open positions keep being managed.

## C7. Restart recovery

On start-up and after any reconnection, **before any new order**: connect,
verify the account, read positions carrying our magic number, match them to
stored trades, restore tracking. Three outcomes:

* a stored trade whose position is **gone** closed while we were away — recorded
  honestly, with `notes` saying the exit price was not observed, rather than
  tracked forever;
* a position we have **no record of** is adopted so it is still managed;
* a **failure to read positions halts execution**, because opening new trades
  against an unknown book is how duplicates happen.

Processed signal ids persist in the market's `state.json`, so a restart cannot
re-execute a live signal. The existing "do not replay old Telegram callbacks"
fix is preserved and re-tested with a queued `demo:on` press.

## C8. Logging and reporting

`data/<market>/executions.csv`, one row carrying **signal, execution and outcome
together** so a fill traces back to its signal.

**Execution results never overwrite signal results.** `outcomes.csv` keeps being
written for every signal whether or not it was executed — which is the entire
point of the layer. `performance.py` gains `analyse_executions()` and
`compare_signal_and_execution()`, and Telegram gains a `🤖 DEMO EXECUTION` view
showing signal average R beside execution average R and the degradation between
them. The comparison is suppressed below 10 demo fills, because a handful of
trades against hundreds of paper signals is not a measurement.

## C9. Telegram

Panel gains `Execution:`, `Open Demo Trades:`, `Today's Demo Trades:` and
`Today's Net P/L:`; the footer switches between `🔬 SIGNAL ONLY` and
`🤖 DEMO AUTO`. New buttons: `💼 OPEN TRADES` and the `🔴/🟢 DEMO AUTO` toggle.

**Enabling is never implicit.** The OFF button routes to `demo:confirm`, which
only renders a warning; only an explicit `demo:on` asks the engine to arm, and
the engine re-runs the whole connect/verify/reconcile sequence before reporting
success. A refused enable reports why rather than showing ON.

Signal cards now say `🔬 SIGNAL … no demo order was placed` in `SIGNAL_ONLY`,
and executed trades get their own `🤖 DEMO TRADE OPENED` and
`📊 DEMO TRADE CLOSED` messages carrying fill, slippage, gross, costs, net and R.

## C10. Known limitations

* **Demo fills are not live fills.** Demo servers typically fill better — less
  requoting, less asymmetric slippage, no market impact. Demo results are an
  *upper bound* on live execution.
* **`money_per_point_per_lot` is an assumption** (gold $1.00, Bitcoin $0.01 per
  point per lot). Contract sizes vary between brokers.
* **`estimated_cost` is modelled, not billed**, and deducted once per trade
  rather than per partial close, so net R is mildly optimistic on a full ladder.
* A position closed while the bot was offline has an unobserved exit price.
* Sizing uses a notional balance and does not track the demo account's equity.
* The MT5 adapter itself is only exercised by the fake in tests — it cannot be
  integration-tested on Linux, so its request shapes are unverified against a
  live terminal.

---

# Build Report — Adding BTCUSD as a Second Market (previous iteration)

Fourth iteration of `xauusd_signal_bot`. **The project was not rebuilt, and no
separate BTC bot was created.** The existing M1 scalping engine was made
market-agnostic by extracting everything instrument-specific into one
`MarketConfig` per market; XAUUSD's shipped values were lifted verbatim from the
previous build so gold's behaviour is unchanged.

* **Tests:** 280 passing (216 before + 64 new two-market tests), ~120 s
* **Static analysis:** `pyflakes` clean across every module
* **Order execution:** still none anywhere — paper/signal only
* **BTCUSD parameters:** INITIAL RESEARCH PARAMETERS. Not optimised, not
  validated, **not claimed to be profitable**. See [§B4](#b4-btcusd-parameters).

---

## B1. The architecture change

One engine, two configurations:

```
ScalpingEngine  ──reads──►  Config view
                                 ▲
              Config.for_market(symbol)
                                 ▲
                          src/markets.py
                          ├── XAUUSD_CONFIG   (existing values, verbatim)
                          └── BTCUSD_CONFIG   (INITIAL RESEARCH PARAMETERS)
```

`Config.for_market()` is the **single seam**. It returns a shallow copy of the
base config with ~35 fields overlaid — price scale, thresholds, target floors,
stop geometry, cost model, pacing, session behaviour and every file path. Nine
analysis engines, the scoring layer, the filter chain, the target builder, the
tracker, the backtester and the performance report all keep reading plain
config attributes; only the values differ. **No analysis code was duplicated.**

`effective_config()` folds a second layer on top: whatever the Telegram panel
has changed *for that market*.

Adding a third market is one `MarketConfig` plus `register_market()`. A test
asserts that this needs no engine change.

### What was NOT assumed to be shared

Volatility, spread, target distances, stop distances, thresholds, cooldowns,
timeouts, session behaviour and liquidity are all per-market fields. The one
thing deliberately shared is the *shape* of the target model — see §B3.

## B2. XAUUSD preservation

`XAUUSD_CONFIG` holds the previous build's numbers unchanged: threshold 68,
TP multiples 0.45/1.00/1.70 × ATR, TP floors 1.0/1.8/3.0 pips, TP3 ceiling 12
pips, stop 0.70 × ATR clamped 0.50–0.80, assumed spread 20 points, slippage
2+2, cooldown 10 / same-direction 20, timeout 15 candles, M5 context.

Two mechanisms keep gold byte-identical:

* the **percentage** target floors added for Bitcoin are **zero** for gold, so
  its pip floors remain in sole control;
* the legacy unprefixed `.env` names (`SCALP_THRESHOLD`, `ASSUMED_SPREAD_POINTS`,
  …) still work and apply to XAUUSD only, so an existing install keeps its
  tuning and Bitcoin never inherits it.

Legacy `data/state.json` from the single-market build is migrated into
`data/xauusd/state.json` — it was gold's, so it is adopted by gold rather than
dropped or, worse, applied to Bitcoin. A test asserts the "worse" case.

## B3. The target model, and what actually transfers

Targets were already sized from the live M1 ATR rather than fixed distances, so
the *model* transfers unchanged: a market that moves more per minute
automatically gets a wider target. The identical ATR ladder is used on both.

What does **not** transfer is the absolute floor underneath it. "At least 1.8
pips" is a sensible tick-grid floor on gold at $2 300 and meaningless on Bitcoin
at $60 000. So `MarketConfig` carries **two** floors per target — an absolute
one in the market's pip unit and one as a fraction of price — and the larger
wins. Gold's percentages are zero; Bitcoin's pip floors are zero.

| | XAUUSD | BTCUSD |
|---|---|---|
| TP floors | 1.0 / 1.8 / 3.0 pips | 0.020% / 0.036% / 0.060% of price |
| TP3 ceiling | 12 pips | 0.240% of price |

A test asserts that Bitcoin's floor at $90 000 is exactly three times its floor
at $30 000, and that gold's floor at $4 600 equals its floor at $2 300.

## B4. BTCUSD parameters

**Every BTCUSD number is an INITIAL RESEARCH PARAMETER.** They were derived from
the arithmetic of price scale and plausible cost, not from a backtest. No
optimisation was performed on any data, synthetic or real.

| Parameter | Value | Where it came from |
|---|---|---|
| `point_value` | 0.01 | MT5 convention: quoted to 2 digits, so a point is 10⁻² |
| `pip_value` | 1.00 (`$`) | One dollar is the readable reporting unit at this scale |
| `threshold` | 68 | **Starts equal to gold**, purely so the first runs are comparable. Not calibrated for Bitcoin |
| `assumed_spread_points` | 1 000 ($10) | Plausible retail crypto CFD. **Not measured** |
| `slippage_points_entry/exit` | 200 ($2) each | Plausible retail. **Not measured** |
| `max_spread_points` | 4 000 ($40) | Rejection ceiling, scaled from the assumption |
| `cooldown_candles` | 10 | **Initial**, equal to gold |
| `max_holding_candles` | 15 | **Initial**, equal to gold |
| `is_24h` | true | Factual: crypto does not close |

`src/markets.py` also ships `BTCUSD_EXCHANGE_COSTS`, an alternative cost block
for a maker/taker spot exchange (tighter spread, real commission), because a
retail CFD, a spot exchange and a perpetual-futures venue have materially
different cost structures and only one of them can be the default.

All of these are overridable from `.env` with a `BTCUSD_` prefix, and there is
no variable that can move both markets at once.

## B5. Cost model

Each market carries its own `assumed_spread_points`, entry/exit slippage,
commission and cost-model name (`XAUUSD_RETAIL`, `BTC_RETAIL_CFD`). RAW R and
NET R stay separate everywhere they were separate before — signal card, CSV,
Telegram performance view, backtest report — and a setup whose TP1 does not
clear `min_tp1_cost_multiple` × the round-trip cost is still rejected outright,
now against its own market's cost.

`estimated_slippage` was added to both `signals.csv` and `outcomes.csv` so the
assumption in force at signal time is recorded rather than inferred later.

**These are research assumptions, not measured execution.** Nothing in this
build claims a small favourable move is profitable; the cost model exists
precisely to stop that mistake.

## B6. Sessions

`check_session` returns immediately for a 24/7 market — Bitcoin is never blocked
by a session filter. The session **label** is still computed and stored on every
BTCUSD row, because knowing which UTC block a setup came from is useful for
analysis even when it does not filter. Gold's session logic is untouched.

The performance report gained a BY UTC HOUR view alongside BY SESSION, which is
the more meaningful cut for a market with no sessions.

## B7. State and storage isolation

```
data/
├── state.json          GLOBAL: status, active_market, near-signal flag
├── xauusd/  signals.csv  evaluations.csv  outcomes.csv  state.json
└── btcusd/  signals.csv  evaluations.csv  outcomes.csv  state.json
```

`RuntimeState` was split into a global part (run status, active market) and one
`MarketRuntime` per market (threshold override, cooldown, holding time, min R:R,
sessions, **and that market's own `last_processed_candle`**).

Every `state.json` has exactly **one writer**: the global file is written only
by `RuntimeState`, each market file only by that market's `MarketRuntime`, whose
store the market's `SignalTracker` shares rather than opening a second handle.
That single-writer rule fixed a clobbering bug in an earlier iteration and is
preserved here; a test asserts it.

Switching markets is a **selection and nothing else**. The market being left
keeps its files, its settings, its cooldown, its processed-candle marker and its
open paper trades — and those open trades keep being tracked against its own
candles on every subsequent cycle. `main._tick()` computes the markets that need
data as *"being evaluated, OR still holding an open signal"*, which is what
keeps a gold scalp alive after a switch to Bitcoin.

## B8. Telegram

* Market row first on the main panel: `[● 🥇 XAUUSD] [○ ₿ BTCUSD]`, active
  marked `●`, and `Market: 🥇 XAUUSD` named on the panel body.
* `Also tracking: ₿ BTCUSD 1 open` when the other market still holds a trade.
* START/PAUSE became one toggle showing the action that is available.
* ANALYSIS evaluates the **selected** market and names it in the header.
* PERFORMANCE is per market and **never merged by default**:
  `[● 🥇 XAUUSD] [○ ₿ BTCUSD] [○ 📊 COMBINED]`, then BY SCORE / BY REGIME /
  BY HOUR / BY SESSION / BY OUTCOME. COMBINED carries a warning that the two
  markets have different cost and volatility regimes.
* SETTINGS is headed `⚙️ XAUUSD SETTINGS` / `⚙️ BTCUSD SETTINGS` and acts on
  the active market only. The threshold panel says so explicitly.
* Signal cards, near-signal diagnostics and outcome alerts are rendered with the
  **signal's own** market icon, digits and pip unit — not whichever market is
  selected when the message is sent.
* `active_market` is persisted and restored on restart. The startup poller still
  discards updates queued while the engine was down, so a restart restores the
  market **without replaying the button press that set it** — the earlier bug
  remains fixed, and a test covers it.

## B9. Backtesting

`backtest.py --symbol XAUUSD|BTCUSD` runs the **same** engine with that market's
config folded on, and writes to that market's own directory, so results are
never mixed. `walkforward.py` gained the same flag.
`make_synthetic_history.py --symbol BTCUSD` produces a BTC-scaled fixture whose
regimes are expressed as fractions of price.

**No profitability claim is made from synthetic fixtures, for either market.**
The BTCUSD synthetic profile is a plausible-looking guess at crypto M1
behaviour, not a calibration against exchange data, and BTCUSD has not been
calibrated or validated on real data at all.

## B9a. Symbol names

The account's broker suffixes both instruments with a lowercase `s`, so the
**canonical symbols are `XAUUSDs` and `BTCUSDs`** — one name per instrument,
used at the MT5 feed, in `data/xauusds/`, in the `symbol` column of every CSV,
in signal ids, in every Telegram panel and on every CLI. `broker_symbol` is left
empty because there is nothing to translate.

Choosing the broker's name as canonical (rather than translating at the feed)
means there is exactly one string per instrument and no boundary at which the
two spellings can drift apart. The cost is that renaming the symbol renames the
data directory, which is handled below.

Four failure modes are handled explicitly:

* **`.upper()` would destroy the suffix.** Symbol matching previously
  upper-cased, which would turn `XAUUSDs` into `XAUUSDS` — the exact character
  that makes the name correct. Matching is now case-insensitive via `casefold`
  and always returns the registry's own casing. The same bug existed in the
  `MultiMarket` test double and was fixed there too.
* **Older spellings must keep working.** `SYMBOL_ALIASES` maps `XAUUSD`,
  `BTCUSD` and `GOLD` onto the canonical names, so a `--symbol BTCUSD`, a
  persisted `active_market`, a CSV row and `XAUUSD_THRESHOLD` in `.env` all
  still resolve. argparse `choices=` compares the raw string and would have
  rejected exactly those aliases, so the CLIs use a `market_argument` type
  function instead. Unknown symbols still raise rather than defaulting to gold.
* **A renamed market orphans its data directory.** The key follows the symbol,
  so `data/xauusd/` became `data/xauusds/`. `Config._adopt_legacy_market_dir()`
  renames an existing legacy directory on first use — only ever into a name that
  does not exist yet, so it cannot overwrite, and it is a no-op afterwards.
* **A wrong symbol is silent.** MT5 returns an empty result, not an error, for a
  symbol it does not have — indistinguishable from a market with no candles.
  `MarketData._resolve_symbol()` checks the configured name at connect time and,
  if missing, searches the broker's own symbol list (across every accepted
  spelling, since `XAUUSDs` would not prefix-match another broker's
  `XAUUSD.m`), adopts the shortest match for that session, and logs the exact
  `.env` line to make it permanent. The guess is never written back to config.

Two hardcoded symbol comparisons were removed while doing this: the Telegram
panel and the startup dashboard both tested `symbol == "BTCUSD"` to decide
whether to show the research-parameters warning, which silently stopped matching
the moment the symbol changed. Both now key off the market's own `note`, and a
test asserts neither module compares a symbol literal again.

## B10. Verification runs

Both markets were replayed through the same engine over 19 251 evaluated M1
bars of their own synthetic history (`--no-evaluations`, default assumed
spreads). **These runs verify the plumbing. They are not evidence of an edge on
either market and must not be read as performance.**

| | XAUUSD | BTCUSD |
|---|---|---|
| Bars evaluated | 19 251 | 19 251 |
| Signals | 45 | 220 |
| Cost charged per trade | 2.4p (0.60R) | $14.0 (0.31R) |
| Raw win rate | 68.9% | 63.2% |
| Average RAW R | +0.151R | −0.033R |
| **Average NET R** | **−0.449R** | **−0.344R** |
| Net profit factor | 0.20 | 0.34 |
| TP1 reached | 48.9% | 63.2% |
| Closed at SL | 42.2% | 87.7% |
| Timed out | 55.6% | 0.5% |
| Average hold | 11.2 min | 3.0 min |
| Output written to | `data/xauusd/` | `data/btcusd/` |

What these runs *do* establish:

* the same engine runs both markets, with no per-market analysis code;
* BTCUSD builds dollar-scaled geometry from its own ATR (e.g. `SL 27.2$
  TP 21.0/46.7/79.3$`) rather than gold-sized distances;
* each market charges its own cost and writes to its own directory, with every
  row carrying its `symbol`, `timeframe` and `estimated_slippage`;
* neither market's files contain a row belonging to the other.

What they do **not** establish: anything about either market's profitability.
Both are negative after costs on this data, and the data is synthetic.

The one structural difference worth flagging for future work is the timeout/SL
split. Gold times out 55.6% of the time; Bitcoin almost never does (0.5%) and
reaches its stop 87.7% of the time. On this fixture BTC's per-minute movement
resolves a trade well inside the 15-candle window, which means **the shared
15-candle timeout is very likely the wrong number for Bitcoin** — it was set
equal to gold's deliberately, as an initial value, and this is exactly the kind
of parameter that must be re-derived from real data rather than inherited.

---

# Build Report — M1 Micro-Scalping Simplification (previous iteration)

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
