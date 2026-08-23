"""Central configuration for the XAUUSD signal engine.

Every tunable lives here.  Secrets are read from environment variables (see
``.env.example``) and are never hard-coded.

TIMEZONE POLICY
---------------
MetaTrader 5 returns candle timestamps in *broker server time*, which for most
gold brokers is UTC+2 (winter) / UTC+3 (summer).  Everything inside this project
works in **UTC**.  ``MT5_SERVER_UTC_OFFSET_HOURS`` tells the system how many
hours the broker clock is ahead of UTC so that candle times can be normalised.

The default is ``0`` (i.e. "treat broker time as UTC").  That default is
deliberately conservative: the session filter defaults to ``ALL_SESSIONS`` so an
unset offset cannot silently mute every signal.  If you enable a session filter
you MUST set the offset correctly - ``main.py`` prints the detected session for
the most recent candle on every evaluation so you can sanity-check it.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

try:  # optional dependency, only needed to read a .env file
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is listed in requirements.txt
    def load_dotenv(*_args, **_kwargs) -> bool:  # type: ignore[misc]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _env_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    return _env_str(key, "yes" if default else "no").lower() in ("1", "true", "yes", "y", "on")


# --------------------------------------------------------------------------- #
# sub-configuration blocks
# --------------------------------------------------------------------------- #
@dataclass
class Weights:
    """Component weights for **M1 scalping**.  They must sum to 100.

    RE-BALANCED FOR M1 - AND NOT CLAIMED TO BE OPTIMAL.
    ---------------------------------------------------
    The previous weights were built for M5 swing-style setups, where a slow
    trend and a higher-timeframe view carry real information over a multi-hour
    hold.  A scalp lives for a handful of minutes, so those components were cut
    back and the weight moved to what actually moves price over that horizon:
    momentum and its acceleration, the shape of the last few candles
    (displacement), and sweeps of the immediate highs/lows.

    These values were chosen by reasoning about the holding period, **not** by
    optimising against backtest results.  Re-derive them for your own data
    before trusting them - see the README "Calibration" section.

    ==================  ===  =========================================
    component           now  rationale for M1
    ==================  ===  =========================================
    momentum             22  short-term momentum + acceleration
    price_action         18  displacement, candle structure, rejection
    liquidity            14  sweeps of recent M1 highs/lows
    structure            12  micro break of structure / character change
    support_resistance   10  immediate levels the next 1-3 pips must clear
    trend                10  M1 EMA direction only (was 20)
    context               6  M5 bias, secondary at this horizon (was 15)
    volatility            5  is the expected move big relative to noise
    volume                3  tick-volume confirmation, weakest signal on M1
    ==================  ===  =========================================
    """

    momentum: float = 22.0
    price_action: float = 18.0
    liquidity: float = 14.0
    structure: float = 12.0
    support_resistance: float = 10.0
    trend: float = 10.0
    #: M5 context.  Field name kept as ``htf`` so the scoring key mapping and
    #: every stored CSV column stay stable.
    htf: float = 6.0
    volatility: float = 5.0
    volume: float = 3.0

    def total(self) -> float:
        return (
            self.trend
            + self.htf
            + self.momentum
            + self.structure
            + self.liquidity
            + self.support_resistance
            + self.volume
            + self.volatility
            + self.price_action
        )

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class IndicatorParams:
    """Indicator periods, shortened for M1.

    On M1 the previous M5 periods spanned hours of real time - a 200-period EMA
    is over three hours, which says nothing useful about the next three minutes.
    Every lookback here is expressed in *minutes* because that is what one M1
    candle is worth.
    """

    ema_fast: int = 5        # 5 minutes
    ema_mid: int = 13        # 13 minutes
    ema_slow: int = 34       # ~34 minutes
    ema_trend: int = 100     # ~1.7 hours - the slowest thing a scalp respects
    adx_period: int = 10
    rsi_period: int = 9
    macd_fast: int = 6
    macd_slow: int = 13
    macd_signal: int = 5
    stoch_k: int = 9
    stoch_d: int = 3
    stoch_smooth: int = 3
    roc_period: int = 5
    atr_period: int = 14     # 14 minutes of true range
    bb_period: int = 20
    bb_stddev: float = 2.0
    volume_ma_period: int = 20
    atr_history_period: int = 120   # two hours, for the volatility band
    # Micro pivots: on M1 a 2-bar fractal is already slow, so structure uses a
    # 1-bar confirmation and only levels use a wider one.
    swing_left: int = 1
    swing_right: int = 1
    level_swing_left: int = 3
    level_swing_right: int = 3
    level_lookback_bars: int = 120


@dataclass
class SessionWindows:
    """Session windows expressed in **UTC** as (start_hour, end_hour).

    ``end`` is exclusive.  Windows may overlap; :func:`src.utils.detect_session`
    resolves a single label using ``session_priority``.
    """

    asian: Tuple[int, int] = (0, 8)
    london: Tuple[int, int] = (7, 16)
    new_york: Tuple[int, int] = (12, 21)
    overlap: Tuple[int, int] = (12, 16)


@dataclass
class Config:
    """Full system configuration."""

    # ---- credentials / connectivity (from environment only) ---------------- #
    mt5_login: int = field(default_factory=lambda: _env_int("MT5_LOGIN", 0))
    mt5_password: str = field(default_factory=lambda: _env_str("MT5_PASSWORD"))
    mt5_server: str = field(default_factory=lambda: _env_str("MT5_SERVER"))
    mt5_terminal_path: str = field(default_factory=lambda: _env_str("MT5_TERMINAL_PATH"))
    telegram_bot_token: str = field(default_factory=lambda: _env_str("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: str = field(default_factory=lambda: _env_str("TELEGRAM_CHAT_ID"))
    telegram_enabled: bool = field(default_factory=lambda: _env_bool("TELEGRAM_ENABLED", True))

    # ---- market ------------------------------------------------------------ #
    #
    # The symbol and every market-specific parameter live in src/markets.py.
    # The fields below are the DEFAULTS that :meth:`for_market` overwrites, and
    # they are kept so that a bare ``Config()`` remains usable (tests, tooling)
    # without having to select a market first.
    #
    symbol: str = field(default_factory=lambda: _env_str("SYMBOL", "XAUUSDs"))
    market_key: str = "xauusds"

    #: Account-currency value of a one-point move on one lot.  Used ONLY by the
    #: demo execution layer to turn price moves into money; the signal engine
    #: never sees it.  Market-specific, overwritten by :meth:`for_market`.
    money_per_point_per_lot: float = 1.0

    #: The ONLY signal timeframe.  This is a dedicated M1 scalping system; the
    #: multi-timeframe selector was removed deliberately.
    signal_timeframe: str = "M1"

    #: Single short-term context timeframe.  M15/H1 were dropped: a one-hour
    #: trend has little bearing on a position held for a few minutes.  Set to
    #: "" to run on pure M1 microstructure - the context component is then
    #: marked not-applicable and its weight is shared out (see src/scoring.py).
    context_timeframe: str = field(default_factory=lambda: _env_str("CONTEXT_TIMEFRAME", "M5"))

    candles_signal: int = 900        # 15 hours of M1
    candles_context: int = 400       # ~33 hours of M5

    digits: int = 2
    point_value: float = 0.01     # one "point" in price terms (the last digit)

    #: One "pip" in price terms.  Market-specific: 0.10 for gold, 1.00 (= $1)
    #: for Bitcoin.  See src/markets.py.
    pip_value: float = field(default_factory=lambda: _env_float("PIP_VALUE", 0.10))
    pip_name: str = "p"

    #: True for a market that never closes (Bitcoin).  The session *label* is
    #: still recorded for analysis; only the session *filter* is bypassed.
    is_24h: bool = False

    mt5_server_utc_offset_hours: float = field(
        default_factory=lambda: _env_float("MT5_SERVER_UTC_OFFSET_HOURS", 0.0)
    )

    # ---- loop timing -------------------------------------------------------- #
    #: M1 closes every 60s, so poll fast enough to catch one promptly.
    poll_seconds: int = field(default_factory=lambda: _env_int("POLL_SECONDS", 5))
    #: An M1 candle more than three minutes old means the feed has stalled.
    max_candle_staleness_seconds: int = field(
        default_factory=lambda: _env_int("MAX_CANDLE_STALENESS_SECONDS", 180)
    )

    # ---- data validation ---------------------------------------------------- #
    min_candles_required: int = 260      #: M1 candles before anything is scored
    min_context_candles: int = 150       #: context candles before it is trusted

    # ---- scoring ------------------------------------------------------------ #
    weights: Weights = field(default_factory=Weights)
    indicators: IndicatorParams = field(default_factory=IndicatorParams)

    # ---- threshold ---------------------------------------------------------- #
    #
    # CALIBRATION NOTE - please read before changing this.
    #
    # The score is an additive weighted sum of nine components, several of them
    # event-driven (a liquidity sweep either happened or it did not).  Those
    # events rarely coincide, so the raw distribution does not run to 100.
    #
    # SCALP_THRESHOLD is a STARTING POINT, not an optimum, and it is deliberately
    # not tuned against the data used to measure performance.  It is set high
    # enough that the engine produces candidates without emitting one on every
    # other candle - on M1 there are 1,440 candles a day, so a threshold that is
    # a few points too low buries the useful setups in noise.
    #
    # Re-derive it from your own broker's data: run backtest.py with a low
    # threshold, then read the percentiles of the score columns in
    # backtest_evaluations.csv (README, "Calibration").
    #
    # The effective threshold for one evaluation is:
    #
    #     scalp_threshold + regime_offset + counter_trend_extra
    #
    # clamped to [min_threshold, max_threshold].
    #
    mode: str = "SCALPING"       #: the only strategy mode
    base_threshold: float = field(default_factory=lambda: _env_float("SCALP_THRESHOLD", 68.0))

    #: Regime adjustment applied on top of the threshold, in points.  Ranges and
    #: violent conditions demand more confirmation than a clean micro-trend.
    regime_threshold_offsets: Dict[str, float] = field(
        default_factory=lambda: {
            "STRONG_BULL_TREND": -4.0,
            "STRONG_BEAR_TREND": -4.0,
            "WEAK_TREND": 0.0,
            "BREAKOUT": 0.0,
            "RANGE": 5.0,
            "HIGH_VOLATILITY": 5.0,
            "LOW_VOLATILITY": 3.0,
        }
    )

    #: Hard limits on any threshold, including ones set from Telegram.
    min_threshold: float = field(default_factory=lambda: _env_float("MIN_THRESHOLD", 40.0))
    max_threshold: float = field(default_factory=lambda: _env_float("MAX_THRESHOLD", 95.0))

    #: Increments offered by the Telegram threshold controls.
    threshold_steps: Tuple[int, ...] = (-5, -1, 1, 5)

    confidence_bands: Tuple[Tuple[float, str], ...] = (
        (80.0, "VERY_STRONG"),
        (70.0, "STRONG"),
        (62.0, "MODERATE"),
    )

    # ---- near-signal diagnostic --------------------------------------------- #
    #: A candidate whose best score lands within this many points *below* the
    #: active threshold is logged as NEAR_SIGNAL.  It is never sent as a trading
    #: signal - the point is to reveal whether the threshold is slightly strict.
    near_signal_margin: float = field(default_factory=lambda: _env_float("NEAR_SIGNAL_MARGIN", 8.0))
    near_signal_alerts: bool = field(default_factory=lambda: _env_bool("NEAR_SIGNAL_ALERTS", False))

    # counter-trend penalty: extra score required when trading against the
    # short-term context timeframe
    counter_trend_extra_score: float = field(
        default_factory=lambda: _env_float("COUNTER_TREND_EXTRA_SCORE", 5.0)
    )

    # ---- filters ------------------------------------------------------------ #
    # Bull/bear conflict guard, scaled to the same calibrated distribution as
    # the thresholds above (nominally 10 points on a full 0-100 spread).
    min_score_separation: float = field(default_factory=lambda: _env_float("MIN_SCORE_SEPARATION", 9.0))
    max_spread_points: float = field(default_factory=lambda: _env_float("MAX_SPREAD_POINTS", 35.0))

    #: Spread must stay small relative to the move the market can realistically
    #: make within the holding window.  A single M1 candle's ATR is barely
    #: larger than the spread, so the old "spread vs one-candle ATR" test was
    #: meaningless here; the expected move is scaled by sqrt(holding candles),
    #: the usual random-walk approximation.
    max_spread_to_expected_move: float = field(
        default_factory=lambda: _env_float("MAX_SPREAD_TO_EXPECTED_MOVE", 0.35)
    )
    #: Cooldown in M1 candles, i.e. minutes.
    cooldown_candles: int = field(default_factory=lambda: _env_int("COOLDOWN_CANDLES", 10))
    same_direction_cooldown_candles: int = field(
        default_factory=lambda: _env_int("SAME_DIRECTION_COOLDOWN_CANDLES", 20)
    )
    #: M1 offers 1,440 candles a day, so the M5-era cap of 8 was binding long
    #: before the engine's own filters were.  This exists to keep Telegram
    #: usable, not as a quality control - the threshold does that.
    max_signals_per_day: int = field(default_factory=lambda: _env_int("MAX_SIGNALS_PER_DAY", 30))
    allowed_sessions: Tuple[str, ...] = field(
        default_factory=lambda: tuple(
            s.strip().upper()
            for s in _env_str("ALLOWED_SESSIONS", "ALL_SESSIONS").split(",")
            if s.strip()
        )
    )
    sessions: SessionWindows = field(default_factory=SessionWindows)
    session_priority: Tuple[str, ...] = ("LONDON_NEW_YORK_OVERLAP", "LONDON", "NEW_YORK", "ASIAN")

    block_on_extreme_volatility: bool = field(
        default_factory=lambda: _env_bool("BLOCK_ON_EXTREME_VOLATILITY", True)
    )
    extreme_volatility_extra_score: float = 10.0
    enable_fakeout_filter: bool = True
    max_concurrent_active_signals: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT_ACTIVE_SIGNALS", 3)
    )

    # ---- volatility classification ------------------------------------------ #
    vol_low_ratio: float = 0.70       # atr / atr_history below this  -> LOW
    vol_high_ratio: float = 1.50      # above this -> HIGH
    vol_extreme_ratio: float = 2.20   # above this -> EXTREME

    # ---- regime ------------------------------------------------------------- #
    adx_trend_threshold: float = 25.0
    adx_strong_threshold: float = 35.0
    adx_range_threshold: float = 18.0

    # ---- micro-scalping targets --------------------------------------------- #
    #
    # Targets are sized from CURRENT MARKET CONDITIONS (M1 ATR and the nearest
    # structure), not from fixed pip distances.  The multipliers below are
    # applied to the M1 ATR, so a quiet session produces tighter targets than a
    # news spike, which is the point.
    #
    # With a typical XAUUSD M1 ATR of 0.25-0.60, the defaults give roughly:
    #     TP1  0.11-0.27  (~1.1-2.7 pips)   <- the "1-3 pip" research concept
    #     TP2  0.25-0.60  (~2.5-6.0 pips)
    #     TP3  0.43-1.02  (~4.3-10.2 pips)
    # NONE OF THIS IS CLAIMED TO BE PROFITABLE.  Whether a target of that size
    # survives the spread is exactly what the cost model below measures.
    #
    sl_mode: str = field(default_factory=lambda: _env_str("SL_MODE", "HYBRID").upper())
    #: The stop must be TIGHTER than TP2, otherwise the reward/risk of the
    #: ladder is structurally <= 1 no matter how good the setup is.  This is a
    #: geometry requirement, not a performance tuning - validate() enforces it.
    sl_atr_multiplier: float = field(default_factory=lambda: _env_float("SL_ATR_MULTIPLIER", 0.70))
    sl_structure_buffer_atr: float = field(
        default_factory=lambda: _env_float("SL_STRUCTURE_BUFFER_ATR", 0.15)
    )
    sl_min_atr_multiplier: float = 0.50
    #: Ceiling on the stop, in ATR.  This is what keeps HYBRID honest on M1: the
    #: structure branch can otherwise put the stop 2-3 ATR away while the targets
    #: stay put, leaving the ladder's reward/risk below 1 by construction.  On a
    #: 15-minute horizon a stop that wide is not a scalp anyway.
    #: With TP2 at 1.00 ATR this guarantees TP2 >= 1.25R (validate() enforces it).
    sl_max_atr_multiplier: float = 0.80
    sl_structure_lookback: int = 12       # 12 minutes of micro structure

    #: Take-profit distances as multiples of the M1 ATR.  With the stop at
    #: 0.70 ATR these give a raw ladder of roughly 0.64R / 1.43R / 2.43R before
    #: the cost floor and any structure truncation.
    tp_atr_multiples: Tuple[float, float, float] = (0.45, 1.00, 1.70)

    #: Absolute floor on each target, in the market's pip unit, so a dead-quiet
    #: minute cannot produce a target smaller than the tick grid.
    min_tp_pips: Tuple[float, float, float] = (1.0, 1.8, 3.0)

    #: Floor on each target as a FRACTION OF PRICE.  The larger of this and the
    #: pip floor applies.  Gold leaves this at zero (its pip floor is meaningful
    #: at any gold price); Bitcoin uses it instead, because a fixed dollar floor
    #: is far too tight at $90,000 and far too loose at $20,000.
    min_tp_pct: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    #: Hard ceiling on TP3.  A "scalp" that needs a huge move is not a scalp;
    #: the setup is rejected rather than silently re-scoped.  Expressed in pips
    #: and/or as a fraction of price - whichever is non-zero, larger wins.
    max_tp3_pips: float = field(default_factory=lambda: _env_float("MAX_TP3_PIPS", 12.0))
    max_tp3_pct: float = 0.0

    # Zone significance.  A "zone" accumulates weight from every level that
    # merges into it; raising these keeps noise pivots out of the S/R engine and
    # stops them truncating take-profits.
    min_zone_weight: float = 2.0
    min_tp_block_zone_weight: float = 3.0
    tp_sr_buffer_atr: float = 0.15

    #: Raw reward gate: TP2 must be worth this multiple of the risk.
    min_tp2_rr: float = field(default_factory=lambda: _env_float("MIN_TP2_RR", 1.2))
    #: Net reward gate: TP2 must STILL be worth something once the round-trip
    #: cost is paid.  Without this a setup can clear the raw gate and still be a
    #: guaranteed loser - the whole reason the cost model exists.
    min_net_tp2_rr: float = field(default_factory=lambda: _env_float("MIN_NET_TP2_RR", 0.30))

    # ---- cost model ----------------------------------------------------------- #
    #
    # CRITICAL FOR SCALPING.  At a 1-3 pip target the spread is not a rounding
    # error - it is often the whole trade.  Every R figure is reported twice:
    # RAW R (price move only) and NET R (after the costs below).
    #
    # All values are in POINTS (0.01 of price) unless stated otherwise.
    #
    #: Used when the live spread is unavailable (backtests, or a broker that
    #: does not report it).  20 points = 0.20 = 2 pips, a common gold spread.
    assumed_spread_points: float = field(
        default_factory=lambda: _env_float("ASSUMED_SPREAD_POINTS", 20.0)
    )
    #: Slippage assumed on entry and on exit, each.  Market orders on M1 gold
    #: rarely fill exactly at the quoted price.
    slippage_points_entry: float = field(
        default_factory=lambda: _env_float("SLIPPAGE_POINTS_ENTRY", 2.0)
    )
    slippage_points_exit: float = field(
        default_factory=lambda: _env_float("SLIPPAGE_POINTS_EXIT", 2.0)
    )
    #: Commission per side, in points of price equivalent.  Zero for most
    #: spread-only retail accounts.
    commission_points_per_side: float = field(
        default_factory=lambda: _env_float("COMMISSION_POINTS_PER_SIDE", 0.0)
    )

    #: TP1 must be at least this multiple of the full round-trip cost, otherwise
    #: the setup is rejected as "target too small to clear costs".  This is the
    #: single most important filter in a scalping system.
    min_tp1_cost_multiple: float = field(
        default_factory=lambda: _env_float("MIN_TP1_COST_MULTIPLE", 1.5)
    )

    #: Human-readable name of the active market's cost assumptions, shown in
    #: Telegram settings so it is obvious which model produced a NET R figure.
    cost_model_name: str = "XAUUSD_RETAIL"

    #: The stop must also clear costs, otherwise a normal spread excursion stops
    #: the trade out on noise alone.
    min_sl_cost_multiple: float = field(
        default_factory=lambda: _env_float("MIN_SL_COST_MULTIPLE", 2.0)
    )

    # ---- outcome tracking ---------------------------------------------------- #
    #: Maximum holding period in M1 candles.  A scalp that has not resolved in
    #: this many minutes is closed at market and recorded as TIMEOUT - it must
    #: never sit "active" indefinitely.
    max_holding_candles: int = field(default_factory=lambda: _env_int("MAX_HOLDING_CANDLES", 15))
    move_sl_to_breakeven_after_tp1: bool = field(
        default_factory=lambda: _env_bool("MOVE_SL_TO_BREAKEVEN_AFTER_TP1", True)
    )
    partial_fractions: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    invalidate_on_opposite_signal: bool = True

    #: Resolve a candle that touches both target and stop by replaying raw
    #: ticks.  Without it the pessimistic assumption (stop first) always
    #: applies - which at M1 scalping scale is a large systematic penalty.
    use_ticks_for_ambiguous_candles: bool = field(
        default_factory=lambda: _env_bool("USE_TICKS_FOR_AMBIGUOUS_CANDLES", True)
    )

    # ---- Telegram control panel --------------------------------------------- #
    telegram_control_enabled: bool = field(
        default_factory=lambda: _env_bool("TELEGRAM_CONTROL_ENABLED", True)
    )
    #: seconds to hold a long-poll open against getUpdates
    telegram_poll_timeout: int = field(default_factory=lambda: _env_int("TELEGRAM_POLL_TIMEOUT", 25))

    #: Only these settings may be changed from Telegram.  Anything not listed -
    #: credentials, tokens, the symbol, file paths, indicator internals - is
    #: deliberately unreachable from chat (spec section 16).
    telegram_editable_settings: Tuple[str, ...] = (
        "threshold",
        "cooldown_candles",
        "min_tp2_rr",
        "allowed_sessions",
        "near_signal_alerts",
        "max_holding_candles",
        "status",
    )

    #: Values offered by the Settings menu for the two cycling options.
    cooldown_choices: Tuple[int, ...] = (0, 2, 3, 5, 10, 15)
    min_rr_choices: Tuple[float, ...] = (0.0, 0.8, 1.0, 1.2, 1.5, 2.0)
    holding_choices: Tuple[int, ...] = (5, 10, 15, 20, 30, 45)
    session_choices: Tuple[str, ...] = (
        "ALL_SESSIONS", "LONDON", "NEW_YORK", "LONDON_NEW_YORK_OVERLAP", "ASIAN",
    )

    # ---- storage ------------------------------------------------------------- #
    #
    # Each market owns a directory so nothing can ever be mixed:
    #
    #     data/state.json              global runtime (active market, run state)
    #     data/xauusds/{evaluations,signals,outcomes,executions}.csv, state.json
    #     data/btcusds/{evaluations,signals,outcomes,executions}.csv, state.json
    #
    # The paths below point at the ACTIVE market and are rewritten by
    # :meth:`for_market`.
    #
    data_dir: Path = DATA_DIR
    market_dir: Path = DATA_DIR / "xauusds"
    signals_csv: Path = DATA_DIR / "xauusds" / "signals.csv"
    evaluations_csv: Path = DATA_DIR / "xauusds" / "evaluations.csv"
    outcomes_csv: Path = DATA_DIR / "xauusds" / "outcomes.csv"
    #: Demo EXECUTION records.  Deliberately a separate file from outcomes.csv:
    #: paper results and demo-fill results must be comparable, which means
    #: neither may overwrite the other (spec section 14).
    executions_csv: Path = DATA_DIR / "xauusds" / "executions.csv"
    #: Per-market runtime state (threshold, cooldown, last processed candle).
    state_file: Path = DATA_DIR / "xauusds" / "state.json"
    #: Global runtime state (active market, run state, alert preferences).
    global_state_file: Path = DATA_DIR / "state.json"
    log_file: Path = DATA_DIR / "system_log.txt"
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO").upper())

    #: Evaluate every configured market on each cycle instead of only the one
    #: selected in Telegram.  Off by default - the panel shows one market and
    #: signalling on an unselected market would be surprising - but useful when
    #: collecting research data for both at once.
    evaluate_all_markets: bool = field(
        default_factory=lambda: _env_bool("EVALUATE_ALL_MARKETS", False)
    )

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration is internally inconsistent."""
        total = self.weights.total()
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"Component weights must sum to 100, got {total}")
        if self.sl_mode not in ("ATR", "STRUCTURE", "HYBRID"):
            raise ValueError(f"Unknown SL_MODE '{self.sl_mode}' (use ATR, STRUCTURE or HYBRID)")
        if not (self.tp_atr_multiples[0] < self.tp_atr_multiples[1] < self.tp_atr_multiples[2]):
            raise ValueError("tp_atr_multiples must be strictly increasing")
        if self.tp_atr_multiples[1] <= self.sl_atr_multiplier:
            raise ValueError(
                "TP2 must be further than the stop "
                f"(TP2 {self.tp_atr_multiples[1]} ATR vs SL {self.sl_atr_multiplier} ATR); "
                "otherwise the ladder's reward/risk is structurally <= 1"
            )
        achievable_rr2 = self.tp_atr_multiples[1] / self.sl_max_atr_multiplier
        if achievable_rr2 < self.min_tp2_rr:
            raise ValueError(
                f"the widest allowed stop ({self.sl_max_atr_multiplier} ATR) makes TP2 worth at "
                f"most {achievable_rr2:.2f}R, below MIN_TP2_RR {self.min_tp2_rr:.2f} - "
                "the R:R gate would reject every setup"
            )
        if not (self.min_tp_pips[0] <= self.min_tp_pips[1] <= self.min_tp_pips[2]):
            raise ValueError("min_tp_pips must be non-decreasing")
        if not (self.min_tp_pct[0] <= self.min_tp_pct[1] <= self.min_tp_pct[2]):
            raise ValueError("min_tp_pct must be non-decreasing")
        if max(self.min_tp_pips) <= 0 and max(self.min_tp_pct) <= 0:
            raise ValueError("a market needs either pip floors or percentage floors")
        if self.max_tp3_pips <= 0 and self.max_tp3_pct <= 0:
            raise ValueError("a market needs a TP3 ceiling in pips or percent")
        if self.sl_min_atr_multiplier >= self.sl_max_atr_multiplier:
            raise ValueError("sl_min_atr_multiplier must be < sl_max_atr_multiplier")
        if self.base_threshold <= 0 or self.base_threshold > 100:
            raise ValueError("threshold must be within (0, 100]")
        if not 0 < self.min_threshold < self.max_threshold <= 100:
            raise ValueError("require 0 < min_threshold < max_threshold <= 100")
        if self.signal_timeframe != "M1":
            raise ValueError("this build is M1-only; signal_timeframe must be 'M1'")
        if self.pip_value <= 0 or self.point_value <= 0:
            raise ValueError("pip_value and point_value must be positive")
        if self.max_holding_candles < 1:
            raise ValueError("max_holding_candles must be at least 1")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._adopt_legacy_market_dir()
        self.market_dir.mkdir(parents=True, exist_ok=True)

    def _adopt_legacy_market_dir(self) -> None:
        """Carry a pre-rename data directory over to the current name.

        The market key follows the symbol, so renaming ``XAUUSD`` to the
        broker's ``XAUUSDs`` moves the directory from ``data/xauusd`` to
        ``data/xauusds``.  Without this the old history would still be on disk
        but invisible, which looks exactly like data loss.

        Only ever renames INTO a name that does not exist yet, so it cannot
        overwrite anything, and it is a no-op on every run after the first.
        """
        from src.markets import SYMBOL_ALIASES

        if self.market_dir.exists():
            return
        candidates = [
            alias.lower() for alias, canonical in SYMBOL_ALIASES.items()
            if canonical.casefold() == str(self.symbol).casefold()
        ]
        for legacy_key in candidates:
            legacy = self.data_dir / legacy_key
            if legacy_key != self.market_key and legacy.is_dir():
                try:
                    legacy.rename(self.market_dir)
                except OSError:
                    # A failed rename must not stop start-up: the engine simply
                    # begins with an empty directory, and the old one is intact.
                    return
                logging.getLogger("scalper.config").info(
                    "Adopted legacy data directory %s -> %s", legacy, self.market_dir
                )
                return

    # ------------------------------------------------------------------ #
    # market selection
    # ------------------------------------------------------------------ #
    def for_market(self, market) -> "Config":
        """Return a copy of this config set up for one market.

        This is the only seam between the engine and the instrument: every
        analysis engine, filter and target calculation keeps reading plain
        config attributes, but the values now describe whichever market is being
        evaluated.  Adding a third instrument therefore needs a
        :class:`~src.markets.MarketConfig` and nothing else.

        ``market`` may be a :class:`~src.markets.MarketConfig` or a symbol.

        The copy is shallow - ``weights``, ``indicators`` and ``sessions`` are
        shared and never mutated here.
        """
        from src.markets import MarketConfig, get_market

        if not isinstance(market, MarketConfig):
            market = get_market(market)

        view = copy.copy(self)
        view.symbol = market.symbol
        view.market_key = market.key
        view.digits = market.digits
        view.point_value = market.point_value
        view.pip_value = market.pip_value
        view.pip_name = market.pip_name
        view.money_per_point_per_lot = market.money_per_point_per_lot
        view.is_24h = market.is_24h
        view.context_timeframe = market.context_timeframe
        view.candles_signal = market.candles_signal
        view.candles_context = market.candles_context

        view.base_threshold = market.threshold
        view.tp_atr_multiples = market.tp_atr_multiples
        view.min_tp_pips = market.min_tp_pips
        view.min_tp_pct = market.min_tp_pct
        view.max_tp3_pips = market.max_tp3_pips
        view.max_tp3_pct = market.max_tp3_pct
        view.sl_atr_multiplier = market.sl_atr_multiplier
        view.sl_min_atr_multiplier = market.sl_min_atr_multiplier
        view.sl_max_atr_multiplier = market.sl_max_atr_multiplier
        view.sl_structure_buffer_atr = market.sl_structure_buffer_atr
        view.sl_structure_lookback = market.sl_structure_lookback
        view.min_tp2_rr = market.min_tp2_rr
        view.min_net_tp2_rr = market.min_net_tp2_rr

        view.assumed_spread_points = market.assumed_spread_points
        view.slippage_points_entry = market.slippage_points_entry
        view.slippage_points_exit = market.slippage_points_exit
        view.commission_points_per_side = market.commission_points_per_side
        view.min_tp1_cost_multiple = market.min_tp1_cost_multiple
        view.min_sl_cost_multiple = market.min_sl_cost_multiple
        view.max_spread_points = market.max_spread_points
        view.max_spread_to_expected_move = market.max_spread_to_expected_move
        view.cost_model_name = market.cost_model_name

        view.cooldown_candles = market.cooldown_candles
        view.same_direction_cooldown_candles = market.same_direction_cooldown_candles
        view.max_signals_per_day = market.max_signals_per_day
        view.max_concurrent_active_signals = market.max_concurrent_active_signals
        view.max_holding_candles = market.max_holding_candles
        view.allowed_sessions = market.default_sessions

        view.market_dir = self.data_dir / market.key
        view.signals_csv = view.market_dir / "signals.csv"
        view.evaluations_csv = view.market_dir / "evaluations.csv"
        view.outcomes_csv = view.market_dir / "outcomes.csv"
        view.executions_csv = view.market_dir / "executions.csv"
        view.state_file = view.market_dir / "state.json"
        return view

    # ------------------------------------------------------------------ #
    def clamp_threshold(self, value: float) -> float:
        """Constrain a threshold to the configured safe band."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.base_threshold
        return max(self.min_threshold, min(self.max_threshold, value))

    # -- cost model ----------------------------------------------------------- #
    def round_trip_cost(self, spread_points: Optional[float] = None) -> float:
        """Full round-trip trading cost in **price** terms.

        ``spread + entry slippage + exit slippage + commission on both sides``.
        A missing or non-finite spread falls back to ``assumed_spread_points``,
        so a backtest still pays a realistic cost instead of trading for free.
        """
        spread = self.effective_spread_points(spread_points)
        points = (
            spread
            + self.slippage_points_entry
            + self.slippage_points_exit
            + 2.0 * self.commission_points_per_side
        )
        return points * self.point_value

    def effective_spread_points(self, spread_points: Optional[float] = None) -> float:
        """The spread to charge: the live one when known, else the assumption."""
        try:
            value = float(spread_points)
        except (TypeError, ValueError):
            return self.assumed_spread_points
        if not math.isfinite(value) or value < 0:
            return self.assumed_spread_points
        return value

    def pips(self, price_distance: float) -> float:
        """Convert a price distance into pips."""
        return float(price_distance) / self.pip_value


def load_config() -> Config:
    """Build and validate the configuration."""
    cfg = Config()
    cfg.validate()
    cfg.ensure_dirs()
    return cfg
