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

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple

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
    """Component weights.  They must sum to 100."""

    trend: float = 20.0
    htf: float = 15.0
    momentum: float = 15.0
    structure: float = 15.0
    liquidity: float = 10.0
    support_resistance: float = 10.0
    volume: float = 5.0
    volatility: float = 5.0
    price_action: float = 5.0

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
    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    ema_trend: int = 200
    adx_period: int = 14
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3
    roc_period: int = 10
    atr_period: int = 14
    bb_period: int = 20
    bb_stddev: float = 2.0
    volume_ma_period: int = 20
    atr_history_period: int = 100
    # Structure pivots stay responsive (BOS/CHoCH must react quickly)...
    swing_left: int = 2
    swing_right: int = 2
    # ...while support/resistance and liquidity levels use a much wider fractal,
    # so only genuinely significant turning points become reference levels.
    level_swing_left: int = 5
    level_swing_right: int = 5
    level_lookback_bars: int = 150


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
    symbol: str = field(default_factory=lambda: _env_str("SYMBOL", "XAUUSD"))
    signal_timeframe: str = field(default_factory=lambda: _env_str("SIGNAL_TIMEFRAME", "M5"))
    intermediate_timeframe: str = field(default_factory=lambda: _env_str("INTERMEDIATE_TIMEFRAME", "M15"))
    higher_timeframe: str = field(default_factory=lambda: _env_str("HIGHER_TIMEFRAME", "H1"))
    micro_timeframe: str = field(default_factory=lambda: _env_str("MICRO_TIMEFRAME", "M1"))

    candles_signal: int = 800
    candles_intermediate: int = 500
    candles_higher: int = 500
    candles_micro: int = 400

    digits: int = 2               # XAUUSD is quoted with 2 decimals on most brokers
    point_value: float = 0.01     # one "point" in price terms

    mt5_server_utc_offset_hours: float = field(
        default_factory=lambda: _env_float("MT5_SERVER_UTC_OFFSET_HOURS", 0.0)
    )

    # ---- loop timing -------------------------------------------------------- #
    poll_seconds: int = field(default_factory=lambda: _env_int("POLL_SECONDS", 15))
    max_candle_staleness_seconds: int = field(
        default_factory=lambda: _env_int("MAX_CANDLE_STALENESS_SECONDS", 900)
    )

    # ---- data validation ---------------------------------------------------- #
    min_candles_required: int = 260
    min_htf_candles_required: int = 220

    # ---- scoring ------------------------------------------------------------ #
    weights: Weights = field(default_factory=Weights)
    indicators: IndicatorParams = field(default_factory=IndicatorParams)

    # ---- thresholds --------------------------------------------------------- #
    #
    # CALIBRATION NOTE - please read before changing these.
    #
    # The score is an additive weighted sum of nine components.  Several of them
    # are *event driven*: the liquidity engine only scores when a sweep actually
    # happens, market structure only scores a CHoCH when character actually
    # changes, and so on.  Those events rarely all coincide, so in practice the
    # raw score distribution on XAUUSD M5 peaks in the low 80s rather than
    # running to 100 - the median bar scores around 47 and the 99th percentile
    # around 76.
    #
    # The thresholds below therefore keep the *structure* of the nominal
    # 90/82/75 confidence bands (very strong / strong / moderate, with a
    # regime-adaptive requirement) but are set to the values the engine's actual
    # distribution supports.  They were derived from the score distribution
    # alone, never from backtest profitability.
    #
    # To re-derive them for your own broker's data, run backtest.py with a low
    # BASE_THRESHOLD, then look at the percentiles of `bullish_score` /
    # `bearish_score` in backtest_evaluations.csv (see README, "Calibration").
    #
    base_threshold: float = field(default_factory=lambda: _env_float("BASE_THRESHOLD", 72.0))
    regime_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "STRONG_BULL_TREND": 68.0,
            "STRONG_BEAR_TREND": 68.0,
            "WEAK_TREND": 72.0,
            "BREAKOUT": 72.0,
            "RANGE": 77.0,
            "HIGH_VOLATILITY": 77.0,
            "LOW_VOLATILITY": 75.0,
        }
    )
    confidence_bands: Tuple[Tuple[float, str], ...] = (
        (80.0, "VERY_STRONG"),
        (72.0, "STRONG"),
        (66.0, "MODERATE"),
    )

    # counter-trend penalty: extra score required when trading against the HTF
    counter_trend_extra_score: float = field(
        default_factory=lambda: _env_float("COUNTER_TREND_EXTRA_SCORE", 5.0)
    )

    # ---- filters ------------------------------------------------------------ #
    # Bull/bear conflict guard, scaled to the same calibrated distribution as
    # the thresholds above (nominally 10 points on a full 0-100 spread).
    min_score_separation: float = field(default_factory=lambda: _env_float("MIN_SCORE_SEPARATION", 9.0))
    max_spread_points: float = field(default_factory=lambda: _env_float("MAX_SPREAD_POINTS", 40.0))
    max_spread_atr_ratio: float = field(default_factory=lambda: _env_float("MAX_SPREAD_ATR_RATIO", 0.15))
    cooldown_candles: int = field(default_factory=lambda: _env_int("COOLDOWN_CANDLES", 10))
    same_direction_cooldown_candles: int = field(
        default_factory=lambda: _env_int("SAME_DIRECTION_COOLDOWN_CANDLES", 20)
    )
    max_signals_per_day: int = field(default_factory=lambda: _env_int("MAX_SIGNALS_PER_DAY", 8))
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
        default_factory=lambda: _env_int("MAX_CONCURRENT_ACTIVE_SIGNALS", 2)
    )

    # ---- volatility classification ------------------------------------------ #
    vol_low_ratio: float = 0.70       # atr / atr_history below this  -> LOW
    vol_high_ratio: float = 1.50      # above this -> HIGH
    vol_extreme_ratio: float = 2.20   # above this -> EXTREME

    # ---- regime ------------------------------------------------------------- #
    adx_trend_threshold: float = 25.0
    adx_strong_threshold: float = 35.0
    adx_range_threshold: float = 18.0

    # ---- targets ------------------------------------------------------------ #
    sl_mode: str = field(default_factory=lambda: _env_str("SL_MODE", "HYBRID").upper())
    sl_atr_multiplier: float = field(default_factory=lambda: _env_float("SL_ATR_MULTIPLIER", 1.5))
    sl_structure_buffer_atr: float = field(
        default_factory=lambda: _env_float("SL_STRUCTURE_BUFFER_ATR", 0.25)
    )
    sl_min_atr_multiplier: float = 0.80
    sl_max_atr_multiplier: float = 3.00
    sl_structure_lookback: int = 20

    # Zone significance.  A "zone" accumulates weight from every level that
    # merges into it (previous day high/low = 3.0, previous session = 2.0, a
    # swing cluster = 1.0 + 0.5 per touch).  Raising these keeps noise pivots
    # out of the S/R engine and stops them from truncating take-profits.
    min_zone_weight: float = 2.0
    min_tp_block_zone_weight: float = 3.0

    tp_r_multiples: Tuple[float, float, float] = (1.0, 1.8, 2.8)
    tp_sr_buffer_atr: float = 0.20
    tp_min_r_after_adjustment: Tuple[float, float, float] = (0.6, 1.2, 1.8)
    min_tp2_rr: float = field(default_factory=lambda: _env_float("MIN_TP2_RR", 1.5))

    # ---- outcome tracking ---------------------------------------------------- #
    signal_expiry_candles: int = field(default_factory=lambda: _env_int("SIGNAL_EXPIRY_CANDLES", 96))
    move_sl_to_breakeven_after_tp1: bool = field(
        default_factory=lambda: _env_bool("MOVE_SL_TO_BREAKEVEN_AFTER_TP1", True)
    )
    partial_fractions: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    invalidate_on_opposite_signal: bool = True

    # ---- storage ------------------------------------------------------------- #
    data_dir: Path = DATA_DIR
    signals_csv: Path = DATA_DIR / "signals.csv"
    evaluations_csv: Path = DATA_DIR / "evaluations.csv"
    outcomes_csv: Path = DATA_DIR / "outcomes.csv"
    log_file: Path = DATA_DIR / "system_log.txt"
    state_file: Path = DATA_DIR / "state.json"
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO").upper())

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration is internally inconsistent."""
        total = self.weights.total()
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"Component weights must sum to 100, got {total}")
        if self.sl_mode not in ("ATR", "STRUCTURE", "HYBRID"):
            raise ValueError(f"Unknown SL_MODE '{self.sl_mode}' (use ATR, STRUCTURE or HYBRID)")
        if not (self.tp_r_multiples[0] < self.tp_r_multiples[1] < self.tp_r_multiples[2]):
            raise ValueError("tp_r_multiples must be strictly increasing")
        if self.sl_min_atr_multiplier >= self.sl_max_atr_multiplier:
            raise ValueError("sl_min_atr_multiplier must be < sl_max_atr_multiplier")
        if self.base_threshold <= 0 or self.base_threshold > 100:
            raise ValueError("base_threshold must be within (0, 100]")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Build and validate the configuration."""
    cfg = Config()
    cfg.validate()
    cfg.ensure_dirs()
    return cfg
