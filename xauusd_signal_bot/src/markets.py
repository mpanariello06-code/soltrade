"""Per-market configuration.

The scalping engine is market-agnostic: it reads plain attributes off a config
object and never mentions a symbol.  Everything that genuinely differs between
instruments lives here, and :func:`config.Config.for_market` folds one of these
onto the global config to produce the view the engine actually runs on.

WHAT IS SHARED AND WHAT IS NOT
------------------------------
Shared (identical for every market): the nine analysis engines, the scoring
weights, indicator periods, regime classification, the filter chain, the target
*shape*, and every anti-lookahead rule.  A market never gets its own code path.

Per-market: price scale, cost assumptions, absolute floors, thresholds,
cooldowns, holding period, session behaviour and the storage directory.

WHY THE TARGET MODEL TRANSFERS BUT THE FLOORS DO NOT
-----------------------------------------------------
Targets and stops are already expressed as multiples of the live M1 ATR, so
they self-scale: a market with a $40 ATR gets $40-scale targets without anyone
configuring that.  What does *not* transfer are the absolute floors - a "1 pip
minimum target" is meaningful on gold at $2,300 and meaningless on Bitcoin at
$90,000.  Each market therefore carries both:

``min_tp_pips``  an absolute floor in that market's pip unit
``min_tp_pct``   a floor as a fraction of price

and the larger of the two applies.  Gold uses the pip floor with the percentage
set to zero, which reproduces its existing behaviour exactly.  Bitcoin uses the
percentage floor, which keeps the geometry sane at any price level.

BITCOIN PARAMETERS ARE **INITIAL RESEARCH PARAMETERS**
------------------------------------------------------
They are starting points chosen from the arithmetic of price scale and typical
cost, not from any backtest, and they are not claimed to be optimal or
profitable.  Re-derive them from your own venue's data before trusting them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from .logger import get_logger

LOGGER = get_logger("markets")

#: Canonical symbols - the exact names THIS ACCOUNT'S BROKER uses.  These are
#: what the MT5 feed is asked for AND what every file, record, menu and report
#: shows, so there is one name per instrument and no translation anywhere.
XAUUSD = "XAUUSDs"
BTCUSD = "BTCUSDs"

#: Older spellings that still resolve to the canonical symbol above.  Data
#: written before the rename, a ``--symbol XAUUSD`` on the command line and a
#: persisted ``active_market`` all keep working.  Matching is case-insensitive.
SYMBOL_ALIASES: Dict[str, str] = {
    "XAUUSD": XAUUSD,
    "GOLD": XAUUSD,
    "BTCUSD": BTCUSD,
}


@dataclass(frozen=True)
class MarketConfig:
    """Everything that differs between one tradeable instrument and another."""

    # -- identity ---------------------------------------------------------- #
    symbol: str
    key: str                       #: lowercase, used for the data directory
    display: str                   #: name shown in Telegram
    icon: str

    # -- price scale -------------------------------------------------------- #
    digits: int                    #: decimals the broker quotes
    point_value: float             #: one point in price terms (10**-digits)
    pip_value: float               #: one "pip" in price terms
    #: short suffix used when printing distances ("1.8p", "12.0$")
    pip_name: str = "p"

    # -- market hours -------------------------------------------------------- #
    #: True for a market that never closes.  The session *label* is still
    #: recorded for analysis; only the session *filter* is bypassed.
    is_24h: bool = False
    default_sessions: Tuple[str, ...] = ("ALL_SESSIONS",)

    # -- decision threshold --------------------------------------------------- #
    threshold: float = 68.0

    # -- targets and stops (ATR multiples are shared; floors are not) --------- #
    tp_atr_multiples: Tuple[float, float, float] = (0.45, 1.00, 1.70)
    min_tp_pips: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    min_tp_pct: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_tp3_pips: float = 0.0
    max_tp3_pct: float = 0.0
    sl_atr_multiplier: float = 0.70
    sl_min_atr_multiplier: float = 0.50
    sl_max_atr_multiplier: float = 0.80
    sl_structure_buffer_atr: float = 0.15
    sl_structure_lookback: int = 12
    min_tp2_rr: float = 1.2
    min_net_tp2_rr: float = 0.30

    # -- cost model (RESEARCH ASSUMPTIONS, not verified execution) ------------ #
    assumed_spread_points: float = 20.0
    slippage_points_entry: float = 2.0
    slippage_points_exit: float = 2.0
    commission_points_per_side: float = 0.0
    min_tp1_cost_multiple: float = 1.5
    min_sl_cost_multiple: float = 2.0
    max_spread_points: float = 35.0
    max_spread_to_expected_move: float = 0.35
    cost_model_name: str = "DEFAULT"

    # -- pacing ---------------------------------------------------------------- #
    cooldown_candles: int = 10
    same_direction_cooldown_candles: int = 20
    max_signals_per_day: int = 30
    max_concurrent_active_signals: int = 3
    max_holding_candles: int = 15

    # -- data ------------------------------------------------------------------ #
    context_timeframe: str = "M5"
    candles_signal: int = 900
    candles_context: int = 400

    #: Set ONLY when the data feed needs a different name from ``symbol``.
    #: Normally empty, because ``symbol`` is already the broker's own name -
    #: one name per instrument, no translation anywhere.  It exists for moving
    #: to a broker that spells an instrument differently without renaming the
    #: market and orphaning its stored history.
    broker_symbol: str = ""

    #: Free-text note surfaced in the README/build report and the settings panel.
    note: str = ""

    def feed_symbol(self) -> str:
        """The symbol to request candles for."""
        return self.broker_symbol or self.symbol

    def label(self) -> str:
        """Icon + symbol, for Telegram."""
        return f"{self.icon} {self.symbol}"

    def pips(self, price_distance: float) -> float:
        """Convert a price distance into this market's pip unit."""
        return float(price_distance) / self.pip_value

    def with_overrides(self, **changes) -> "MarketConfig":
        """A copy with individual fields replaced (used by tests and tuning)."""
        return replace(self, **changes)


# --------------------------------------------------------------------------- #
# XAUUSD - the existing, working configuration, unchanged
# --------------------------------------------------------------------------- #
#
# Every value below was lifted verbatim from the single-market build so that the
# refactor cannot alter gold's behaviour.  ``min_tp_pct`` and ``max_tp3_pct``
# are zero, which makes the percentage floors inactive and leaves the pip floors
# in sole control exactly as before.
#
XAUUSD_CONFIG = MarketConfig(
    symbol=XAUUSD,
    key="xauusds",
    display=XAUUSD,
    icon="🥇",
    # broker_symbol is left empty: `symbol` above IS this broker's name, so
    # there is nothing to translate.  Set XAUUSD_BROKER_SYMBOL only if you move
    # to a broker that spells it differently again.
    digits=2,
    point_value=0.01,
    # Gold is quoted to 2 decimals and a pip is conventionally the first
    # decimal, so 10 points = 0.10.
    pip_value=0.10,
    pip_name="p",
    is_24h=False,
    threshold=68.0,
    tp_atr_multiples=(0.45, 1.00, 1.70),
    min_tp_pips=(1.0, 1.8, 3.0),
    max_tp3_pips=12.0,
    sl_atr_multiplier=0.70,
    sl_min_atr_multiplier=0.50,
    sl_max_atr_multiplier=0.80,
    sl_structure_buffer_atr=0.15,
    sl_structure_lookback=12,
    min_tp2_rr=1.2,
    min_net_tp2_rr=0.30,
    # 20 points = 0.20 = 2 pips, a common retail gold spread.
    assumed_spread_points=20.0,
    slippage_points_entry=2.0,
    slippage_points_exit=2.0,
    commission_points_per_side=0.0,
    min_tp1_cost_multiple=1.5,
    min_sl_cost_multiple=2.0,
    max_spread_points=35.0,
    max_spread_to_expected_move=0.35,
    cost_model_name="XAUUSD_RETAIL",
    cooldown_candles=10,
    same_direction_cooldown_candles=20,
    max_signals_per_day=30,
    max_concurrent_active_signals=3,
    max_holding_candles=15,
    context_timeframe="M5",
    note="Existing, unchanged M1 scalping configuration.",
)


# --------------------------------------------------------------------------- #
# BTCUSD - INITIAL RESEARCH PARAMETERS
# --------------------------------------------------------------------------- #
#
# NOT OPTIMISED AND NOT CLAIMED TO BE PROFITABLE.  Each value below is derived
# from arithmetic about price scale and typical cost, never from a backtest:
#
# * ``pip_value = 1.0`` - a "pip" on Bitcoin is defined here as one US dollar.
#   There is no market convention to inherit; $1 keeps the reported numbers
#   readable at any price level.
# * ``point_value = 0.01`` - MT5 reports spread in points of 10**-digits, and
#   BTCUSD is normally quoted to 2 decimals, so a $10 spread is 1,000 points.
#   The awkward magnitude is the broker's convention, not a choice.
# * Floors are PERCENTAGE based.  A fixed dollar floor would be far too tight at
#   $90,000 and far too loose at $20,000; 0.02% of price is ~$18 at $90k.
# * Cost assumptions describe a *typical retail CFD* spread on Bitcoin, which is
#   an order of magnitude wider in relative terms than a raw crypto exchange
#   fee.  If you trade a spot exchange, replace them - see ``BTCUSD_EXCHANGE``.
# * The threshold deliberately starts at the same value as gold so the first
#   Bitcoin dataset is directly comparable.  It is separately configurable and
#   must be re-derived from Bitcoin's own score distribution before use.
# * The holding window also starts equal, for the same comparability reason.
#
BTCUSD_CONFIG = MarketConfig(
    symbol=BTCUSD,
    key="btcusds",
    display=BTCUSD,
    icon="₿",
    digits=2,
    point_value=0.01,
    pip_value=1.0,            # one "pip" == one US dollar
    pip_name="$",
    is_24h=True,              # crypto never closes
    default_sessions=("ALL_SESSIONS",),
    threshold=68.0,           # INITIAL - starts equal to gold for comparability
    # The ATR multiples are shared with gold on purpose: the target *shape* is
    # the strategy, and it self-scales through the ATR.
    tp_atr_multiples=(0.45, 1.00, 1.70),
    min_tp_pips=(0.0, 0.0, 0.0),          # unused; the percentage floors apply
    min_tp_pct=(0.00020, 0.00036, 0.00060),   # 0.020% / 0.036% / 0.060% of price
    max_tp3_pips=0.0,
    max_tp3_pct=0.00240,                  # 0.24% of price is the scalp ceiling
    sl_atr_multiplier=0.70,
    sl_min_atr_multiplier=0.50,
    sl_max_atr_multiplier=0.80,
    sl_structure_buffer_atr=0.15,
    sl_structure_lookback=12,
    min_tp2_rr=1.2,
    min_net_tp2_rr=0.30,
    # $10 spread = 1,000 points; $2 slippage per side = 200 points.
    assumed_spread_points=1000.0,
    slippage_points_entry=200.0,
    slippage_points_exit=200.0,
    commission_points_per_side=0.0,
    min_tp1_cost_multiple=1.5,
    min_sl_cost_multiple=2.0,
    max_spread_points=4000.0,             # $40 - reject obvious blowouts
    max_spread_to_expected_move=0.35,
    cost_model_name="BTC_RETAIL_CFD",
    cooldown_candles=10,
    same_direction_cooldown_candles=20,
    max_signals_per_day=30,
    max_concurrent_active_signals=3,
    max_holding_candles=15,               # INITIAL - equal to gold, see above
    context_timeframe="M5",
    note="INITIAL RESEARCH PARAMETERS - not optimised, not proven profitable.",
)


#: An alternative Bitcoin cost model for a raw spot exchange (maker/taker fees
#: rather than a CFD spread).  Not wired to a market by default; swap it in via
#: ``BTCUSD_CONFIG.with_overrides(...)`` when you know your venue.
BTCUSD_EXCHANGE_COSTS = {
    "assumed_spread_points": 100.0,       # $1
    "slippage_points_entry": 50.0,        # $0.50
    "slippage_points_exit": 50.0,
    "commission_points_per_side": 500.0,  # ~0.05% taker on a $100k notional
    "cost_model_name": "BTC_SPOT_EXCHANGE",
}


# --------------------------------------------------------------------------- #
# environment overrides
# --------------------------------------------------------------------------- #
#: Fields a user may retune from ``.env``, and how to parse them.  Deliberately
#: a short list: these are the knobs worth changing per broker or per research
#: run.  Anything else is edited here, in the open, where its rationale is.
_ENV_FIELDS: Dict[str, type] = {
    "broker_symbol": str,
    "threshold": float,
    "assumed_spread_points": float,
    "slippage_points_entry": float,
    "slippage_points_exit": float,
    "commission_points_per_side": float,
    "max_spread_points": float,
    "cooldown_candles": int,
    "same_direction_cooldown_candles": int,
    "max_signals_per_day": int,
    "max_concurrent_active_signals": int,
    "max_holding_candles": int,
    "context_timeframe": str,
}

#: Names the single-market build used, kept working for XAUUSD only.  A user
#: upgrading from that build keeps their tuning; Bitcoin never inherits it.
_LEGACY_ENV_ALIASES: Dict[str, str] = {
    "broker_symbol": "SYMBOL",
    "threshold": "SCALP_THRESHOLD",
    "assumed_spread_points": "ASSUMED_SPREAD_POINTS",
    "slippage_points_entry": "SLIPPAGE_POINTS_ENTRY",
    "slippage_points_exit": "SLIPPAGE_POINTS_EXIT",
    "max_spread_points": "MAX_SPREAD_POINTS",
    "cooldown_candles": "COOLDOWN_CANDLES",
    "same_direction_cooldown_candles": "SAME_DIRECTION_COOLDOWN_CANDLES",
    "max_signals_per_day": "MAX_SIGNALS_PER_DAY",
    "max_concurrent_active_signals": "MAX_CONCURRENT_ACTIVE_SIGNALS",
    "max_holding_candles": "MAX_HOLDING_CANDLES",
    "context_timeframe": "CONTEXT_TIMEFRAME",
}


def env_prefixes(market: MarketConfig) -> Tuple[str, ...]:
    """Environment-variable prefixes accepted for ``market``, best first.

    The canonical symbol carries the broker's lowercase suffix, and shouting
    ``XAUUSDS_THRESHOLD`` in a ``.env`` file is unpleasant, so the un-suffixed
    spelling is accepted too: ``XAUUSD_THRESHOLD`` and ``XAUUSDS_THRESHOLD``
    both retune gold.
    """
    prefixes = [market.symbol.upper()]
    for alias, canonical in SYMBOL_ALIASES.items():
        if canonical == market.symbol and alias not in prefixes:
            prefixes.append(alias)
    return tuple(prefixes)


def env_prefix_hint(market: MarketConfig) -> str:
    """The friendliest accepted prefix, for messages that tell a user what to set.

    Every prefix in :func:`env_prefixes` works; this picks the shortest one that
    is genuinely a prefix of the canonical symbol, so gold is suggested as
    ``XAUUSD_...`` rather than the shoutier ``XAUUSDS_...``.
    """
    canonical = market.symbol.upper()
    usable = [p for p in env_prefixes(market) if canonical.startswith(p)]
    return min(usable, key=len) if usable else canonical


def apply_env_overrides(market: MarketConfig) -> MarketConfig:
    """Overlay ``<SYMBOL>_<FIELD>`` environment variables onto ``market``.

    ``BTCUSD_THRESHOLD=72`` retunes Bitcoin and nothing else; there is no
    variable that can move both markets at once, which is the point.  A value
    that will not parse is ignored with a warning rather than crashing the
    engine at start-up.
    """
    changes: Dict[str, object] = {}
    prefixes = env_prefixes(market)
    for field_name, caster in _ENV_FIELDS.items():
        names = [f"{prefix}_{field_name.upper()}" for prefix in prefixes]
        if market.symbol == XAUUSD and field_name in _LEGACY_ENV_ALIASES:
            names.append(_LEGACY_ENV_ALIASES[field_name])
        for name in names:
            raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                continue
            try:
                changes[field_name] = caster(raw.strip())
            except (TypeError, ValueError):
                LOGGER.warning("Ignoring %s=%r: not a valid %s", name, raw, caster.__name__)
            break
    if not changes:
        return market
    LOGGER.info("%s: applied environment overrides %s", market.symbol, sorted(changes))
    return replace(market, **changes)


MARKETS: Dict[str, MarketConfig] = {
    XAUUSD: apply_env_overrides(XAUUSD_CONFIG),
    BTCUSD: apply_env_overrides(BTCUSD_CONFIG),
}

#: Order used by the Telegram keyboard and every report.
MARKET_ORDER: Tuple[str, ...] = (XAUUSD, BTCUSD)


def _default_market() -> str:
    """Market selected on a first run, before any state file exists.

    Only a first run: once the user has chosen a market in Telegram, the
    persisted choice wins and this is not consulted again.
    """
    wanted = str(os.getenv("DEFAULT_MARKET", "") or "").strip().upper()
    if wanted and wanted in MARKETS:
        return wanted
    if wanted:
        LOGGER.warning("DEFAULT_MARKET=%r is not a supported market - using %s", wanted, XAUUSD)
    return XAUUSD


DEFAULT_MARKET = _default_market()


def get_market(symbol: str) -> MarketConfig:
    """Look up a market by symbol.  Raises ``KeyError`` for an unknown one."""
    return MARKETS[normalise_market(symbol, default="")]


def _match(symbol: Optional[str]) -> Optional[str]:
    """Resolve any spelling of a symbol to its canonical form, or ``None``.

    Matching is case-insensitive because the canonical symbols now carry the
    broker's lowercase suffix (``XAUUSDs``), so the old ``.upper()`` would have
    destroyed the very character that makes the name correct.  The value
    returned is always the registry's own casing, never the caller's.
    """
    candidate = str(symbol or "").strip()
    if not candidate:
        return None
    folded = candidate.casefold()
    for known in MARKETS:
        if known.casefold() == folded:
            return known
    alias = SYMBOL_ALIASES.get(candidate.upper())
    return alias if alias in MARKETS else None


def normalise_market(symbol: Optional[str], default: str = DEFAULT_MARKET) -> str:
    """Validate a market symbol, falling back to ``default``.

    Accepts the canonical name in any casing, plus the legacy spellings in
    :data:`SYMBOL_ALIASES`, so a ``--symbol XAUUSD`` or an ``active_market``
    persisted before the rename still resolves.

    Passing ``default=""`` makes an unknown symbol raise instead of silently
    resolving - used by :func:`get_market` so a typo cannot select gold by
    accident.
    """
    matched = _match(symbol)
    if matched is not None:
        return matched
    if default == "":
        raise KeyError(f"unknown market '{symbol}' (known: {', '.join(MARKET_ORDER)})")
    return default


def is_supported(symbol: str) -> bool:
    """True when ``symbol`` is a configured market, under any accepted spelling."""
    return _match(symbol) is not None


def market_argument(value: str) -> str:
    """``argparse`` type for ``--symbol``: accepts any spelling, returns canonical.

    ``choices=`` cannot be used for this, because it compares the raw string and
    would reject the very aliases :func:`normalise_market` exists to accept.
    """
    import argparse

    try:
        return normalise_market(value, default="")
    except KeyError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def register_market(market: MarketConfig) -> None:
    """Add a market at runtime.

    Adding a third instrument needs a :class:`MarketConfig` and nothing else -
    no engine changes, no new code path.
    """
    global MARKET_ORDER
    MARKETS[market.symbol] = market
    if market.symbol not in MARKET_ORDER:
        MARKET_ORDER = MARKET_ORDER + (market.symbol,)
