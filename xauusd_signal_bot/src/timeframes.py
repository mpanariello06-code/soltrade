"""Timeframe hierarchy, mode definitions and per-timeframe thresholds.

CONFIRMATION HIERARCHY
----------------------
Changing the signal timeframe must also change what "higher timeframe" means -
confirming an M1 setup against H1 is a different (and much slower) statement
than confirming an M5 setup against H1.  The mapping below is applied whenever
the signal timeframe changes, so the whole hierarchy moves together:

======  ==========================  ==================================
signal  intermediate confirmation   higher confirmation
======  ==========================  ==================================
M1      M5                          M15
M5      M15                         H1
M15     M30                         H1
M30     H1                          H4
H1      H4                          (none)
H4      (configurable, default none)(none)
======  ==========================  ==================================

When only one confirmation timeframe exists, the HTF component is computed from
that timeframe alone.  When none exists (H4 by default), the HTF component is
marked *not applicable* and :func:`src.scoring.compute_scorecard` renormalises
the remaining weights so the score stays on a 0-100 scale rather than being
silently capped at 85.

The ``micro`` timeframe is only used to resolve intrabar TP/SL ordering during
outcome tracking; it is never an input to the signal decision.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

#: Signal timeframes selectable from Telegram, slowest last.
SUPPORTED_TIMEFRAMES: Tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4")

#: signal timeframe -> (intermediate confirmation, higher confirmation)
CONFIRMATION_HIERARCHY: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "M1": ("M5", "M15"),
    "M5": ("M15", "H1"),
    "M15": ("M30", "H1"),
    "M30": ("H1", "H4"),
    "H1": ("H4", None),
    "H4": (None, None),
}

#: signal timeframe -> timeframe used to resolve intrabar TP/SL ordering
MICRO_TIMEFRAME: Dict[str, Optional[str]] = {
    "M1": None,       # nothing finer is available
    "M5": "M1",
    "M15": "M1",
    "M30": "M5",
    "H1": "M5",
    "H4": "M15",
}

#: Operating modes.  RESEARCH exists to *observe and collect* candidate setups,
#: not to assert that they are tradeable.
MODE_RESEARCH = "RESEARCH"
MODE_STANDARD = "STANDARD"
MODE_CONSERVATIVE = "CONSERVATIVE"
MODES: Tuple[str, ...] = (MODE_RESEARCH, MODE_STANDARD, MODE_CONSERVATIVE)

#: Nominal headline threshold per mode, shown in the Telegram panel.
#: The threshold actually applied is the per-timeframe value below.
MODE_NOMINAL_THRESHOLD: Dict[str, float] = {
    MODE_RESEARCH: 50.0,
    MODE_STANDARD: 72.0,
    MODE_CONSERVATIVE: 80.0,
}

MODE_ICONS: Dict[str, str] = {
    MODE_RESEARCH: "🔬",
    MODE_STANDARD: "📊",
    MODE_CONSERVATIVE: "🛡",
}

#: Engine run states (persisted, so a restart resumes where it left off).
STATUS_RUNNING = "RUNNING"
STATUS_PAUSED = "PAUSED"
STATUS_STOPPED = "STOPPED"
RUN_STATES: Tuple[str, ...] = (STATUS_RUNNING, STATUS_PAUSED, STATUS_STOPPED)


def confirmation_timeframes(signal_timeframe: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(intermediate, higher)`` confirmation timeframes, either may be ``None``."""
    return CONFIRMATION_HIERARCHY.get(str(signal_timeframe).upper(), (None, None))


def confirmation_label(signal_timeframe: str) -> str:
    """Human/CSV-friendly description, e.g. ``"M15+H1"`` or ``"NONE"``."""
    parts = [tf for tf in confirmation_timeframes(signal_timeframe) if tf]
    return "+".join(parts) if parts else "NONE"


def micro_timeframe(signal_timeframe: str) -> Optional[str]:
    """Timeframe used to resolve intrabar TP/SL ordering, or ``None``."""
    return MICRO_TIMEFRAME.get(str(signal_timeframe).upper())


def is_supported(timeframe: str) -> bool:
    """True when ``timeframe`` can be used as the signal timeframe."""
    return str(timeframe).upper() in SUPPORTED_TIMEFRAMES


def normalise_timeframe(timeframe: str, default: str = "M5") -> str:
    """Upper-case and validate a timeframe label, falling back to ``default``."""
    label = str(timeframe or "").upper()
    return label if label in SUPPORTED_TIMEFRAMES else default


def normalise_mode(mode: str, default: str = MODE_STANDARD) -> str:
    """Upper-case and validate a mode name, falling back to ``default``."""
    label = str(mode or "").upper()
    return label if label in MODES else default


def timeframes_in_use(signal_timeframe: str) -> List[str]:
    """Every timeframe the engine needs for ``signal_timeframe`` (signal first)."""
    intermediate, higher = confirmation_timeframes(signal_timeframe)
    micro = micro_timeframe(signal_timeframe)
    ordered = [signal_timeframe, intermediate, higher, micro]
    seen: List[str] = []
    for timeframe in ordered:
        if timeframe and timeframe not in seen:
            seen.append(timeframe)
    return seen
