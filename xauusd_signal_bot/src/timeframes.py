"""Timeframe and mode constants for the M1 scalping build.

This system is deliberately single-purpose:

* **one signal timeframe** - M1
* **one strategy mode** - SCALPING

The multi-timeframe selector and the RESEARCH/STANDARD/CONSERVATIVE modes were
removed. What remains is one optional *context* timeframe (M5 by default) used
only as a short-term bias filter.  M15/H1/H4 context was dropped as well: a
one-hour trend says very little about a position held for a few minutes, and
carrying it invited the engine to reject good scalps for disagreeing with a
timeframe that will not resolve inside the holding period.

Set ``CONTEXT_TIMEFRAME=""`` to run on pure M1 microstructure; the context
component is then marked not-applicable and :func:`src.scoring.compute_scorecard`
shares its weight out across the remaining components.
"""

from __future__ import annotations

from typing import Optional, Tuple

#: The only signal timeframe.
SIGNAL_TIMEFRAME = "M1"

#: The only strategy mode.
MODE_SCALPING = "SCALPING"

#: Engine run states (persisted, so a restart resumes where it left off).
STATUS_RUNNING = "RUNNING"
STATUS_PAUSED = "PAUSED"
STATUS_STOPPED = "STOPPED"
RUN_STATES: Tuple[str, ...] = (STATUS_RUNNING, STATUS_PAUSED, STATUS_STOPPED)

#: Expected holding-period label shown on the signal card.  Scalps are short by
#: construction; the label reflects how many M1 candles the target implies.
HOLD_VERY_SHORT = "VERY SHORT"
HOLD_SHORT = "SHORT"
HOLD_MEDIUM = "MEDIUM"


def timeframes_in_use(context_timeframe: Optional[str] = None) -> Tuple[str, ...]:
    """Every timeframe the engine fetches: the signal one, plus context if set."""
    if context_timeframe:
        return (SIGNAL_TIMEFRAME, context_timeframe)
    return (SIGNAL_TIMEFRAME,)


def expected_hold_label(tp3_pips: float, atr_pips: float) -> str:
    """Rough holding-period label from how many ATRs the far target sits away.

    Purely descriptive - it sets expectations on the signal card and is not used
    in any decision.
    """
    if atr_pips <= 0:
        return HOLD_SHORT
    ratio = float(tp3_pips) / float(atr_pips)
    if ratio <= 1.2:
        return HOLD_VERY_SHORT
    if ratio <= 2.5:
        return HOLD_SHORT
    return HOLD_MEDIUM
