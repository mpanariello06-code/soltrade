"""Who decides a trade: the rule engine, or PPO.

DELIBERATELY SEPARATE FROM ``ExecutionMode``
--------------------------------------------
``src.execution_config.ExecutionMode`` answers "may we send orders at all", and
it asserts it has exactly two members so a live mode cannot be smuggled in.
This answers a different question - "whose signal is it" - and conflating them
would either break that assertion or let a strategy change quietly widen what
the system is permitted to do.

The two compose:

    StrategyMode.PPO_DEMO  AND  ExecutionMode.DEMO_AUTO  AND  verified demo
    -> PPO signals reach the broker, through every existing safety gate

Any weaker combination means no order.  ``RULE_ONLY`` is the default and needs
none of the RL dependencies installed.
"""

from __future__ import annotations

import os
from enum import Enum

from src.logger import get_logger

LOGGER = get_logger("ai.mode")


class StrategyMode(str, Enum):
    """Which decision-maker is in charge.

    ``RULE_ONLY``   the existing scalper decides.  PPO is not loaded.
    ``PPO_SHADOW``  the rule engine still decides; PPO observes and records.
    ``PPO_DEMO``    PPO decides, and its signals go through the SAME safety
                    gates a rule signal does.  Manually enabled only.
    """

    RULE_ONLY = "RULE_ONLY"
    PPO_SHADOW = "PPO_SHADOW"
    PPO_DEMO = "PPO_DEMO"

    @property
    def uses_ppo(self) -> bool:
        """Whether a model needs to be loaded at all."""
        return self is not StrategyMode.RULE_ONLY

    @property
    def ppo_decides(self) -> bool:
        """Whether PPO's action can become an order."""
        return self is StrategyMode.PPO_DEMO


DEFAULT_STRATEGY_MODE = StrategyMode.RULE_ONLY


def parse_strategy_mode(value) -> StrategyMode:
    """Read a mode from config, defaulting to ``RULE_ONLY``.

    Anything unrecognised resolves to the safe mode with a warning: there is no
    spelling of this setting that hands control to a model by accident.
    """
    text = str(value or "").strip().upper()
    if not text:
        return DEFAULT_STRATEGY_MODE
    try:
        return StrategyMode(text)
    except ValueError:
        LOGGER.warning(
            "STRATEGY_MODE=%r is not valid (%s) - using %s",
            value, ", ".join(m.value for m in StrategyMode), DEFAULT_STRATEGY_MODE.value,
        )
        return DEFAULT_STRATEGY_MODE


def load_strategy_mode() -> StrategyMode:
    return parse_strategy_mode(os.getenv("STRATEGY_MODE", ""))
