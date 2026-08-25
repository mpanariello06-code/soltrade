"""Gymnasium environment: M1 bracket scalping with realistic costs.

WHAT THE AGENT IS ACTUALLY LEARNING
-----------------------------------
Not "will price go up", but "is a short bracket trade taken here expected to
return positive **NET R after costs**".  Those are different questions, and at a
three-pip target the difference is the entire edge: a move can be correctly
predicted and still lose money.  The reward is therefore NET R - never dollars,
never trade count.

EXECUTION REALISM (the part that decides whether any of this means anything)
---------------------------------------------------------------------------
* **Entry crosses the spread.** A BUY fills at the ask, a SELL at the bid.
  Using the candle close for both would hand the agent half a spread of free
  edge on every trade.
* **Slippage is charged**, configurably, on entry and exit.
* **Costs come from the live bot's own** ``config.round_trip_cost()``.  The
  agent cannot be given better fills than the rule engine's backtester gets.
* **Ambiguity resolves pessimistically.** If one M1 candle trades through both
  the stop and a target, the stop is assumed first - the same rule
  ``signal_tracker.py`` uses, imported rather than re-derived, because two
  copies of a pessimism rule eventually disagree and the optimistic one wins.
* **Timeout** closes a stale scalp at market, using the existing holding window.

NO LOOKAHEAD
------------
``step`` advances one candle and returns the observation for the NEW candle,
built from data at or before it.  Bracket outcomes are evaluated on the candle
*after* entry, because entry happens at the close of the decision candle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - gymnasium is an optional research dependency
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except Exception:  # noqa: BLE001
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    GYM_AVAILABLE = False

HOLD, BUY, SELL = 0, 1, 2
ACTION_NAMES = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}

#: Outcome labels, matching the rest of the project's vocabulary exactly so a
#: PPO trade log and a rule-engine outcome log can be compared column by column.
RESULT_TP1, RESULT_TP2, RESULT_TP3 = "TP1_HIT", "TP2_HIT", "TP3_HIT"
RESULT_SL, RESULT_TIMEOUT = "SL_HIT", "TIMEOUT"


@dataclass
class EnvConfig:
    """Everything about the simulation, all configurable, none of it claimed optimal.

    The defaults mirror the rule engine so the first comparison is like-for-like;
    the brief is explicit that they must not simply be inherited as correct, so
    the walk-forward harness sweeps the ones that matter (holding window, target
    multiples) on VALIDATION data only.
    """

    # -- bracket geometry, in ATR multiples ------------------------------- #
    sl_atr: float = 0.70
    tp_atr: Tuple[float, float, float] = (0.45, 1.00, 1.70)
    #: Buckets for the advanced MultiDiscrete action space.  Small on purpose:
    #: a large action space is a slower thing to learn, not a richer one.
    sl_buckets: Tuple[float, ...] = (0.50, 0.70, 0.90)
    tp_scale_buckets: Tuple[float, ...] = (0.75, 1.00, 1.40)
    #: Fraction of the position closed at each target.
    tp_fractions: Tuple[float, float, float] = (0.33, 0.33, 0.34)

    # -- holding window ---------------------------------------------------- #
    max_holding_candles: int = 15
    move_sl_to_breakeven_after_tp1: bool = False

    # -- costs -------------------------------------------------------------- #
    point_value: float = 0.01
    #: Used when the data carries no per-candle spread.
    assumed_spread_points: float = 20.0
    slippage_points_entry: float = 2.0
    slippage_points_exit: float = 2.0
    commission_points_per_side: float = 0.0
    #: Refuse to enter above this spread; 0 disables the check.
    max_spread_points: float = 0.0
    use_data_spread: bool = True

    # -- reward shaping ------------------------------------------------------ #
    #: Small per-trade cost so the agent must believe a setup is worth more than
    #: nothing.  Discourages churn without dictating a trade count.
    trade_penalty: float = 0.02
    #: Charged per candle held, so a scalp that drifts is not free.
    holding_penalty_per_candle: float = 0.002
    #: Applied to new equity lows, so two paths to the same total R are not
    #: equally rewarded when one drew down harder getting there.
    drawdown_penalty: float = 0.05
    #: Reward for HOLD.  Deliberately 0.0: paying an agent to do nothing is the
    #: easiest possible policy to learn and teaches nothing about trading.
    hold_reward: float = 0.0
    #: R is clipped so one freak candle cannot dominate a batch's gradient.
    reward_clip: float = 5.0

    # -- episode ------------------------------------------------------------- #
    #: 0 means "run to the end of the data".
    episode_length: int = 0
    random_start: bool = True
    #: Extra observation channels describing the open position.
    include_position_features: bool = True

    def cost_points(self, spread_points: float) -> float:
        """Round-trip cost in POINTS: spread + both slippages + commission."""
        return (
            float(spread_points)
            + self.slippage_points_entry
            + self.slippage_points_exit
            + 2.0 * self.commission_points_per_side
        )

    @classmethod
    def from_market_config(cls, config, **overrides) -> "EnvConfig":
        """Build from the live bot's config so costs match the rule engine's."""
        env = cls(
            sl_atr=float(getattr(config, "sl_atr_multiplier", 0.70)),
            tp_atr=tuple(getattr(config, "tp_atr_multiples", (0.45, 1.00, 1.70))),
            max_holding_candles=int(getattr(config, "max_holding_candles", 15)),
            move_sl_to_breakeven_after_tp1=bool(
                getattr(config, "move_sl_to_breakeven_after_tp1", False)
            ),
            point_value=float(getattr(config, "point_value", 0.01)),
            assumed_spread_points=float(getattr(config, "assumed_spread_points", 20.0)),
            slippage_points_entry=float(getattr(config, "slippage_points_entry", 2.0)),
            slippage_points_exit=float(getattr(config, "slippage_points_exit", 2.0)),
            commission_points_per_side=float(
                getattr(config, "commission_points_per_side", 0.0)
            ),
            max_spread_points=float(getattr(config, "max_spread_points", 0.0) or 0.0),
        )
        for key, value in overrides.items():
            setattr(env, key, value)
        return env


@dataclass
class OpenPosition:
    """A live bracket trade inside the simulation."""

    direction: int
    entry_price: float
    stop_loss: float
    targets: Tuple[float, float, float]
    entry_index: int
    initial_risk: float
    remaining: float = 1.0
    tp_hits: int = 0
    realised_r: float = 0.0
    cost_r: float = 0.0
    breakeven_applied: bool = False

    @property
    def is_long(self) -> bool:
        return self.direction == BUY


class ScalpingEnv(gym.Env if GYM_AVAILABLE else object):
    """One market, one M1 series, bracket trades, NET-R reward.

    Market-agnostic: it receives a prepared feature frame and an
    :class:`EnvConfig`, so XAUUSDs and BTCUSDs use the same class with their own
    costs and their own data - and never share a model.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: List[str],
        config: Optional[EnvConfig] = None,
        seed: Optional[int] = None,
        advanced_actions: bool = False,
    ) -> None:
        if not GYM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for the RL environment. "
                "Install it with: pip install gymnasium"
            )
        super().__init__()
        if frame.empty:
            raise ValueError("the environment needs a non-empty feature frame")

        self.config = config or EnvConfig()
        self.feature_columns = list(feature_columns)
        self.advanced_actions = bool(advanced_actions)

        # Everything is pulled into plain numpy up front: the step loop runs
        # millions of times during training, and pandas indexing there would
        # dominate the run time.
        self._features = np.nan_to_num(
            frame[self.feature_columns].to_numpy(dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        self._open = frame["open"].to_numpy(dtype=np.float64)
        self._high = frame["high"].to_numpy(dtype=np.float64)
        self._low = frame["low"].to_numpy(dtype=np.float64)
        self._close = frame["close"].to_numpy(dtype=np.float64)
        self._atr = frame["f_atr"].to_numpy(dtype=np.float64)
        self._time = pd.to_datetime(frame["time"], utc=True).to_numpy()
        if "spread" in frame.columns and self.config.use_data_spread:
            self._spread = pd.to_numeric(
                frame["spread"], errors="coerce"
            ).fillna(self.config.assumed_spread_points).to_numpy(dtype=np.float64)
        else:
            self._spread = np.full(
                len(frame), self.config.assumed_spread_points, dtype=np.float64
            )

        self.n = len(frame)
        self._position_features = 4 if self.config.include_position_features else 0
        observation_size = len(self.feature_columns) + self._position_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32
        )
        if self.advanced_actions:
            # direction x SL bucket x TP bucket - 27 combinations, small enough
            # for PPO to actually explore.
            self.action_space = spaces.MultiDiscrete(
                [3, len(self.config.sl_buckets), len(self.config.tp_scale_buckets)]
            )
        else:
            self.action_space = spaces.Discrete(3)

        self._rng = np.random.default_rng(seed)
        self.position: Optional[OpenPosition] = None
        self.trades: List[Dict[str, Any]] = []
        self._index = 0
        self._start = 0
        self._end = self.n - 1
        self._equity_r = 0.0
        self._peak_r = 0.0

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        length = self.config.episode_length or self.n
        # One candle of head-room at each end: entry needs a NEXT candle to be
        # evaluated against, and warm-up rows are already gone.
        usable = max(self.n - 2, 1)
        if self.config.random_start and length < usable:
            self._start = int(self._rng.integers(0, usable - length))
        else:
            self._start = 0
        self._end = min(self._start + length, self.n - 1)

        self._index = self._start
        self.position = None
        self.trades = []
        self._equity_r = 0.0
        self._peak_r = 0.0
        return self._observation(), {}

    def step(self, action):
        """Advance one M1 candle.

        Order of operations matters and is deliberate:

        1. an OPEN position is advanced against **this** candle's high/low;
        2. only if flat may a new position be opened, entered at THIS candle's
           close (crossing the spread);
        3. the index advances and the NEXT candle's observation is returned.

        A position opened at step *t* is therefore never evaluated against
        candle *t* - it could not have been, since entry is that candle's close.
        """
        direction, sl_bucket, tp_bucket = self._decode(action)
        reward = 0.0
        info: Dict[str, Any] = {}

        if self.position is not None:
            reward += self._advance_position(self._index)
        elif direction in (BUY, SELL):
            opened = self._open_position(self._index, direction, sl_bucket, tp_bucket)
            if opened:
                reward -= self.config.trade_penalty
                info["opened"] = ACTION_NAMES[direction]
            else:
                info["rejected"] = True
        else:
            reward += self.config.hold_reward

        self._index += 1
        truncated = self._index >= self._end

        if truncated and self.position is not None:
            # Close at market rather than abandoning the trade: an unrecorded
            # open position would quietly remove every loser at an episode edge.
            reward += self._close_position(
                self._index - 1, self._exit_price(self._index - 1), RESULT_TIMEOUT
            )

        reward = float(np.clip(reward, -self.config.reward_clip, self.config.reward_clip))
        info.update(
            equity_r=self._equity_r,
            trades=len(self.trades),
            position=None if self.position is None else ACTION_NAMES[self.position.direction],
        )
        return self._observation(), reward, False, truncated, info

    # ------------------------------------------------------------------ #
    # mechanics
    # ------------------------------------------------------------------ #
    def _decode(self, action) -> Tuple[int, int, int]:
        """Turn either action-space shape into (direction, sl bucket, tp bucket)."""
        if self.advanced_actions:
            values = np.asarray(action).ravel()
            if values.size < 3:
                return HOLD, 0, 0
            direction = int(values[0])
            sl_bucket = int(np.clip(values[1], 0, len(self.config.sl_buckets) - 1))
            tp_bucket = int(np.clip(values[2], 0, len(self.config.tp_scale_buckets) - 1))
        else:
            direction = int(np.asarray(action).ravel()[0]) if np.ndim(action) else int(action)
            sl_bucket = tp_bucket = -1
        if direction not in (HOLD, BUY, SELL):
            direction = HOLD
        return direction, sl_bucket, tp_bucket

    def _spread_price(self, index: int) -> float:
        return float(self._spread[index]) * self.config.point_value

    def _entry_price(self, index: int, direction: int) -> float:
        """Entry crosses the spread: BUY lifts the ask, SELL hits the bid."""
        close = float(self._close[index])
        half = self._spread_price(index) / 2.0
        slip = self.config.slippage_points_entry * self.config.point_value
        return close + half + slip if direction == BUY else close - half - slip

    def _exit_price(self, index: int) -> float:
        return float(self._close[index])

    def _open_position(
        self, index: int, direction: int, sl_bucket: int, tp_bucket: int
    ) -> bool:
        """Open a bracket, or refuse when the setup is untradeable."""
        atr_value = float(self._atr[index])
        if not np.isfinite(atr_value) or atr_value <= 0:
            return False
        spread_points = float(self._spread[index])
        if self.config.max_spread_points > 0 and spread_points > self.config.max_spread_points:
            return False

        sl_multiple = (
            self.config.sl_buckets[sl_bucket] if sl_bucket >= 0 else self.config.sl_atr
        )
        tp_scale = (
            self.config.tp_scale_buckets[tp_bucket] if tp_bucket >= 0 else 1.0
        )
        risk = sl_multiple * atr_value
        if risk <= 0:
            return False

        entry = self._entry_price(index, direction)
        sign = 1.0 if direction == BUY else -1.0
        stop = entry - sign * risk
        targets = tuple(entry + sign * m * tp_scale * atr_value for m in self.config.tp_atr)

        self.position = OpenPosition(
            direction=direction, entry_price=entry, stop_loss=stop,
            targets=targets, entry_index=index, initial_risk=risk,
        )
        return True

    def _advance_position(self, index: int) -> float:
        """Score the open bracket against candle ``index``.  Returns reward in R."""
        position = self.position
        assert position is not None
        high, low = float(self._high[index]), float(self._low[index])
        long = position.is_long

        stop_touched = low <= position.stop_loss if long else high >= position.stop_loss
        target_touched = [
            (high >= t if long else low <= t) for t in position.targets
        ]

        # PESSIMISTIC RESOLUTION.  M1 OHLC cannot say which came first, so when
        # both the stop and a target trade inside one candle, the stop wins.
        # Identical to signal_tracker.py's rule - deliberately, so PPO's paper
        # results and the rule engine's are computed on the same assumption.
        if stop_touched:
            return self._close_position(index, position.stop_loss, RESULT_SL)

        reward = 0.0
        for level, touched in enumerate(target_touched):
            if position.tp_hits > level or not touched:
                continue
            is_final = level >= len(position.targets) - 1
            if is_final:
                return reward + self._close_position(
                    index, position.targets[level], RESULT_TP3
                )
            reward += self._take_partial(index, level)
            if self.position is None:
                return reward

        if index - position.entry_index >= self.config.max_holding_candles:
            return reward + self._close_position(
                index, self._exit_price(index), RESULT_TIMEOUT
            )

        reward -= self.config.holding_penalty_per_candle
        return reward

    def _take_partial(self, index: int, level: int) -> float:
        """Bank the configured fraction at TP1/TP2 and keep the rest running."""
        position = self.position
        assert position is not None
        fraction = min(self.config.tp_fractions[level], position.remaining)
        if fraction <= 0:
            position.tp_hits = level + 1
            return 0.0

        gross_r = self._gross_r(position, position.targets[level]) * fraction
        cost_r = self._cost_r(index, position) * fraction
        position.realised_r += gross_r
        position.cost_r += cost_r
        position.remaining = round(position.remaining - fraction, 8)
        position.tp_hits = level + 1

        if (
            level == 0
            and self.config.move_sl_to_breakeven_after_tp1
            and not position.breakeven_applied
        ):
            position.stop_loss = position.entry_price
            position.breakeven_applied = True

        return gross_r - cost_r

    def _gross_r(self, position: OpenPosition, exit_price: float) -> float:
        """Price move in units of the risk taken at entry."""
        sign = 1.0 if position.is_long else -1.0
        return (exit_price - position.entry_price) * sign / position.initial_risk

    def _cost_r(self, index: int, position: OpenPosition) -> float:
        """Round-trip cost expressed in R - which is what makes it comparable.

        A two-pip cost is trivial against a 50-pip stop and ruinous against a
        three-pip one; only the ratio tells the agent which situation it is in.
        """
        points = self.config.cost_points(self._spread[index])
        return points * self.config.point_value / position.initial_risk

    def _close_position(self, index: int, exit_price: float, result: str) -> float:
        """Close whatever remains, record the trade, return the reward in R."""
        position = self.position
        assert position is not None
        remaining = max(position.remaining, 0.0)
        gross_r = self._gross_r(position, exit_price) * remaining
        cost_r = self._cost_r(index, position) * remaining if remaining > 0 else 0.0

        total_gross = position.realised_r + gross_r
        total_cost = position.cost_r + cost_r
        net_r = total_gross - total_cost
        # Reward is for the portion being closed NOW.  The partials already paid
        # their own share when they fired, so returning the trade total here
        # would count them twice - and would make a laddered winner look better
        # than the same move taken in one piece.
        closing_reward = gross_r - cost_r

        holding = index - position.entry_index
        if result == RESULT_TP3:
            position.tp_hits = len(position.targets)
        self.trades.append({
            "entry_time": pd.Timestamp(self._time[position.entry_index]).isoformat(),
            "exit_time": pd.Timestamp(self._time[index]).isoformat(),
            "direction": ACTION_NAMES[position.direction],
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "stop_loss": position.stop_loss,
            "tp1": position.targets[0], "tp2": position.targets[1],
            "tp3": position.targets[2],
            "result": result,
            "tp_hits": position.tp_hits,
            "gross_r": round(total_gross, 6),
            "cost_r": round(total_cost, 6),
            "net_r": round(net_r, 6),
            "holding_candles": holding,
            "spread_points": float(self._spread[index]),
            "initial_risk": position.initial_risk,
        })

        self._equity_r += net_r
        self._peak_r = max(self._peak_r, self._equity_r)
        drawdown = self._peak_r - self._equity_r

        self.position = None

        reward = closing_reward
        reward -= self.config.holding_penalty_per_candle * max(holding, 0)
        if drawdown > 0:
            # Penalise the depth of the hole, not merely the final total: two
            # paths to the same R are not equally good.
            reward -= self.config.drawdown_penalty * drawdown
        return reward

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def _observation(self) -> np.ndarray:
        index = min(self._index, self.n - 1)
        features = self._features[index]
        if not self._position_features:
            return features.astype(np.float32)

        if self.position is None:
            extra = np.zeros(4, dtype=np.float32)
        else:
            position = self.position
            unrealised = self._gross_r(position, float(self._close[index]))
            age = (index - position.entry_index) / max(self.config.max_holding_candles, 1)
            extra = np.array([
                1.0 if position.is_long else -1.0,   # which way we are facing
                np.clip(unrealised, -5.0, 5.0),      # open P/L in R
                np.clip(age, 0.0, 2.0),              # how much of the window is spent
                position.remaining,                  # how much is still on
            ], dtype=np.float32)
        return np.concatenate([features, extra]).astype(np.float32)

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def summary(self) -> Dict[str, Any]:
        """Episode statistics, NET first because NET is what matters."""
        frame = self.trades_frame()
        if frame.empty:
            return {
                "trades": 0, "net_r": 0.0, "average_net_r": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "max_drawdown_r": 0.0,
                "average_holding": 0.0, "median_holding": 0.0,
            }
        net = frame["net_r"]
        wins, losses = net[net > 0], net[net < 0]
        equity = net.cumsum()
        gross_loss = float(-losses.sum())
        return {
            "trades": int(len(frame)),
            "net_r": round(float(net.sum()), 4),
            "average_net_r": round(float(net.mean()), 4),
            "gross_r": round(float(frame["gross_r"].sum()), 4),
            "cost_r": round(float(frame["cost_r"].sum()), 4),
            "win_rate": round(100.0 * len(wins) / len(frame), 2),
            "profit_factor": round(float(wins.sum()) / gross_loss, 3) if gross_loss > 0 else float("inf"),
            "max_drawdown_r": round(float((equity.cummax() - equity).max()), 4),
            "average_holding": round(float(frame["holding_candles"].mean()), 2),
            "median_holding": round(float(frame["holding_candles"].median()), 2),
            "tp1_rate": round(100.0 * float((frame["tp_hits"] >= 1).mean()), 2),
            "tp2_rate": round(100.0 * float((frame["tp_hits"] >= 2).mean()), 2),
            "tp3_rate": round(100.0 * float((frame["result"] == RESULT_TP3).mean()), 2),
            "sl_rate": round(100.0 * float((frame["result"] == RESULT_SL).mean()), 2),
            "timeout_rate": round(100.0 * float((frame["result"] == RESULT_TIMEOUT).mean()), 2),
        }
