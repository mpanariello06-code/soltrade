"""The Gymnasium environment: brackets, costs, pessimism, timeout, reward."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.environment.scalping_env import (
    BUY, GYM_AVAILABLE, HOLD, RESULT_SL, RESULT_TIMEOUT, RESULT_TP3, SELL,
    EnvConfig, ScalpingEnv,
)
from ai.features.feature_pipeline import build_dataset
from tests.test_ai_features import make_candles

pytestmark = pytest.mark.skipif(not GYM_AVAILABLE, reason="gymnasium is not installed")


def dataset(n=1500, **kwargs):
    frame, spec = build_dataset(make_candles(n, **kwargs))
    return frame, spec


def scripted(prices, spread=20, atr=0.30):
    """A hand-built frame with an exact price path, for deterministic outcomes.

    Features are stubbed to zero: these tests are about execution mechanics,
    and a real feature block would only add noise to them.
    """
    n = len(prices)
    frame = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": prices, "high": [p + 0.01 for p in prices],
        "low": [p - 0.01 for p in prices], "close": prices,
        "tick_volume": 50.0, "spread": spread, "f_atr": atr,
        "f_stub": 0.0,
    })
    return frame, ["f_stub"]


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def test_reset_and_step_obey_the_gym_contract():
    frame, spec = dataset()
    env = ScalpingEnv(frame, spec.columns, EnvConfig())

    observation, info = env.reset(seed=1)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    assert isinstance(info, dict)

    observation, reward, terminated, truncated, info = env.step(HOLD)
    assert observation.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_the_observation_includes_position_state():
    frame, spec = dataset()
    env = ScalpingEnv(frame, spec.columns, EnvConfig(include_position_features=True))
    assert env.observation_space.shape[0] == len(spec.columns) + 4

    flat = ScalpingEnv(frame, spec.columns, EnvConfig(include_position_features=False))
    assert flat.observation_space.shape[0] == len(spec.columns)


def test_an_invalid_action_is_treated_as_hold():
    """Never trade because something upstream produced nonsense."""
    frame, spec = dataset()
    env = ScalpingEnv(frame, spec.columns, EnvConfig())
    env.reset(seed=1)
    for action in (7, -3, 99):
        env.step(action)
    assert env.summary()["trades"] == 0


def test_hold_never_opens_a_position():
    frame, spec = dataset()
    env = ScalpingEnv(frame, spec.columns, EnvConfig())
    env.reset(seed=2)
    for _ in range(300):
        env.step(HOLD)
    assert env.position is None
    assert env.summary()["trades"] == 0


def test_the_advanced_action_space_is_multidiscrete():
    frame, spec = dataset()
    env = ScalpingEnv(frame, spec.columns, EnvConfig(), advanced_actions=True)
    assert list(env.action_space.nvec) == [3, 3, 3]
    env.reset(seed=1)
    env.step([BUY, 0, 0])
    assert env.position is not None


def test_sl_and_tp_buckets_change_the_geometry():
    """The point of the advanced space: the agent picks how much it risks."""
    frame, columns = scripted([2300.0] * 60)
    config = EnvConfig(sl_buckets=(0.25, 0.70, 1.50), tp_scale_buckets=(0.5, 1.0, 2.0))

    tight = ScalpingEnv(frame, columns, config, advanced_actions=True)
    tight.reset(seed=1)
    tight.step([BUY, 0, 0])
    wide = ScalpingEnv(frame, columns, config, advanced_actions=True)
    wide.reset(seed=1)
    wide.step([BUY, 2, 2])

    assert tight.position.initial_risk < wide.position.initial_risk
    assert tight.position.targets[2] < wide.position.targets[2]


# --------------------------------------------------------------------------- #
# execution realism
# --------------------------------------------------------------------------- #
def test_a_buy_pays_the_spread_on_entry():
    """The candle close is NOT the fill: a BUY lifts the ask."""
    frame, columns = scripted([2300.0] * 40, spread=20)
    config = EnvConfig(slippage_points_entry=2.0)
    env = ScalpingEnv(frame, columns, config)
    env.reset(seed=1)
    env.step(BUY)

    expected = 2300.0 + (20 * 0.01) / 2 + 2 * 0.01
    assert env.position.entry_price == pytest.approx(expected)
    assert env.position.entry_price > 2300.0


def test_a_sell_enters_on_the_other_side():
    frame, columns = scripted([2300.0] * 40, spread=20)
    env = ScalpingEnv(frame, columns, EnvConfig(slippage_points_entry=2.0))
    env.reset(seed=1)
    env.step(SELL)
    assert env.position.entry_price < 2300.0


def test_costs_are_charged_in_r_and_scale_with_the_stop():
    """A 2-pip cost is trivial against a wide stop and ruinous against a tight one."""
    frame, columns = scripted([2300.0] * 40, spread=20, atr=0.30)
    config = EnvConfig(sl_atr=0.70, slippage_points_entry=2.0, slippage_points_exit=2.0)
    env = ScalpingEnv(frame, columns, config)
    env.reset(seed=1)
    env.step(BUY)

    risk = env.position.initial_risk
    expected = (20 + 2 + 2) * 0.01 / risk
    assert env._cost_r(1, env.position) == pytest.approx(expected)
    assert expected > 0


def test_a_wide_spread_blocks_entry():
    frame, columns = scripted([2300.0] * 40, spread=200)
    env = ScalpingEnv(frame, columns, EnvConfig(max_spread_points=50))
    env.reset(seed=1)
    _obs, _r, _t, _tr, info = env.step(BUY)
    assert env.position is None
    assert info.get("rejected") is True


# --------------------------------------------------------------------------- #
# outcomes
# --------------------------------------------------------------------------- #
def test_a_stop_out_is_recorded_as_sl():
    frame, columns = scripted([2300.0] + [2290.0] * 30)
    env = ScalpingEnv(frame, columns, EnvConfig())
    env.reset(seed=1)
    env.step(BUY)
    env.step(HOLD)

    trades = env.trades_frame()
    assert len(trades) == 1
    assert trades.iloc[0]["result"] == RESULT_SL
    assert trades.iloc[0]["net_r"] < 0


def test_reaching_the_final_target_closes_the_trade():
    frame, columns = scripted([2300.0] + [2310.0] * 30)
    env = ScalpingEnv(frame, columns, EnvConfig())
    env.reset(seed=1)
    env.step(BUY)
    env.step(HOLD)

    trades = env.trades_frame()
    assert trades.iloc[0]["result"] == RESULT_TP3
    assert trades.iloc[0]["gross_r"] > 0


def test_ambiguity_resolves_to_the_stop():
    """THE pessimism rule (spec section 12).

    One M1 candle that trades through both the stop and the target cannot say
    which came first, so the stop is assumed. Anything else flatters the result.
    """
    prices = [2300.0] * 5
    n = len(prices) + 1
    frame = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": 2300.0,
        # A candle spanning far above the targets AND far below the stop.
        "high": [2300.01] + [2400.0] * (n - 1),
        "low": [2299.99] + [2200.0] * (n - 1),
        "close": 2300.0, "tick_volume": 50.0, "spread": 20,
        "f_atr": 0.30, "f_stub": 0.0,
    })
    env = ScalpingEnv(frame, ["f_stub"], EnvConfig())
    env.reset(seed=1)
    env.step(BUY)
    env.step(HOLD)

    trades = env.trades_frame()
    assert trades.iloc[0]["result"] == RESULT_SL, (
        "an ambiguous candle was scored optimistically"
    )


def test_a_position_times_out_on_the_configured_window():
    frame, columns = scripted([2300.0] * 40)
    env = ScalpingEnv(frame, columns, EnvConfig(max_holding_candles=5))
    env.reset(seed=1)
    env.step(BUY)
    for _ in range(10):
        env.step(HOLD)

    trades = env.trades_frame()
    assert trades.iloc[0]["result"] == RESULT_TIMEOUT
    assert trades.iloc[0]["holding_candles"] == 5


def test_an_open_position_is_closed_at_the_end_of_an_episode():
    """Otherwise every loser open at an episode edge silently disappears."""
    frame, columns = scripted([2300.0] * 30)
    env = ScalpingEnv(frame, columns, EnvConfig(max_holding_candles=999,
                                                episode_length=0, random_start=False))
    env.reset(seed=1)
    env.step(BUY)
    truncated = False
    while not truncated:
        _obs, _r, _t, truncated, _i = env.step(HOLD)

    assert env.position is None
    assert len(env.trades_frame()) == 1


def test_partial_targets_bank_and_leave_a_runner():
    # entry ~2300.12, so TP1 sits at ~2300.255 and TP2 at ~2300.42:
    # 2300.30 reaches the first target and not the second.
    frame, columns = scripted([2300.0] + [2300.30] * 30, atr=0.30)
    env = ScalpingEnv(frame, columns, EnvConfig(tp_fractions=(0.5, 0.25, 0.25)))
    env.reset(seed=1)
    env.step(BUY)
    env.step(HOLD)

    assert env.position is not None, "TP1 closed the whole position"
    assert env.position.tp_hits == 1
    assert env.position.remaining == pytest.approx(0.5)


def test_breakeven_follows_the_configuration_in_both_directions():
    frame, columns = scripted([2300.0] + [2300.30] * 30, atr=0.30)

    on = ScalpingEnv(frame, columns, EnvConfig(move_sl_to_breakeven_after_tp1=True))
    on.reset(seed=1)
    on.step(BUY)
    entry = on.position.entry_price
    on.step(HOLD)
    if on.position is not None and on.position.tp_hits >= 1:
        assert on.position.stop_loss == pytest.approx(entry)

    off = ScalpingEnv(frame, columns, EnvConfig(move_sl_to_breakeven_after_tp1=False))
    off.reset(seed=1)
    off.step(BUY)
    original = off.position.stop_loss
    off.step(HOLD)
    if off.position is not None:
        assert off.position.stop_loss == pytest.approx(original)


# --------------------------------------------------------------------------- #
# reward
# --------------------------------------------------------------------------- #
def test_the_reward_is_net_r_not_dollars():
    """Reward must track NET R, so it is scale-free and cost-aware."""
    frame, columns = scripted([2300.0] + [2310.0] * 30)
    env = ScalpingEnv(frame, columns, EnvConfig(trade_penalty=0.0,
                                                holding_penalty_per_candle=0.0,
                                                drawdown_penalty=0.0))
    env.reset(seed=1)
    env.step(BUY)
    _obs, reward, _t, _tr, _i = env.step(HOLD)

    trade = env.trades_frame().iloc[0]
    # Exact to float rounding: the reward for a trade IS its NET R, with the
    # partials paid as they fire rather than counted twice at the close.
    assert reward == pytest.approx(float(trade["net_r"]), abs=1e-5)


def test_trading_costs_something_so_churn_is_not_free():
    frame, columns = scripted([2300.0] * 40)
    env = ScalpingEnv(frame, columns, EnvConfig(trade_penalty=0.5))
    env.reset(seed=1)
    _obs, reward, _t, _tr, info = env.step(BUY)
    assert info.get("opened") == "BUY"
    assert reward == pytest.approx(-0.5)


def test_the_reward_is_clipped():
    """One freak candle must not dominate a batch's gradient."""
    frame, columns = scripted([2300.0] + [9000.0] * 20)
    env = ScalpingEnv(frame, columns, EnvConfig(reward_clip=2.0))
    env.reset(seed=1)
    env.step(BUY)
    _obs, reward, _t, _tr, _i = env.step(HOLD)
    assert abs(reward) <= 2.0


def test_holding_is_penalised_so_a_scalp_does_not_drift():
    frame, columns = scripted([2300.0] * 40)
    env = ScalpingEnv(frame, columns, EnvConfig(holding_penalty_per_candle=0.01,
                                                trade_penalty=0.0))
    env.reset(seed=1)
    env.step(BUY)
    _obs, reward, _t, _tr, _i = env.step(HOLD)
    assert reward < 0


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_the_summary_reports_net_first_and_costs_separately():
    frame, spec = dataset()
    env = ScalpingEnv(frame, spec.columns, EnvConfig())
    env.reset(seed=3)
    for _ in range(400):
        _obs, _r, _t, truncated, _i = env.step(BUY)
        if truncated:
            break

    summary = env.summary()
    for key in ("trades", "net_r", "gross_r", "cost_r", "win_rate",
                "max_drawdown_r", "sl_rate", "timeout_rate", "median_holding"):
        assert key in summary, key
    if summary["trades"]:
        assert summary["net_r"] < summary["gross_r"], "costs were not deducted"


def test_two_markets_run_the_same_class_with_their_own_costs():
    """XAUUSDs and BTCUSDs share the environment but never a model or a cost."""
    gold_frame, gold_spec = build_dataset(make_candles(800, volatility=0.16, start=2300.0))
    btc_frame, _ = build_dataset(
        make_candles(800, volatility=30.0, start=60_000.0, spread=1000), spec=gold_spec
    )

    gold = ScalpingEnv(gold_frame, gold_spec.columns,
                       EnvConfig(assumed_spread_points=20.0))
    btc = ScalpingEnv(btc_frame, gold_spec.columns,
                      EnvConfig(assumed_spread_points=1000.0, point_value=0.01))

    gold.reset(seed=1)
    gold.step(BUY)
    btc.reset(seed=1)
    btc.step(BUY)

    assert gold.position.entry_price != btc.position.entry_price
    assert gold.trades == [] and btc.trades == []
    assert gold._spread[0] != btc._spread[0]
