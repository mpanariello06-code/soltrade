"""PPO training, inference fail-safe, registry, walk-forward, holdout, shadow."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ai.environment.scalping_env import BUY, GYM_AVAILABLE, HOLD, SELL, EnvConfig
from ai.evaluation.holdout import (
    create_seal, holdout_frame, load_seal, record_opening, training_frame,
)
from ai.evaluation.metrics import equity_curve, summarise_trades
from ai.evaluation.walk_forward import aggregate, build_folds, slice_period
from ai.features.feature_pipeline import FeatureSpec, build_dataset
from ai.model_registry import (
    STATUS_EXPERIMENTAL, STATUS_PRODUCTION, STATUS_RETIRED,
    ModelRecord, ModelRegistry, PromotionGates,
)
from ai.ppo.inference import PPODecision, PPOPolicy
from ai.strategy_mode import StrategyMode, parse_strategy_mode
from tests.test_ai_features import make_candles

SB3_AVAILABLE = False
try:  # pragma: no cover
    import stable_baselines3 as _sb3

    SB3_AVAILABLE = _sb3 is not None
except ImportError:
    pass

needs_sb3 = pytest.mark.skipif(
    not (SB3_AVAILABLE and GYM_AVAILABLE),
    reason="stable-baselines3 / gymnasium not installed",
)


# --------------------------------------------------------------------------- #
# strategy mode
# --------------------------------------------------------------------------- #
def test_the_default_strategy_mode_is_rule_only():
    assert parse_strategy_mode(None) is StrategyMode.RULE_ONLY
    assert parse_strategy_mode("") is StrategyMode.RULE_ONLY
    assert StrategyMode.RULE_ONLY.uses_ppo is False
    assert StrategyMode.RULE_ONLY.ppo_decides is False


def test_only_ppo_demo_lets_the_model_decide():
    assert StrategyMode.PPO_SHADOW.uses_ppo is True
    assert StrategyMode.PPO_SHADOW.ppo_decides is False, "shadow must not trade"
    assert StrategyMode.PPO_DEMO.ppo_decides is True


@pytest.mark.parametrize("attempt", ["PPO_LIVE", "REAL", "typo", "ppo demo"])
def test_an_unrecognised_mode_falls_back_to_rule_only(attempt):
    assert parse_strategy_mode(attempt) is StrategyMode.RULE_ONLY


def test_strategy_mode_is_separate_from_execution_mode():
    """They answer different questions and must not be conflated.

    ``ExecutionMode`` asserts it has exactly two members so a live mode cannot
    be added; folding PPO modes into it would break that guarantee.
    """
    from src.execution_config import ExecutionMode, assert_no_live_mode

    assert_no_live_mode()
    assert len(list(ExecutionMode)) == 2
    assert {m.name for m in StrategyMode} == {"RULE_ONLY", "PPO_SHADOW", "PPO_DEMO"}
    for mode in StrategyMode:
        assert "LIVE" not in mode.value.upper()


# --------------------------------------------------------------------------- #
# inference fail-safe  (the most important behaviour in the stack)
# --------------------------------------------------------------------------- #
class _BrokenModel:
    """A model that always raises, to prove the fail-safe holds."""

    class _Space:
        shape = (10,)

    observation_space = _Space()

    def predict(self, *_args, **_kwargs):
        raise RuntimeError("the model exploded")


class _RogueModel:
    """A model returning an action outside the action space."""

    class _Space:
        shape = (3,)

    observation_space = _Space()

    def predict(self, *_args, **_kwargs):
        return np.array([99]), None


def _policy(model, columns):
    return PPOPolicy(model, FeatureSpec(columns=columns), position_features=0)


def test_an_exploding_model_holds_rather_than_trading():
    """Never place a trade because the AI errored."""
    frame = pd.DataFrame({f"f_{i}": [0.0] for i in range(10)})
    policy = _policy(_BrokenModel(), [f"f_{i}" for i in range(10)])

    decision = policy.decide(frame)
    assert decision.action == HOLD
    assert decision.is_trade is False
    assert decision.ok is False
    assert "exploded" in decision.error
    assert policy.failures == 1


def test_an_out_of_range_action_holds():
    frame = pd.DataFrame({"f_a": [0.0], "f_b": [0.0], "f_c": [0.0]})
    policy = _policy(_RogueModel(), ["f_a", "f_b", "f_c"])

    decision = policy.decide(frame)
    assert decision.action == HOLD
    assert decision.ok is False
    assert "out-of-range" in decision.error


def test_missing_features_hold_rather_than_guessing():
    """A partially satisfied contract is worse than none: the model would still
    answer confidently, from the wrong inputs."""
    policy = _policy(_RogueModel(), ["f_a", "f_b", "f_missing"])
    decision = policy.decide(pd.DataFrame({"f_a": [0.0], "f_b": [0.0]}))
    assert decision.action == HOLD
    assert decision.ok is False


def test_non_finite_observations_hold():
    policy = _policy(_RogueModel(), ["f_a", "f_b", "f_c"])
    frame = pd.DataFrame({"f_a": [np.nan], "f_b": [0.0], "f_c": [0.0]})
    assert policy.decide(frame).action == HOLD

    frame = pd.DataFrame({"f_a": [np.inf], "f_b": [0.0], "f_c": [0.0]})
    assert policy.decide(frame).action == HOLD


def test_an_empty_frame_holds():
    policy = _policy(_RogueModel(), ["f_a"])
    assert policy.decide(pd.DataFrame()).action == HOLD


def test_a_shape_mismatch_holds():
    """A model trained on a different feature count must not be fed anyway."""
    policy = _policy(_RogueModel(), ["f_a", "f_b", "f_c", "f_d"])  # 4 vs model's 3
    decision = policy.decide(pd.DataFrame({c: [0.0] for c in ("f_a", "f_b", "f_c", "f_d")}))
    assert decision.action == HOLD
    assert "wide" in decision.error


def test_loading_a_missing_model_returns_none_rather_than_raising(tmp_path):
    """A missing model must not stop the rule engine from starting."""
    assert PPOPolicy.load(tmp_path / "nope") is None


def test_a_model_without_its_feature_spec_is_refused(tmp_path):
    """Without the contract the model could silently get the wrong columns."""
    (tmp_path / "best_model.zip").write_bytes(b"not really a model")
    assert PPOPolicy.load(tmp_path) is None


def test_a_decision_reports_its_probabilities():
    decision = PPODecision(action=BUY, action_name="BUY",
                           probabilities={"HOLD": 0.2, "BUY": 0.7, "SELL": 0.1},
                           confidence=0.7)
    assert decision.is_trade is True
    assert decision.probabilities["BUY"] == 0.7


# --------------------------------------------------------------------------- #
# model registry
# --------------------------------------------------------------------------- #
def test_versions_increment_and_are_never_reused(tmp_path):
    registry = ModelRegistry(tmp_path)
    assert registry.next_version("XAUUSDs") == "ppo_v001"

    registry.register(ModelRecord(version="ppo_v001", symbol="XAUUSDs"))
    assert registry.next_version("XAUUSDs") == "ppo_v002"

    registry.register(ModelRecord(version="ppo_v002", symbol="XAUUSDs"))
    assert registry.next_version("XAUUSDs") == "ppo_v003"


def test_registering_an_existing_version_is_refused(tmp_path):
    """Models are never overwritten - that is the whole point of versioning."""
    registry = ModelRegistry(tmp_path)
    registry.register(ModelRecord(version="ppo_v001", symbol="XAUUSDs"))
    with pytest.raises(ValueError, match="never overwritten"):
        registry.register(ModelRecord(version="ppo_v001", symbol="XAUUSDs"))


def test_the_two_markets_have_independent_version_lines(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register(ModelRecord(version="ppo_v001", symbol="XAUUSDs"))
    registry.register(ModelRecord(version="ppo_v002", symbol="XAUUSDs"))

    assert registry.next_version("BTCUSDs") == "ppo_v001", "BTC inherited gold's versions"
    registry.register(ModelRecord(version="ppo_v001", symbol="BTCUSDs"))
    assert len(registry.records("XAUUSDs")) == 2
    assert len(registry.records("BTCUSDs")) == 1
    assert registry.model_dir("XAUUSDs", "ppo_v001") != registry.model_dir("BTCUSDs", "ppo_v001")


def test_promoting_retires_the_previous_production_model(tmp_path):
    """"Which one is live?" must have exactly one answer."""
    registry = ModelRegistry(tmp_path)
    registry.register(ModelRecord(version="ppo_v001", symbol="XAUUSDs"))
    registry.register(ModelRecord(version="ppo_v002", symbol="XAUUSDs"))

    registry.set_status("XAUUSDs", "ppo_v001", STATUS_PRODUCTION)
    assert registry.production("XAUUSDs").version == "ppo_v001"

    registry.set_status("XAUUSDs", "ppo_v002", STATUS_PRODUCTION)
    assert registry.production("XAUUSDs").version == "ppo_v002"
    assert registry.get("XAUUSDs", "ppo_v001").status == STATUS_RETIRED


def test_a_new_model_starts_experimental(tmp_path):
    registry = ModelRegistry(tmp_path)
    record = registry.register(ModelRecord(version="ppo_v001", symbol="XAUUSDs"))
    assert record.status == STATUS_EXPERIMENTAL
    assert registry.production("XAUUSDs") is None, "an untested model was live"


def test_promotion_gates_reject_a_weak_candidate():
    gates = PromotionGates(min_trades=50, min_net_r=0.0, min_profit_factor=1.0)

    verdict = gates.evaluate({
        "trades": 10, "net_r": -5.0, "average_net_r": -0.5,
        "max_drawdown_r": 40.0, "profit_factor": 0.4, "average_holding": 30.0,
    })
    assert verdict["passed"] is False
    assert set(verdict["failures"]) >= {"min_trades", "min_net_r", "min_profit_factor"}

    verdict = gates.evaluate({
        "trades": 120, "net_r": 8.0, "average_net_r": 0.07,
        "max_drawdown_r": 6.0, "profit_factor": 1.3, "average_holding": 6.0,
    })
    assert verdict["passed"] is True


def test_a_corrupt_registry_does_not_destroy_the_models(tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register(ModelRecord(version="ppo_v001", symbol="XAUUSDs"))
    registry.path.write_text("{ this is not json", encoding="utf-8")

    assert registry.records("XAUUSDs") == []
    assert (tmp_path / "model_registry.json.corrupt").exists()


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #
def test_folds_are_chronological_and_never_overlap_their_tests():
    frame = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=60 * 24 * 400, freq="1min", tz="UTC")
    })
    folds = build_folds(frame, train_days=100, validation_days=30, test_days=30)
    assert len(folds) >= 2

    for fold in folds:
        assert fold.train_start < fold.train_end < fold.validation_end < fold.test_end
    for earlier, later in zip(folds, folds[1:]):
        assert later.train_start > earlier.train_start, "the window did not slide"
        # test windows abut but never overlap: an overlap counts a period twice
        assert later.validation_end >= earlier.test_end


def test_a_fold_never_trains_on_its_own_test_data():
    frame = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=60 * 24 * 300, freq="1min", tz="UTC")
    })
    folds = build_folds(frame, train_days=100, validation_days=30, test_days=30)
    for fold in folds:
        train = slice_period(frame, fold.train_start, fold.train_end)
        test = slice_period(frame, fold.validation_end, fold.test_end)
        assert train["time"].max() < test["time"].min()


def test_aggregate_reports_the_spread_not_just_the_total():
    """One good fold beside several poor ones is noise, not an edge."""
    from ai.evaluation.walk_forward import FoldResult

    results = [
        FoldResult(fold=1, test_metrics={"net_r": 10.0, "trades": 50, "average_net_r": 0.2}),
        FoldResult(fold=2, test_metrics={"net_r": -8.0, "trades": 40, "average_net_r": -0.2}),
        FoldResult(fold=3, test_metrics={"net_r": -1.0, "trades": 30, "average_net_r": -0.03}),
    ]
    summary = aggregate(results)
    assert summary["folds"] == 3
    assert summary["positive_folds"] == 1
    assert summary["negative_folds"] == 2
    assert summary["std_fold_net_r"] > 0
    assert summary["worst_fold_net_r"] == -8.0


def test_a_skipped_fold_is_reported_not_hidden():
    from ai.evaluation.walk_forward import FoldResult

    results = [FoldResult(fold=1, skipped="no candles")]
    assert aggregate(results)["folds"] == 0


# --------------------------------------------------------------------------- #
# sealed holdout
# --------------------------------------------------------------------------- #
def test_a_fresh_seal_is_intact(tmp_path):
    seal = create_seal(tmp_path, "XAUUSDs", "2025-01-01", "2026-01-01")
    assert seal.intact is True
    assert seal.times_opened == 0


def test_opening_the_holdout_is_recorded_permanently(tmp_path):
    create_seal(tmp_path, "XAUUSDs", "2025-01-01", "2026-01-01")
    record_opening(tmp_path, "XAUUSDs", "ppo_v001", "final evaluation")

    seal = load_seal(tmp_path, "XAUUSDs")
    assert seal.intact is False
    assert seal.times_opened == 1
    assert seal.openings[0]["model_version"] == "ppo_v001"

    record_opening(tmp_path, "XAUUSDs", "ppo_v002", "second look")
    assert load_seal(tmp_path, "XAUUSDs").times_opened == 2


def test_moving_a_seal_after_the_fact_is_refused(tmp_path):
    """Redefining the boundary after seeing results is how a good holdout
    number gets manufactured."""
    create_seal(tmp_path, "XAUUSDs", "2025-01-01", "2026-01-01")
    with pytest.raises(ValueError, match="already sealed"):
        create_seal(tmp_path, "XAUUSDs", "2024-01-01", "2025-01-01")


def test_training_data_excludes_the_sealed_period(tmp_path):
    """Exclusion happens at the data layer, not by discipline."""
    frame = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=1000, freq="1h", tz="UTC")
    })
    seal = create_seal(tmp_path, "XAUUSDs", "2024-02-01", "2024-03-01")

    train = training_frame(frame, seal)
    held = holdout_frame(frame, seal)

    assert train["time"].max() < pd.Timestamp("2024-02-01", tz="UTC")
    assert held["time"].min() >= pd.Timestamp("2024-02-01", tz="UTC")
    assert not set(train["time"]) & set(held["time"]), "the seal leaked into training"


def test_the_two_markets_have_separate_seals(tmp_path):
    create_seal(tmp_path, "XAUUSDs", "2025-01-01", "2026-01-01")
    assert load_seal(tmp_path, "BTCUSDs") is None


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_metrics_lead_with_net_and_expose_the_cost_share():
    trades = pd.DataFrame({
        "net_r": [1.0, -1.0, 0.5, -0.2],
        "gross_r": [1.5, -0.6, 0.9, 0.2],
        "cost_r": [0.5, 0.4, 0.4, 0.4],
        "holding_candles": [3, 5, 2, 8],
        "result": ["TP3_HIT", "SL_HIT", "TP2_HIT", "TIMEOUT"],
        "tp_hits": [3, 0, 2, 1],
        "spread_points": [20, 20, 20, 20],
    })
    summary = summarise_trades(trades, "test")
    assert summary["trades"] == 4
    assert summary["net_r"] == 0.3
    assert summary["wins"] == 2 and summary["losses"] == 2
    assert summary["sl_rate"] == 25.0
    assert summary["timeout_rate"] == 25.0
    assert summary["cost_share"] > 0, "the cost share was not computed"


def test_metrics_handle_an_empty_log():
    summary = summarise_trades(pd.DataFrame(), "nothing")
    assert summary["trades"] == 0 and summary["net_r"] == 0.0


def test_the_equity_curve_tracks_drawdown():
    trades = pd.DataFrame({"net_r": [1.0, -0.5, -0.5, 2.0],
                           "exit_time": ["a", "b", "c", "d"]})
    curve = equity_curve(trades)
    assert list(curve["equity_r"]) == [1.0, 0.5, 0.0, 2.0]
    assert curve["drawdown_r"].max() == 1.0


# --------------------------------------------------------------------------- #
# shadow mode
# --------------------------------------------------------------------------- #
def _shadow(tmp_path, symbol="XAUUSDs"):
    from config import Config
    from ai.shadow import ShadowRecorder
    from tests.conftest import isolate

    config = isolate(Config().for_market(symbol), tmp_path, symbol)
    return config, ShadowRecorder(config, model_version="ppo_v001",
                                  env_config=EnvConfig())


def _candle(minute=0, close=2300.0, high=None, low=None, spread=20, atr=0.30):
    return {
        "time": f"2024-01-01T00:{minute:02d}:00Z",
        "open": close, "close": close,
        "high": close + 0.1 if high is None else high,
        "low": close - 0.1 if low is None else low,
        "spread": spread, "atr": atr,
    }


def _decision(action=BUY):
    names = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}
    return PPODecision(action=action, action_name=names[action],
                       probabilities={"HOLD": 0.2, "BUY": 0.7, "SELL": 0.1},
                       confidence=0.7)


def test_shadow_records_every_decision_including_hold(tmp_path):
    """Knowing when the agent declined is as informative as when it acted."""
    from src.signal_tracker import read_csv_rows

    _config, recorder = _shadow(tmp_path)
    recorder.record_decision(_decision(HOLD), _candle(0), "PPO_SHADOW")
    recorder.record_decision(_decision(BUY), _candle(1), "PPO_SHADOW")

    rows = read_csv_rows(recorder.decisions_path)
    assert len(rows) == 2
    assert [r["ppo_action"] for r in rows] == ["HOLD", "BUY"]
    assert rows[0]["model_version"] == "ppo_v001"


def test_shadow_never_places_an_order_only_simulates(tmp_path):
    _config, recorder = _shadow(tmp_path)
    trade = recorder.maybe_open(_decision(BUY), _candle(0))

    assert trade is not None
    assert recorder.summary()["simulated"] is True
    # nothing here can reach a broker: the recorder has no broker at all
    assert not hasattr(recorder, "broker")


def test_a_shadow_trade_stops_out_pessimistically(tmp_path):
    _config, recorder = _shadow(tmp_path)
    recorder.maybe_open(_decision(BUY), _candle(0, close=2300.0))
    # a candle that trades through both the stop and the targets
    closed = recorder.observe_candle(_candle(1, close=2300.0, high=2400.0, low=2200.0))

    assert closed is not None
    assert closed["result"] == "SL_HIT", "an ambiguous candle was scored optimistically"
    assert closed["net_r"] < 0


def test_a_shadow_trade_times_out(tmp_path):
    from config import Config
    from ai.shadow import ShadowRecorder
    from tests.conftest import isolate

    config = isolate(Config().for_market("XAUUSDs"), tmp_path, "XAUUSDs")
    recorder = ShadowRecorder(config, model_version="v1",
                              env_config=EnvConfig(max_holding_candles=3))
    recorder.maybe_open(_decision(BUY), _candle(0))

    # Flat candles that reach neither the stop nor a target, so the ONLY way
    # out is the holding window.  A wider candle would stop out first and the
    # test would pass for the wrong reason.
    closed = None
    for minute in range(1, 8):
        flat = _candle(minute, close=2300.12, high=2300.13, low=2300.11)
        closed = recorder.observe_candle(flat) or closed
    assert closed is not None and closed["result"] == "TIMEOUT"


def test_only_one_shadow_trade_runs_at_a_time(tmp_path):
    _config, recorder = _shadow(tmp_path)
    first = recorder.maybe_open(_decision(BUY), _candle(0))
    second = recorder.maybe_open(_decision(SELL), _candle(1))

    assert first is not None
    assert second is None, "a second simulated trade opened while one was live"


def test_shadow_results_are_kept_apart_from_the_other_books(tmp_path):
    """PPO simulated, rule paper and real demo fills are three separate books."""
    config, recorder = _shadow(tmp_path)
    recorder.maybe_open(_decision(BUY), _candle(0))
    recorder.observe_candle(_candle(1, close=2290.0, high=2290.1, low=2280.0))

    assert recorder.trades_path.exists()
    assert recorder.trades_path != config.outcomes_csv
    assert recorder.trades_path != config.executions_csv
    assert "ai" in recorder.trades_path.parts


def test_the_two_markets_shadow_independently(tmp_path):
    _gold_config, gold = _shadow(tmp_path, "XAUUSDs")
    _btc_config, btc = _shadow(tmp_path, "BTCUSDs")

    gold.maybe_open(_decision(BUY), _candle(0))
    assert gold.open_trade is not None
    assert btc.open_trade is None, "a gold shadow trade appeared on Bitcoin"
    assert gold.trades_path != btc.trades_path


# --------------------------------------------------------------------------- #
# training  (slow; skipped without the RL stack)
# --------------------------------------------------------------------------- #
@needs_sb3
def test_a_short_training_run_produces_every_checkpoint(tmp_path):
    from ai.ppo.train import PPOHyperParameters, train_ppo

    frame, spec = build_dataset(make_candles(3000, seed=21))
    train = frame.iloc[:2000].reset_index(drop=True)
    validation = frame.iloc[2000:].reset_index(drop=True)

    result = train_ppo(
        train, validation, spec, EnvConfig(episode_length=500),
        tmp_path / "ppo_v001",
        hyper=PPOHyperParameters(seed=7, n_steps=256, batch_size=64),
        total_timesteps=1024, eval_every=512,
        symbol="XAUUSDs", version="ppo_v001",
    )

    for name in ("best_model.zip", "latest.zip", "final_model.zip",
                 "feature_spec.json", "training_result.json"):
        assert (tmp_path / "ppo_v001" / name).exists(), name
    assert result.feature_fingerprint == spec.fingerprint()
    assert result.hyperparameters["seed"] == 7


@needs_sb3
def test_a_trained_model_round_trips_through_disk(tmp_path):
    """The saved model must load back with its feature contract intact."""
    from ai.ppo.train import PPOHyperParameters, train_ppo

    frame, spec = build_dataset(make_candles(2000, seed=23))
    train_ppo(
        frame.iloc[:1500], frame.iloc[1500:], spec, EnvConfig(episode_length=400),
        tmp_path / "ppo_v001",
        hyper=PPOHyperParameters(seed=5, n_steps=256, batch_size=64),
        total_timesteps=512, eval_every=512, symbol="XAUUSDs", version="ppo_v001",
    )

    policy = PPOPolicy.load(tmp_path / "ppo_v001", symbol="XAUUSDs", version="ppo_v001")
    assert policy is not None
    assert policy.spec.fingerprint() == spec.fingerprint()

    decision = policy.decide(frame.tail(1))
    assert decision.action in (HOLD, BUY, SELL)
    assert decision.ok is True


@needs_sb3
def test_training_is_reproducible_from_a_seed(tmp_path):
    """Two runs with the same seed must give the same validation result."""
    from ai.ppo.train import PPOHyperParameters, train_ppo

    frame, spec = build_dataset(make_candles(2000, seed=31))
    train, validation = frame.iloc[:1500], frame.iloc[1500:]

    def run(directory):
        return train_ppo(
            train, validation, spec,
            EnvConfig(episode_length=400, random_start=False),
            tmp_path / directory,
            hyper=PPOHyperParameters(seed=99, n_steps=256, batch_size=64),
            total_timesteps=512, eval_every=512,
            symbol="XAUUSDs", version=directory,
        )

    first, second = run("a"), run("b")
    assert first.validation_metrics.get("net_r") == second.validation_metrics.get("net_r")
    assert first.validation_metrics.get("trades") == second.validation_metrics.get("trades")


# --------------------------------------------------------------------------- #
# checkpoints  (spec section 19)
# --------------------------------------------------------------------------- #
def _fake_checkpoint(directory, names=("best_model", "latest", "final_model"),
                     spec=True, fingerprint="abc123"):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / f"{name}.zip").write_bytes(b"weights")
    if spec:
        FeatureSpec(columns=["f_a", "f_b"]).save(directory / "feature_spec.json")
        if fingerprint == "override":
            (directory / "feature_spec.json").write_text(
                json.dumps({"columns": ["f_z"], "clip": 10.0}), encoding="utf-8"
            )
    return directory


def test_a_complete_checkpoint_is_usable(tmp_path):
    from ai.ppo.checkpoints import inspect

    info = inspect(_fake_checkpoint(tmp_path / "ppo_v001"))
    assert info.usable is True
    assert set(info.available) == {"best_model", "latest", "final_model"}
    assert info.has_feature_spec is True


def test_weights_without_a_feature_spec_are_not_usable(tmp_path):
    """The model would still answer confidently, from the wrong columns."""
    from ai.ppo.checkpoints import inspect

    info = inspect(_fake_checkpoint(tmp_path / "ppo_v002", spec=False))
    assert info.usable is False


def test_a_missing_directory_reports_cleanly(tmp_path):
    from ai.ppo.checkpoints import inspect

    info = inspect(tmp_path / "nope")
    assert info.usable is False and info.available == []


def test_resume_uses_latest_not_best(tmp_path):
    """Resuming from `best` would discard whatever trained after it."""
    from ai.ppo.checkpoints import resume_from

    directory = _fake_checkpoint(tmp_path / "ppo_v003")
    assert resume_from(directory).name == "latest.zip"


def test_verify_catches_a_feature_fingerprint_mismatch(tmp_path):
    """Caught here it is a config problem; caught at inference it is silent."""
    from ai.ppo.checkpoints import verify

    directory = _fake_checkpoint(tmp_path / "ppo_v004")
    real = FeatureSpec(columns=["f_a", "f_b"]).fingerprint()

    assert verify(directory, expected_fingerprint=real)["ok"] is True
    bad = verify(directory, expected_fingerprint="totally-different")
    assert bad["ok"] is False
    assert any("fingerprint" in p for p in bad["problems"])


def test_archiving_moves_a_checkpoint_rather_than_deleting_it(tmp_path):
    """A model that produced a published number must stay reproducible."""
    from ai.ppo.checkpoints import archive

    directory = _fake_checkpoint(tmp_path / "ppo_v005")
    moved = archive(directory, reason="superseded")

    assert not directory.exists()
    assert moved.exists() and (moved / "best_model.zip").exists()
    assert (moved / "ARCHIVED.txt").exists()


def test_checkpoints_are_listed_per_market(tmp_path):
    from ai.ppo.checkpoints import list_checkpoints

    _fake_checkpoint(tmp_path / "ppo" / "XAUUSDs" / "ppo_v001")
    _fake_checkpoint(tmp_path / "ppo" / "XAUUSDs" / "ppo_v002")
    _fake_checkpoint(tmp_path / "ppo" / "BTCUSDs" / "ppo_v001")

    assert len(list_checkpoints(tmp_path, "XAUUSDs")) == 2
    assert len(list_checkpoints(tmp_path, "BTCUSDs")) == 1
    assert list_checkpoints(tmp_path, "ETHUSD") == []


# --------------------------------------------------------------------------- #
# reports  (spec sections 32, 33)
# --------------------------------------------------------------------------- #
def _trade_log():
    return pd.DataFrame({
        "entry_time": pd.date_range("2024-01-01 08:00", periods=6, freq="1h",
                                    tz="UTC").astype(str),
        "exit_time": pd.date_range("2024-01-01 08:05", periods=6, freq="1h",
                                   tz="UTC").astype(str),
        "net_r": [1.0, -1.0, 0.5, -0.5, 2.0, -0.2],
        "gross_r": [1.5, -0.6, 0.9, -0.1, 2.4, 0.2],
        "cost_r": [0.5, 0.4, 0.4, 0.4, 0.4, 0.4],
        "holding_candles": [3, 5, 2, 8, 4, 6],
        "result": ["TP3_HIT", "SL_HIT", "TP2_HIT", "TIMEOUT", "TP3_HIT", "SL_HIT"],
        "tp_hits": [3, 0, 2, 1, 3, 0],
        "spread_points": [20] * 6,
        "initial_risk": [0.2, 0.3, 0.2, 0.4, 0.2, 0.3],
    })


def test_the_report_writes_every_specified_artefact(tmp_path):
    from ai.evaluation.report import REPORT_FILES, write_report

    written = write_report(_trade_log(), tmp_path, label="test")
    for name in REPORT_FILES:
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "by_regime.csv").exists(), "spec section 33"
    assert (tmp_path / "performance.json").exists()
    assert set(written) >= {"performance", "equity", "trades", "drawdown", "walk_forward"}


def test_an_empty_trade_log_still_produces_a_report(tmp_path):
    """"The agent took no trades" is a result; a missing report is not."""
    from ai.evaluation.report import REPORT_FILES, write_report

    write_report(pd.DataFrame(), tmp_path, label="nothing")
    for name in REPORT_FILES:
        assert (tmp_path / name).exists(), name
    assert pd.read_csv(tmp_path / "performance.csv").iloc[0]["trades"] == 0


def test_the_drawdown_series_tracks_depth_and_length():
    from ai.evaluation.report import drawdown_series

    trades = pd.DataFrame({"net_r": [1.0, -0.5, -0.5, -0.5, 2.0],
                           "exit_time": list("abcde")})
    frame = drawdown_series(trades)

    assert list(frame["equity_r"]) == [1.0, 0.5, 0.0, -0.5, 1.5]
    assert frame["drawdown_r"].max() == 1.5
    # three consecutive trades below the peak, then a new high resets it
    assert list(frame["underwater_trades"]) == [0, 1, 2, 3, 0]


def test_regime_analysis_splits_by_session_and_volatility():
    """A single blended number hides an agent that works in one regime only."""
    from ai.evaluation.metrics import by_regime

    frame = by_regime(_trade_log())
    labels = set(frame["label"])
    assert any(label.startswith("session:") for label in labels)
    assert any(label.startswith("volatility:") for label in labels)


def test_the_comparison_table_puts_both_strategies_on_one_scale():
    from ai.evaluation.report import compare

    text = compare([
        {"label": "RULE_ONLY", "trades": 3, "net_r": -1.19, "average_net_r": -0.397,
         "win_rate": 33.3, "profit_factor": 0.37, "max_drawdown_r": 1.9,
         "average_holding": 4.0},
        {"label": "PPO ppo_v001", "trades": 0, "net_r": 0.0, "average_net_r": 0.0,
         "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown_r": 0.0,
         "average_holding": 0.0},
    ])
    assert "RULE_ONLY" in text and "PPO ppo_v001" in text
    assert "netR" in text and "maxDD" in text


def test_the_summary_leads_with_net_and_shows_the_cost_share():
    from ai.evaluation.metrics import summarise_trades
    from ai.evaluation.report import render_summary

    text = render_summary(summarise_trades(_trade_log(), "x"))
    assert "AFTER COSTS (the numbers that matter)" in text
    assert "BEFORE COSTS (reference only)" in text
    assert "Cost share" in text
    assert text.index("AFTER COSTS") < text.index("BEFORE COSTS"), "NET must lead"
