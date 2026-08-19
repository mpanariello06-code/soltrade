"""Weighted aggregation, confidence bands and the individual engines' bounds."""

from __future__ import annotations

import pytest

from src.liquidity import MAX_SCORE as LIQUIDITY_MAX, analyze_liquidity
from src.momentum import MAX_SCORE as MOMENTUM_MAX, analyze_momentum
from src.price_action import MAX_SCORE as PA_MAX, analyze_price_action
from src.regime import REGIMES, classify_regime
from src.scoring import compute_scorecard, confidence_band, reason_summary
from src.structure import MAX_SCORE as STRUCTURE_MAX, analyze_structure
from src.support_resistance import MAX_SCORE as SR_MAX, analyze_support_resistance
from src.trend import HTF_MAX_SCORE, MAX_SCORE as TREND_MAX, analyze_htf, analyze_trend, htf_alignment
from src.utils import ComponentScore
from src.volatility import MAX_SCORE as VOL_MAX, analyze_volatility
from src.volume import MAX_SCORE as VOLUME_MAX, analyze_volume
from src.market_data import resample_candles
from src.indicators import compute_indicators
from tests.conftest import make_candles


def _full_components(bull_fraction: float, bear_fraction: float) -> dict:
    """Build a component set where every engine scores the given fractions."""
    maxima = {
        "trend": TREND_MAX, "htf": HTF_MAX_SCORE, "momentum": MOMENTUM_MAX,
        "structure": STRUCTURE_MAX, "liquidity": LIQUIDITY_MAX,
        "support_resistance": SR_MAX, "volume": VOLUME_MAX,
        "volatility": VOL_MAX, "price_action": PA_MAX,
    }
    return {
        name: ComponentScore(name, maximum * bull_fraction, maximum * bear_fraction, maximum)
        for name, maximum in maxima.items()
    }


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_weights_sum_to_one_hundred(config):
    assert config.weights.total() == pytest.approx(100.0)


def test_all_components_at_maximum_scores_one_hundred(config):
    card = compute_scorecard(_full_components(1.0, 0.0), config)
    assert card.bullish_score == pytest.approx(100.0)
    assert card.bearish_score == pytest.approx(0.0)
    assert card.direction == "BUY"


def test_all_components_at_zero_scores_zero(config):
    card = compute_scorecard(_full_components(0.0, 0.0), config)
    assert card.bullish_score == 0.0 and card.bearish_score == 0.0
    assert card.direction == "NONE"


def test_scores_scale_linearly_with_component_fill(config):
    card = compute_scorecard(_full_components(0.5, 0.0), config)
    assert card.bullish_score == pytest.approx(50.0)


def test_component_contribution_equals_its_configured_weight(config):
    card = compute_scorecard(_full_components(1.0, 0.0), config)
    assert card.component_score("trend", "BUY") == pytest.approx(config.weights.trend)
    assert card.component_score("volume", "BUY") == pytest.approx(config.weights.volume)


def test_reweighting_changes_the_score(config):
    config.weights.trend = 30.0
    config.weights.momentum = 5.0
    config.validate()
    card = compute_scorecard(
        {
            "trend": ComponentScore("trend", TREND_MAX, 0.0, TREND_MAX),
            "momentum": ComponentScore("momentum", 0.0, 0.0, MOMENTUM_MAX),
        },
        config,
    )
    assert card.bullish_score == pytest.approx(30.0)


def test_component_score_is_clamped_to_its_maximum():
    component = ComponentScore("trend", 999.0, -5.0, 20.0)
    assert component.bull == 20.0
    assert component.bear == 0.0
    assert component.normalised() == (1.0, 0.0)


def test_separation_is_the_absolute_gap(config):
    card = compute_scorecard(_full_components(0.9, 0.3), config)
    assert card.separation == pytest.approx(abs(card.bullish_score - card.bearish_score))


def test_confirmation_count_and_flags(config):
    card = compute_scorecard(_full_components(1.0, 0.0), config)
    assert card.confirmation_count("BUY") == 9
    assert card.confirmation_count("SELL") == 0
    assert all(card.confirmations("BUY").values())


def test_reason_summary_lists_the_strongest_components(config):
    card = compute_scorecard(_full_components(1.0, 0.0), config)
    summary = reason_summary(card, "BUY")
    assert "trend" in summary
    assert summary.count(";") <= 3


# --------------------------------------------------------------------------- #
# confidence bands
# --------------------------------------------------------------------------- #
def test_confidence_bands_are_ordered_and_exhaustive(config):
    bands = sorted(config.confidence_bands, key=lambda item: item[0])
    lowest = bands[0][0]
    assert confidence_band(lowest - 0.01, config) == "NO_SIGNAL"
    assert confidence_band(lowest, config) == bands[0][1]
    assert confidence_band(100.0, config) == bands[-1][1]


def test_confidence_band_matches_configuration(config):
    for minimum, label in config.confidence_bands:
        assert confidence_band(minimum, config) == label
        assert confidence_band(minimum + 0.5, config) == label


# --------------------------------------------------------------------------- #
# engines stay inside their declared ranges
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_every_engine_stays_within_its_maximum(config, seed):
    frame = compute_indicators(make_candles(1200, seed=seed), config.indicators)
    engines = (
        (analyze_trend(frame, config), TREND_MAX),
        (analyze_momentum(frame, config), MOMENTUM_MAX),
        (analyze_structure(frame, config), STRUCTURE_MAX),
        (analyze_liquidity(frame, config), LIQUIDITY_MAX),
        (analyze_support_resistance(frame, config), SR_MAX),
        (analyze_volume(frame, config), VOLUME_MAX),
        (analyze_volatility(frame, config), VOL_MAX),
        (analyze_price_action(frame, config), PA_MAX),
    )
    for component, maximum in engines:
        assert 0.0 <= component.bull <= maximum, component.name
        assert 0.0 <= component.bear <= maximum, component.name


def test_volatility_is_direction_neutral(config):
    frame = compute_indicators(make_candles(800, seed=21), config.indicators)
    component = analyze_volatility(frame, config)
    assert component.bull == component.bear


def test_engines_return_zero_on_insufficient_data(config):
    """Below their minimum history the engines must abstain, not guess."""
    frame = compute_indicators(make_candles(20, seed=23), config.indicators)
    for engine in (analyze_structure, analyze_liquidity, analyze_support_resistance):
        component = engine(frame, config)
        assert component.bull == 0.0 and component.bear == 0.0, component.name
        assert "insufficient" in str(component.details.get("reason", "")).lower()


def test_htf_alignment_classification(config):
    aligned = ComponentScore("htf", 10, 2, 15, {"h1_direction": "BULL", "m15_direction": "BULL"})
    counter = ComponentScore("htf", 2, 10, 15, {"h1_direction": "BEAR", "m15_direction": "BEAR"})
    neutral = ComponentScore("htf", 5, 5, 15, {"h1_direction": "NEUTRAL", "m15_direction": "NEUTRAL"})
    assert htf_alignment(aligned, "BUY") == "ALIGNED"
    assert htf_alignment(counter, "BUY") == "COUNTER"
    assert htf_alignment(neutral, "BUY") == "NEUTRAL"
    assert htf_alignment(counter, "SELL") == "ALIGNED"


def test_htf_engine_blends_both_timeframes(config):
    candles = make_candles(4000, seed=27, drift=0.06)
    m15 = compute_indicators(resample_candles(candles, "M5", "M15"), config.indicators)
    h1 = compute_indicators(resample_candles(candles, "M5", "H1"), config.indicators)
    component = analyze_htf(m15, h1, config)
    assert 0.0 <= component.bull <= HTF_MAX_SCORE
    assert "h1_direction" in component.details and "m15_direction" in component.details


# --------------------------------------------------------------------------- #
# regime
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [31, 33, 35])
def test_regime_is_always_a_known_label(config, seed):
    frame = compute_indicators(make_candles(1500, seed=seed), config.indicators)
    components = {
        "volatility": analyze_volatility(frame, config),
        "structure": analyze_structure(frame, config),
        "support_resistance": analyze_support_resistance(frame, config),
    }
    regime, details = classify_regime(frame, components, config)
    assert regime in REGIMES
    assert "adx" in details


def test_extreme_volatility_forces_high_volatility_regime(config, enriched):
    components = {
        "volatility": ComponentScore("volatility", 0, 0, 5, {"band": "EXTREME"}),
        "structure": ComponentScore("structure", 0, 0, 15, {}),
        "support_resistance": ComponentScore("support_resistance", 0, 0, 10, {}),
    }
    regime, _ = classify_regime(enriched, components, config)
    assert regime == "HIGH_VOLATILITY"


def test_every_regime_has_a_configured_threshold_offset(config):
    for regime in REGIMES:
        assert regime in config.regime_threshold_offsets, regime


def test_inapplicable_component_releases_its_weight(config):
    """H4 has no confirmation timeframe; the score must still reach 100."""
    components = _full_components(1.0, 0.0)
    components["htf"] = ComponentScore(
        "htf", 0.0, 0.0, HTF_MAX_SCORE, {"reason": "no confirmation timeframe"}, applicable=False
    )
    card = compute_scorecard(components, config)
    assert card.bullish_score == pytest.approx(100.0)
    assert card.excluded_components == ("htf",)
    assert card.weight_scale > 1.0
    assert card.component_score("htf", "BUY") == 0.0
    assert card.confirmations("BUY")["htf"] is False
