"""Tests for the exposure limit engine."""

import pytest

from riskguard import ExposureLimits, LimitEngine


def gross(weights):
    return sum(abs(w) for w in weights.values())


def test_compliant_book_unchanged():
    engine = LimitEngine(ExposureLimits(max_position=0.25, max_gross=1.0, max_net=1.0))
    proposed = {"A": 0.20, "B": 0.20, "C": -0.10}
    result = engine.apply(proposed)
    assert result.compliant
    assert result.weights == proposed
    assert result.violations == []


def test_per_position_cap_clips_and_reports():
    engine = LimitEngine(ExposureLimits(max_position=0.25, max_gross=2.0, max_net=2.0))
    result = engine.apply({"A": 0.50, "B": 0.30, "C": -0.40})
    assert result.weights["A"] == pytest.approx(0.25)
    assert result.weights["B"] == pytest.approx(0.25)
    assert result.weights["C"] == pytest.approx(-0.25)  # sign preserved
    assert len(result.violations) == 3
    assert all(v.rule == "max_position" for v in result.violations)


def test_gross_cap_rescales_proportionally():
    engine = LimitEngine(ExposureLimits(max_position=None, max_gross=1.0, max_net=None))
    result = engine.apply({"A": 0.80, "B": 0.80})
    assert gross(result.weights) == pytest.approx(1.0)
    # Proportional scaling preserves relative sizes.
    assert result.weights["A"] == pytest.approx(result.weights["B"])
    assert result.violations[0].rule == "max_gross"


def test_net_cap_applies_to_absolute_net():
    engine = LimitEngine(ExposureLimits(max_position=None, max_gross=None, max_net=0.5))
    result = engine.apply({"A": -0.6, "B": -0.4})
    assert sum(result.weights.values()) == pytest.approx(-0.5)
    assert result.violations[0].rule == "max_net"


def test_sector_cap_scales_only_offending_sector():
    limits = ExposureLimits(max_position=None, max_gross=None, max_net=None, max_sector=0.30)
    engine = LimitEngine(limits)
    sectors = {"A": "tech", "B": "tech", "C": "energy"}
    result = engine.apply({"A": 0.30, "B": 0.20, "C": 0.10}, sectors=sectors)
    # Tech gross 0.50 -> scaled to 0.30; energy untouched.
    assert result.weights["A"] == pytest.approx(0.18)
    assert result.weights["B"] == pytest.approx(0.12)
    assert result.weights["C"] == pytest.approx(0.10)
    assert len(result.violations) == 1
    assert result.violations[0].rule == "max_sector"
    assert result.violations[0].scope == "tech"


def test_sector_cap_ignored_without_sector_map():
    limits = ExposureLimits(max_position=None, max_gross=None, max_net=None, max_sector=0.30)
    engine = LimitEngine(limits)
    result = engine.apply({"A": 0.50, "B": 0.50})
    assert result.compliant


def test_cascade_position_then_gross():
    engine = LimitEngine(ExposureLimits(max_position=0.40, max_gross=1.0, max_net=2.0))
    result = engine.apply({"A": 0.60, "B": 0.60, "C": 0.60})
    # After clipping: 0.40 each, gross 1.20 -> scaled by 1/1.2.
    for asset in "ABC":
        assert result.weights[asset] == pytest.approx(0.40 / 1.2)
    rules = [v.rule for v in result.violations]
    assert rules.count("max_position") == 3
    assert rules.count("max_gross") == 1


def test_final_weights_satisfy_all_limits():
    limits = ExposureLimits(max_position=0.15, max_gross=1.0, max_net=0.6, max_sector=0.35)
    engine = LimitEngine(limits)
    proposed = {
        "A": 0.30,
        "B": 0.25,
        "C": 0.20,
        "D": -0.50,
        "E": 0.40,
    }
    sectors = {"A": "s1", "B": "s1", "C": "s2", "D": "s2", "E": "s3"}
    result = engine.apply(proposed, sectors=sectors)
    w = result.weights
    assert all(abs(x) <= 0.15 + 1e-9 for x in w.values())
    assert gross(w) <= 1.0 + 1e-9
    assert abs(sum(w.values())) <= 0.6 + 1e-9
    for sector in {"s1", "s2", "s3"}:
        sector_gross = sum(abs(x) for a, x in w.items() if sectors[a] == sector)
        assert sector_gross <= 0.35 + 1e-9


def test_violation_records_original_value():
    engine = LimitEngine(ExposureLimits(max_position=0.10, max_gross=None, max_net=None))
    result = engine.apply({"A": 0.42})
    v = result.violations[0]
    assert v.value == pytest.approx(0.42)
    assert v.limit == pytest.approx(0.10)
    assert v.scope == "A"
    assert "clipped" in v.action


def test_limit_result_exposure_properties():
    engine = LimitEngine(ExposureLimits(max_position=None, max_gross=None, max_net=None))
    result = engine.apply({"A": 0.5, "B": -0.2})
    assert result.gross_exposure == pytest.approx(0.7)
    assert result.net_exposure == pytest.approx(0.3)


def test_disabled_limits_pass_everything():
    engine = LimitEngine(
        ExposureLimits(max_position=None, max_gross=None, max_net=None, max_sector=None)
    )
    result = engine.apply({"A": 5.0, "B": -3.0})
    assert result.compliant
    assert result.weights == {"A": 5.0, "B": -3.0}


def test_limits_validation():
    with pytest.raises(ValueError):
        ExposureLimits(max_position=0.0)
    with pytest.raises(ValueError):
        ExposureLimits(max_gross=-1.0)
