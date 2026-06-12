"""Tests for the FactorZoo registry and batch evaluation."""

import numpy as np
import pandas as pd
import pytest

from alphalab.factors import Factor, Momentum, ShortTermReversal
from alphalab.panel import Panel
from alphalab.zoo import FactorZoo


def test_register_and_lookup() -> None:
    zoo = FactorZoo()
    factor = zoo.register(Momentum(21))
    assert len(zoo) == 1
    assert "momentum_21d" in zoo
    assert zoo["momentum_21d"] is factor
    assert zoo.names == ["momentum_21d"]


def test_register_duplicate_raises() -> None:
    zoo = FactorZoo([Momentum(21)])
    with pytest.raises(ValueError, match="already registered"):
        zoo.register(Momentum(21))


def test_register_type_checked() -> None:
    zoo = FactorZoo()
    with pytest.raises(TypeError):
        zoo.register("not a factor")  # type: ignore[arg-type]


def test_default_zoo_has_full_library() -> None:
    zoo = FactorZoo.default()
    assert len(zoo) >= 15
    assert "alpha_101" in zoo
    assert "momentum_21d" in zoo


def test_empty_zoo_evaluate_raises(synthetic_panel: Panel) -> None:
    with pytest.raises(ValueError, match="empty"):
        FactorZoo().evaluate(synthetic_panel)


def test_compute_returns_all_factors(synthetic_panel: Panel) -> None:
    zoo = FactorZoo([Momentum(21), ShortTermReversal(5)])
    values = zoo.compute(synthetic_panel)
    assert set(values) == {"momentum_21d", "reversal_5d"}
    for frame in values.values():
        assert frame.shape == synthetic_panel.shape


def test_evaluate_ranks_by_abs_rank_ic_ir(synthetic_panel: Panel) -> None:
    report = FactorZoo.default().evaluate(synthetic_panel, horizon=5)
    irs = [abs(ev.rank_ic.ir) for ev in report.evaluations if pd.notna(ev.rank_ic.ir)]
    assert irs == sorted(irs, reverse=True)
    assert report.best is report.evaluations[0]
    assert report.horizon == 5


def test_evaluate_report_contents(synthetic_panel: Panel) -> None:
    zoo = FactorZoo.default()
    report = zoo.evaluate(synthetic_panel, horizon=5)
    assert len(report.evaluations) == len(zoo)
    # correlation matrix covers every factor
    assert sorted(report.correlation.columns) == sorted(zoo.names)
    assert np.allclose(np.diag(report.correlation), 1.0)
    # summary frame
    frame = report.to_frame()
    assert len(frame) == len(zoo)
    assert {"ic_mean", "rank_ic_ir", "spread_mean", "turnover"} <= set(frame.columns)
    assert frame["n_obs"].min() > 0


def test_synthetic_panel_momentum_and_reversal_are_real(synthetic_panel: Panel) -> None:
    """The generator embeds momentum and reversal: both should score positive rank IC."""
    report = FactorZoo([Momentum(63), ShortTermReversal(5)]).evaluate(
        synthetic_panel, horizon=5
    )
    by_name = {ev.name: ev for ev in report.evaluations}
    assert by_name["momentum_63d"].rank_ic.mean > 0
    assert by_name["reversal_5d"].rank_ic.mean > 0


def test_top_correlated_pairs(synthetic_panel: Panel) -> None:
    report = FactorZoo.default().evaluate(synthetic_panel, horizon=5)
    pairs = report.top_correlated_pairs(3)
    assert len(pairs) == 3
    rhos = [abs(rho) for _, _, rho in pairs]
    assert rhos == sorted(rhos, reverse=True)
    for a, b, rho in pairs:
        assert a != b
        assert -1.0 <= rho <= 1.0


def test_custom_factor_in_zoo(synthetic_panel: Panel) -> None:
    class Noise(Factor):
        name = "noise"
        category = "test"

        def _compute(self, panel: Panel) -> pd.DataFrame:
            rng = np.random.default_rng(0)
            return pd.DataFrame(
                rng.normal(size=panel.shape),
                index=panel.dates,
                columns=panel.symbols,
            )

    report = FactorZoo([Noise(), Momentum(63)]).evaluate(synthetic_panel, horizon=5)
    by_name = {ev.name: ev for ev in report.evaluations}
    assert abs(by_name["noise"].rank_ic.mean) < abs(by_name["momentum_63d"].rank_ic.mean)
