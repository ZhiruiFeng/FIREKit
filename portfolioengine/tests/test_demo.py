"""Smoke tests for the demo payload (hub schema v1)."""

import json
import math

import pytest

from portfolioengine.demo import build_payload


@pytest.fixture(scope="module")
def payload():
    return build_payload()


def _walk_numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_numbers(v)
    elif isinstance(obj, float):
        yield obj


def test_payload_schema_fields(payload):
    assert payload["schema_version"] == 1
    assert payload["product"] == "portfolioengine"
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert 4 <= len(payload["summary_metrics"]) <= 8
    assert payload["charts"] and payload["tables"]


def test_payload_serializes_without_nan(payload):
    text = json.dumps(payload, allow_nan=False)
    assert len(text) > 1000
    for num in _walk_numbers(payload):
        assert math.isfinite(num)


def test_charts_respect_limits(payload):
    for chart in payload["charts"]:
        assert chart["type"] in {"line", "bar", "scatter"}
        assert len(chart["x"]) <= 500
        assert 1 <= len(chart["series"]) <= 5
        for series in chart["series"]:
            assert len(series["data"]) == len(chart["x"])


def test_comparison_table_has_all_six_methods(payload):
    table = next(t for t in payload["tables"] if t["id"] == "optimizer_comparison")
    methods = {row[0] for row in table["rows"]}
    assert methods == {
        "EqualWeight",
        "InverseVol",
        "MinVariance",
        "MaxSharpe",
        "RiskParity",
        "HRP",
    }
    n_cols = len(table["columns"])
    assert all(len(row) == n_cols for row in table["rows"])


def test_frontier_chart_has_max_sharpe_point(payload):
    chart = next(c for c in payload["charts"] if c["id"] == "efficient_frontier")
    ms = next(s for s in chart["series"] if s["name"] == "Max Sharpe")
    non_null = [v for v in ms["data"] if v is not None]
    assert len(non_null) == 1


def test_weights_chart_columns_sum_to_100(payload):
    chart = next(c for c in payload["charts"] if c["id"] == "weights")
    for series in chart["series"]:
        total = sum(v for v in series["data"] if v is not None)
        assert total == pytest.approx(100.0, abs=0.5)
