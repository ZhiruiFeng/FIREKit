"""Smoke tests for the demo payload (hub schema v1)."""

import json
import math

import pytest

from riskguard.demo import build_payload, downsample, make_strategy_returns


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
    assert payload["product"] == "riskguard"
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"
    assert len(payload["summary_metrics"]) >= 4
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


def test_tables_are_rectangular(payload):
    for table in payload["tables"]:
        n_cols = len(table["columns"])
        assert table["rows"], table["id"]
        for row in table["rows"]:
            assert len(row) == n_cols


def test_demo_is_deterministic():
    a = make_strategy_returns()
    b = make_strategy_returns()
    assert a.equals(b)


def test_downsample_keeps_endpoints():
    s = make_strategy_returns()
    d = downsample(s, max_points=500)
    assert len(d) <= 501
    assert d.index[0] == s.index[0]
    assert d.index[-1] == s.index[-1]


def test_crash_is_present_in_strategy():
    s = make_strategy_returns()
    crash_window = s.iloc[580:620]
    assert crash_window.mean() < -0.005
