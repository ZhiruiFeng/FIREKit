"""Smoke tests for the demo payload (schema contract)."""

import math

import pytest

from alphalab.demo import run_pipeline


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_pipeline()


def _walk_numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_numbers(v)
    elif isinstance(obj, float):
        yield obj


def test_payload_required_keys(payload: dict) -> None:
    required = {
        "schema_version", "product", "title", "tagline", "version",
        "generated_at", "status", "summary_metrics", "charts", "tables", "notes",
    }
    assert required <= set(payload)
    assert payload["schema_version"] == 1
    assert payload["product"] == "alphalab"
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"


def test_payload_metrics(payload: dict) -> None:
    assert 4 <= len(payload["summary_metrics"]) <= 8
    metrics = {m["label"]: m["value"] for m in payload["summary_metrics"]}
    assert metrics["Factors evaluated"] == "20"
    assert "50 symbols" in metrics["Universe"]


def test_payload_charts_shape(payload: dict) -> None:
    ids = {c["id"] for c in payload["charts"]}
    assert {"best_factor_ic", "quantile_returns"} <= ids
    for chart in payload["charts"]:
        assert chart["type"] in {"line", "bar", "scatter"}
        assert 0 < len(chart["x"]) <= 500
        assert 1 <= len(chart["series"]) <= 5
        for series in chart["series"]:
            assert len(series["data"]) == len(chart["x"])


def test_payload_tables(payload: dict) -> None:
    tables = {t["id"]: t for t in payload["tables"]}
    top = tables["top_factors"]
    assert len(top["rows"]) == 10
    assert len(top["columns"]) == len(top["rows"][0])
    assert tables["correlated_pairs"]["rows"]


def test_payload_no_nan(payload: dict) -> None:
    for value in _walk_numbers(payload):
        assert math.isfinite(value)
