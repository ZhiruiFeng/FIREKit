"""Smoke tests for the demo payload (schema contract)."""

import math

import pytest

from datastream.demo import run_pipeline


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
    assert payload["product"] == "datastream"
    assert payload["status"] == "ok"
    assert payload["version"] == "0.1.0"


def test_payload_metrics_nonempty(payload: dict) -> None:
    assert 4 <= len(payload["summary_metrics"]) <= 8
    for metric in payload["summary_metrics"]:
        assert metric["label"]
        assert str(metric["value"])


def test_payload_charts_shape(payload: dict) -> None:
    assert payload["charts"]
    for chart in payload["charts"]:
        assert chart["type"] in {"line", "bar", "scatter"}
        assert len(chart["x"]) <= 500
        assert 1 <= len(chart["series"]) <= 5
        for series in chart["series"]:
            assert len(series["data"]) == len(chart["x"])


def test_payload_no_nan(payload: dict) -> None:
    for value in _walk_numbers(payload):
        assert math.isfinite(value)


def test_payload_demo_scale(payload: dict) -> None:
    metrics = {m["label"]: m["value"] for m in payload["summary_metrics"]}
    assert metrics["Symbols ingested"] == "50"
    assert int(metrics["Rows ingested"].replace(",", "")) > 30_000
    assert int(metrics["Issues found"]) > 0
    assert 0 < float(metrics["Quality score"]) <= 100
