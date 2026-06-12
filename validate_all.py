#!/usr/bin/env python3
"""
FIREKit ecosystem validation harness.

One command to answer "does everything still work?":

    Layer 1  environment   Python version + required third-party packages
    Layer 2  smoke         every product package imports and reports a version
    Layer 3  tests         every product's pytest suite
    Layer 4  demos         run_all.py (all demos + hub JSON schema + bundle)
    Layer 5  pipeline      pipeline.py --fast (cross-product integration)

Usage:
    python3 validate_all.py                  # all five layers
    python3 validate_all.py --skip-tests     # skip the pytest layer (faster)
    python3 validate_all.py --skip-demos     # skip demos + hub bundling
    python3 validate_all.py --skip-pipeline  # skip the integration pipeline
    python3 validate_all.py --only signalml,riskguard   # restrict layers 2-3

Exit code 0 when every executed check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PRODUCTS = [
    "vectorforge", "datastream", "alphalab", "signalml", "sentimentpulse",
    "deeptrader", "executioncore", "riskguard", "portfolioengine",
]

REQUIRED_PACKAGES = [
    "numpy", "pandas", "scipy", "sklearn", "pydantic", "pyarrow", "yaml", "pytest",
]

CHECKS: list[tuple[str, str, bool, float, str]] = []  # (layer, name, ok, secs, detail)


def record(layer: str, name: str, ok: bool, secs: float, detail: str = "") -> bool:
    CHECKS.append((layer, name, ok, secs, detail))
    mark = "PASS" if ok else "FAIL"
    extra = f"  ({detail})" if detail and not ok else ""
    print(f"  {mark}  {name:<20} {secs:6.1f}s{extra}")
    return ok


def run(cmd: list[str], cwd: Path, timeout: int = 600) -> tuple[bool, float, str]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - start, f"timeout after {timeout}s"
    secs = time.perf_counter() - start
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return False, secs, " | ".join(tail)
    return True, secs, ""


def layer_environment() -> None:
    print("\nLayer 1: environment")
    v = sys.version_info
    record("environment", "python>=3.11", (v.major, v.minor) >= (3, 11), 0.0,
           f"found {v.major}.{v.minor}.{v.micro}")
    for pkg in REQUIRED_PACKAGES:
        start = time.perf_counter()
        try:
            __import__(pkg)
            record("environment", pkg, True, time.perf_counter() - start)
        except ImportError as exc:
            record("environment", pkg, False, time.perf_counter() - start, str(exc))


def layer_smoke(targets: list[str]) -> None:
    print("\nLayer 2: smoke imports")
    for product in targets:
        code = f"import {product}; print({product}.__version__)"
        ok, secs, detail = run([sys.executable, "-c", code], cwd=ROOT / product, timeout=120)
        record("smoke", product, ok, secs, detail)


def layer_tests(targets: list[str]) -> None:
    print("\nLayer 3: unit tests")
    for product in targets:
        ok, secs, detail = run(
            [sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT / product
        )
        record("tests", product, ok, secs, detail)


def layer_demos() -> None:
    print("\nLayer 4: demos + hub bundle (run_all.py)")
    ok, secs, detail = run([sys.executable, "run_all.py"], cwd=ROOT)
    record("demos", "run_all.py", ok, secs, detail)


def layer_pipeline() -> None:
    print("\nLayer 5: cross-product pipeline (pipeline.py --fast)")
    ok, secs, detail = run([sys.executable, "pipeline.py", "--fast"], cwd=ROOT)
    record("pipeline", "pipeline.py", ok, secs, detail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-demos", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--only", default="", help="comma-separated product subset for layers 2-3")
    args = parser.parse_args()

    subset = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = subset - set(PRODUCTS)
    if unknown:
        print(f"unknown products: {sorted(unknown)}", file=sys.stderr)
        return 1
    targets = [p for p in PRODUCTS if not subset or p in subset]

    t0 = time.perf_counter()
    layer_environment()
    layer_smoke(targets)
    if not args.skip_tests:
        layer_tests(targets)
    if not args.skip_demos:
        layer_demos()
    if not args.skip_pipeline:
        layer_pipeline()

    failures = [(layer, name, detail) for layer, name, ok, _, detail in CHECKS if not ok]
    n_pass = sum(1 for _, _, ok, _, _ in CHECKS if ok)
    print(f"\n{'=' * 60}")
    print(f"{n_pass}/{len(CHECKS)} checks passed in {time.perf_counter() - t0:.0f}s")
    if failures:
        print("\nFAILED:")
        for layer, name, detail in failures:
            print(f"  [{layer}] {name}: {detail}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
