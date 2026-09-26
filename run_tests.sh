#!/usr/bin/env bash
# Run all three test suites.
#
# They cannot share a single pytest invocation: each paper folder is deliberately
# self-contained and they have identically named subpackages (solver/, utils/,
# tests/, data/), so whichever is imported first shadows the others. Running them
# as three processes is the fix, not a workaround.
#
# Usage:  ./run_tests.sh [python]     (default: .venv/bin/python, else python3)
set -u

PY="${1:-}"
if [ -z "$PY" ]; then
    if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi

echo "interpreter: $($PY --version 2>&1) at $PY"
echo

fail=0

echo "── paper1_gmrf_laplacian (solver assertion checks) ───────────────────────"
( cd paper1_gmrf_laplacian && "../$PY" tests/test_solver.py ) || fail=1
echo

echo "── paper2_ticc (pytest) ──────────────────────────────────────────────────"
( cd paper2_ticc && "../$PY" -m pytest tests/ -q ) || fail=1
echo

echo "── paper3_market_modularity (pytest) ─────────────────────────────────────"
( cd paper3_market_modularity && "../$PY" -m pytest tests/ -q ) || fail=1
echo

if [ "$fail" -eq 0 ]; then
    echo "All three suites passed."
else
    echo "At least one suite FAILED."
fi
exit "$fail"
