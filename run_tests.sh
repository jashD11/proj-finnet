#!/usr/bin/env bash
# Run all four test suites.
#
# They cannot share a single pytest invocation: each paper folder is deliberately
# self-contained and they have identically named subpackages (solver/, utils/,
# tests/, data/), so whichever is imported first shadows the others. Running them
# as separate processes is the fix, not a workaround.
#
# Usage:  ./run_tests.sh [python]     (default: .venv/bin/python, else python3)
set -u

PY="${1:-}"
if [ -z "$PY" ]; then
    if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi

# Resolve once: each suite runs from inside its own folder, so a relative path
# must become absolute and a bare command (python3) must be looked up on PATH.
case "$PY" in
    /*) ;;
    */*) PY="$(pwd)/$PY" ;;
    *) PY="$(command -v "$PY")" ;;
esac

echo "interpreter: $($PY --version 2>&1) at $PY"
echo

fail=0

echo "── paper1_gmrf_laplacian (solver assertion checks) ───────────────────────"
( cd paper1_gmrf_laplacian && "$PY" tests/test_solver.py ) || fail=1
echo

echo "── paper2_ticc (pytest) ──────────────────────────────────────────────────"
( cd paper2_ticc && "$PY" -m pytest tests/ -q ) || fail=1
echo

echo "── paper3_market_modularity (pytest) ─────────────────────────────────────"
( cd paper3_market_modularity && "$PY" -m pytest tests/ -q ) || fail=1
echo

echo "── modularity_graph_learning (pytest) ────────────────────────────────────"
( cd modularity_graph_learning && "$PY" -m pytest tests/ -q ) || fail=1
echo

if [ "$fail" -eq 0 ]; then
    echo "All four suites passed."
else
    echo "At least one suite FAILED."
fi
exit "$fail"
