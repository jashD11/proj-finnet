"""
Part II identities, checked on random graphs.

  tr(LS) = tr(L JSJ) for every S       (the Laplacian only sees JSJ)
  tr(L bb') = b'Lb = sum_{i<j} W_ij (b_i - b_j)^2
  sum_{i<j} (b_i - b_j)^2 = p ||Jb||^2 = p ||b||^2 sin^2(theta)
plus the synthetic generator's beta patterns and the explicit-null inputs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import laplacian  # noqa: E402
from part2 import (market_mode_removed, residual_corr, sin2theta,  # noqa: E402
                   synthetic_betas, synthetic_returns)


def random_graph(p, rng):
    A = np.triu(rng.random((p, p)), 1)
    return A + A.T


@pytest.mark.parametrize("seed", range(3))
def test_laplacian_only_sees_JSJ(seed):
    rng = np.random.default_rng(seed)
    p = 12
    L = laplacian(random_graph(p, rng))
    M = rng.standard_normal((p, p))
    S = M @ M.T
    J = np.eye(p) - np.ones((p, p)) / p
    assert np.isclose(np.trace(L @ S), np.trace(L @ J @ S @ J))


@pytest.mark.parametrize("seed", range(3))
def test_market_term_is_beta_smoothness(seed):
    rng = np.random.default_rng(seed)
    p = 10
    W = random_graph(p, rng)
    b = 1 + 0.4 * rng.standard_normal(p)
    iu = np.triu_indices(p, 1)
    pairwise = np.sum(W[iu] * (b[iu[0]] - b[iu[1]]) ** 2)
    assert np.isclose(np.trace(laplacian(W) @ np.outer(b, b)), pairwise)
    assert np.isclose(b @ laplacian(W) @ b, pairwise)
    J = np.eye(p) - np.ones((p, p)) / p
    total = np.sum((b[iu[0]] - b[iu[1]]) ** 2)
    assert np.isclose(total, p * np.sum((J @ b) ** 2))
    assert np.isclose(total, p * (b @ b) * sin2theta(b))


def test_equal_betas_leave_nothing_behind():
    p = 8
    W = random_graph(p, np.random.default_rng(0))
    b = 1.3 * np.ones(p)
    assert np.isclose(b @ laplacian(W) @ b, 0.0)
    assert np.isclose(sin2theta(b), 0.0)


def test_synthetic_beta_patterns():
    b, sector = synthetic_betas("aligned", 0.5)
    assert np.allclose(np.unique(b), [0.5, 1.0, 1.5])
    assert all(len(np.unique(b[sector == s])) == 1 for s in range(3))   # one beta per sector
    b, sector = synthetic_betas("cross", 0.5)
    for s in range(3):
        assert sorted(np.unique(b[sector == s])) == [0.5, 1.5]            # both betas in every sector
        assert np.sum(b[sector == s] > 1) == np.sum(b[sector == s] < 1)
    b, _ = synthetic_betas("equal", 0.0)
    assert np.allclose(b, 1.0)


def test_explicit_nulls_are_correlation_matrices():
    X, f, _, _ = synthetic_returns("cross", 0.6, 0)
    S = np.corrcoef(X.T)
    for M in (market_mode_removed(S), residual_corr(X, f)):
        assert np.allclose(np.diag(M), 1.0)
        assert np.allclose(M, M.T)
        assert np.all(np.abs(M[np.triu_indices(len(M), 1)]) < 1)
