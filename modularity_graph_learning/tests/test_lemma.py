"""
Part I lemma of the proposal, checked on random graphs.

Under diag(L) = 1 (every degree one): B = I - L - 11'/p, so on 1-perp the
spectrum of B is 1 - spec(L). Then
  (a) rank(L) = p - k  <=>  B has eigenvalue 1 with multiplicity k - 1;
  (b) if lambda_k(L) <= 1: sum of k smallest eig(L) = (k-1) - sum of k-1 largest eig(B);
  (c) lambda_max(B) = max(0, 1 - lambda_2(L)).
Plus the counterexample for (b), the Part III identity and the normalised case.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import laplacian, modularity_matrix, unit_degree  # noqa: E402

TOL = 1e-9


def random_unit_degree(p, rng, density=0.6):
    A = rng.random((p, p)) * (rng.random((p, p)) < density)
    A = np.triu(A, 1)
    A = A + A.T
    A += 1e-3 * (np.ones((p, p)) - np.eye(p))  # connected
    return unit_degree(A)


def block_unit_degree(sizes, rng):
    p = sum(sizes)
    W = np.zeros((p, p))
    start = 0
    for n in sizes:
        W[start:start + n, start:start + n] = random_unit_degree(n, rng)
        start += n
    return W


def eig(M):
    return np.linalg.eigvalsh((M + M.T) / 2)


@pytest.mark.parametrize("seed", range(5))
def test_B_is_reflected_L_on_unit_degree(seed):
    rng = np.random.default_rng(seed)
    p = 12
    W = random_unit_degree(p, rng)
    L, B = laplacian(W), modularity_matrix(W)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(B, np.eye(p) - L - np.ones((p, p)) / p, atol=TOL)
    lam = eig(L)
    expected = np.sort(np.concatenate([[0.0], 1.0 - lam[1:]]))
    assert np.allclose(eig(B), expected, atol=TOL)


@pytest.mark.parametrize("sizes", [[5, 4, 3], [6, 6], [4, 3, 3, 2], [10]])
def test_a_rank_iff_multiplicity(sizes):
    rng = np.random.default_rng(1)
    W = block_unit_degree(sizes, rng)
    L, B = laplacian(W), modularity_matrix(W)
    k = len(sizes)
    p = W.shape[0]
    assert np.sum(eig(L) > 1e-9) == p - k
    assert np.sum(np.abs(eig(B) - 1.0) < 1e-9) == k - 1


@pytest.mark.parametrize("sizes", [[5, 4, 3], [6, 6], [4, 3, 3, 2]])
def test_b_fan_identity_when_lambda_k_le_1(sizes):
    rng = np.random.default_rng(2)
    W = block_unit_degree(sizes, rng)
    L, B = laplacian(W), modularity_matrix(W)
    k = len(sizes)
    lam, mu = eig(L), eig(B)[::-1]
    assert lam[k - 1] <= 1.0
    assert np.isclose(lam[:k].sum(), (k - 1) - mu[:k - 1].sum(), atol=TOL)


def test_b_fails_on_unit_degree_complete_graph():
    """K_p scaled to unit degree: lambda_2 = p/(p-1) > 1, and (b) breaks for k = 2."""
    p, k = 8, 2
    W = (np.ones((p, p)) - np.eye(p)) / (p - 1)
    L, B = laplacian(W), modularity_matrix(W)
    lam, mu = eig(L), eig(B)[::-1]
    assert np.isclose(lam[1], p / (p - 1))
    lhs = lam[:k].sum()                    # p/(p-1)
    rhs = (k - 1) - mu[:k - 1].sum()       # 1 - 0
    assert np.isclose(lhs, p / (p - 1)) and np.isclose(rhs, 1.0)
    assert not np.isclose(lhs, rhs)


@pytest.mark.parametrize("seed", range(5))
def test_c_lambda_max_B(seed):
    rng = np.random.default_rng(seed)
    two_blocks = block_unit_degree([5, 5], rng)
    cross = 0.05 * np.triu(rng.random(two_blocks.shape), 1)
    noise = 0.1 * np.triu(rng.random((10, 10)), 1)
    near_complete = unit_degree(np.ones((10, 10)) - np.eye(10) + noise + noise.T)
    for W, censored in ((near_complete, True),                   # lambda_2 > 1: max clips to 0
                        (unit_degree(two_blocks + cross + cross.T), False)):  # lambda_2 < 1
        assert np.allclose(W.sum(axis=1), 1.0, atol=1e-12)
        lam2 = eig(laplacian(W))[1]
        assert (lam2 > 1.0) == censored
        assert np.isclose(eig(modularity_matrix(W))[-1], max(0.0, 1.0 - lam2), atol=TOL)


def test_modularity_prior_identity_heterogeneous_degrees():
    """-tr(V'BV) = tr(V'LV) - sum_i d_i ||v_i - c_d||^2 for any V and any degrees."""
    rng = np.random.default_rng(3)
    p, r = 15, 2
    A = np.triu(rng.random((p, p)) * 3, 1)
    W = A + A.T
    d = W.sum(axis=1)
    V, _ = np.linalg.qr(rng.standard_normal((p, r)))
    lhs = -np.trace(V.T @ modularity_matrix(W) @ V)
    c_d = (d @ V) / d.sum()
    rhs = np.trace(V.T @ laplacian(W) @ V) - np.sum(d * np.sum((V - c_d) ** 2, axis=1))
    assert np.isclose(lhs, rhs, atol=1e-10)


def test_normalised_case_contains_unit_degree_case():
    """B_n = D^-1/2 B D^-1/2 = I - L_sym - uu', u = sqrt(d / 2m); u = 1/sqrt(p) when D = I."""
    rng = np.random.default_rng(4)
    p = 11
    A = np.triu(rng.random((p, p)), 1)
    W = A + A.T
    d = W.sum(axis=1)
    Dm = np.diag(d ** -0.5)
    u = np.sqrt(d / d.sum())
    Lsym = np.eye(p) - Dm @ W @ Dm
    assert np.allclose(Dm @ modularity_matrix(W) @ Dm, np.eye(p) - Lsym - np.outer(u, u), atol=TOL)
    Wu = random_unit_degree(p, rng)
    assert np.allclose(np.sqrt(Wu.sum(axis=1) / Wu.sum()), 1 / np.sqrt(p), atol=1e-10)
