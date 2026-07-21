"""
Sanity tests for the TICC implementation. Run: pytest paper2_ticc/tests/

Covers the three subcomponents:
  1. Toeplitz graphical-lasso ADMM  -> valid (symmetric PD block-Toeplitz) Theta
     that recovers a planted sparse support.
  2. Dynamic-programming assignment  -> matches brute-force optimum on a toy trellis.
  3. End-to-end TICC                 -> high macro-F1 on a simple 1,2,1 sequence.
"""

import os
import sys
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.toeplitz_admm import solve_toeplitz_graphical_lasso, _toeplitz_groups
from solver.dp_assign import assign_points
from data.synthetic import generate_series, random_toeplitz_theta
from solver.ticc import fit_ticc, stack_windows
from utils.metrics import macro_f1, _edge_support


N, W = 5, 5


def _is_block_toeplitz(M, n, w, tol=1e-8):
    for a in range(w):
        for b in range(w):
            m = b - a
            ref_a = 0 if m >= 0 else -m
            ref_b = m if m >= 0 else 0
            blk = M[a * n:(a + 1) * n, b * n:(b + 1) * n]
            ref = M[ref_a * n:(ref_a + 1) * n, ref_b * n:(ref_b + 1) * n]
            if not np.allclose(blk, ref, atol=tol):
                return False
    return True


def test_admm_valid_and_recovers_support():
    rng = np.random.default_rng(0)
    Theta_true = random_toeplitz_theta(N, W, rng)
    Sigma = np.linalg.inv(Theta_true)
    # empirical covariance from many samples of the length-w window
    X = rng.multivariate_normal(np.zeros(N * W), Sigma, size=8000)
    S = X.T @ X / X.shape[0]

    gid, pen = _toeplitz_groups(N, W)
    Theta = solve_toeplitz_graphical_lasso(S, lam=0.07, n=N, w=W, gid=gid, penalize=pen)

    assert np.allclose(Theta, Theta.T, atol=1e-6), "Theta must be symmetric"
    assert np.linalg.eigvalsh(Theta).min() > 0, "Theta must be positive definite"
    assert _is_block_toeplitz(Theta, N, W), "Theta must be block-Toeplitz"

    from sklearn.metrics import f1_score
    f1 = f1_score(_edge_support(Theta_true, N, W), _edge_support(Theta, N, W),
                  zero_division=0)
    assert f1 > 0.7, f"support recovery F1 too low: {f1:.3f}"


def test_dp_matches_bruteforce():
    rng = np.random.default_rng(1)
    T, K = 7, 3
    nll = rng.uniform(0, 5, size=(T, K))
    beta = 1.5

    labels = assign_points(nll, beta)
    dp_cost = nll[np.arange(T), labels].sum() + beta * np.sum(labels[1:] != labels[:-1])

    best = np.inf
    for path in product(range(K), repeat=T):
        p = np.array(path)
        c = nll[np.arange(T), p].sum() + beta * np.sum(p[1:] != p[:-1])
        best = min(best, c)
    assert abs(dp_cost - best) < 1e-9, f"DP {dp_cost} != brute force {best}"


def test_end_to_end_high_f1():
    n, w = N, W
    X, true_labels, _ = generate_series([1, 2, 1], seg_len=300, n=n, w=w, seed=3)
    labels, thetas, _ = fit_ticc(X, K=2, w=w, lam=0.07, beta=60, n_restarts=15, seed=42)
    f1 = macro_f1(true_labels[w - 1:], labels)
    assert f1 > 0.85, f"end-to-end macro-F1 too low: {f1:.3f}"


if __name__ == "__main__":
    test_admm_valid_and_recovers_support()
    test_dp_matches_bruteforce()
    test_end_to_end_high_f1()
    print("all tests passed")
