"""
Sanity checks for the GMRF Laplacian solver.

Generates a synthetic p=10 path-graph GMRF, samples n=500 observations,
computes the sample correlation, runs learn_laplacian, then asserts:
  - P1: L1 ≈ 0  (row sums are zero)
  - P2: L_ij ≤ 0 for all i ≠ j
  - PSD: all eigenvalues ≥ 0
  - rank(L) = p-1  (exactly one zero eigenvalue → connected graph)
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from solver.admm_graph import learn_laplacian


def _path_laplacian(p):
    """Laplacian of the path graph 0-1-2-...(p-1)."""
    L = np.zeros((p, p))
    for i in range(p - 1):
        L[i, i] += 1.0
        L[i + 1, i + 1] += 1.0
        L[i, i + 1] -= 1.0
        L[i + 1, i] -= 1.0
    return L


def _sample_gmrf(L_true, n, eps=0.01, seed=42):
    """Sample n observations from a GMRF with Laplacian L_true."""
    rng = np.random.default_rng(seed)
    Sigma = np.linalg.inv(L_true + eps * np.eye(L_true.shape[0]))
    return rng.multivariate_normal(np.zeros(L_true.shape[0]), Sigma, size=n)


def _sample_correlation(X):
    """Sample correlation matrix from data matrix X (n × p)."""
    X_c = X - X.mean(axis=0)
    std = X_c.std(axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    X_s = X_c / std
    S = (X_s.T @ X_s) / X_s.shape[0]
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    return S


def test_p1_row_sums(L, tol=1e-8):
    p = L.shape[0]
    row_sums = L @ np.ones(p)
    max_err = np.max(np.abs(row_sums))
    assert max_err < tol, f"P1 failed: max |L·1| = {max_err:.2e}"
    print(f"  [P1] row-sum max |L·1| = {max_err:.2e}  ✓")


def test_p2_off_diag(L, tol=1e-10):
    p = L.shape[0]
    mask = ~np.eye(p, dtype=bool)
    max_off = L[mask].max()
    assert max_off <= tol, f"P2 failed: max off-diag L_ij = {max_off:.2e}"
    print(f"  [P2] max off-diag = {max_off:.2e}  ✓")


def test_psd(L, tol=-1e-8):
    vals = np.linalg.eigvalsh(L)
    min_eig = vals.min()
    assert min_eig >= tol, f"PSD failed: min eigenvalue = {min_eig:.2e}"
    print(f"  [PSD] min eigenvalue = {min_eig:.2e}  ✓")
    return vals


def test_rank(vals, p, tol=1e-6):
    n_zero = int(np.sum(vals < tol))
    assert n_zero == 1, (
        f"Rank check failed: {n_zero} near-zero eigenvalues (expected 1). "
        f"Eigenvalues: {vals}"
    )
    print(f"  [rank] zero eigenvalues (< {tol}) = {n_zero}  ✓  (rank = {p-1})")


def run():
    p, n = 10, 500
    print(f"Synthetic GMRF test: p={p} nodes, n={n} samples")

    L_true = _path_laplacian(p)
    X = _sample_gmrf(L_true, n)
    S = _sample_correlation(X)

    print(f"  Running learn_laplacian (beta=0.5) …")
    L = learn_laplacian(S, beta=0.5, max_iter=2000, tol=1e-7)

    test_p1_row_sums(L)
    test_p2_off_diag(L)
    vals = test_psd(L)
    test_rank(vals, p)

    off_mask = ~np.eye(p, dtype=bool)
    n_edges = int((L[off_mask] < -1e-8).sum()) // 2
    print(f"  Positive-weight edges: {n_edges} (out of {p*(p-1)//2} possible)")
    print(f"\n[PASS] All assertions passed.")


if __name__ == "__main__":
    run()
