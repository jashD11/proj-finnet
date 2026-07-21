"""
Toeplitz graphical lasso via ADMM  (TICC paper, Problem 4 / eqs 5-9).

Solves, for one cluster:
    minimize_{Theta}  -log det Theta + tr(S Theta) + ||lam o Theta||_1
    subject to        Theta is block-Toeplitz (w x w grid of n x n blocks)

Theta is the (n*w) x (n*w) inverse covariance of a length-w window of an
n-dimensional time series. Block-Toeplitz means block (a, b) depends only on the
offset m = b - a, so the free parameters are the blocks A(0), A(1), ..., A(w-1)
(with A(0) symmetric and A(m<0) = A(-m)^T by the symmetry of Theta).

ADMM (Boyd et al. 2011), with the consensus split Theta = Z, Z in T:
    Theta-update : closed form via eigendecomposition of rho(Z-U) - S   (eq 6)
    Z-update     : block-Toeplitz projection (group-average) + soft-threshold (eq 9)
    U-update     : U += Theta - Z
"""

import numpy as np
from scipy.linalg import eigh


def _toeplitz_groups(n, w):
    """
    Build, once, the mapping from each (nw x nw) matrix position to the logical
    block-Toeplitz parameter that governs it.

    Returns
    -------
    gid : (nw, nw) int array
        gid[x, y] = index of the shared parameter for position (x, y). All
        positions with the same id are constrained equal (that is what enforces
        both block-Toeplitz structure and the symmetry of Theta).
    penalize : (n_groups,) bool array
        True where the l1 penalty applies (everything except the main diagonal,
        i.e. the variance terms, which are never penalised in a graphical lasso).
    """
    N = n * w
    gid = np.full((N, N), -1, dtype=int)
    groups = {}
    penalize = []

    def key(m, i, j):
        # A(0) is symmetric -> canonicalise (i, j) so (i,j) and (j,i) share a param
        if m == 0:
            i, j = min(i, j), max(i, j)
        return (m, i, j)

    for a in range(w):
        for b in range(w):
            m = b - a
            for i in range(n):
                for j in range(n):
                    x, y = a * n + i, b * n + j
                    # block (a, b) with b < a equals A(-m)^T -> logical (-m, j, i)
                    k = key(m, i, j) if m >= 0 else key(-m, j, i)
                    if k not in groups:
                        groups[k] = len(groups)
                        is_main_diag = (k[0] == 0 and k[1] == k[2])
                        penalize.append(not is_main_diag)
                    gid[x, y] = groups[k]
    return gid, np.array(penalize, dtype=bool)


def _project_toeplitz_soft(M, gid, penalize, thr):
    """Z-update: average M over each Toeplitz group, then soft-threshold (eq 9)."""
    flat = gid.ravel()
    n_groups = penalize.size
    sums = np.bincount(flat, weights=M.ravel(), minlength=n_groups)
    counts = np.bincount(flat, minlength=n_groups)
    means = sums / counts
    out = means.copy()
    p = penalize
    out[p] = np.sign(means[p]) * np.maximum(np.abs(means[p]) - thr, 0.0)
    return out[gid]


def solve_toeplitz_graphical_lasso(
    S, lam, n, w, rho=1.0, max_iter=1000, tol=1e-4, gid=None, penalize=None
):
    """
    Solve the Toeplitz graphical lasso for a single cluster.

    Parameters
    ----------
    S : (nw, nw) empirical covariance of the stacked windows in the cluster.
    lam : float, l1 regularisation strength.
    n, w : block dimension and window size (so nw = n*w).
    rho : ADMM penalty parameter.
    gid, penalize : optional precomputed outputs of `_toeplitz_groups(n, w)`
        (pass them in to avoid rebuilding across ADMM calls / clusters).

    Returns
    -------
    Theta : (nw, nw) exactly block-Toeplitz, symmetric, positive-definite inverse
        covariance. This is the consensus variable Z (the structured, sparse iterate);
        at convergence it equals the smooth update to numerical precision, and a tiny
        diagonal shift is applied if needed to guarantee positive-definiteness.
    """
    N = n * w
    if gid is None or penalize is None:
        gid, penalize = _toeplitz_groups(n, w)

    Z = np.eye(N)
    U = np.zeros((N, N))
    thr = lam / rho

    for _ in range(max_iter):
        # Theta-update (eq 6): analytical prox of -log det
        M = rho * (Z - U) - S
        M = 0.5 * (M + M.T)                     # symmetrise for numerical safety
        d, Q = eigh(M)
        d_theta = (d + np.sqrt(d ** 2 + 4.0 * rho)) / (2.0 * rho)
        Theta = (Q * d_theta) @ Q.T

        # Z-update (eqs 7-9): block-Toeplitz projection + soft-threshold
        Z_old = Z
        Z = _project_toeplitz_soft(Theta + U, gid, penalize, thr)

        # U-update
        U = U + Theta - Z

        # stopping: primal + dual residuals (Boyd sec. 3.3)
        r = np.linalg.norm(Theta - Z)
        s = rho * np.linalg.norm(Z - Z_old)
        eps = tol * N
        if r < eps and s < eps:
            break

    # Z is exactly block-Toeplitz & symmetric; ensure positive-definiteness (adding
    # to the diagonal keeps it block-Toeplitz) so the Gaussian likelihood is valid.
    min_eig = np.linalg.eigvalsh(Z).min()
    if min_eig < 1e-6:
        Z = Z + (1e-6 - min_eig) * np.eye(N)
    return Z
