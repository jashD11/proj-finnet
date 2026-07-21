"""
TICC: Toeplitz Inverse Covariance-Based Clustering  (paper Algorithm 2).

Alternating minimisation (EM-style):
    M-step: for each cluster, fit a block-Toeplitz inverse covariance to its
            assigned windows via the Toeplitz graphical lasso (ADMM).
    E-step: reassign every window to a cluster by dynamic programming, trading off
            log-likelihood against a temporal-consistency (switching) penalty beta.
Repeat until the assignments stop changing.

Initialisation note. The clusters are zero-mean and differ only in second-order
structure, so a Euclidean init (K-means/GMM on the raw windows) cannot see them and
EM gets trapped for K >= 3. We instead run several restarts from random
*contiguous-segment* labelings and keep the one with the lowest penalised objective
    sum_t nll(x_t, Theta_{z_t}) + beta * #switches,
which the experiments confirm reliably recovers the global optimum (the true
segmentation is the lowest-objective fixed point).
"""

import numpy as np

from .toeplitz_admm import solve_toeplitz_graphical_lasso, _toeplitz_groups
from .dp_assign import gaussian_neg_ll, assign_points


def stack_windows(X, w):
    """
    Turn an (T, n) series into (T-w+1, n*w) look-back windows: row t holds
    [X[t-w+1], ..., X[t]] flattened. Window ending at time t carries label t, so
    the stacked rows correspond to original times w-1, w, ..., T-1.
    """
    T, n = X.shape
    M = T - w + 1
    out = np.empty((M, n * w))
    for r in range(M):
        out[r] = X[r:r + w].reshape(-1)
    return out


def _random_contiguous_init(M, K, rng):
    """Random labeling built from random contiguous segments (structure-agnostic seed)."""
    n_seg = int(rng.integers(2 * K, 6 * K + 1))
    n_bp = min(n_seg - 1, M - 1)
    bps = np.sort(rng.choice(np.arange(1, M), size=n_bp, replace=False))
    init = np.empty(M, dtype=int)
    for seg in np.split(np.arange(M), bps):
        init[seg] = int(rng.integers(0, K))
    return init


def _em_once(Xst, K, n, w, lam, beta, gid, penalize, init,
             max_iter, rho, admm_iter, admm_tol):
    """Run EM to convergence from one initial labeling. Returns (labels, thetas, objective)."""
    M = Xst.shape[0]
    S_global = Xst.T @ Xst / M
    min_pts = n * w + 1
    labels = init.copy()
    thetas = [None] * K
    prev = None
    nll = None

    for _ in range(max_iter):
        # M-step
        for k in range(K):
            idx = labels == k
            if idx.sum() >= min_pts:
                Sk = Xst[idx].T @ Xst[idx] / idx.sum()
            elif thetas[k] is not None:
                continue
            else:
                Sk = S_global
            thetas[k] = solve_toeplitz_graphical_lasso(
                Sk, lam, n, w, rho=rho, max_iter=admm_iter, tol=admm_tol,
                gid=gid, penalize=penalize,
            )
        # E-step
        nll = gaussian_neg_ll(Xst, thetas)
        labels = assign_points(nll, beta)
        if prev is not None and np.array_equal(labels, prev):
            break
        prev = labels.copy()

    obj = nll[np.arange(M), labels].sum() + beta * np.sum(labels[1:] != labels[:-1])
    return labels, thetas, float(obj)


def fit_ticc(
    X, K, w, lam, beta,
    n_restarts=20, max_iter=30, rho=1.0, admm_iter=1000, admm_tol=1e-4,
    seed=42, verbose=False,
):
    """
    Fit TICC to a single multivariate time series.

    Parameters
    ----------
    X : (T, n) time series.
    K : number of clusters.
    w : window size.
    lam : Toeplitz-graphical-lasso l1 strength.
    beta : temporal-consistency switching penalty (beta=0 -> no consistency).
    n_restarts : random-contiguous-init restarts; the best (lowest objective) is kept.

    Returns
    -------
    labels : (T-w+1,) cluster label per window (window ending at time t = index t).
    thetas : list of K block-Toeplitz inverse covariances (nw x nw).
    Xst    : (T-w+1, n*w) stacked windows (returned for downstream metrics).
    """
    T, n = X.shape
    Xst = stack_windows(X, w)
    M = Xst.shape[0]
    gid, penalize = _toeplitz_groups(n, w)
    rng = np.random.default_rng(seed)

    best = None
    for r in range(n_restarts):
        init = _random_contiguous_init(M, K, rng)
        labels, thetas, obj = _em_once(
            Xst, K, n, w, lam, beta, gid, penalize, init,
            max_iter, rho, admm_iter, admm_tol,
        )
        if verbose:
            print(f"  restart {r:2d}  objective={obj:.1f}")
        if best is None or obj < best[2]:
            best = (labels, thetas, obj)

    return best[0], best[1], Xst
