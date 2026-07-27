"""
Algorithm 2 — Time-varying graph estimation (Section 5, Fig 3).

Causal rolling-window estimator with temporal-consistency penalty δ:

    min  Σ_t n_t [tr(S_t L_t) - log gdet(L_t)]  +  δ Σ_t ||L_t - L_{t-1}||²_F

Solved by a sequential greedy update: at each step t, solve (dividing by n_t)
    min  tr(S_t L) - log gdet(L)  +  (δ/n_t) ||L - L_{t-1}||²_F
degree_control must be False for Algorithm 2 (degree control is Algorithm 1 only).

The Frobenius temporal penalty is implemented EXACTLY in edge-weight space by
learn_laplacian (w_prev/gamma parameters): it is a plain quadratic in w with a
closed-form gradient. Each window also warm-starts L-BFGS-B from the previous
window's weights (w0=w_prev).

History (dev_log §18): an earlier version approximated the penalty by
augmenting the similarity matrix with S_aug = S_t + (2δ/n_t)·L_prev. The sign
is inverted — the cross-term of the Frobenius expansion is −2δ·tr(L·L_prev) —
so that augmentation *repelled* each window from the previous graph and shrank
all edge weights, collapsing the λ₂ scale (65/243 windows exactly 0). Do not
reintroduce it.
"""

import numpy as np
from scipy.linalg import eigh

from solver.admm_graph import learn_laplacian, _build_incidence
from utils.preprocessing import build_similarity


def _laplacian_to_weights(L, K, m):
    """
    Project L back to edge weights via least-squares (K^T diag(w) K ≈ L).
    Used to warm-start the next window.
    Returns w (m,) with w >= 0.
    """
    # For complete-graph K, the weight of edge (i,j) is -L[i,j]
    p = L.shape[0]
    K_arr, edges = _build_incidence(p)
    w = np.array([-L[i, j] for i, j in edges])
    return np.maximum(w, 0.0)


def algorithm2(
    returns,
    window=30,
    k=1,
    eta=0.0,
    beta=0.0,
    delta=100.0,
    degree_control=False,
    rho_degree=100.0,
    max_inner=500,
    inner_tol=1e-5,
    verbose=False,
):
    """
    Time-varying graph estimation from a return matrix.

    Parameters
    ----------
    returns        : (T, p) log-return matrix, rows = days, cols = assets
    window         : rolling window length in days (paper uses 30)
    k              : spectral penalty components (k=1 = no spectral penalty)
    eta            : spectral penalty weight
    beta           : sparsity regulariser (Problem 1; 0 for time-varying)
    delta          : temporal smoothness weight (paper uses 100)
    degree_control : if True, enforce diag(L)=1 (Algorithm 1 only; False for Alg 2)
    rho_degree     : quadratic penalty strength for degree constraint
    max_inner      : L-BFGS-B iterations per window
    inner_tol      : solver tolerance
    verbose        : print progress

    Returns
    -------
    graphs : list of (p, p) Laplacian arrays, one per valid window
             (length = T - window + 1, causal: graph[t] uses returns[t:t+window])
    dates  : corresponding integer indices of the last day of each window
    """
    T, p = returns.shape
    K, edges = _build_incidence(p)
    m = len(edges)

    graphs = []
    dates = []

    L_prev = None

    for t in range(window - 1, T):
        window_ret = returns[t - window + 1 : t + 1]  # (window, p) — causal
        S_t = build_similarity(window_ret, use_correlation=True)

        # Exact temporal penalty (δ/n_t)·||L - L_prev||²_F, handled inside
        # learn_laplacian in edge-weight space; warm-start from w_prev.
        if L_prev is not None:
            w_prev = _laplacian_to_weights(L_prev, K, m)
            gamma = delta / window
        else:
            w_prev = None
            gamma = 0.0

        # V-step: eigenvectors of L_prev (or of S_t for the first window)
        if L_prev is not None:
            vals, vecs = eigh(L_prev)
        else:
            vals, vecs = eigh(S_t)
        V = vecs[:, :k]  # k smallest eigenvectors

        L_t = learn_laplacian(
            S_t,
            beta=beta,
            eta=eta,
            V=V,
            degree_control=degree_control,
            rho_degree=rho_degree,
            w_prev=w_prev,
            gamma=gamma,
            w0=w_prev,
            max_iter=max_inner,
            tol=inner_tol,
        )

        graphs.append(L_t)
        dates.append(t)
        L_prev = L_t

        if verbose and (t - window + 1) % 20 == 0:
            from utils.graph_metrics import algebraic_connectivity
            lam2 = algebraic_connectivity(L_t)
            print(f"  window ending t={t}: λ₂ = {lam2:.4f}")

    return graphs, dates
