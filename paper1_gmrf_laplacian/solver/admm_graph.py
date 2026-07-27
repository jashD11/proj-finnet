"""
GMRF Laplacian solver via gradient descent on edge weights.

Solves Problem 1 (base) and Problem 10 (Algorithm 1 L-step):
    min_w  (q + β)ᵀw - log gdet(L(w))  [+ degree penalty if requested]
    s.t.   w ≥ 0
where L(w) = Kᵀ diag(w) K is automatically a valid Laplacian.

K is the signed incidence matrix (m × p) for the complete graph on p nodes.
This parameterization guarantees: L1=0, L_ij≤0 (i≠j), L ≽ 0 for any w≥0.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize


def _build_incidence(p):
    """Return incidence matrix K (m × p) and list of edges for complete graph."""
    m = p * (p - 1) // 2
    K = np.zeros((m, p))
    edges = []
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            K[idx, i] = 1.0
            K[idx, j] = -1.0
            edges.append((i, j))
            idx += 1
    return K, edges


def log_gdet(L, tol=1e-10):
    """Log pseudo-determinant = sum of logs of strictly positive eigenvalues."""
    vals = eigh(L, eigvals_only=True)
    return float(np.sum(np.log(vals[vals > tol])))


def learn_laplacian(
    S,
    beta=1.0,
    eta=0.0,
    V=None,
    degree_control=False,
    rho_degree=100.0,
    w_prev=None,
    gamma=0.0,
    w0=None,
    max_iter=1000,
    tol=1e-6,
):
    """
    Learn a graph Laplacian from similarity matrix S.

    Problem 1  (degree_control=False):
        min_L  tr(L S) - log gdet(L) + β · Σ_{i<j} |L_ij|
        s.t.   L1=0, L_ij≤0 (i≠j), L ≽ 0

    Problem 10 (degree_control=True, eta>0, V given):
        min_L  tr(L (S + η VVᵀ)) - log gdet(L)
        s.t.   L1=0, L_ij≤0 (i≠j), diag(L)=1, L ≽ 0

    Temporal penalty (Algorithm 2, Problem 11): if w_prev is given, adds the
    EXACT Frobenius term γ‖L(w) − L(w_prev)‖²_F, which in edge-weight space is
        γ (2‖w − w_prev‖² + ‖K_sqᵀ(w − w_prev)‖²)
    (off-diagonals contribute twice, diagonals via the degree map) — a plain
    quadratic in w with closed-form gradient γ(4Δw + 2 K_sq K_sqᵀ Δw).

    Parameters
    ----------
    S              : (p, p) similarity matrix (use correlation, not covariance)
    beta           : sparsity regularizer (β in Problem 1; 0 for Problem 10)
    eta            : spectral penalty weight (Algorithm 1 L-step; 0 otherwise)
    V              : (p, k) eigenvectors for spectral penalty (Algorithm 1)
    degree_control : if True, adds quadratic penalty to enforce diag(L)≈1
    rho_degree     : penalty strength for degree constraint
    w_prev         : (m,) edge weights of the previous graph (Algorithm 2)
    gamma          : temporal penalty weight γ = δ/n_t (0 disables)
    w0             : (m,) initial point for L-BFGS-B (warm start); default 1/p
    max_iter       : max L-BFGS-B iterations
    tol            : convergence tolerance

    Returns
    -------
    L : (p, p) estimated graph Laplacian
    """
    p = S.shape[0]
    S_tilde = S.copy()
    if eta > 0.0 and V is not None:
        S_tilde = S + eta * (V @ V.T)

    K, edges = _build_incidence(p)
    m = len(edges)
    K_sq = K ** 2  # (m, p): K_sq[l, i] = 1 if node i is endpoint of edge l

    # Linear term: tr(L(w) S) = Σ_l w_l (S_ii + S_jj - 2 S_ij)
    q = np.array(
        [S_tilde[i, i] + S_tilde[j, j] - 2.0 * S_tilde[i, j] for i, j in edges]
    )
    q_eff = q + beta  # adds the ||w||_1 sparsity penalty (w_l ≥ 0 so |w_l|=w_l)

    _psd_tol = 1e-10

    def _obj_grad(w):
        L = (K.T * w) @ K  # K^T diag(w) K, shape (p, p)
        vals, vecs = eigh(L)
        pos = vals > _psd_tol
        if not np.any(pos):
            return 1e12, np.ones(m) * 1e6

        log_gdet_L = float(np.sum(np.log(vals[pos])))
        L_pinv = (vecs[:, pos] * (1.0 / vals[pos])) @ vecs[:, pos].T

        # d/dw_l [-log gdet(L)] = -(K L† Kᵀ)_ll
        KL = K @ L_pinv  # (m, p)
        KLK_diag = np.einsum("ij,ij->i", KL, K)  # diagonal of K L† Kᵀ, shape (m,)

        obj = float(q_eff @ w) - log_gdet_L
        grad = q_eff - KLK_diag

        if degree_control:
            # Penalty: (rho/2) ||diag(L) - 1||^2 = (rho/2) ||Dw - 1||^2
            # where D = K_sq^T, so d = K_sq^T w, grad_w = K_sq (d-1)
            d = K_sq.T @ w  # node degrees, shape (p,)
            diff = d - 1.0
            obj += 0.5 * rho_degree * float(np.dot(diff, diff))
            grad = grad + rho_degree * (K_sq @ diff)

        if w_prev is not None and gamma > 0.0:
            # γ‖L(w) − L(w_prev)‖²_F = γ(2‖Δw‖² + ‖K_sqᵀΔw‖²)
            dw = w - w_prev
            kd = K_sq.T @ dw  # (p,) degree differences
            obj += gamma * (2.0 * float(dw @ dw) + float(kd @ kd))
            grad = grad + gamma * (4.0 * dw + 2.0 * (K_sq @ kd))

        return obj, grad

    w_init = np.maximum(w0, 0.0) if w0 is not None else np.full(m, 1.0 / p)
    bounds = [(0.0, None)] * m

    res = minimize(
        _obj_grad,
        w_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": max_iter,
            "ftol": tol ** 2,
            "gtol": tol,
            "maxls": 50,
        },
    )

    w_opt = np.maximum(res.x, 0.0)
    return (K.T * w_opt) @ K
