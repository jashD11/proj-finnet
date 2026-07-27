"""
Algorithm 1 — k-component GMRF Laplacian estimator (Section 4.1).

Alternating minimisation:
  V-step: V ← k eigenvectors of L for the k SMALLEST eigenvalues
  L-step: learn_laplacian(S + η V Vᵀ, degree_control=True)

Converges when ||L_new - L_old||_F / (||L_old||_F + 1) < tol.
"""

import numpy as np
from scipy.linalg import eigh

from solver.admm_graph import learn_laplacian


def algorithm1(
    S,
    k=3,
    beta=10.0,
    eta=10.0,
    rho_degree=100.0,
    max_outer=50,
    tol=1e-4,
    max_inner=1000,
    inner_tol=1e-6,
):
    """
    Learn a k-component graph Laplacian from similarity matrix S.

    Parameters
    ----------
    S          : (p, p) similarity / correlation matrix
    k          : number of connected components to enforce
    beta       : sparsity penalty (set 0 when degree_control=True per paper)
    eta        : spectral penalty weight for V-step augmentation
    rho_degree : quadratic penalty strength for diag(L)=1 constraint
    max_outer  : max alternating iterations
    tol        : outer convergence tolerance (relative Frobenius change)
    max_inner  : max L-BFGS-B iterations inside learn_laplacian
    inner_tol  : tolerance passed to learn_laplacian

    Returns
    -------
    L : (p, p) estimated k-component graph Laplacian
    """
    p = S.shape[0]
    np.random.seed(42)

    # Initialise L as the identity-like valid Laplacian (degree=1 for all nodes)
    L = np.eye(p) * (p - 1) / p - np.ones((p, p)) / p

    for iteration in range(max_outer):
        # V-step: k eigenvectors for k smallest eigenvalues of L
        vals, vecs = eigh(L)
        V = vecs[:, :k]  # (p, k)

        # L-step: solve Problem 10
        L_new = learn_laplacian(
            S,
            beta=0.0,          # degree_control mode uses no sparsity penalty
            eta=eta,
            V=V,
            degree_control=True,
            rho_degree=rho_degree,
            max_iter=max_inner,
            tol=inner_tol,
        )

        # Check convergence
        rel_change = np.linalg.norm(L_new - L, "fro") / (np.linalg.norm(L, "fro") + 1.0)
        L = L_new

        if rel_change < tol:
            break

    return L
