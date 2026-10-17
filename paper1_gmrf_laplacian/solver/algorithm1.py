"""
Algorithm 1 — k-component GMRF Laplacian estimator (Section 4.1).

Alternating minimisation:
  V-step: V ← k eigenvectors of L for the k SMALLEST eigenvalues
  L-step: learn_laplacian(S + η V Vᵀ, degree_control=True)

Converges when ||L_new - L_old||_F / (||L_old||_F + 1) < tol.

Initialisation matters. The paper's Algorithm 1 takes an "initial estimate L0".
The uniform complete graph is NOT a usable L0: its nonzero eigenvalue p/(p-1)
has multiplicity p-1, so the first V-step returns an arbitrary basis of that
eigenspace and the components that form are arbitrary too. Measured at eta=300:
on the old 1,005-day window, components of 77/12/8 stocks (ARI 0.012 against
sectors); on the paper's 753-day window, 96 stocks plus one ISOLATED node (degree
0.0, ARI -0.009), the very failure degree control is meant to prevent. The
default L0 is therefore the Problem 1 estimate (beta = 0), whose bottom
eigenvectors already carry the cluster structure: on the 753-day window the
components are 52/28/17, exactly the sectors (ARI 1.0).
"""

import numpy as np
from scipy.linalg import eigh

from solver.laplacian_solver import learn_laplacian


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
    L0=None,
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
    L0         : (p, p) initial Laplacian; default is the Problem 1 estimate
                 learn_laplacian(S, beta=0) (see module docstring)

    Returns
    -------
    L : (p, p) estimated k-component graph Laplacian
    """
    p = S.shape[0]
    np.random.seed(42)

    # Initialise from the Problem 1 estimate, not the uniform complete graph,
    # whose degenerate spectrum makes the first V-step arbitrary.
    L = L0 if L0 is not None else learn_laplacian(S, beta=0.0, max_iter=max_inner, tol=inner_tol)

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
