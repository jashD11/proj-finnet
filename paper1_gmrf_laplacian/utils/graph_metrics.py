"""
Graph-level metrics extracted from a Laplacian matrix.
"""

import numpy as np
from scipy.linalg import eigh


def eigenvalues(L):
    """Return eigenvalues of L in ascending order."""
    return eigh(L, eigvals_only=True)


def algebraic_connectivity(L, tol=1e-8):
    """
    Second smallest eigenvalue λ₂(L) — the Fiedler value.
    Measures how well-connected the graph is (0 = disconnected).
    """
    vals = eigh(L, eigvals_only=True)
    return float(vals[1]) if vals[1] > tol else 0.0


def effective_rank(L, tol=1e-8):
    """Number of positive eigenvalues = p - number_of_components."""
    vals = eigh(L, eigvals_only=True)
    return int(np.sum(vals > tol))


def is_psd(L, tol=1e-8):
    """True if all eigenvalues >= -tol (numerically PSD)."""
    vals = eigh(L, eigvals_only=True)
    return bool(np.all(vals >= -tol))


def check_laplacian(L, tol=1e-6):
    """
    Verify P1 (L1=0) and P2 (off-diagonals <= 0).
    Returns dict with pass/fail and max-violation magnitudes.
    """
    p = L.shape[0]
    ones = np.ones(p)
    row_sum_err = float(np.max(np.abs(L @ ones)))

    mask = ~np.eye(p, dtype=bool)
    max_offdiag = float(np.max(L[mask]))  # should be <= 0

    return {
        "P1_L1_eq_0": row_sum_err < tol,
        "P2_offdiag_leq_0": max_offdiag <= tol,
        "P3_psd": is_psd(L, tol=tol),
        "row_sum_max_err": row_sum_err,
        "max_offdiag": max_offdiag,
    }


def log_gdet(L, tol=1e-10):
    """Log pseudo-determinant (sum of logs of positive eigenvalues)."""
    vals = eigh(L, eigvals_only=True)
    return float(np.sum(np.log(vals[vals > tol])))
