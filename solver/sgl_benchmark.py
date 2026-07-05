"""
SGL baseline — Structured Graph Learning [Kumar et al. 2016].

SGL solves the same GMRF problem but WITHOUT the degree-control constraint
(diag(L) = 1 is not enforced).  This causes isolated nodes (zero-degree
rows/cols) when the sparsity penalty β is high, which is the failure mode
that Algorithm 1 fixes.  Used in Fig 2 for comparison.

Implementation: identical call to learn_laplacian with degree_control=False
and beta=10 (same β as Algorithm 1 per the paper).
"""

import numpy as np
from solver.admm_graph import learn_laplacian


def sgl(S, beta=10.0, max_iter=1000, tol=1e-6):
    """
    SGL estimate of the graph Laplacian.

    Parameters
    ----------
    S        : (p, p) similarity / correlation matrix
    beta     : sparsity regulariser (paper uses 10)
    max_iter : max L-BFGS-B iterations
    tol      : convergence tolerance

    Returns
    -------
    L : (p, p) estimated Laplacian (may have isolated nodes)
    """
    return learn_laplacian(
        S,
        beta=beta,
        eta=0.0,
        V=None,
        degree_control=False,
        max_iter=max_iter,
        tol=tol,
    )
