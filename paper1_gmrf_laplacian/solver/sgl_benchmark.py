"""
Fig 2 baseline. NOT SGL, despite the file name (kept so cache paths and history
stay stable).

What this is: Problem 1, the plain GMRF Laplacian MLE with an l1 penalty
(beta = 10, the paper's value), with NO degree control and NO rank/spectral
constraint. It returns one connected component (rank p-1).

What the paper compares against: SGL, Kumar, Ying, Cardoso and Palomar, "A
unified framework for structured graph learning via spectral constraints"
(JMLR 2020), which imposes the k-component spectral constraint WITHOUT degree
control and so produces isolated nodes. That method is not implemented here (its
reference implementation is the R package spectralGraphTopology). The paper's
Fig 2 failure mode is therefore not reproduced by this baseline; see CLAUDE.md
deviation D3/D5.
"""

import numpy as np
from solver.laplacian_solver import learn_laplacian


def sgl(S, beta=10.0, max_iter=1000, tol=1e-6):
    """
    Problem 1 + l1 baseline estimate of the graph Laplacian (not SGL).

    Parameters
    ----------
    S        : (p, p) similarity / correlation matrix
    beta     : sparsity regulariser (paper uses 10)
    max_iter : max L-BFGS-B iterations
    tol      : convergence tolerance

    Returns
    -------
    L : (p, p) estimated Laplacian (connected; no isolated nodes in practice)
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
