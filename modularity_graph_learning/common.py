"""
Shared helpers for the proposal work ("Modularity Matrix Approach for Graph
Learning?", Anguluri and Dalal, October 2026).

Paper 1's code is reused, not copied: its folder goes on sys.path, the same way
modularity_indicator/build_series.py does it.
"""

import os
import sys

import numpy as np
import pandas as pd
from numpy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P1 = os.path.join(ROOT, "paper1_gmrf_laplacian")
sys.path.insert(0, P1)

from data.tickers import SP500_SECTORS  # noqa: E402

K = 3  # three sectors, as in Paper 1's Fig 2


def load_sp():
    """Paper 1's S&P 500 three-sector returns (753 days, 2016-01-05 to 2018-12-31).

    Returns (X, S, sector): the (T, p) return frame, the (p, p) sample
    correlation matrix and the (p,) array of sector labels.
    """
    X = pd.read_csv(os.path.join(P1, "data", "sp500_3sectors.csv"),
                    index_col=0, parse_dates=True)
    S = np.corrcoef(X.values.T)
    sector = np.array([SP500_SECTORS[c] for c in X.columns])
    return X, S, sector


def adjacency(L):
    """Weighted adjacency W = -L off the diagonal, clipped at zero."""
    W = -np.asarray(L, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    return np.maximum(W, 0.0)


def laplacian(W):
    return np.diag(W.sum(axis=1)) - W


def modularity_matrix(W):
    """B(W) = W - d d' / 2m, the configuration-model modularity matrix."""
    d = W.sum(axis=1)
    return W - np.outer(d, d) / d.sum()


def top_eig(M, r):
    """Largest r eigenvalues (descending) and their eigenvectors."""
    w, V = eigh((M + M.T) / 2.0)
    return w[::-1][:r], V[:, ::-1][:, :r]


def bottom_eig(M, r):
    """Smallest r eigenvalues (ascending) and their eigenvectors."""
    w, V = eigh((M + M.T) / 2.0)
    return w[:r], V[:, :r]


def unit_degree(W, iters=5000, tol=1e-13):
    """Symmetric Sinkhorn scaling: W -> D^-1/2 W D^-1/2 until every degree is 1.

    Keeps W symmetric with zero diagonal; the limit is doubly stochastic, i.e.
    the unit-degree (diag(L) = 1) graph with W's support.
    """
    W = np.asarray(W, dtype=float).copy()
    for _ in range(iters):
        d = W.sum(axis=1)
        if np.max(np.abs(d - 1.0)) < tol:
            break
        s = 1.0 / np.sqrt(d)
        W = W * np.outer(s, s)
    return W


def sector_block_graph(sector):
    """Unit-degree graph with a complete graph inside each sector, nothing across."""
    same = (sector[:, None] == sector[None, :]).astype(float)
    np.fill_diagonal(same, 0.0)
    return same / same.sum(axis=1, keepdims=True)
