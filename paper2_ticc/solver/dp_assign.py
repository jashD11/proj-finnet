"""
Cluster assignment via dynamic programming  (TICC paper, Algorithm 1).

Given fixed cluster parameters, assign each of the T stacked windows to one of K
clusters to minimise total negative log-likelihood plus a switching penalty beta
on every change of cluster between consecutive time steps. This is the minimum-cost
Viterbi path over a length-T trellis and is solved exactly in O(T*K).
"""

import numpy as np


def gaussian_neg_ll(X, thetas):
    """
    Negative log-likelihood of each stacked point under each cluster's zero-mean
    Gaussian with inverse covariance Theta_k.

    Parameters
    ----------
    X : (T, nw) stacked windows.
    thetas : list of K arrays, each (nw, nw) positive-definite inverse covariance.

    Returns
    -------
    nll : (T, K) array,  nll[t, k] = -log N(X_t | 0, Theta_k^{-1}).
    """
    T, d = X.shape
    K = len(thetas)
    nll = np.empty((T, K))
    const = d * np.log(2.0 * np.pi)
    for k, Th in enumerate(thetas):
        sign, logdet = np.linalg.slogdet(Th)
        # Mahalanobis term x^T Theta x for every row, vectorised
        maha = np.einsum("ti,ij,tj->t", X, Th, X)
        nll[:, k] = 0.5 * (const - logdet + maha)
    return nll


def assign_points(nll, beta):
    """
    Exact DP (Viterbi) solution of TICC Problem (3).

    Parameters
    ----------
    nll : (T, K) negative log-likelihood, node costs.
    beta : float >= 0, penalty added once for every cluster switch.

    Returns
    -------
    labels : (T,) int array of cluster assignments minimising
             sum_t nll[t, label_t] + beta * #{t : label_t != label_{t-1}}.
    """
    T, K = nll.shape
    if T == 0:
        return np.empty(0, dtype=int)

    cost = np.empty((T, K))
    back = np.empty((T, K), dtype=int)
    cost[0] = nll[0]
    back[0] = np.arange(K)

    for t in range(1, T):
        prev = cost[t - 1]
        best_prev = np.argmin(prev)
        min_prev = prev[best_prev]
        for j in range(K):
            stay = prev[j]                 # no switch
            switch = min_prev + beta       # come from cheapest other state + penalty
            if stay <= switch:
                cost[t, j] = stay + nll[t, j]
                back[t, j] = j
            else:
                cost[t, j] = switch + nll[t, j]
                back[t, j] = best_prev

    labels = np.empty(T, dtype=int)
    labels[-1] = np.argmin(cost[-1])
    for t in range(T - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels
