"""
Preprocessing variants used by the paper's Fig 1 experiment.

Four combinations: {raw covariance, correlation} × {market in, market out}.
The paper's best panel (d) = correlation + market factor NOT explicitly removed.
"""

import numpy as np


def to_correlation(S):
    """Convert covariance matrix to correlation matrix."""
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def scale_data(X):
    """
    Standardise each asset column to zero mean, unit variance.
    X : (T, p) return matrix.  Returns (T, p) scaled matrix.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    return (X - mu) / sigma


def remove_market_factor(X):
    """
    Remove the equal-weighted market factor from each column via OLS.
    X : (T, p) return matrix.
    Returns residual (T, p) matrix.
    """
    mkt = X.mean(axis=1, keepdims=True)  # (T, 1)
    # OLS beta for each asset: beta_i = cov(x_i, mkt) / var(mkt)
    beta = (X - X.mean(axis=0)).T @ (mkt - mkt.mean()) / (
        ((mkt - mkt.mean()) ** 2).sum()
    )  # (p, 1)
    return X - mkt * beta.T  # (T, p)


def build_similarity(X, use_correlation=True):
    """
    Build similarity matrix S from return matrix X.
    use_correlation=True  → S = correlation matrix  (paper default).
    use_correlation=False → S = covariance matrix.
    """
    T, p = X.shape
    S = np.cov(X.T, ddof=1)  # (p, p) covariance
    if use_correlation:
        S = to_correlation(S)
    return S


def four_variants(X):
    """
    Return the 4 preprocessing variants of X as (S_raw, label) pairs:
      (a) covariance,              market in
      (b) covariance,              market out
      (c) correlation,             market in
      (d) correlation,             market out  ← paper's best
    """
    X_scaled = scale_data(X)
    X_no_mkt = remove_market_factor(X_scaled)

    return [
        (build_similarity(X_scaled, use_correlation=False), "raw cov, market in"),
        (build_similarity(X_no_mkt,  use_correlation=False), "raw cov, market out"),
        (build_similarity(X_scaled,  use_correlation=True),  "corr, market in"),
        (build_similarity(X_no_mkt,  use_correlation=True),  "corr, market out"),
    ]
