"""
Part II of the proposal: the market factor as an implicit null model.

Since L1 = 0, tr(LS) = tr(L JSJ) with J = I - 11'/p, so the Laplacian learner
only sees S with each day's cross-sectional mean removed. For a one-factor model
S = s2 bb' + rest, the market's contribution to the objective is
    tr(L s2 bb') = s2 b'Lb = s2 sum_{i<j} W_ij (b_i - b_j)^2,
which vanishes when all loadings are equal and otherwise penalises edges between
stocks with different betas.

This module holds the inputs (S and its market-removed variants), the two
learners, the synthetic one-factor generator and the metrics. part2_runs.py runs
the jobs; report_numbers.py (section H onwards) prints the results.

Naming: "loading" b_i = corr(x_i, f) is what a correlation matrix sees; "beta"
is the plain OLS slope. The solvers' sparsity parameter is also called beta in
Paper 1's code; here it is passed by keyword as SPARSITY to avoid the clash.
"""

import os

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

from common import HERE, K, adjacency, bottom_eig, top_eig

from solver.algorithm1 import algorithm1  # noqa: E402  (Paper 1)
from solver.laplacian_solver import learn_laplacian  # noqa: E402  (Paper 1)

CACHE = os.path.join(HERE, "part2_cache")
ETA, RHO = 300.0, 100.0   # Algorithm 1, as in the repaired Fig 2
SPARSITY = 10.0           # Problem 1, as in Fig 1 (exp1_preprocessing.py)


# ------------------------------------------------------------------ data
def load_index():
    """S&P 500 (^GSPC) daily log returns on the 753 days of the stock panel."""
    return pd.read_csv(os.path.join(HERE, "sp500_index.csv"), index_col=0,
                       parse_dates=True)["SPX"]


def to_corr(M):
    s = np.sqrt(np.diag(M))
    return M / np.outer(s, s)


def loadings(X, f):
    """corr(x_i, f) for each column of X (the market loading a correlation matrix sees)."""
    Xc, fc = X - X.mean(axis=0), f - f.mean()
    return (Xc.T @ fc) / (np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((fc ** 2).sum()))


def ols_betas(X, f):
    Xc, fc = X - X.mean(axis=0), f - f.mean()
    return (Xc.T @ fc) / (fc @ fc)


def sin2theta(b):
    """sin^2 of the angle between b and 1: the share of ||b||^2 that J leaves behind."""
    return 1.0 - b.sum() ** 2 / (len(b) * (b @ b))


def residual_corr(X, f):
    """Correlation of the residuals after regressing each stock on f (explicit null)."""
    Xc, fc = X - X.mean(axis=0), f - f.mean()
    R = Xc - np.outer(fc, ols_betas(X, f))
    return np.corrcoef(R.T)


def market_mode_removed(S):
    """S - lambda_1 u_1 u_1', rescaled back to unit diagonal (recorded deviation)."""
    lam, U = top_eig(S, 1)
    return to_corr(S - lam[0] * np.outer(U[:, 0], U[:, 0]))


def rmt_group(S, T):
    """MacMahon-Garlaschelli group component: eigenmodes above the Marchenko-Pastur
    edge (1 + sqrt(p/T))^2, minus the market mode, with the unit diagonal put back
    (the 'noise' variance) so the learner sees a correlation-like matrix."""
    p = S.shape[0]
    edge = (1.0 + np.sqrt(p / T)) ** 2
    w, V = np.linalg.eigh(S)
    keep = (w > edge) & (np.arange(p) != p - 1)
    C = (V[:, keep] * w[keep]) @ V[:, keep].T
    np.fill_diagonal(C, 1.0)
    return C, int(keep.sum()), edge


def real_inputs(X, f):
    """The six inputs for the S&P panel. X: (T, p) returns, f: (T,) index returns."""
    S = np.corrcoef(X.T)
    p, T = S.shape[0], X.shape[0]
    J = np.eye(p) - np.ones((p, p)) / p
    ew = X.mean(axis=1)
    return {
        "S": S,
        "JSJ": J @ S @ J,
        "S-mode": market_mode_removed(S),
        "resid-SPX": residual_corr(X, f),
        "resid-EW": residual_corr(X, ew),
        "RMT": rmt_group(S, T)[0],
    }


# ------------------------------------------------------------------ synthetic
SYN_P, SYN_T = 60, 753          # 3 sectors of 20; same length as the real panel
SYN_SIG = dict(m=1.0, s=0.6, e=1.2)
DELTAS = (0.0, 0.2, 0.4, 0.6, 0.8)


def synthetic_betas(pattern, delta, p=SYN_P, k=K):
    """'aligned': betas 1-delta, 1, 1+delta by sector. 'cross': within every sector,
    half the stocks at 1+delta and half at 1-delta, so beta cuts across sectors."""
    sector = np.repeat(np.arange(k), p // k)
    if pattern == "aligned":
        beta = 1.0 + delta * (sector - (k - 1) / 2)
    else:
        half = (np.arange(p) % (p // k)) < (p // k) // 2
        beta = np.where(half, 1.0 + delta, 1.0 - delta)
    return beta, sector


def synthetic_returns(pattern, delta, seed):
    """x_t = beta f_t + s_sector(t) + e_t, all Gaussian. Returns X, f, beta, sector."""
    rng = np.random.default_rng(seed)
    beta, sector = synthetic_betas(pattern, delta)
    f = SYN_SIG["m"] * rng.standard_normal(SYN_T)
    s = SYN_SIG["s"] * rng.standard_normal((SYN_T, K))
    e = SYN_SIG["e"] * rng.standard_normal((SYN_T, SYN_P))
    X = np.outer(f, beta) + s[:, sector] + e
    return X, f, beta, sector


def synthetic_inputs(X, f):
    S = np.corrcoef(X.T)
    return {"S": S, "S-mode": market_mode_removed(S), "oracle": residual_corr(X, f)}


def synthetic_configs():
    """(pattern, delta): delta = 0 once (all betas equal), then both patterns."""
    out = [("equal", 0.0)]
    for pattern in ("aligned", "cross"):
        out += [(pattern, d) for d in DELTAS if d > 0]
    return out


# ------------------------------------------------------------------ learners
def learn(S, learner):
    if learner == "alg1":
        return algorithm1(S, k=K, beta=0.0, eta=ETA, rho_degree=RHO, max_outer=80)
    return learn_laplacian(S, beta=SPARSITY, max_iter=2000, tol=1e-6)


def clusters(L, learner, k=K):
    """Algorithm 1: connected components (if there are k of them, else spectral).
    Problem 1: k-means on the bottom k eigenvectors of L."""
    W = adjacency(L)
    if learner == "alg1":
        n, lab = connected_components(W > 1e-4)
        if n == k:
            return lab
    _, V = bottom_eig(L, k)
    return KMeans(k, n_init=20, random_state=0).fit_predict(V)


def beta_buckets(b, n=K):
    """Beta groups: one per distinct value when there are at most n of them (the
    synthetic designs), else terciles (real loadings)."""
    vals, idx = np.unique(np.round(b, 12), return_inverse=True)
    if len(vals) <= n:
        return idx
    return pd.qcut(b, n, labels=False)


def within_sector_beta_ratio(W, b, sector):
    """Edge-weighted mean |b_i - b_j| over within-sector pairs, divided by the plain
    mean over the same pairs. Below 1: inside a sector, the graph favours links
    between stocks with similar loadings."""
    iu = np.triu_indices(len(b), 1)
    same = sector[iu[0]] == sector[iu[1]]
    gap = np.abs(b[iu[0]] - b[iu[1]])[same]
    w = W[iu][same]
    if w.sum() == 0 or gap.mean() == 0:   # no edges, or one loading per sector
        return np.nan
    return (w @ gap / w.sum()) / gap.mean()


def metrics(L, learner, sector, b):
    W = adjacency(L)
    d = W.sum(axis=1)
    lab = clusters(L, learner)
    n, _ = connected_components(W > 1e-4)
    return dict(ari_sector=ari(sector, lab), ari_beta=ari(beta_buckets(b), lab),
                components=n, isolated=int(np.sum(d < 1e-6)),
                dmin=d.min(), dmax=d.max(),
                beta_ratio=within_sector_beta_ratio(W, b, sector),
                sizes=sorted(np.bincount(lab).tolist(), reverse=True))


def cache_file(key):
    return os.path.join(CACHE, key.replace("/", "_") + ".npy")
