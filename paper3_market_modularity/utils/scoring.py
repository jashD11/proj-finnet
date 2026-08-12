"""
Scoring a null model against the real networks — the paper's way, then honestly.

The paper compares each measure's real series against each model's series with
two numbers: a chi-square (mean squared difference) and a Pearson correlation,
computed separately before and after 2002. Both are reproduced here exactly as
specified, and both are then given the context they need:

*Normalized MSE* (audit E2). The paper's raw chi-square values span 0.0188 to
2,522 across measures and are printed in one table as if comparable. They are
not: betweenness is O(100) and transitivity is O(0.1), so the chi-square is
dominated by the units. Dividing by the variance of the real series makes the
number mean "what fraction of the real variation the model fails to capture",
which is comparable across measures and has 1.0 as the natural failure point.

*Overlap-corrected correlation* (audit E1) — the single most consequential
correction in this replication. Every series is computed on 30-day windows slid
by one day, so consecutive points share 29/30 of their data. The 5,978 points
the paper correlates are not 5,978 independent observations; they are roughly
N_w / dt ~ 200. Correlation of two smooth, heavily overlapping series is
inflated toward 1 almost regardless of what the underlying processes do, and
every headline number in Fig. 6 is such a correlation.

*Block bootstrap* (audit E4). Confidence intervals for those correlations must
resample blocks at least as long as the window, or they inherit exactly the
same dependence they are supposed to be measuring.
"""

import numpy as np


def _clean(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def chi_square(real, model) -> float:
    """The paper's chi-square: mean squared difference between the series."""
    r, m = _clean(real, model)
    return float(np.mean((r - m) ** 2)) if len(r) else np.nan


def normalized_mse(real, model) -> float:
    """chi-square / var(real). 0 is perfect; 1 means the model does no better
    than predicting the real series' own mean."""
    r, m = _clean(real, model)
    v = np.var(r)
    return float(np.mean((r - m) ** 2) / v) if len(r) and v > 0 else np.nan


def pearson(real, model) -> float:
    r, m = _clean(real, model)
    if len(r) < 3 or np.std(r) == 0 or np.std(m) == 0:
        return np.nan
    return float(np.corrcoef(r, m)[0, 1])


def nonoverlapping_slice(n: int, delta_t: int) -> np.ndarray:
    """Indices of windows that share no data: every delta_t-th day.

    Window w covers price indices w .. w+delta_t, so windows w and w+delta_t
    are the closest pair with disjoint return sets.
    """
    return np.arange(0, n, delta_t)


def effective_sample_size(n: int, delta_t: int) -> int:
    return int(np.ceil(n / delta_t))


def block_bootstrap_ci(real, model, block: int, n_boot: int = 1000,
                       seed: int = 42, alpha: float = 0.05):
    """Moving-block bootstrap CI for the Pearson correlation.

    Blocks are drawn with replacement and concatenated to the original length,
    preserving dependence up to the block length. `block` must be at least the
    window length, or the resample destroys exactly the overlap it exists to
    account for.
    """
    r, m = _clean(real, model)
    n = len(r)
    if n < 3 * block:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        rb, mb = r[idx], m[idx]
        if np.std(rb) == 0 or np.std(mb) == 0:
            out[b] = np.nan
        else:
            out[b] = np.corrcoef(rb, mb)[0, 1]
    lo, hi = np.nanpercentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.nanmean(out)), float(lo), float(hi)


def inter_measure_correlation(series_by_measure: dict):
    """Audit E5: how many of the "eight independent facts" are independent.

    Returns (measure order, correlation matrix, summary). The summary reports
    the mean absolute off-diagonal correlation and an effective dimensionality
    -- the number of principal components needed to reach 90 % of the variance
    of the standardized measure matrix.
    """
    names = list(series_by_measure)
    X = np.column_stack([np.asarray(series_by_measure[k], dtype=float)
                         for k in names])
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    C = np.corrcoef(X, rowvar=False)
    off = C[~np.eye(len(names), dtype=bool)]

    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    ev = np.linalg.eigvalsh(np.cov(Z, rowvar=False))[::-1]
    ratio = ev / ev.sum()
    n_90 = int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)
    return names, C, {
        "mean_abs_offdiag": float(np.mean(np.abs(off))),
        "max_abs_offdiag": float(np.max(np.abs(off))),
        "n_components_for_90pct": n_90,
        "explained_variance_ratio": [float(v) for v in ratio],
        "n_measures": len(names),
        "n_days": int(len(X)),
    }


def compression_cost(k, n_nodes: int):
    """Audit D1: is the community summary actually smaller than the network?

    The community model stores k community sizes plus the k(k+1)/2 entries of
    the mixing matrix; the configuration model stores N degrees. The paper
    asserts the former is a compression without ever checking the arithmetic.

    Note the audit's "equal at k = 26" counts only Pi's k(k+1)/2 entries. The
    sizes are counted here too, because Eq. 2 cannot generate a network without
    them, which moves the break-even down slightly (k = 25 at N = 360).
    """
    k = np.asarray(k, dtype=float)
    community = k + k * (k + 1) / 2.0
    return community, float(n_nodes), community < n_nodes
