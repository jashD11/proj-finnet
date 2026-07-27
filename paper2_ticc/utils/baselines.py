"""
Baseline clustering methods for TICC Table 1 / Figure 3.

Every baseline has the same signature `f(Xst, K, seed) -> labels`, operating on the
stacked windows `Xst` (shape (M, n*w)) so the comparison is apples-to-apples with
TICC. Distance-based methods (DTW, Neural Gas) act on the raw window vectors and are
expected to score poorly on these zero-mean, structure-only clusters -- exactly the
paper's point.

EEV, DTW and Neural Gas are approximations of the paper's exact toolchain (mclust /
custom implementations); they are the low-scoring rows and any divergence there does
not affect the headline TICC-vs-GMM-vs-K-means comparison.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def gmm_labels(Xst, K, seed=42):
    """Full-covariance Gaussian Mixture Model."""
    gm = GaussianMixture(n_components=K, covariance_type="full",
                         reg_covar=1e-4, random_state=seed)
    return gm.fit_predict(Xst)


def eev_labels(Xst, K, seed=42):
    """
    Approximation of EEV (equal-volume regularised GMM). sklearn has no exact EEV
    covariance family; 'spherical' (equal volume, equal shape) is the closest
    constrained GMM and is used here as a documented stand-in.
    """
    gm = GaussianMixture(n_components=K, covariance_type="spherical",
                         reg_covar=1e-4, random_state=seed)
    return gm.fit_predict(Xst)


def kmeans_labels(Xst, K, seed=42):
    """Standard K-means (Euclidean)."""
    return KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Xst)


def _dtw_labels(Xst, K, n, w, seed, metric):
    """DTW-based clustering via tslearn; each window reshaped to (w timesteps, n)."""
    from tslearn.clustering import TimeSeriesKMeans
    M = Xst.shape[0]
    series = Xst.reshape(M, w, n)
    km = TimeSeriesKMeans(n_clusters=K, metric=metric, max_iter=10,
                          n_init=1, random_state=seed)
    return km.fit_predict(series)


def dtw_gak_labels(Xst, K, n, w, seed=42):
    """DTW clustering with a global alignment kernel (soft-DTW)."""
    return _dtw_labels(Xst, K, n, w, seed, metric="softdtw")


def dtw_euclidean_labels(Xst, K, n, w, seed=42):
    """DTW clustering with a Euclidean local metric."""
    return _dtw_labels(Xst, K, n, w, seed, metric="dtw")


def neural_gas_labels(Xst, K, seed=42, epochs=20, lam0=None, lam_f=0.5,
                      eps0=0.5, eps_f=0.05):
    """
    Compact Neural Gas clustering (Martinetz & Schulten). K prototypes are moved
    toward each presented point with a rank-based neighbourhood; points are then
    assigned to the nearest prototype. A distance-based method (expected low score).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(Xst, dtype=float)
    M = X.shape[0]
    protos = X[rng.choice(M, size=K, replace=False)].copy()
    lam0 = lam0 if lam0 is not None else K / 2.0
    total = epochs * M

    step = 0
    for _ in range(epochs):
        for idx in rng.permutation(M):
            t = step / total
            lam = lam0 * (lam_f / lam0) ** t
            eps = eps0 * (eps_f / eps0) ** t
            x = X[idx]
            d = np.linalg.norm(protos - x, axis=1)
            ranks = np.empty(K, dtype=int)
            ranks[np.argsort(d)] = np.arange(K)
            h = np.exp(-ranks / lam)
            protos += (eps * h)[:, None] * (x - protos)
            step += 1

    d = np.linalg.norm(X[:, None, :] - protos[None, :, :], axis=2)
    return np.argmin(d, axis=1)
