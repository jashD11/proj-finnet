"""
Section II of Silva et al. (2015): returns -> correlation -> thresholded network.

The whole of the paper's network construction is three steps, and each one
throws information away. This module implements them and, at every step, also
records what was discarded, because several audit items live in the discards:

    log returns              -> (nothing lost)
    30-day Pearson rho       -> A4: 30 observations for 377 series
    top-f% threshold         -> A7 tau(t), A8 negative correlations
    binarize                 -> A9 edge weights

Nothing here knows about communities; that is net/communities.py.
"""

import numpy as np

# ── Returns ────────────────────────────────────────────────────────────────

def log_returns(prices: np.ndarray) -> np.ndarray:
    """(N, T) prices -> (N, T-1) log returns. Column r uses prices r and r+1."""
    if np.any(prices <= 0):
        raise ValueError("non-positive prices: log return is undefined")
    return np.diff(np.log(prices), axis=1)


# ── Windowed correlation ───────────────────────────────────────────────────

def n_windows(n_prices: int, delta_t: int) -> int:
    """N_w = C_p - Delta_t. Window w spans price indices w .. w+delta_t."""
    return n_prices - delta_t


def _standardize(block: np.ndarray):
    """Row-standardize an (N, delta_t) return block for a correlation matmul.

    Returns (Z, n_flat) where rows of Z with zero variance are zeroed out and
    counted. A stock that does not move for 30 straight days has an undefined
    correlation with everything; the paper never says what happens to it. We
    give it correlation 0 with every other stock, which makes it a guaranteed
    non-edge, and we count how often this happens so the choice is auditable.
    """
    centered = block - block.mean(axis=1, keepdims=True)
    norm = np.sqrt((centered ** 2).sum(axis=1))
    flat = norm == 0
    norm_safe = np.where(flat, 1.0, norm)
    return centered / norm_safe[:, None], int(flat.sum())


def window_correlation(returns: np.ndarray, w: int, delta_t: int) -> np.ndarray:
    """Pearson correlation matrix (Eq. 1) for window `w`. (N, N)."""
    Z, _ = _standardize(returns[:, w:w + delta_t])
    return Z @ Z.T


def mean_correlation_series(returns: np.ndarray, delta_t: int) -> np.ndarray:
    """Cross-sectional mean off-diagonal correlation per window. (N_w,)

    This is the crisis signal the paper's fixed-density construction discards
    (audit A7), so it is worth having cheaply. It is computed without ever
    forming a correlation matrix: with Z row-standardized,

        sum(C) = 1' Z Z' 1 = || Z' 1 ||^2

    which costs O(N * delta_t) per window instead of O(N^2 * delta_t).
    """
    N = returns.shape[0]
    n_w = n_windows(returns.shape[1] + 1, delta_t)
    out = np.empty(n_w)
    for w in range(n_w):
        Z, _ = _standardize(returns[:, w:w + delta_t])
        colsum = Z.sum(axis=0)                     # (delta_t,) = Z' 1
        total = float(colsum @ colsum)             # = sum of all entries of C
        diag = float((Z * Z).sum())                # = trace(C); N minus flat rows
        out[w] = (total - diag) / (N * (N - 1))
    return out


# ── Fixed-density thresholding ─────────────────────────────────────────────

def target_edge_count(n_nodes: int, f: float) -> int:
    """Number of edges kept per day: f * N(N-1)/2, rounded to an integer.

    The paper asserts the average degree is held fixed at f*(N-1). That is only
    achievable up to integer rounding, since the number of edges must be a whole
    number; the residual is reported rather than hidden.
    """
    return int(round(f * n_nodes * (n_nodes - 1) / 2))


def threshold_window(corr: np.ndarray, n_edges: int, iu):
    """Keep the `n_edges` largest correlations. Returns (tau, pair_idx, weights).

    `iu` is a precomputed np.triu_indices(N, 1) pair. `pair_idx` is the flat
    row-major index i*N + j of each retained pair, which is directly usable as
    an igraph edge list and costs 4 bytes per edge.
    """
    upper = corr[iu]
    # np.argpartition puts the n_edges largest in the tail; no full sort needed.
    cut = len(upper) - n_edges
    part = np.argpartition(upper, cut)[cut:]
    weights = upper[part]
    tau = float(weights.min())
    N = corr.shape[0]
    pair_idx = (iu[0][part].astype(np.uint32) * np.uint32(N)
                + iu[1][part].astype(np.uint32))
    order = np.argsort(pair_idx)
    return tau, pair_idx[order], weights[order]


def decode_pairs(pair_idx: np.ndarray, n_nodes: int) -> np.ndarray:
    """Flat row-major pair indices -> (n_edges, 2) integer edge list."""
    idx = pair_idx.astype(np.int64)
    return np.column_stack((idx // n_nodes, idx % n_nodes))


# ── The full sequence ──────────────────────────────────────────────────────

def build_network_sequence(returns: np.ndarray, delta_t: int, f: float,
                           progress_every: int = 1000, verbose: bool = True) -> dict:
    """Build every daily network. Returns arrays, not graphs.

    Graphs are reconstructed on demand from `edges` (see decode_pairs); storing
    igraph objects for thousands of days would be far more memory than the
    edge lists themselves.

    Recorded per window, beyond the edges themselves:
        tau              the threshold that achieved density f  (audit A7)
        n_isolated       nodes with degree 0                    (Fig. 3B signature)
        edge_w_mean/min/max   spread of the weights binarization discards (A9)
        n_below_neg_tau  pairs with rho < -tau: the anticorrelated relationships
                         the construction is structurally blind to  (audit A8)
        n_flat           rows with zero variance in this window
    """
    N, n_ret = returns.shape
    n_w = n_windows(n_ret + 1, delta_t)
    n_edges = target_edge_count(N, f)
    iu = np.triu_indices(N, 1)

    edges = np.empty((n_w, n_edges), dtype=np.uint32)
    tau = np.empty(n_w)
    n_isolated = np.empty(n_w, dtype=np.int32)
    w_mean = np.empty(n_w)
    w_min = np.empty(n_w)
    w_max = np.empty(n_w)
    n_below_neg_tau = np.empty(n_w, dtype=np.int32)
    n_flat = np.empty(n_w, dtype=np.int32)

    degree = np.empty(N, dtype=np.int32)
    for w in range(n_w):
        Z, flat = _standardize(returns[:, w:w + delta_t])
        corr = Z @ Z.T
        t, pair_idx, weights = threshold_window(corr, n_edges, iu)

        edges[w] = pair_idx
        tau[w] = t
        w_mean[w] = weights.mean()
        w_min[w] = weights.min()
        w_max[w] = weights.max()
        n_flat[w] = flat
        n_below_neg_tau[w] = int((corr[iu] < -t).sum())

        degree[:] = 0
        pairs = decode_pairs(pair_idx, N)
        np.add.at(degree, pairs[:, 0], 1)
        np.add.at(degree, pairs[:, 1], 1)
        n_isolated[w] = int((degree == 0).sum())

        if verbose and progress_every and (w + 1) % progress_every == 0:
            print(f"[networks] {w + 1}/{n_w} windows")

    return {
        "edges": edges, "tau": tau, "n_isolated": n_isolated,
        "edge_w_mean": w_mean, "edge_w_min": w_min, "edge_w_max": w_max,
        "n_below_neg_tau": n_below_neg_tau, "n_flat": n_flat,
        "n_nodes": N, "n_edges": n_edges, "n_windows": n_w,
        "delta_t": delta_t, "f": f,
    }


def window_timestamps(price_dates, delta_t: int, convention: str):
    """Calendar stamp for each window under a given convention (audit A10).

    Window w spans price indices w .. w+delta_t, so:
        t1       -> price_dates[w]
        midpoint -> price_dates[w + delta_t // 2]
        t2       -> price_dates[w + delta_t]
    """
    offsets = {"t1": 0, "midpoint": delta_t // 2, "t2": delta_t}
    if convention not in offsets:
        raise ValueError(f"unknown convention {convention!r}; "
                         f"expected one of {sorted(offsets)}")
    off = offsets[convention]
    n_w = n_windows(len(price_dates), delta_t)
    return price_dates[off:off + n_w]
