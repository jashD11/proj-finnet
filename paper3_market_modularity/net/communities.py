"""
Community detection and modularity (Silva et al. 2015, Sec. II and Figs. 3-5).

The paper runs Louvain on each daily network and reports three modularities
that differ only in *which partition* is scored:

    dynamical   Q(g_t, C(t))          re-detected every day
    fixed       Q(g_t, C(0))          the first day's partition, held forever
    lagged      Q(g_t, C(t - 100))    a partition 100 trading days stale

It runs Louvain once. We run it ten times per day with different seeds. Seed 0
is the canonical C(t) used everywhere downstream (Pi, fixed, lagged, Phase 4's
null model) so the pipeline stays faithful to the paper; the other nine exist
only to measure how much of the daily modularity series is algorithmic noise
rather than market structure (audit B1/B2, reported in Phase 8f).

Louvain is stochastic in node visit order, and python-igraph draws that order
from Python's `random` module, so `random.seed(s)` is what makes a run
reproducible. `louvain_partitions` does that seeding explicitly.
"""

import random

import igraph as ig
import numpy as np


def graph_from_pairs(pair_idx, n_nodes: int) -> ig.Graph:
    """Build an igraph Graph from the packed uint32 pair indices of Phase 2."""
    idx = np.asarray(pair_idx, dtype=np.int64)
    ij = np.column_stack((idx // n_nodes, idx % n_nodes))
    return ig.Graph(n=n_nodes, edges=ij.tolist(), directed=False)


def edge_endpoints(pair_idx, n_nodes: int) -> np.ndarray:
    """(n_edges, 2) endpoint array, i < j, without building a graph."""
    idx = np.asarray(pair_idx, dtype=np.int64)
    return np.column_stack((idx // n_nodes, idx % n_nodes))


def louvain_partitions(g: ig.Graph, seeds) -> np.ndarray:
    """Run Louvain once per seed. Returns (n_seeds, n_nodes) int16 memberships.

    Communities are relabelled by descending size so that labels are stable
    across seeds and days; this matters only for readability of stored Pi, not
    for modularity, which is label-invariant.
    """
    out = np.empty((len(seeds), g.vcount()), dtype=np.int16)
    for r, s in enumerate(seeds):
        random.seed(int(s))
        out[r] = relabel_by_size(np.asarray(g.community_multilevel().membership))
    return out


def relabel_by_size(membership: np.ndarray) -> np.ndarray:
    """Relabel communities 0..k-1 in descending order of size."""
    labels, counts = np.unique(membership, return_counts=True)
    order = labels[np.argsort(-counts, kind="stable")]
    lookup = np.empty(int(labels.max()) + 1, dtype=np.int16)
    lookup[order] = np.arange(len(order), dtype=np.int16)
    return lookup[membership]


def modularity(g: ig.Graph, membership) -> float:
    """Q = sum_c [ e_c/m - (d_c/2m)^2 ], igraph's implementation."""
    return float(g.modularity(np.asarray(membership).tolist()))


def community_sizes(membership: np.ndarray, k: int = None) -> np.ndarray:
    k = k if k is not None else int(membership.max()) + 1
    return np.bincount(membership, minlength=k).astype(np.int64)


def mixing_matrix(endpoints: np.ndarray, membership: np.ndarray,
                  k: int = None) -> np.ndarray:
    """Pi_ab: edges within community a (diagonal) and between a and b.

    Upper-triangular including the diagonal; Pi[a, b] for a < b holds *all*
    edges running between the two communities. This is the entire input to the
    paper's Eq. 2 null model, together with the community sizes.
    """
    k = k if k is not None else int(membership.max()) + 1
    a = membership[endpoints[:, 0]].astype(np.int64)
    b = membership[endpoints[:, 1]].astype(np.int64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    Pi = np.zeros((k, k), dtype=np.int64)
    np.add.at(Pi, (lo, hi), 1)
    return Pi


class MixingStore:
    """Ragged per-day storage for Pi and the community sizes.

    k varies enormously across days -- typically 6-9, but on Black Monday every
    isolated stock is its own singleton community and k jumps into the hundreds.
    A dense (n_windows, k_max, k_max) array would therefore be ~700 MB of mostly
    zeros, so days are concatenated flat with an offset index instead.
    """

    def __init__(self, pi_flat, pi_ptr, sizes_flat, sizes_ptr, k):
        self.pi_flat, self.pi_ptr = pi_flat, pi_ptr
        self.sizes_flat, self.sizes_ptr = sizes_flat, sizes_ptr
        self.k = k

    def __len__(self):
        return len(self.k)

    def __getitem__(self, w: int):
        """(Pi, sizes) for day w, Pi rebuilt as a dense upper-triangular k x k."""
        kw = int(self.k[w])
        flat = self.pi_flat[self.pi_ptr[w]:self.pi_ptr[w + 1]]
        Pi = np.zeros((kw, kw), dtype=np.int64)
        Pi[np.triu_indices(kw)] = flat
        return Pi, self.sizes_flat[self.sizes_ptr[w]:self.sizes_ptr[w + 1]]

    @classmethod
    def build(cls, pis, sizes_list):
        k = np.array([len(s) for s in sizes_list], dtype=np.int32)
        pi_flat = np.concatenate(
            [p[np.triu_indices(len(p))] for p in pis]).astype(np.int32)
        sizes_flat = np.concatenate(sizes_list).astype(np.int32)
        pi_ptr = np.concatenate(([0], np.cumsum(k * (k + 1) // 2))).astype(np.int64)
        sizes_ptr = np.concatenate(([0], np.cumsum(k))).astype(np.int64)
        return cls(pi_flat, pi_ptr, sizes_flat, sizes_ptr, k)

    def save(self, path):
        np.savez_compressed(path, pi_flat=self.pi_flat, pi_ptr=self.pi_ptr,
                            sizes_flat=self.sizes_flat, sizes_ptr=self.sizes_ptr,
                            k=self.k)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["pi_flat"], z["pi_ptr"], z["sizes_flat"], z["sizes_ptr"], z["k"])


def modularity_from_mixing(Pi: np.ndarray, sizes: np.ndarray) -> float:
    """Q computed from the compressed description alone.

    Phase 4's correctness check: if Pi and the community sizes really are a
    sufficient summary for modularity, this must equal `modularity()` on the
    graph it came from. Degrees are recovered as d_c = 2*Pi_cc + sum_{b!=c} Pi_cb.
    """
    m = Pi.sum()
    if m == 0:
        return 0.0
    within = np.diag(Pi).astype(np.float64)
    total = Pi + Pi.T - np.diag(np.diag(Pi))
    # row sum counts each within-edge once and each between-edge once, so the
    # degree sum of community c is (2 x within) + (row sum - within).
    d = within + total.sum(axis=1)
    return float((within / m - (d / (2.0 * m)) ** 2).sum())
