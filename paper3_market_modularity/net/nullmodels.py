"""
The two null models (Silva et al. 2015, Sec. III and Eq. 2).

Both take a real daily network and throw almost all of it away, then generate a
new random network from what is left. The paper's claim is that one of these
summaries — community sizes plus the inter-community mixing matrix Π — is
enough to reproduce the real network's topology.

    community model     keeps  Π and the community sizes        (Eq. 2)
    configuration model keeps  the degree sequence only

The configuration model is the baseline: whatever it reproduces is reproduced
by the degree distribution alone and is not evidence for anything about
communities. Its modularity is the noise floor that belongs on every modularity
figure (audit B6).

Deliberately *not* fixed here (audit D2): Eq. 2 connects each pair with an
independent coin flip, so the generated edge count is a random variable and
only matches the real one in expectation. Making it exact would be a different
model from the paper's. The realized-vs-target discrepancy is measured instead.
"""

import random

import igraph as ig
import numpy as np


def block_probabilities(Pi: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Eq. 2: edge counts -> connection probabilities.

        pi_aa = 2 * Pi_aa / (|a| * (|a| - 1))
        pi_ab =     Pi_ab / (|a| * |b|)

    Singleton communities have no internal pair to connect, so pi_aa is 0 for
    |a| = 1 rather than a division by zero. That case is not exotic: on Black
    Monday 152 of the 156 communities are singletons.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    k = len(sizes)
    full = Pi + Pi.T - np.diag(np.diag(Pi))          # symmetric edge counts
    denom = np.outer(sizes, sizes).astype(np.float64)
    np.fill_diagonal(denom, sizes * (sizes - 1) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(denom > 0, full / np.maximum(denom, 1e-300), 0.0)
    if np.any(P > 1.0 + 1e-9):
        P = np.clip(P, 0.0, 1.0)
    return np.clip(P, 0.0, 1.0).reshape(k, k)


def community_graph(sizes, Pi, rng: np.random.Generator, n_nodes: int = None):
    """Generate one Eq. 2 realization. Returns (igraph Graph, n_edges).

    Nodes are laid out community by community, which is exchangeable with any
    other assignment because the generating probabilities depend only on the
    community labels.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    n = int(sizes.sum()) if n_nodes is None else n_nodes
    labels = np.repeat(np.arange(len(sizes)), sizes)
    P = block_probabilities(Pi, sizes)
    iu = np.triu_indices(n, 1)
    p_pair = P[labels[iu[0]], labels[iu[1]]]
    hit = rng.random(p_pair.shape[0]) < p_pair
    edges = np.column_stack((iu[0][hit], iu[1][hit]))
    return ig.Graph(n=n, edges=edges.tolist(), directed=False), int(hit.sum())


# How to realize "same degree sequence, otherwise random". The two readings do
# not agree, and the difference is not cosmetic (audit D4):
#
#   "rewire"  degree-preserving double-edge swaps starting from the real graph.
#             Keeps the degree sequence AND the edge count exactly, and yields
#             a simple graph. This is the default.
#   "stub"    classical stub matching (igraph's Degree_Sequence). Produces
#             self-loops and multi-edges which must be collapsed, and the
#             collapse destroys ~13 % of the edges -- so the resulting null has
#             a visibly lower density than the network it is compared against,
#             confounding every density-sensitive measure (transitivity, path
#             length, clique number) with an artifact of the generator.
#
# igraph's "vl" method is not an option here: it requires a connected
# realization to exist, and our crisis networks contain hundreds of degree-0
# nodes -- so it fails on exactly the days the paper is about.
CONFIG_METHOD = "rewire"
REWIRE_SWEEPS = 5          # swap attempts as a multiple of the edge count


def configuration_graph(g_or_degrees, seed: int, method: str = CONFIG_METHOD):
    """Generate one degree-preserving realization. Returns (Graph, loss dict).

    igraph draws its randomness from Python's `random` module, so the seed is
    set there rather than passed in.
    """
    random.seed(int(seed))
    if method == "rewire":
        g = g_or_degrees.copy()
        m = g.ecount()
        g.rewire(n=REWIRE_SWEEPS * m, allowed_edge_types="simple")
        return g, {"edges_before": m, "edges_after": g.ecount(),
                   "n_self_loops": 0, "n_multi_collapsed": 0}

    if method == "stub":
        degrees = (degree_sequence(g_or_degrees)
                   if isinstance(g_or_degrees, ig.Graph)
                   else np.asarray(g_or_degrees, dtype=np.int64))
        g = ig.Graph.Degree_Sequence(degrees.tolist(), method="configuration")
        raw = g.ecount()
        n_loops = int(sum(g.is_loop()))
        g = g.simplify(multiple=True, loops=True)
        return g, {"edges_before": raw, "edges_after": g.ecount(),
                   "n_self_loops": n_loops,
                   "n_multi_collapsed": raw - n_loops - g.ecount()}

    raise ValueError(f"unknown configuration-model method: {method!r}")


def degree_sequence(g: ig.Graph) -> np.ndarray:
    return np.asarray(g.degree(), dtype=np.int64)
