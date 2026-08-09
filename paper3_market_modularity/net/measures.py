"""
The eight topological measures (Silva et al. 2015, Figs. 6 and S3).

    modularity            Louvain Q, re-detected on whatever graph is passed
    average shortest path
    degree assortativity
    transitivity
    average betweenness
    clique number         exact, not a heuristic
    rich club             phi(k), raw and configuration-normalized
    average matching index

Two of these are undefined on a disconnected graph, and the paper never says
what it does about that (audit E6) -- which matters because Black Monday, its
showcase event, is exactly when the graph shatters into a hub plus 152 isolated
stocks. Both readings are therefore computed and stored side by side:

    *_lcc        restricted to the largest connected component
    *_harmonic   harmonic mean over all pairs, infinite distances contributing
                 zero (equivalently 1 / global efficiency)

They are not interchangeable: the first silently changes the population being
averaged over from day to day, the second keeps it fixed and lets the crisis
show up as a number rather than as a shrinking denominator.

Everything here takes an igraph Graph and returns plain floats, so it works
identically on real networks and on regenerated null-model graphs.
"""

import numpy as np

import igraph as ig

from net.communities import louvain_partitions, modularity

#: Order used everywhere downstream -- figures, scoring tables, PCA columns.
MEASURES = (
    "modularity",
    "path_length",
    "assortativity",
    "transitivity",
    "betweenness",
    "clique_number",
    "rich_club",
    "matching_index",
)

#: The measures whose disconnection convention is ambiguous (audit E6).
DISCONNECTION_SENSITIVE = ("path_length", "betweenness")


def adjacency(g: ig.Graph) -> np.ndarray:
    """Dense boolean adjacency. Much faster than igraph's get_adjacency(),
    which builds a Python list of lists."""
    n = g.vcount()
    A = np.zeros((n, n), dtype=bool)
    if g.ecount():
        ij = np.asarray(g.get_edgelist(), dtype=np.int64)
        A[ij[:, 0], ij[:, 1]] = True
        A[ij[:, 1], ij[:, 0]] = True
    return A


def path_lengths(g: ig.Graph):
    """Both disconnection conventions from a single all-pairs BFS.

    Returns (asp_lcc, harmonic, efficiency, lcc_frac).
    """
    n = g.vcount()
    D = np.asarray(g.distances(), dtype=np.float64)
    finite = np.isfinite(D) & (D > 0)

    # Harmonic convention: unreachable pairs contribute 1/inf = 0, so the
    # denominator stays n(n-1) on every day and a shattered graph reads as a
    # low number rather than as a different average.
    inv = np.zeros_like(D)
    np.divide(1.0, D, out=inv, where=finite)
    efficiency = inv.sum() / (n * (n - 1)) if n > 1 else np.nan
    harmonic = 1.0 / efficiency if efficiency > 0 else np.inf

    # Largest-component convention: the population being averaged over changes
    # with the graph, which is exactly the objection in E6.
    comp = g.connected_components()
    sizes = np.asarray(comp.sizes())
    giant = int(np.argmax(sizes))
    memb = np.asarray(comp.membership)
    sel = memb == giant
    n_g = int(sel.sum())
    if n_g > 1:
        sub = D[np.ix_(sel, sel)]
        m = np.isfinite(sub) & (sub > 0)
        asp_lcc = float(sub[m].mean())
    else:
        asp_lcc = np.nan
    return asp_lcc, float(harmonic), float(efficiency), n_g / n


def betweenness(g: ig.Graph):
    """(mean over all vertices, mean over the largest component only)."""
    b = np.asarray(g.betweenness(), dtype=np.float64)
    comp = g.connected_components()
    memb = np.asarray(comp.membership)
    giant = int(np.argmax(np.asarray(comp.sizes())))
    sel = memb == giant
    return float(b.mean()), float(b[sel].mean()) if sel.any() else np.nan


def rich_club_curve(A: np.ndarray, k_max: int = None) -> np.ndarray:
    """phi(k) for k = 0 .. k_max, as an array with NaN where undefined.

    phi(k) is the edge density among the nodes of degree > k. Computed with a
    single degree-sorted prefix scan rather than one submatrix sum per k, which
    is what makes it cheap enough to run on 132,615 graphs.
    """
    d = A.sum(axis=1).astype(np.int64)
    n = len(d)
    k_max = int(d.max()) if k_max is None else int(k_max)
    order = np.argsort(-d, kind="stable")          # descending degree
    As = A[np.ix_(order, order)]
    d_sorted = d[order]

    # edges_within[p] = number of edges inside the first p nodes of `order`.
    # Row i of the strict lower triangle counts node i's edges to the nodes
    # already in the prefix, so one cumulative sum gives every p at once.
    inc = np.tril(As, -1).sum(axis=1)
    edges_within = np.concatenate(([0], np.cumsum(inc)))

    phi = np.full(k_max + 1, np.nan)
    # For threshold k the qualifying set {d > k} is a prefix of `order`.
    counts = np.searchsorted(-d_sorted, -np.arange(k_max + 1), side="left")
    for k in range(k_max + 1):
        p = int(counts[k])
        if p > 1:
            phi[k] = 2.0 * edges_within[p] / (p * (p - 1))
    return phi


def matching_index(A: np.ndarray) -> float:
    """Mean over all pairs of |N(i) ∩ N(j)| / |N(i) ∪ N(j)|, with i and j
    themselves excluded from both neighbourhoods."""
    n = A.shape[0]
    Af = A.astype(np.float32)
    inter = Af @ Af                       # i and j drop out: A has no diagonal
    d = Af.sum(axis=1)
    union = d[:, None] + d[None, :] - inter - 2.0 * Af
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    iu = np.triu_indices(n, 1)
    return float(M[iu].mean())


def measure_all(g: ig.Graph, seed: int, k_max: int = None) -> dict:
    """All eight measures on one graph. Returns floats plus the phi(k) curve."""
    A = adjacency(g)
    asp_lcc, harmonic, efficiency, lcc_frac = path_lengths(g)
    b_all, b_lcc = betweenness(g)
    phi = rich_club_curve(A, k_max=k_max)

    return {
        "modularity": modularity(g, louvain_partitions(g, [seed])[0]),
        "path_length": harmonic,          # headline convention: no shrinking denominator
        "path_length_lcc": asp_lcc,
        "efficiency": efficiency,
        "lcc_frac": lcc_frac,
        "assortativity": float(g.assortativity_degree()),
        "transitivity": float(g.transitivity_undirected(mode="zero")),
        "betweenness": b_all,
        "betweenness_lcc": b_lcc,
        "clique_number": float(g.clique_number()),
        "rich_club": float(np.nanmean(phi)) if np.isfinite(phi).any() else np.nan,
        "matching_index": matching_index(A),
        "_phi": phi,
    }


def normalized_rich_club(phi_real: np.ndarray, phi_null: np.ndarray) -> float:
    """Audit D4: phi(k) rises with k in random graphs too, so the raw curve is
    not evidence of a rich club. Normalize curve-wise against the degree-
    preserving null, then average -- not a ratio of the two averages, which is
    a different and worse statistic."""
    both = np.isfinite(phi_real) & np.isfinite(phi_null) & (phi_null > 0)
    if not both.any():
        return np.nan
    return float(np.mean(phi_real[both] / phi_null[both]))
