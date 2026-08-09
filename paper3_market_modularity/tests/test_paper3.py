"""
Sanity tests for the Silva et al. (2015) replication.

Run:
    /opt/anaconda3/bin/python -m pytest tests/

Tests are added phase by phase; everything here is Phase 0.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.fetch_data import build_panel  # noqa: E402
from data.quality import longest_zero_run, quality_gate  # noqa: E402
from data.universe import to_yahoo_symbol  # noqa: E402
from net.communities import (MixingStore, community_sizes,  # noqa: E402
                             louvain_partitions, mixing_matrix, modularity,
                             modularity_from_mixing, relabel_by_size)
from net.nullmodels import (block_probabilities, community_graph,  # noqa: E402
                            configuration_graph)
from net.construct import (build_network_sequence, decode_pairs,  # noqa: E402
                           log_returns, mean_correlation_series, n_windows,
                           target_edge_count, threshold_window,
                           window_correlation, window_timestamps)
from utils.crisis_dates import CRISES, coverage, crisis_frame, crisis_mask  # noqa: E402


# ── Phase 0: universe ──────────────────────────────────────────────────────

def test_symbol_translation():
    """NASDAQ Trader writes share classes with a dot; Yahoo wants a dash."""
    assert to_yahoo_symbol("BRK.B") == "BRK-B"
    assert to_yahoo_symbol("IBM") == "IBM"
    assert to_yahoo_symbol("  GE ") == "GE"


def _ragged_frame():
    """A 100-day, 4-ticker frame exercising every filter branch."""
    idx = pd.bdate_range("2000-01-03", periods=100)
    df = pd.DataFrame(100.0, index=idx, columns=["FULL", "LATE", "EARLY", "GAPPY"])
    df.loc[idx[:20], "LATE"] = np.nan       # starts late  -> fails full-span
    df.loc[idx[-20:], "EARLY"] = np.nan     # ends early   -> fails full-span
    df.loc[idx[50], "GAPPY"] = np.nan       # one missing day
    return df, idx


def test_full_span_filter_drops_late_starters_and_early_enders():
    df, _ = _ragged_frame()
    panel, rep = build_panel(df, span_tol=5, min_coverage=0.0,
                             missing_policy="drop_days", verbose=False)
    assert rep["dropped_not_full_span"] == 2
    assert set(panel.columns) == {"FULL", "GAPPY"}


def test_strict_coverage_drops_the_gappy_ticker():
    """min_coverage=1.0 is the strict reading of 'complete history'."""
    df, _ = _ragged_frame()
    panel, rep = build_panel(df, span_tol=5, min_coverage=1.0,
                             missing_policy="drop_days", verbose=False)
    assert rep["dropped_incomplete_coverage"] == 1
    assert list(panel.columns) == ["FULL"]
    assert rep["n_dropped_days"] == 0


def test_missing_day_policies_trade_days_against_fabricated_returns():
    """drop_days loses the day; ffill keeps it and manufactures a zero return.

    This is audit A3 made concrete: the two policies disagree, and the
    disagreement is exactly one trading day per gap.
    """
    df, _ = _ragged_frame()
    dropped, rep_d = build_panel(df, span_tol=5, min_coverage=0.99,
                                 missing_policy="drop_days", verbose=False)
    filled, rep_f = build_panel(df, span_tol=5, min_coverage=0.99,
                                missing_policy="ffill", verbose=False)
    assert set(dropped.columns) == set(filled.columns) == {"FULL", "GAPPY"}
    assert rep_d["n_days_final"] == rep_f["n_days_final"] - 1
    assert rep_d["n_dropped_days"] == 1
    assert not dropped.isna().any().any()
    assert not filled.isna().any().any()


def test_build_panel_rejects_unknown_policy():
    df, _ = _ragged_frame()
    with pytest.raises(ValueError):
        build_panel(df, missing_policy="interpolate", verbose=False)


# ── Phase 0: pre-registered crisis windows ─────────────────────────────────

def test_crisis_registry_is_well_formed():
    """crisis_frame() asserts internally; this pins the registry's shape."""
    df = crisis_frame()
    assert len(df) == 17, "the paper names exactly 17 crises in Fig. 4"
    assert set(df["tier"]) == {"sharp", "diffuse"}
    assert df["name"].is_unique
    # Black Monday is the one crisis the paper dates explicitly.
    bm = df[df["name"] == "Black Monday"].iloc[0]
    assert str(bm["onset"].date()) == "1987-10-19"


def test_crisis_mask_and_coverage_agree():
    idx = pd.bdate_range("1987-01-01", "2011-02-28")
    cov = coverage(idx)
    assert cov["frac_sharp"] < cov["frac_any"] <= 1.0
    assert cov["n_sharp"] + cov["n_diffuse"] == 17
    # The C2 headline: bands cover a large minority of the timeline, so a
    # detector must be scored against that base rate, not against zero.
    assert 0.25 < cov["frac_any"] < 0.45
    assert cov["frac_sharp"] < 0.06
    assert np.isclose(crisis_mask(idx).mean(), cov["frac_any"])


def test_black_monday_is_inside_its_own_band():
    idx = pd.bdate_range("1987-01-01", "1988-01-01")
    mask = crisis_mask(idx, tiers=("sharp",))
    assert mask.loc[pd.Timestamp("1987-10-19")]
    assert not mask.loc[pd.Timestamp("1987-06-01")]


# ── Phase 1: returns and quality gates ─────────────────────────────────────

def test_log_returns_rejects_non_positive_prices():
    """VHI came back from yfinance with a negative adjusted close."""
    with pytest.raises(ValueError):
        log_returns(np.array([[1.0, 2.0, -3.0]]))


def test_longest_zero_run():
    assert longest_zero_run(np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])) == 3
    assert longest_zero_run(np.array([0.1, 0.2])) == 0


def _gate_fixture(seed=0):
    """Four series: one clean, one frozen, one bad print, one split artifact."""
    rng = np.random.default_rng(seed)
    Y = rng.normal(0, 0.02, size=(4, 400))
    Y[1, 100:160] = 0.0           # frozen for 60 days  -> G1
    Y[2, 200] = 0.8               # spike ...
    Y[2, 201] = -0.8              # ... the next day undoes -> G2
    Y[3, 300] = -1.4              # unadjusted split      -> G3
    return Y, ["CLEAN", "FROZEN", "BADPRINT", "SPLIT"]


def test_quality_gate_catches_each_failure_mode_separately():
    Y, tickers = _gate_fixture()
    keep, rep = quality_gate(Y, tickers, delta_t=30, verbose=False)
    assert rep["g1_frozen_run"] == ["FROZEN"]
    assert rep["g2_reversing_spike"] == ["BADPRINT"]
    assert rep["g3_extreme_return"] == ["SPLIT"]
    assert list(np.array(tickers)[keep]) == ["CLEAN"]
    assert rep["max_frozen_run_after"] < 30


def test_quality_gate_keeps_a_real_crash():
    """AIG fell 60.8 % the day Lehman filed. A gate that removes that is
    removing the crisis from a paper about crises."""
    rng = np.random.default_rng(1)
    Y = rng.normal(0, 0.02, size=(1, 400))
    Y[0, 200] = -0.936
    keep, rep = quality_gate(Y, ["AIG"], delta_t=30, verbose=False)
    assert keep.all() and rep["n_after"] == 1


def test_frozen_run_gate_is_tied_to_delta_t():
    """A 40-day frozen run is fatal at Δt=30 but tolerable at Δt=60... and the
    point of the gate is that it tracks the window length, not a round number."""
    Y, tickers = _gate_fixture()
    assert "FROZEN" in quality_gate(Y, tickers, delta_t=30, verbose=False)[1]["g1_frozen_run"]
    assert "FROZEN" not in quality_gate(Y, tickers, delta_t=90, verbose=False)[1]["g1_frozen_run"]


# ── Phase 2: network construction ──────────────────────────────────────────

def _returns(n=40, t=200, seed=3):
    return np.random.default_rng(seed).normal(0, 0.02, size=(n, t))


def test_window_correlation_matches_numpy():
    Y = _returns()
    C = window_correlation(Y, 10, 30)
    ref = np.corrcoef(Y[:, 10:40])
    assert np.allclose(C, ref, atol=1e-12)
    assert np.allclose(np.diag(C), 1.0)


def test_zero_variance_row_becomes_a_guaranteed_non_edge():
    """A stock frozen through the window gets correlation 0, not NaN."""
    Y = _returns()
    Y[7, 10:40] = 0.0
    C = window_correlation(Y, 10, 30)
    assert not np.isnan(C).any()
    assert np.allclose(C[7, :], 0.0)


def test_n_windows_matches_the_papers_own_relation():
    """N_w = C_p - Delta_t; the paper's 6008 - 30 = 5978 pins the convention."""
    assert n_windows(6008, 30) == 5978


def test_threshold_keeps_exactly_the_target_count_and_tau_is_the_cutoff():
    Y = _returns()
    N = Y.shape[0]
    C = window_correlation(Y, 0, 30)
    k = target_edge_count(N, 0.10)
    iu = np.triu_indices(N, 1)
    tau, pairs, weights = threshold_window(C, k, iu)
    assert len(pairs) == k == len(np.unique(pairs))
    assert np.isclose(tau, weights.min())
    # Exactly k upper-triangle entries are >= tau, and none kept is below it.
    assert (C[iu] >= tau).sum() == k
    assert (weights >= tau).all()


def test_decode_pairs_round_trips_and_is_upper_triangular():
    Y = _returns()
    N = Y.shape[0]
    C = window_correlation(Y, 0, 30)
    iu = np.triu_indices(N, 1)
    _, pairs, _ = threshold_window(C, target_edge_count(N, 0.10), iu)
    ij = decode_pairs(pairs, N)
    assert (ij[:, 0] < ij[:, 1]).all()
    assert np.array_equal(ij[:, 0] * N + ij[:, 1], pairs.astype(np.int64))


def test_every_day_has_the_same_edge_count():
    """The paper's fixed-density claim: average degree is constant by
    construction, so a varying edge count means the thresholding is wrong."""
    Y = _returns(t=90)
    seq = build_network_sequence(Y, 30, 0.10, verbose=False)
    assert seq["edges"].shape == (seq["n_windows"], seq["n_edges"])
    for w in range(seq["n_windows"]):
        assert len(np.unique(seq["edges"][w])) == seq["n_edges"]


def test_mean_correlation_series_matches_the_brute_force_value():
    """The O(N·Δt) shortcut must equal the O(N²·Δt) computation exactly."""
    Y = _returns(t=70)
    fast = mean_correlation_series(Y, 30)
    N = Y.shape[0]
    iu = np.triu_indices(N, 1)
    slow = np.array([window_correlation(Y, w, 30)[iu].mean()
                     for w in range(len(fast))])
    assert np.allclose(fast, slow, atol=1e-12)


def test_window_timestamps_offsets_and_validation():
    dates = pd.bdate_range("2000-01-03", periods=100)
    assert window_timestamps(dates, 30, "t1")[0] == dates[0]
    assert window_timestamps(dates, 30, "midpoint")[0] == dates[15]
    assert window_timestamps(dates, 30, "t2")[0] == dates[30]
    assert len(window_timestamps(dates, 30, "t2")) == n_windows(len(dates), 30)
    with pytest.raises(ValueError):
        window_timestamps(dates, 30, "close")


# ── Phase 3: communities and modularity ────────────────────────────────────

def _two_cliques(n=12):
    """Two disjoint cliques joined by a single edge: k=2, Q close to 0.5."""
    import igraph as ig
    half = n // 2
    edges = [(i, j) for i in range(half) for j in range(i + 1, half)]
    edges += [(i, j) for i in range(half, n) for j in range(i + 1, n)]
    edges.append((0, half))
    return ig.Graph(n=n, edges=edges, directed=False)


def test_louvain_finds_a_planted_two_clique_split():
    g = _two_cliques()
    P = louvain_partitions(g, [0])
    assert P.shape == (1, 12)
    assert int(P[0].max()) + 1 == 2
    assert modularity(g, P[0]) > 0.4


def test_louvain_is_reproducible_and_seeds_actually_differ():
    """Both halves matter: without seeding the run is irreproducible, and if
    the seed did nothing the ten-seed stability analysis would be vacuous."""
    import igraph as ig
    rng = np.random.default_rng(5)
    n = 120
    ij = set()
    while len(ij) < 900:
        a, b = rng.integers(0, n, 2)
        if a != b:
            ij.add((min(a, b), max(a, b)))
    g = ig.Graph(n=n, edges=sorted(ij), directed=False)
    assert np.array_equal(louvain_partitions(g, [1]), louvain_partitions(g, [1]))
    many = louvain_partitions(g, range(12))
    assert len({m.tobytes() for m in many}) > 1


def test_relabel_by_size_orders_communities_descending():
    m = np.array([2, 2, 0, 1, 1, 1, 2, 2])       # sizes: 2->4, 1->3, 0->1
    out = relabel_by_size(m)
    _, counts = np.unique(out, return_counts=True)
    assert list(counts) == sorted(counts, reverse=True)
    # Relabelling must not change the grouping, only the names.
    assert len(np.unique(out)) == len(np.unique(m))
    for a in range(len(m)):
        for b in range(len(m)):
            assert (m[a] == m[b]) == (out[a] == out[b])


def test_mixing_matrix_counts_every_edge_exactly_once():
    g = _two_cliques()
    ep = np.array(g.get_edgelist())
    ep = np.column_stack((ep.min(axis=1), ep.max(axis=1)))
    memb = relabel_by_size(np.asarray(louvain_partitions(g, [0])[0]))
    Pi = mixing_matrix(ep, memb)
    assert Pi.sum() == g.ecount()
    assert np.allclose(Pi, np.triu(Pi))          # upper-triangular by contract
    assert Pi[0, 1] == 1                          # the single bridging edge


def test_modularity_is_recoverable_from_the_compressed_summary():
    """Phase 4's null model only ever sees Pi and the community sizes. If those
    two objects do not determine Q, the paper's compression claim is untestable
    before it is even tested."""
    g = _two_cliques(16)
    memb = relabel_by_size(np.asarray(louvain_partitions(g, [0])[0]))
    ep = np.array(g.get_edgelist())
    ep = np.column_stack((ep.min(axis=1), ep.max(axis=1)))
    Pi = mixing_matrix(ep, memb)
    assert np.isclose(modularity_from_mixing(Pi, community_sizes(memb)),
                      modularity(g, memb), atol=1e-12)


def test_random_partition_scores_about_zero():
    """The floor every modularity number in the paper must be read against."""
    g = _two_cliques(20)
    rng = np.random.default_rng(0)
    qs = []
    for _ in range(30):
        m = rng.integers(0, 2, size=20)
        qs.append(modularity(g, m))
    assert abs(np.mean(qs)) < 0.15


def test_mixing_store_round_trips_ragged_days(tmp_path):
    """k swings from 4 to 156 across the sample, so storage must be ragged."""
    pis, sizes = [], []
    for k in (2, 5, 3):
        Pi = np.triu(np.arange(k * k).reshape(k, k))
        pis.append(Pi)
        sizes.append(np.arange(1, k + 1))
    store = MixingStore.build(pis, sizes)
    path = str(tmp_path / "mix.npz")
    store.save(path)
    back = MixingStore.load(path)
    assert len(back) == 3
    for w in range(3):
        Pi, sz = back[w]
        assert np.array_equal(Pi, pis[w])
        assert np.array_equal(sz, sizes[w])


# ── Phase 4: null models ───────────────────────────────────────────────────

def test_block_probabilities_follow_equation_2():
    """pi_aa = 2*Pi_aa/(|a|(|a|-1)), pi_ab = Pi_ab/(|a||b|)."""
    Pi = np.array([[6, 4], [0, 3]], dtype=np.int64)   # upper-triangular
    sizes = np.array([4, 3])
    P = block_probabilities(Pi, sizes)
    assert np.isclose(P[0, 0], 2 * 6 / (4 * 3))
    assert np.isclose(P[1, 1], 2 * 3 / (3 * 2))
    assert np.isclose(P[0, 1], 4 / (4 * 3))
    assert np.allclose(P, P.T)
    assert ((P >= 0) & (P <= 1)).all()


def test_block_probabilities_handle_singleton_communities():
    """Black Monday leaves 152 singleton communities; |a|(|a|-1) = 0 for each,
    and a division by zero there would poison the whole crisis period."""
    Pi = np.array([[0, 1], [0, 0]], dtype=np.int64)
    P = block_probabilities(Pi, np.array([1, 1]))
    assert np.isfinite(P).all()
    assert P[0, 0] == 0.0 and P[1, 1] == 0.0
    assert np.isclose(P[0, 1], 1.0)


def test_community_graph_reproduces_the_planted_modularity():
    """The unit test for Eq. 2: a network generated from Pi and the sizes must
    have roughly the modularity of the partition those came from."""
    sizes = np.array([40, 40, 40])
    Pi = np.array([[300, 30, 30], [0, 300, 30], [0, 0, 300]], dtype=np.int64)
    planted = np.repeat(np.arange(3), sizes)
    rng = np.random.default_rng(0)
    qs = []
    for _ in range(8):
        g, ne = community_graph(sizes, Pi, rng)
        assert g.vcount() == 120
        qs.append(modularity(g, planted))
    target = modularity_from_mixing(Pi, sizes)
    assert abs(np.mean(qs) - target) < 0.03


def test_community_graph_edge_count_is_random_not_exact():
    """Audit D2, kept deliberately: independent coin flips match the target
    edge count only in expectation. A generator that matched it exactly would
    be a different model from the paper's."""
    sizes = np.array([30, 30])
    Pi = np.array([[200, 50], [0, 200]], dtype=np.int64)
    rng = np.random.default_rng(1)
    counts = [community_graph(sizes, Pi, rng)[1] for _ in range(40)]
    assert len(set(counts)) > 1
    assert abs(np.mean(counts) - Pi.sum()) < 0.05 * Pi.sum()


def test_configuration_rewire_preserves_degrees_and_edge_count_exactly():
    g = _two_cliques(30)
    before = sorted(g.degree())
    g2, loss = configuration_graph(g, seed=3, method="rewire")
    assert sorted(g2.degree()) == before
    assert g2.ecount() == g.ecount()
    assert loss["n_multi_collapsed"] == 0
    assert not any(g2.is_loop())
    assert max(g2.count_multiple()) == 1


def test_configuration_rewire_actually_randomizes():
    """Preserving the degree sequence is necessary but not sufficient: if the
    rewiring did nothing, the null would equal the real network."""
    g = _two_cliques(40)
    g2, _ = configuration_graph(g, seed=3, method="rewire")
    shared = len(set(map(tuple, map(sorted, g2.get_edgelist())))
                 & set(map(tuple, map(sorted, g.get_edgelist()))))
    assert shared < g.ecount()
    assert modularity(g2, louvain_partitions(g2, [0])[0]) < modularity(
        g, louvain_partitions(g, [0])[0])


def test_stub_matching_loses_edges_which_is_why_it_is_not_the_default():
    """Audit D4 made concrete: collapsing multi-edges lowers the null's density
    below the graph it is compared against."""
    g = _two_cliques(40)
    _, loss = configuration_graph(g, seed=3, method="stub")
    assert loss["edges_after"] < loss["edges_before"]
    assert loss["n_multi_collapsed"] + loss["n_self_loops"] > 0


def test_configuration_graph_rejects_unknown_method():
    with pytest.raises(ValueError):
        configuration_graph(_two_cliques(10), seed=0, method="vl")


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
