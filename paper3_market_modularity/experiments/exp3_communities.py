"""
Phase 3 — Communities and the three modularities (Silva et al. 2015, Figs. 3-5).

Louvain is run on every daily network, ten times with different seeds. Seed 0's
partition is the canonical C(t): it feeds Pi, the fixed and lagged modularities,
and Phase 4's null model, so the pipeline stays exactly as faithful to the paper
as a single Louvain run would be. The other nine seeds are kept for one purpose
only — to measure how much of the daily modularity series is Louvain's
randomness rather than the market (audit B1/B2). That comparison is the point:
a day-to-day wiggle smaller than the seed-to-seed spread on a single day is not
a finding about markets.

Three modularities, identical formula, only the partition changes:
    dynamical  Q(g_t, C(t))          re-detected daily
    lagged     Q(g_t, C(t - 100))    a partition 100 trading days stale
    fixed      Q(g_t, C(0))          the first day's partition, held forever

The paper plots all three but never takes their difference. Q_dyn - Q_lag is
the amount of modularity that exists *only* because the communities were
allowed to move, which is the quantity its "modular dynamics" title is about.

Run:
    /opt/anaconda3/bin/python experiments/exp3_communities.py
    /opt/anaconda3/bin/python experiments/exp3_communities.py --force
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp1_data import (FIT_SEED, LAG_T_DELTA, N_LOUVAIN_SEEDS,  # noqa: E402
                       RESULTS, TIMESTAMP_CONVENTION)
from net.communities import (MixingStore, community_sizes,  # noqa: E402
                             edge_endpoints, graph_from_pairs,
                             louvain_partitions, mixing_matrix, modularity,
                             modularity_from_mixing)
from utils.crisis_dates import crisis_mask  # noqa: E402

CACHE = os.path.join(RESULTS, "phase3_communities.json")
EDGES_NPY = os.path.join(ROOT, "data", "edges.npy")
PARTITIONS_NPY = os.path.join(ROOT, "data", "partitions.npy")
MIXING_NPZ = os.path.join(ROOT, "data", "mixing.npz")
NET_SERIES = os.path.join(RESULTS, "series", "networks.parquet")
SERIES_PARQUET = os.path.join(RESULTS, "series", "communities.parquet")

# The Fig. 3 panel, carried over from Phase 2 so the two zooms line up.
FIG3_PANEL = ("1987-08-01", "1988-03-31")
BLACK_MONDAY = ("1987-10-14", "1987-11-13")

# Louvain seeds. Derived from FIT_SEED so the whole run is reproducible from
# the two seeds recorded in the config module.
SEEDS = [FIT_SEED + i for i in range(N_LOUVAIN_SEEDS)]


def run(force: bool = False, verbose: bool = True) -> dict:
    if os.path.exists(CACHE) and os.path.exists(PARTITIONS_NPY) and not force:
        with open(CACHE) as f:
            report = json.load(f)
        if verbose:
            print(f"[skip] {CACHE} exists — pass force=True to rebuild.")
        return report

    net = pd.read_parquet(NET_SERIES)
    edges = np.load(EDGES_NPY, mmap_mode="r")
    n_w = edges.shape[0]
    n_nodes = int(json.load(open(os.path.join(RESULTS, "phase2_networks.json")))["n_nodes"])
    if verbose:
        print(f"[communities] {n_w} networks, {n_nodes} nodes, "
              f"Louvain x {len(SEEDS)} seeds, lag t_Δ = {LAG_T_DELTA}")

    parts = np.empty((n_w, len(SEEDS), n_nodes), dtype=np.int16)
    q_seeds = np.empty((n_w, len(SEEDS)), dtype=np.float64)
    k_seeds = np.empty((n_w, len(SEEDS)), dtype=np.int32)
    q_fixed = np.empty(n_w)
    q_lag = np.full(n_w, np.nan)
    largest_frac = np.empty(n_w)
    n_singleton = np.empty(n_w, dtype=np.int32)
    pis, sizes_list = [], []

    t0 = time.time()
    for w in range(n_w):
        g = graph_from_pairs(edges[w], n_nodes)
        P = louvain_partitions(g, SEEDS)
        parts[w] = P
        for r in range(len(SEEDS)):
            q_seeds[w, r] = modularity(g, P[r])
            k_seeds[w, r] = int(P[r].max()) + 1

        canon = P[0]
        sizes = community_sizes(canon)
        Pi = mixing_matrix(edge_endpoints(edges[w], n_nodes), canon, len(sizes))
        pis.append(Pi)
        sizes_list.append(sizes)
        largest_frac[w] = sizes.max() / n_nodes
        n_singleton[w] = int((sizes == 1).sum())

        q_fixed[w] = modularity(g, parts[0, 0])
        if w >= LAG_T_DELTA:
            q_lag[w] = modularity(g, parts[w - LAG_T_DELTA, 0])

        if verbose and w and w % 1000 == 0:
            el = time.time() - t0
            print(f"  window {w:5d}/{n_w}  {el:6.1f}s elapsed, "
                  f"~{el / w * (n_w - w):5.1f}s left")

    np.save(PARTITIONS_NPY, parts)
    MixingStore.build(pis, sizes_list).save(MIXING_NPZ)
    diag = structure_diagnostics(edges, parts, n_nodes, verbose=verbose)

    q_dyn = q_seeds[:, 0]
    series = pd.DataFrame({
        "q_dyn": q_dyn,
        "q_fixed": q_fixed,
        "q_lagged": q_lag,
        "q_dyn_minus_lag": q_dyn - q_lag,
        "q_seed_mean": q_seeds.mean(axis=1),
        "q_seed_sd": q_seeds.std(axis=1, ddof=1),
        "q_seed_range": q_seeds.max(axis=1) - q_seeds.min(axis=1),
        "k": k_seeds[:, 0],
        "k_seed_mean": k_seeds.mean(axis=1),
        "largest_community_frac": largest_frac,
        "n_singleton": n_singleton,
    }, index=net.index)
    series.to_parquet(SERIES_PARQUET)

    report = _summarize(series, q_seeds, net, n_nodes, elapsed=time.time() - t0)
    report["diagnostics"] = diag
    with open(CACHE, "w") as f:
        json.dump(report, f, indent=2)
    return report


# Lags for the partition-decay curve. The paper picks t_Delta = 100 and never
# says why; this brackets it by two orders of magnitude on both sides.
DECAY_LAGS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)
N_DIAG_DAYS = 200
N_NOISE_WINDOWS = 300


def structure_diagnostics(edges, parts, n_nodes, verbose: bool = True) -> dict:
    """Three things the paper's modularity numbers cannot be read without.

    1. **Partition decay.** Q(g_t, C(t-L)) as a function of L. The paper's
       t_Delta = 100 is one arbitrary point on this curve; the curve itself is
       the timescale on which market community structure turns over, which is
       the quantity the phrase "modular dynamics" actually refers to.
    2. **The random-partition floor.** Q of a label-shuffled partition with the
       identical community sizes. Everything above is structure, and it puts
       the fixed and lagged modularities on a scale.
    3. **Louvain on the A4 noise networks.** Phase 2 showed that i.i.d. Gaussian
       returns still produce a full, plausible network under fixed-density
       thresholding. The question this answers is whether they also produce a
       *modular* one -- i.e. whether the paper's headline Q is evidence of
       market community structure at all, or the generic modularity of any
       sparse thresholded correlation graph.

    Note that (3) is bounded by audit A5 (market-mode removal) being out of
    scope: real returns carry a dominant market factor that i.i.d. noise does
    not, and that factor pushes real modularity *down*. So (3) compares the
    paper's construction against a noise baseline, not against a factor-matched
    one, and should be read as "the level of Q is not by itself evidence",
    not as "the real market is indistinguishable from noise".
    """
    from net.construct import build_network_sequence
    from exp1_data import DATA_SEED, DELTA_T, EDGE_FRACTION

    rng = np.random.default_rng(DATA_SEED)
    n_w = edges.shape[0]
    days = np.sort(rng.choice(np.arange(max(DECAY_LAGS), n_w), N_DIAG_DAYS,
                              replace=False))
    graphs = {int(w): graph_from_pairs(edges[w], n_nodes) for w in days}

    decay = {}
    for L in DECAY_LAGS:
        decay[L] = float(np.mean([modularity(graphs[int(w)], np.asarray(parts[w - L, 0]))
                                  for w in days]))

    floor = []
    for w in days:
        m = np.asarray(parts[w, 0]).copy()
        rng.shuffle(m)
        floor.append(modularity(graphs[int(w)], m))

    # Louvain on pure noise, identical pipeline.
    fake = np.random.default_rng(DATA_SEED).standard_normal(
        (n_nodes, N_NOISE_WINDOWS + DELTA_T))
    seq = build_network_sequence(fake, DELTA_T, EDGE_FRACTION, verbose=False)
    q_noise, k_noise = [], []
    for w in range(seq["n_windows"]):
        g = graph_from_pairs(seq["edges"][w], n_nodes)
        p = louvain_partitions(g, [FIT_SEED])[0]
        q_noise.append(modularity(g, p))
        k_noise.append(int(p.max()) + 1)

    out = {
        "decay_lags": list(DECAY_LAGS),
        "decay_q": [decay[L] for L in DECAY_LAGS],
        "decay_half_life_days": _half_life(DECAY_LAGS, [decay[L] for L in DECAY_LAGS]),
        "decay_is_monotone": bool(all(decay[a] >= decay[b] - 1e-9 for a, b in
                                      zip(DECAY_LAGS, DECAY_LAGS[1:]))),
        "decay_retained_at_100": float(decay[100] / decay[0]),
        "random_partition_floor": float(np.mean(floor)),
        "random_partition_floor_sd": float(np.std(floor)),
        "noise_q_mean": float(np.mean(q_noise)),
        "noise_q_sd": float(np.std(q_noise)),
        "noise_k_median": float(np.median(k_noise)),
        "noise_q_values": [round(float(v), 6) for v in q_noise],
        "n_noise_windows": int(seq["n_windows"]),
        "n_diag_days": int(len(days)),
    }
    if verbose:
        print(f"[diagnostics] partition half-life ≈ {out['decay_half_life_days']:.0f} "
              f"trading days; Q retained at t_Δ=100: {out['decay_retained_at_100']:.1%}")
        print(f"[diagnostics] random-partition floor Q = "
              f"{out['random_partition_floor']:+.4f}; "
              f"i.i.d. noise networks Q = {out['noise_q_mean']:.4f} ± "
              f"{out['noise_q_sd']:.4f}")
    return out


def _half_life(lags, qs) -> float:
    """First lag at which Q falls below half its lag-0 value, interpolated."""
    target = qs[0] / 2.0
    for (a, qa), (b, qb) in zip(zip(lags, qs), zip(lags[1:], qs[1:])):
        if qb <= target <= qa:
            if qa == qb:
                return float(a)
            return float(a + (qa - target) / (qa - qb) * (b - a))
    return float("nan")


def _summarize(series, q_seeds, net, n_nodes, elapsed) -> dict:
    """Everything the acceptance tests and the report need, as plain numbers."""
    q = series["q_dyn"]
    lag_ok = series.dropna(subset=["q_lagged"])
    # What is actually required: a partition re-detected on today's graph must
    # score at least as well as any stale partition. This has no exceptions.
    viol_dyn_fix = int((lag_ok["q_fixed"] > lag_ok["q_dyn"] + 1e-12).sum())
    viol_dyn_lag = int((lag_ok["q_lagged"] > lag_ok["q_dyn"] + 1e-12).sum())
    # What is NOT required, and does not hold: that a 100-day-old partition
    # beats a 25-year-old one. See `structure_diagnostics` -- both sit close to
    # the random-partition floor, and C(0) additionally encodes a persistent
    # split that a 100-day-old Louvain fit does not. The violations are not
    # scattered noise; they are concentrated before 1999, so they are reported
    # by era rather than asserted away.
    fx = lag_ok["q_fixed"] > lag_ok["q_lagged"]
    by_year = fx.groupby(lag_ok.index.year).mean()

    # Audit B1's headline comparison, computed rather than asserted.
    day_moves = q.diff().abs().dropna()
    seed_sd = series["q_seed_sd"]

    crisis = crisis_mask(series.index, tiers=("sharp",))
    bm = series.loc[BLACK_MONDAY[0]:BLACK_MONDAY[1]]

    return {
        "n_windows": int(len(series)),
        "n_nodes": n_nodes,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "lag_t_delta": LAG_T_DELTA,
        "elapsed_sec": round(elapsed, 1),
        "q_dyn": {"mean": float(q.mean()), "sd": float(q.std()),
                  "min": float(q.min()), "max": float(q.max()),
                  "argmin_date": str(q.idxmin().date()),
                  "argmax_date": str(q.idxmax().date())},
        "q_fixed_mean": float(series["q_fixed"].mean()),
        "q_lagged_mean": float(lag_ok["q_lagged"].mean()),
        "q_dyn_minus_lag_mean": float(lag_ok["q_dyn_minus_lag"].mean()),
        "ordering": {
            "fixed_over_dyn": viol_dyn_fix,
            "lagged_over_dyn": viol_dyn_lag,
            "n_days_tested": int(len(lag_ok)),
            "worst_stale_minus_dyn": float(
                max((lag_ok["q_lagged"] - lag_ok["q_dyn"]).max(),
                    (lag_ok["q_fixed"] - lag_ok["q_dyn"]).max())),
            "fixed_over_lagged_days": int(fx.sum()),
            "fixed_over_lagged_frac": float(fx.mean()),
            "fixed_over_lagged_frac_pre1999": float(fx[lag_ok.index.year < 1999].mean()),
            "fixed_over_lagged_frac_post1999": float(fx[lag_ok.index.year >= 1999].mean()),
            "fixed_lagged_gap_mean_abs": float(
                (lag_ok["q_fixed"] - lag_ok["q_lagged"]).abs().mean()),
            "by_year": {int(y): round(float(v), 3) for y, v in by_year.items()},
        },
        "seed_noise": {
            "sd_median": float(seed_sd.median()),
            "range_median": float(series["q_seed_range"].median()),
            "day_move_median": float(day_moves.median()),
            "ratio_seed_range_to_day_move": float(
                series["q_seed_range"].median() / day_moves.median()),
            "frac_days_seed_range_exceeds_day_move": float(
                (series["q_seed_range"].to_numpy()[1:] > day_moves.to_numpy()).mean()),
            "k_disagrees_across_seeds_frac": float(
                (series["k"] != series["k_seed_mean"]).mean()),
        },
        "k": {"median": float(series["k"].median()),
              "min": int(series["k"].min()), "max": int(series["k"].max()),
              "argmax_date": str(series["k"].idxmax().date()),
              "crisis_median": float(series.loc[crisis, "k"].median()),
              "calm_median": float(series.loc[~crisis, "k"].median())},
        "largest_community_frac_median": float(series["largest_community_frac"].median()),
        "black_monday": {
            "q_min": float(bm["q_dyn"].min()),
            "q_min_date": str(bm["q_dyn"].idxmin().date()),
            "q_pctile": float((q < bm["q_dyn"].min()).mean()),
            "k_max": int(bm["k"].max()),
            "n_singleton_max": int(bm["n_singleton"].max()),
        },
        "crisis_vs_calm": {
            "q_crisis_median": float(q[crisis].median()),
            "q_calm_median": float(q[~crisis].median()),
        },
        "corr_q_with_tau": float(np.corrcoef(q, net["tau"])[0, 1]),
        "corr_q_with_isolated": float(np.corrcoef(q, net["n_isolated"])[0, 1]),
    }


def plot(save_dir: str = None, verbose: bool = True):
    from utils.plots import (plot_event_zoom, plot_modularity_overview,
                             plot_partition_decay, plot_three_modularities)
    save_dir = save_dir or os.path.join(ROOT, "figures")
    os.makedirs(save_dir, exist_ok=True)
    s = pd.read_parquet(SERIES_PARQUET)
    net = pd.read_parquet(NET_SERIES)
    with open(CACHE) as f:
        d = json.load(f)["diagnostics"]

    plot_event_zoom(
        s.index,
        {"dynamical modularity Q(t)": s["q_dyn"].to_numpy(),
         "number of communities k(t)": s["k"].to_numpy().astype(float),
         "isolated nodes": net["n_isolated"].to_numpy().astype(float)},
        FIG3_PANEL[0], FIG3_PANEL[1],
        "Fig. 3 — Black Monday: what the crash does to community structure",
        onset="1987-10-19",
        save=os.path.join(save_dir, "fig3_blackmonday.png"))

    plot_modularity_overview(
        s.index, s["q_dyn"].to_numpy(), s["k"].to_numpy().astype(float),
        seed_lo=(s["q_seed_mean"] - s["q_seed_sd"]).to_numpy(),
        seed_hi=(s["q_seed_mean"] + s["q_seed_sd"]).to_numpy(),
        stamp=TIMESTAMP_CONVENTION,
        save=os.path.join(save_dir, "fig4_modularity.png"))

    plot_three_modularities(
        s.index, s["q_dyn"].to_numpy(), s["q_lagged"].to_numpy(),
        s["q_fixed"].to_numpy(), s["q_dyn_minus_lag"].to_numpy(),
        lag=LAG_T_DELTA, stamp=TIMESTAMP_CONVENTION,
        save=os.path.join(save_dir, "fig5_three_modularities.png"))

    plot_partition_decay(
        d["decay_lags"], d["decay_q"], d["random_partition_floor"],
        q_real=s["q_dyn"].to_numpy(),
        q_noise=np.asarray(d["noise_q_values"]),
        lag_marker=LAG_T_DELTA,
        save=os.path.join(save_dir, "fig5b_partition_decay.png"))


def print_report(r: dict) -> None:
    q, sn, k = r["q_dyn"], r["seed_noise"], r["k"]
    print("\n── Phase 3 communities " + "─" * 45)
    print(f"  windows / nodes / seeds: {r['n_windows']} / {r['n_nodes']} / {r['n_seeds']}"
          f"   ({r['elapsed_sec']:.0f}s)")
    print(f"  dynamical Q            : mean {q['mean']:.4f} ± {q['sd']:.4f}, "
          f"range {q['min']:.4f}–{q['max']:.4f}")
    print(f"     min on {q['argmin_date']}, max on {q['argmax_date']}")
    print(f"  lagged Q (t_Δ={r['lag_t_delta']:>3})    : mean {r['q_lagged_mean']:.4f}")
    print(f"  fixed Q  (C(0))        : mean {r['q_fixed_mean']:.4f}")
    print(f"  Q_dyn − Q_lag          : mean {r['q_dyn_minus_lag_mean']:.4f}"
          "   <- modularity that exists only because communities may move")
    v = r["ordering"]
    print(f"  stale beats fresh       : {v['fixed_over_dyn'] + v['lagged_over_dyn']} "
          f"days of {v['n_days_tested']}  (worst stale−dyn "
          f"{v['worst_stale_minus_dyn']:+.5f})   <- must be 0")
    print(f"  fixed beats lagged      : {v['fixed_over_lagged_days']} days "
          f"({v['fixed_over_lagged_frac']:.1%}) — "
          f"{v['fixed_over_lagged_frac_pre1999']:.0%} of days before 1999, "
          f"{v['fixed_over_lagged_frac_post1999']:.0%} after"
          "   <- not an ordering, see diagnostics")
    print(f"  communities k(t)       : median {k['median']:.0f}, "
          f"range {k['min']}–{k['max']} (max on {k['argmax_date']}); "
          f"crisis {k['crisis_median']:.0f} vs calm {k['calm_median']:.0f}")
    bm = r["black_monday"]
    print(f"  Black Monday window    : Q falls to {bm['q_min']:.4f} on {bm['q_min_date']} "
          f"({bm['q_pctile']:.2%} of days lower), k → {bm['k_max']}, "
          f"{bm['n_singleton_max']} singleton communities")
    cv = r["crisis_vs_calm"]
    print(f"  Q median crisis vs calm: {cv['q_crisis_median']:.4f} vs "
          f"{cv['q_calm_median']:.4f}  (sharp-tier bands only)")
    print("  ── audit B1: how much of the Q series is Louvain, not the market ──")
    print(f"     seed-to-seed spread on one day : sd {sn['sd_median']:.5f}, "
          f"range {sn['range_median']:.5f}  (median day)")
    print(f"     day-to-day |ΔQ|                : {sn['day_move_median']:.5f}  (median)")
    print(f"     ratio                          : {sn['ratio_seed_range_to_day_move']:.2f}×"
          f"   — seed noise exceeds the daily move on "
          f"{sn['frac_days_seed_range_exceeds_day_move']:.1%} of days")
    print(f"  Q vs τ(t) correlation  : {r['corr_q_with_tau']:+.3f}"
          f"    Q vs isolated nodes: {r['corr_q_with_isolated']:+.3f}")
    d = r["diagnostics"]
    print("  ── how long a partition stays valid, and what Q = 0.22 is worth ──")
    print("     Q(g_t, C(t−L)) by lag L:")
    print("        L  " + "".join(f"{L:>8d}" for L in d["decay_lags"]))
    print("        Q  " + "".join(f"{v:>8.4f}" for v in d["decay_q"]))
    print(f"     half-life ≈ {d['decay_half_life_days']:.0f} trading days; at the "
          f"paper's t_Δ = 100 only {d['decay_retained_at_100']:.1%} of Q survives")
    print(f"     random partition, same sizes : Q = {d['random_partition_floor']:+.4f}"
          f" ± {d['random_partition_floor_sd']:.4f}  (the floor)")
    print(f"     i.i.d. NOISE networks        : Q = {d['noise_q_mean']:.4f}"
          f" ± {d['noise_q_sd']:.4f}, k median {d['noise_k_median']:.0f}")
    print(f"     real networks                : Q = {r['q_dyn']['mean']:.4f}"
          f" ± {r['q_dyn']['sd']:.4f}, k median {r['k']['median']:.0f}")
    print("     -> the LEVEL of modularity is not evidence of market community")
    print("        structure; only its VARIATION is. (Bounded by A5, untested.)")


def acceptance(r: dict, verbose: bool = True) -> bool:
    s = pd.read_parquet(SERIES_PARQUET)
    net = pd.read_parquet(NET_SERIES)
    edges = np.load(EDGES_NPY, mmap_mode="r")
    parts = np.load(PARTITIONS_NPY, mmap_mode="r")
    store = MixingStore.load(MIXING_NPZ)

    # Independent re-derivation of one day: rebuild the graph, re-run Louvain
    # with the recorded seed, and check the stored partition and Q come back.
    w = 4321
    g = graph_from_pairs(edges[w], r["n_nodes"])
    redo = louvain_partitions(g, [SEEDS[0]])[0]
    repro = bool(np.array_equal(redo, np.asarray(parts[w, 0])))

    # Pi + sizes must be a sufficient statistic for modularity (Phase 4 leans
    # on this: the null model only ever sees these two objects).
    Pi, sizes = store[w]
    q_from_summary = modularity_from_mixing(Pi, sizes)
    v = r["ordering"]
    d = r["diagnostics"]
    q = s["q_dyn"]

    checks = [
        ("dynamical Q lies in [0.05, 0.55] every day",
         bool((q > 0.05).all() and (q < 0.55).all()),
         f"range {q.min():.4f}–{q.max():.4f}"),
        ("a re-detected partition beats every stale one, on every single day",
         v["fixed_over_dyn"] == 0 and v["lagged_over_dyn"] == 0,
         f"{v['fixed_over_dyn']} + {v['lagged_over_dyn']} violations "
         f"in {v['n_days_tested']} days"),
        ("partition quality decays monotonically with lag",
         d["decay_is_monotone"],
         "Q(g_t, C(t−L)) non-increasing in L over "
         f"L = {d['decay_lags'][0]}…{d['decay_lags'][-1]}"),
        ("the paper's t_Δ = 100 sits past the decay, not on it",
         d["decay_retained_at_100"] < 0.25,
         f"{d['decay_retained_at_100']:.1%} of Q survives 100 days; "
         f"half-life ≈ {d['decay_half_life_days']:.0f} days"),
        ("stale modularities sit near the random-partition floor",
         abs(d["random_partition_floor"]) < 0.02
         and r["q_lagged_mean"] < 0.25 * r["q_dyn"]["mean"],
         f"floor {d['random_partition_floor']:+.4f}, "
         f"lagged {r['q_lagged_mean']:.4f} vs dynamical {r['q_dyn']['mean']:.4f}"),
        ("Louvain Q on i.i.d. noise networks is reported next to the real one",
         "noise_q_mean" in d and d["n_noise_windows"] > 100,
         f"noise {d['noise_q_mean']:.4f} ± {d['noise_q_sd']:.4f} vs "
         f"real {r['q_dyn']['mean']:.4f} ± {r['q_dyn']['sd']:.4f}"),
        ("Q drops sharply inside the Black Monday window",
         r["black_monday"]["q_pctile"] <= 0.05,
         f"Q = {r['black_monday']['q_min']:.4f} on {r['black_monday']['q_min_date']}, "
         f"{r['black_monday']['q_pctile']:.2%} of days lower"),
        ("every series has length N_w with no NaNs outside the lag warm-up",
         len(s) == len(net) and int(s.drop(columns=["q_lagged", "q_dyn_minus_lag"])
                                    .isna().sum().sum()) == 0
         and int(s["q_lagged"].isna().sum()) == LAG_T_DELTA,
         f"{len(s)} rows, {int(s['q_lagged'].isna().sum())} lag warm-up NaNs"),
        ("stored partition reproduces from its recorded seed",
         repro, f"window {w}, seed {SEEDS[0]}"),
        ("Π and community sizes reproduce Q exactly (sufficient-statistic check)",
         abs(q_from_summary - float(q.iloc[w])) < 1e-10,
         f"{q_from_summary:.10f} vs {float(q.iloc[w]):.10f}"),
        ("Π edge total equals the network's edge count on every sampled day",
         all(store[i][0].sum() == edges.shape[1] for i in range(0, len(store), 500)),
         f"{edges.shape[1]} edges/day"),
        ("k(t) is plausible: median in [3, 20] and never 1",
         3 <= r["k"]["median"] <= 20 and r["k"]["min"] > 1,
         f"median {r['k']['median']:.0f}, min {r['k']['min']}, max {r['k']['max']}"),
        ("seed-to-seed spread is reported against the day-to-day move (B1)",
         "ratio_seed_range_to_day_move" in r["seed_noise"],
         f"{r['seed_noise']['ratio_seed_range_to_day_move']:.2f}× the median daily move"),
    ]
    if verbose:
        print("\n── Phase 3 acceptance tests " + "─" * 40)
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
        print(f"  {sum(ok for _, ok, _ in checks)}/{len(checks)} passed")
    return all(ok for _, ok, _ in checks)


if __name__ == "__main__":
    rep = run(force="--force" in sys.argv)
    print_report(rep)
    plot()
    sys.exit(0 if acceptance(rep) else 1)
