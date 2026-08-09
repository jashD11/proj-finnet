"""
Phase 4 — The two null models and the modularity noise floor (Sec. III, Eq. 2).

Generates, for every one of the 6,315 days, ten realizations of each model:

    community model      from Pi(t) and the community sizes alone   (Eq. 2)
    configuration model  from the degree sequence alone

and validates each generator against what it is supposed to preserve, before
Phase 5 starts measuring anything with them. A generator that is wrong here
would produce eight plausible-looking measure series and a completely false
conclusion, so the checks are the point of this phase.

The configuration model's modularity is the number this phase exists to
produce: it is the level of modularity you get from the degree distribution
alone, and every modularity figure from here on is unreadable without it
(audit B6). The paper draws it in Fig. 6(a1) and never states its value.

Null-model graphs are never stored -- 126,300 of them would be ~700 M edges.
They are regenerated on demand from Pi, the sizes, and the recorded seed, which
is also why every seed here is a pure function of (FIT_SEED, window, replicate).

Run:
    /opt/anaconda3/bin/python experiments/exp4_nulls.py
    /opt/anaconda3/bin/python experiments/exp4_nulls.py --force
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

from exp1_data import FIT_SEED, N_REPLICATES, RESULTS  # noqa: E402
from net.communities import (MixingStore, graph_from_pairs,  # noqa: E402
                             louvain_partitions, modularity)
from net.nullmodels import (CONFIG_METHOD, REWIRE_SWEEPS,  # noqa: E402
                            community_graph, configuration_graph,
                            degree_sequence)
from utils.parallel import map_windows, stack  # noqa: E402

CACHE = os.path.join(RESULTS, "phase4_nulls.json")
EDGES_NPY = os.path.join(ROOT, "data", "edges.npy")
PARTITIONS_NPY = os.path.join(ROOT, "data", "partitions.npy")
MIXING_NPZ = os.path.join(ROOT, "data", "mixing.npz")
COMM_SERIES = os.path.join(RESULTS, "series", "communities.parquet")
SERIES_PARQUET = os.path.join(RESULTS, "series", "nulls.parquet")

# Days sampled for the stub-matching comparison. The alternative reading is
# documented, not adopted, so it is priced accordingly.
N_STUB_DAYS = 250

_G = {}


def _init(edges_path, parts_path, mixing_path, n_nodes):
    """Open the big arrays once per worker process."""
    _G["edges"] = np.load(edges_path, mmap_mode="r")
    _G["parts"] = np.load(parts_path, mmap_mode="r")
    _G["store"] = MixingStore.load(mixing_path)
    _G["n"] = n_nodes


def _seed(w: int, r: int) -> int:
    """Deterministic per-(day, replicate) seed, independent of scheduling."""
    return int(FIT_SEED * 1_000_003 + w * 131 + r)


def _worker(span):
    lo, hi = span
    n = _G["n"]
    R = N_REPLICATES
    out = {k: np.empty((hi - lo, R), dtype=np.float64)
           for k in ("q_comm", "q_comm_planted", "q_conf")}
    out["edges_comm"] = np.empty((hi - lo, R), dtype=np.int64)
    out["deg_ok"] = np.ones(hi - lo, dtype=bool)
    out["edges_conf"] = np.empty((hi - lo, R), dtype=np.int64)

    for i, w in enumerate(range(lo, hi)):
        real = graph_from_pairs(_G["edges"][w], n)
        deg = degree_sequence(real)
        Pi, sizes = _G["store"][w]
        planted = np.repeat(np.arange(len(sizes)), sizes)

        for r in range(R):
            rng = np.random.default_rng(_seed(w, r))
            gb, ne = community_graph(sizes, Pi, rng, n)
            out["edges_comm"][i, r] = ne
            out["q_comm_planted"][i, r] = modularity(gb, planted)
            out["q_comm"][i, r] = modularity(gb, louvain_partitions(gb, [_seed(w, r)])[0])

            gc, _ = configuration_graph(real, _seed(w, r) + 7)
            out["edges_conf"][i, r] = gc.ecount()
            out["q_conf"][i, r] = modularity(gc, louvain_partitions(gc, [_seed(w, r)])[0])
            if r == 0:
                out["deg_ok"][i] = np.array_equal(np.sort(degree_sequence(gc)),
                                                  np.sort(deg))
    return out


def run(force: bool = False, verbose: bool = True, n_workers: int = None) -> dict:
    if os.path.exists(CACHE) and os.path.exists(SERIES_PARQUET) and not force:
        with open(CACHE) as f:
            report = json.load(f)
        if verbose:
            print(f"[skip] {CACHE} exists — pass force=True to rebuild.")
        return report

    comm = pd.read_parquet(COMM_SERIES)
    edges = np.load(EDGES_NPY, mmap_mode="r")
    n_w, n_real_edges = edges.shape
    n_nodes = int(json.load(open(os.path.join(RESULTS, "phase2_networks.json")))["n_nodes"])
    if verbose:
        print(f"[nulls] {n_w} days x {N_REPLICATES} replicates x 2 models = "
              f"{n_w * N_REPLICATES * 2:,} graphs; config method = "
              f"{CONFIG_METHOD} ({REWIRE_SWEEPS}x m swaps)")

    t0 = time.time()
    chunks = map_windows(_worker, n_w,
                         initializer=_init,
                         initargs=(EDGES_NPY, PARTITIONS_NPY, MIXING_NPZ, n_nodes),
                         n_workers=n_workers, chunk_size=64,
                         verbose=verbose, label="nulls")
    q_comm = stack(chunks, "q_comm")
    q_comm_planted = stack(chunks, "q_comm_planted")
    q_conf = stack(chunks, "q_conf")
    edges_comm = stack(chunks, "edges_comm")
    edges_conf = stack(chunks, "edges_conf")
    deg_ok = stack(chunks, "deg_ok")
    elapsed = time.time() - t0

    series = pd.DataFrame({
        "q_comm_mean": q_comm.mean(axis=1), "q_comm_sd": q_comm.std(axis=1, ddof=1),
        "q_comm_planted_mean": q_comm_planted.mean(axis=1),
        "q_conf_mean": q_conf.mean(axis=1), "q_conf_sd": q_conf.std(axis=1, ddof=1),
        "edges_comm_mean": edges_comm.mean(axis=1),
        "edges_comm_sd": edges_comm.std(axis=1, ddof=1),
        "edges_conf_mean": edges_conf.mean(axis=1),
    }, index=comm.index)
    series.to_parquet(SERIES_PARQUET)

    report = _summarize(series, comm, n_real_edges, deg_ok, n_nodes, elapsed)
    report["stub_comparison"] = stub_comparison(edges, n_nodes, verbose=verbose)
    report["noise_excess"] = noise_excess_reference(n_nodes, report, verbose=verbose)
    with open(CACHE, "w") as f:
        json.dump(report, f, indent=2)
    return report


def _summarize(series, comm, n_real_edges, deg_ok, n_nodes, elapsed) -> dict:
    q_real = comm["q_dyn"]
    rel = series["edges_comm_mean"] / n_real_edges - 1.0
    planted_err = (series["q_comm_planted_mean"] - q_real).abs()
    return {
        "n_windows": int(len(series)),
        "n_nodes": n_nodes,
        "n_replicates": N_REPLICATES,
        "config_method": CONFIG_METHOD,
        "rewire_sweeps": REWIRE_SWEEPS,
        "n_graphs_generated": int(len(series) * N_REPLICATES * 2),
        "elapsed_sec": round(elapsed, 1),
        "real_edges_per_day": int(n_real_edges),
        "community_model": {
            "edges_mean": float(series["edges_comm_mean"].mean()),
            "edges_rel_bias": float(rel.mean()),
            "edges_rel_abs_max": float(rel.abs().max()),
            "edges_sd_mean": float(series["edges_comm_sd"].mean()),
            "frac_days_within_3pct": float((rel.abs() < 0.03).mean()),
            "q_planted_mean": float(series["q_comm_planted_mean"].mean()),
            "q_planted_vs_real_mae": float(planted_err.mean()),
            "q_planted_vs_real_max": float(planted_err.max()),
            "q_planted_vs_real_corr": float(
                np.corrcoef(series["q_comm_planted_mean"], q_real)[0, 1]),
            "q_louvain_mean": float(series["q_comm_mean"].mean()),
            "q_louvain_vs_real_corr": float(
                np.corrcoef(series["q_comm_mean"], q_real)[0, 1]),
        },
        "configuration_model": {
            "q_mean": float(series["q_conf_mean"].mean()),
            "q_sd_across_days": float(series["q_conf_mean"].std()),
            "q_min": float(series["q_conf_mean"].min()),
            "q_max": float(series["q_conf_mean"].max()),
            "q_replicate_sd_mean": float(series["q_conf_sd"].mean()),
            "degrees_preserved_frac": float(np.mean(deg_ok)),
            "edges_preserved_frac": float(
                (series["edges_conf_mean"] == n_real_edges).mean()),
            "paper_noise_floor": 0.10,
        },
        "real": {"q_mean": float(q_real.mean()), "q_sd": float(q_real.std())},
    }


def stub_comparison(edges, n_nodes, verbose: bool = True) -> dict:
    """Audit D4: what the other reading of "configuration model" would cost.

    Classical stub matching emits self-loops and multi-edges; collapsing them
    is the only way to get a simple graph, and the collapse silently lowers the
    density of the null below the network it is being compared against. This
    measures the loss on a sample of days so the choice of `rewire` is recorded
    with a number attached rather than as a preference.
    """
    rng = np.random.default_rng(FIT_SEED)
    days = np.sort(rng.choice(edges.shape[0], N_STUB_DAYS, replace=False))
    lost, q_stub = [], []
    for w in days:
        g = graph_from_pairs(edges[w], n_nodes)
        gs, loss = configuration_graph(g, _seed(int(w), 0), method="stub")
        lost.append(1.0 - loss["edges_after"] / loss["edges_before"])
        q_stub.append(modularity(gs, louvain_partitions(gs, [_seed(int(w), 0)])[0]))
    out = {"n_days": int(len(days)),
           "edge_loss_frac_mean": float(np.mean(lost)),
           "edge_loss_frac_max": float(np.max(lost)),
           "q_stub_mean": float(np.mean(q_stub))}
    if verbose:
        print(f"[D4] stub matching would discard "
              f"{out['edge_loss_frac_mean']:.1%} of edges (max "
              f"{out['edge_loss_frac_max']:.1%}); its Q = {out['q_stub_mean']:.4f}")
    return out


N_NOISE_WINDOWS = 300


def noise_excess_reference(n_nodes, report, verbose: bool = True) -> dict:
    """The control the paper's central comparison needs, and does not have.

    Phase 4's headline is that the real market scores Q = 0.2215 against a
    degree-sequence floor of 0.1065, an excess of +0.115. That gap is the
    paper's evidence that the market is modularly organised.

    The control is to run the *entire pipeline* -- returns, correlation,
    fixed-density threshold, Louvain, configuration null -- on i.i.d. Gaussian
    returns, which contain no communities of any kind, and ask what excess
    *they* produce. If the answer is "about the same", then the excess measures
    the geometry of thresholded correlation matrices rather than the market:
    correlation matrices are positive semi-definite, so high rho_ij and rho_ik
    force rho_jk up, manufacturing transitivity and hence modularity out of
    nothing. A degree-sequence null cannot remove that, because it destroys the
    geometry along with everything else.
    """
    from net.construct import build_network_sequence
    from exp1_data import DATA_SEED, DELTA_T, EDGE_FRACTION

    fake = np.random.default_rng(DATA_SEED).standard_normal(
        (n_nodes, N_NOISE_WINDOWS + DELTA_T))
    seq = build_network_sequence(fake, DELTA_T, EDGE_FRACTION, verbose=False)
    q_noise, q_null, trans = [], [], []
    for w in range(0, seq["n_windows"], 3):
        g = graph_from_pairs(seq["edges"][w], n_nodes)
        q_noise.append(modularity(g, louvain_partitions(g, [FIT_SEED])[0]))
        trans.append(g.transitivity_undirected())
        gc, _ = configuration_graph(g, FIT_SEED + w)
        q_null.append(modularity(gc, louvain_partitions(gc, [FIT_SEED])[0]))

    real_excess = report["real"]["q_mean"] - report["configuration_model"]["q_mean"]
    noise_excess = float(np.mean(q_noise) - np.mean(q_null))
    out = {
        "n_days": len(q_noise),
        "q_noise": float(np.mean(q_noise)), "q_noise_sd": float(np.std(q_noise)),
        "q_noise_null": float(np.mean(q_null)),
        "noise_transitivity": float(np.mean(trans)),
        "noise_excess": noise_excess,
        "real_excess": float(real_excess),
        "frac_of_real_excess_explained": float(noise_excess / real_excess),
    }
    if verbose:
        print(f"[control] i.i.d.-noise networks Q = {out['q_noise']:.4f} over their own "
              f"degree null {out['q_noise_null']:.4f} → excess {noise_excess:+.4f}; "
              f"real excess {real_excess:+.4f} "
              f"({out['frac_of_real_excess_explained']:.0%} reproduced by noise)")
    return out


def plot(save_dir: str = None, verbose: bool = True):
    from utils.plots import plot_noise_floor, plot_null_validation
    save_dir = save_dir or os.path.join(ROOT, "figures")
    os.makedirs(save_dir, exist_ok=True)
    s = pd.read_parquet(SERIES_PARQUET)
    comm = pd.read_parquet(COMM_SERIES)
    with open(CACHE) as f:
        r = json.load(f)

    plot_noise_floor(
        s.index, comm["q_dyn"].to_numpy(), s["q_comm_mean"].to_numpy(),
        s["q_conf_mean"].to_numpy(), conf_sd=s["q_conf_sd"].to_numpy(),
        paper_floor=r["configuration_model"]["paper_noise_floor"],
        save=os.path.join(save_dir, "fig6a_noise_floor.png"))

    plot_null_validation(
        comm["q_dyn"].to_numpy(), s["q_comm_planted_mean"].to_numpy(),
        s["edges_comm_mean"].to_numpy(), r["real_edges_per_day"],
        save=os.path.join(save_dir, "fig6b_null_validation.png"))


def print_report(r: dict) -> None:
    cm, cf = r["community_model"], r["configuration_model"]
    print("\n── Phase 4 null models " + "─" * 45)
    print(f"  generated              : {r['n_graphs_generated']:,} graphs "
          f"({r['n_windows']} days × {r['n_replicates']} reps × 2 models) "
          f"in {r['elapsed_sec']:.0f}s")
    print(f"  configuration method   : {r['config_method']} "
          f"({r['rewire_sweeps']}× m swaps)")
    print("  ── community model (Eq. 2) ──")
    print(f"     edges/day           : {cm['edges_mean']:.0f} ± {cm['edges_sd_mean']:.0f}"
          f"   vs real {r['real_edges_per_day']}"
          f"   (bias {cm['edges_rel_bias']:+.3%}, worst day {cm['edges_rel_abs_max']:.2%})")
    print(f"     within 3 % of real  : {cm['frac_days_within_3pct']:.2%} of days"
          "     <- D2: independent coin flips, so this is a random variable")
    print(f"     Q vs planted C(t)   : {cm['q_planted_mean']:.4f}"
          f"   real Q {r['real']['q_mean']:.4f}"
          f"   MAE {cm['q_planted_vs_real_mae']:.4f}, ρ = {cm['q_planted_vs_real_corr']:.4f}")
    print(f"     Q via Louvain       : {cm['q_louvain_mean']:.4f}"
          f"   (ρ with real Q = {cm['q_louvain_vs_real_corr']:.4f})")
    print("  ── configuration model — THE NOISE FLOOR (audit B6) ──")
    print(f"     Q                   : {cf['q_mean']:.4f} ± {cf['q_sd_across_days']:.4f}"
          f"   range {cf['q_min']:.4f}–{cf['q_max']:.4f}")
    print(f"     paper's drawn floor : ≈ {cf['paper_noise_floor']:.2f}"
          f"      real market Q: {r['real']['q_mean']:.4f} ± {r['real']['q_sd']:.4f}")
    print(f"     degrees preserved   : {cf['degrees_preserved_frac']:.1%} of days; "
          f"edge count preserved {cf['edges_preserved_frac']:.1%}")
    sc = r["stub_comparison"]
    print(f"  D4 alternative reading : stub matching loses "
          f"{sc['edge_loss_frac_mean']:.1%} of edges (max {sc['edge_loss_frac_max']:.1%}) "
          f"→ Q = {sc['q_stub_mean']:.4f} on a lower-density graph")
    ne = r["noise_excess"]
    print("  ── the control the paper's central comparison lacks ──")
    print(f"     real market   : Q {r['real']['q_mean']:.4f} over floor "
          f"{cf['q_mean']:.4f}  → excess {ne['real_excess']:+.4f}")
    print(f"     i.i.d. noise  : Q {ne['q_noise']:.4f} over floor "
          f"{ne['q_noise_null']:.4f}  → excess {ne['noise_excess']:+.4f}")
    print(f"     -> {ne['frac_of_real_excess_explained']:.0%} of the market's excess "
          "modularity is reproduced by structureless data.")
    print("        Thresholded correlation matrices are transitive by construction "
          f"(noise transitivity {ne['noise_transitivity']:.3f});")
    print("        a degree-sequence null cannot control for that. (Bounded by A5.)")


def acceptance(r: dict, verbose: bool = True) -> bool:
    s = pd.read_parquet(SERIES_PARQUET)
    comm = pd.read_parquet(COMM_SERIES)
    cm, cf = r["community_model"], r["configuration_model"]

    checks = [
        ("Eq. 2 reproduces the modularity of the partition it was built from",
         cm["q_planted_vs_real_mae"] < 0.01 and cm["q_planted_vs_real_corr"] > 0.99,
         f"MAE {cm['q_planted_vs_real_mae']:.5f}, ρ = {cm['q_planted_vs_real_corr']:.5f}"),
        ("community-model edge count matches the real one within 3 % every day",
         cm["edges_rel_abs_max"] < 0.03,
         f"bias {cm['edges_rel_bias']:+.3%}, worst day {cm['edges_rel_abs_max']:.3%}"),
        ("configuration model preserves the degree sequence exactly",
         cf["degrees_preserved_frac"] == 1.0,
         f"{cf['degrees_preserved_frac']:.1%} of days"),
        ("configuration model preserves the edge count exactly",
         cf["edges_preserved_frac"] == 1.0,
         f"{cf['edges_preserved_frac']:.1%} of days"),
        ("configuration-model modularity is the ≈0.10 noise floor the paper draws",
         0.05 <= cf["q_mean"] <= 0.15,
         f"Q = {cf['q_mean']:.4f} vs the paper's ≈{cf['paper_noise_floor']:.2f}"),
        ("that floor is flat — it does not track the real modularity series",
         cf["q_sd_across_days"] < 0.5 * r["real"]["q_sd"],
         f"floor sd {cf['q_sd_across_days']:.4f} vs real sd {r['real']['q_sd']:.4f}"),
        ("the real market sits above its own degree-sequence null",
         r["real"]["q_mean"] > cf["q_mean"] + 0.05,
         f"real {r['real']['q_mean']:.4f} vs floor {cf['q_mean']:.4f} "
         f"(+{r['real']['q_mean'] - cf['q_mean']:.4f})"),
        ("every series has length N_w with no NaNs",
         len(s) == len(comm) and int(s.isna().sum().sum()) == 0,
         f"{len(s)} rows"),
        ("the rejected D4 reading is quantified, not just asserted",
         r["stub_comparison"]["edge_loss_frac_mean"] > 0,
         f"stub matching loses {r['stub_comparison']['edge_loss_frac_mean']:.1%} of edges"),
        ("the market's excess over its null is reported against a noise control",
         "frac_of_real_excess_explained" in r["noise_excess"],
         f"noise reproduces {r['noise_excess']['frac_of_real_excess_explained']:.0%} "
         f"of the real excess ({r['noise_excess']['noise_excess']:+.4f} vs "
         f"{r['noise_excess']['real_excess']:+.4f})"),
    ]
    if verbose:
        print("\n── Phase 4 acceptance tests " + "─" * 40)
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
        print(f"  {sum(ok for _, ok, _ in checks)}/{len(checks)} passed")
    return all(ok for _, ok, _ in checks)


if __name__ == "__main__":
    rep = run(force="--force" in sys.argv)
    print_report(rep)
    plot()
    sys.exit(0 if acceptance(rep) else 1)
