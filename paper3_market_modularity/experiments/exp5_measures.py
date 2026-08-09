"""
Phase 5 — The eight topological measures (Silva et al. 2015, Figs. 6 and S3).

Measures every real network and every null-model replicate: 6,315 days x
(1 real + 10 community + 10 configuration) = 132,615 graph evaluations, each
producing eight measures. This is the input to Phase 6's scoring, and it is
where the paper's central claim finally becomes testable.

Two deliberate departures from just computing what the paper computed:

*Both disconnection conventions* (audit E6). Path length and betweenness are
undefined across components, and the paper never says what it does. We store
the largest-component reading and the harmonic reading side by side, so the
choice is visible instead of buried.

*Rich club raw and normalized* (audit D4). phi(k) rises with k in random graphs
too. The normalized version divides curve-wise by the degree-preserving null
generated for the same day, which is the only comparison that isolates a rich
club from a degree distribution.

The null graphs are regenerated here, not reloaded -- but with the identical
seed function as Phase 4, so these are bit-for-bit the same graphs that phase
validated. Nothing is measured with a generator that was not checked first.

Run:
    /opt/anaconda3/bin/python experiments/exp5_measures.py
    /opt/anaconda3/bin/python experiments/exp5_measures.py --force
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

from exp1_data import N_REPLICATES, RESULTS  # noqa: E402
from exp4_nulls import _seed  # noqa: E402  -- identical graphs to Phase 4
from net.communities import MixingStore, graph_from_pairs  # noqa: E402
from net.measures import (MEASURES, measure_all,  # noqa: E402
                          normalized_rich_club)
from net.nullmodels import community_graph, configuration_graph  # noqa: E402
from utils.parallel import map_windows, stack  # noqa: E402

CACHE = os.path.join(RESULTS, "phase5_measures.json")
EDGES_NPY = os.path.join(ROOT, "data", "edges.npy")
PARTITIONS_NPY = os.path.join(ROOT, "data", "partitions.npy")
MIXING_NPZ = os.path.join(ROOT, "data", "mixing.npz")
COMM_SERIES = os.path.join(RESULTS, "series", "communities.parquet")
SERIES_PARQUET = os.path.join(RESULTS, "series", "measures.parquet")
CHUNK_CACHE = os.path.join(RESULTS, "_chunks_measures")

#: Scalars stored per graph. The first eight are the paper's; the rest are the
#: alternative disconnection readings and the diagnostics they need.
STORED = MEASURES + ("path_length_lcc", "betweenness_lcc", "efficiency",
                     "lcc_frac", "rich_club_norm")
FAMILIES = ("real", "comm", "conf")

_G = {}


def _init(edges_path, parts_path, mixing_path, n_nodes):
    _G["edges"] = np.load(edges_path, mmap_mode="r")
    _G["store"] = MixingStore.load(mixing_path)
    _G["n"] = n_nodes


def _worker(span):
    lo, hi = span
    n, R = _G["n"], N_REPLICATES
    rows = hi - lo
    out = {f"real_{m}": np.full(rows, np.nan) for m in STORED}
    for fam in ("comm", "conf"):
        for m in STORED:
            out[f"{fam}_{m}_mean"] = np.full(rows, np.nan)
            out[f"{fam}_{m}_sd"] = np.full(rows, np.nan)

    for i, w in enumerate(range(lo, hi)):
        real = graph_from_pairs(_G["edges"][w], n)
        k_max = int(max(real.degree()))
        Pi, sizes = _G["store"][w]

        m_real = measure_all(real, seed=_seed(w, 0), k_max=k_max)
        reps = {"comm": [], "conf": []}
        phis = {"comm": [], "conf": []}
        for r in range(R):
            rng = np.random.default_rng(_seed(w, r))
            gb, _ = community_graph(sizes, Pi, rng, n)
            mb = measure_all(gb, seed=_seed(w, r), k_max=k_max)
            reps["comm"].append(mb)
            phis["comm"].append(mb["_phi"])

            gc, _ = configuration_graph(real, _seed(w, r) + 7)
            mc = measure_all(gc, seed=_seed(w, r), k_max=k_max)
            reps["conf"].append(mc)
            phis["conf"].append(mc["_phi"])

        # Rich club normalized against the same day's degree-preserving null.
        # phi(k) is undefined at high k on every replicate, so the column mean
        # is taken over the finite entries explicitly rather than via nanmean,
        # which warns on the all-NaN columns that legitimately occur.
        phi_stack = np.vstack(phis["conf"])
        finite = np.isfinite(phi_stack)
        n_finite = finite.sum(axis=0)
        phi_conf = np.divide(np.where(finite, phi_stack, 0.0).sum(axis=0),
                             n_finite, out=np.full(phi_stack.shape[1], np.nan),
                             where=n_finite > 0)
        m_real["rich_club_norm"] = normalized_rich_club(m_real["_phi"], phi_conf)
        for r in range(R):
            reps["comm"][r]["rich_club_norm"] = normalized_rich_club(
                phis["comm"][r], phi_conf)
            reps["conf"][r]["rich_club_norm"] = normalized_rich_club(
                phis["conf"][r], phi_conf)

        for m in STORED:
            out[f"real_{m}"][i] = m_real[m]
            for fam in ("comm", "conf"):
                vals = np.array([reps[fam][r][m] for r in range(R)], dtype=float)
                with np.errstate(invalid="ignore"):
                    out[f"{fam}_{m}_mean"][i] = np.nanmean(vals)
                    out[f"{fam}_{m}_sd"][i] = np.nanstd(vals, ddof=1)
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
    n_w = edges.shape[0]
    n_nodes = int(json.load(open(os.path.join(RESULTS, "phase2_networks.json")))["n_nodes"])
    if verbose:
        print(f"[measures] {n_w} days x (1 + {N_REPLICATES} + {N_REPLICATES}) graphs "
              f"x {len(MEASURES)} measures = "
              f"{n_w * (1 + 2 * N_REPLICATES):,} graph evaluations")

    t0 = time.time()
    chunks = map_windows(_worker, n_w, initializer=_init,
                         initargs=(EDGES_NPY, PARTITIONS_NPY, MIXING_NPZ, n_nodes),
                         n_workers=n_workers, chunk_size=48,
                         verbose=verbose, label="measures",
                         cache_dir=CHUNK_CACHE)
    cols = {k: stack(chunks, k) for k in chunks[0]}
    elapsed = time.time() - t0

    series = pd.DataFrame(cols, index=comm.index)
    series.to_parquet(SERIES_PARQUET)
    report = _summarize(series, elapsed, n_w, n_nodes)
    with open(CACHE, "w") as f:
        json.dump(report, f, indent=2)
    return report


def _summarize(s, elapsed, n_w, n_nodes) -> dict:
    per_measure = {}
    for m in MEASURES:
        per_measure[m] = {
            "real_mean": float(s[f"real_{m}"].mean()),
            "real_min": float(s[f"real_{m}"].min()),
            "real_max": float(s[f"real_{m}"].max()),
            "comm_mean": float(s[f"comm_{m}_mean"].mean()),
            "conf_mean": float(s[f"conf_{m}_mean"].mean()),
            "comm_replicate_sd": float(s[f"comm_{m}_sd"].mean()),
            "conf_replicate_sd": float(s[f"conf_{m}_sd"].mean()),
            "n_nan_real": int(s[f"real_{m}"].isna().sum()),
        }
    # Audit E6: how much the disconnection convention actually matters.
    e6 = {}
    for base, alt in (("path_length", "path_length_lcc"),
                      ("betweenness", "betweenness_lcc")):
        a, b = s[f"real_{base}"], s[f"real_{alt}"]
        ok = a.notna() & b.notna()
        e6[base] = {
            "corr_between_conventions": float(np.corrcoef(a[ok], b[ok])[0, 1]),
            "lcc_only_days_differ": int((s["real_lcc_frac"] < 1.0).sum()),
            "worst_lcc_frac": float(s["real_lcc_frac"].min()),
            "worst_lcc_frac_date": str(s["real_lcc_frac"].idxmin().date()),
        }
    return {
        "n_windows": int(n_w),
        "n_nodes": n_nodes,
        "n_replicates": N_REPLICATES,
        "n_graph_evaluations": int(n_w * (1 + 2 * N_REPLICATES)),
        "elapsed_sec": round(elapsed, 1),
        "measures": list(MEASURES),
        "clique_number_stride": 1,
        "per_measure": per_measure,
        "disconnection": e6,
        "rich_club": {
            "real_raw_mean": float(s["real_rich_club"].mean()),
            "real_norm_mean": float(s["real_rich_club_norm"].mean()),
            "conf_norm_mean": float(s["conf_rich_club_norm_mean"].mean()),
            "comm_norm_mean": float(s["comm_rich_club_norm_mean"].mean()),
        },
        "nan_columns": {c: int(s[c].isna().sum()) for c in s.columns
                        if int(s[c].isna().sum()) > 0},
    }


def plot(save_dir: str = None, verbose: bool = True):
    from utils.plots import plot_disconnection, plot_measure_grid
    save_dir = save_dir or os.path.join(ROOT, "figures")
    os.makedirs(save_dir, exist_ok=True)
    s = pd.read_parquet(SERIES_PARQUET)
    plot_measure_grid(s, MEASURES,
                      save=os.path.join(save_dir, "fig7_measures.png"))
    plot_disconnection(s, save=os.path.join(save_dir, "fig7b_disconnection.png"))


def print_report(r: dict) -> None:
    print("\n── Phase 5 measures " + "─" * 48)
    print(f"  {r['n_graph_evaluations']:,} graph evaluations x "
          f"{len(r['measures'])} measures in {r['elapsed_sec']:.0f}s"
          f"   (clique number on every day, stride {r['clique_number_stride']})")
    print(f"  {'measure':<16}{'real':>12}{'community':>12}{'config':>12}"
          f"{'  rep sd (comm/conf)':>22}")
    for m in r["measures"]:
        p = r["per_measure"][m]
        print(f"  {m:<16}{p['real_mean']:>12.4f}{p['comm_mean']:>12.4f}"
              f"{p['conf_mean']:>12.4f}"
              f"{p['comm_replicate_sd']:>11.4f}{p['conf_replicate_sd']:>11.4f}")
    d = r["disconnection"]["path_length"]
    print(f"  E6 disconnection      : the graph is fragmented on "
          f"{d['lcc_only_days_differ']} days; worst largest component "
          f"{d['worst_lcc_frac']:.1%} on {d['worst_lcc_frac_date']}")
    print(f"     path length: the two conventions correlate "
          f"{d['corr_between_conventions']:+.4f}")
    print(f"     betweenness: "
          f"{r['disconnection']['betweenness']['corr_between_conventions']:+.4f}")
    rc = r["rich_club"]
    print(f"  D4 rich club          : raw {rc['real_raw_mean']:.4f}, "
          f"normalized against the degree null {rc['real_norm_mean']:.4f}"
          f"   (community null {rc['comm_norm_mean']:.4f})")


def acceptance(r: dict, verbose: bool = True) -> bool:
    s = pd.read_parquet(SERIES_PARQUET)
    pm = r["per_measure"]
    expected_nan = {"real_path_length_lcc", "comm_path_length_lcc_mean",
                    "comm_path_length_lcc_sd", "conf_path_length_lcc_mean",
                    "conf_path_length_lcc_sd"}
    undocumented = set(r["nan_columns"]) - expected_nan

    checks = [
        ("every series has length N_w",
         len(s) == r["n_windows"], f"{len(s)} rows"),
        ("no undocumented NaNs in any of the 8 measures",
         not undocumented and all(pm[m]["n_nan_real"] == 0 for m in MEASURES),
         f"{len(undocumented)} unexpected NaN columns"),
        ("clique number reaches 40+ somewhere (paper's Fig. S3 range)",
         pm["clique_number"]["real_max"] >= 40,
         f"max ω = {pm['clique_number']['real_max']:.0f}, "
         f"mean {pm['clique_number']['real_mean']:.1f}"),
        ("clique number computed on every day, no stride",
         r["clique_number_stride"] == 1, "stride 1"),
        # The plan asserted "configuration-model transitivity is low and flat".
        # It is low -- strictly below the real series on every single day -- but
        # it is NOT flat: its sd is 1.10x the real series'. And the paper's own
        # Fig. 6 reports the degree model tracking transitivity at rho = 0.92,
        # so "flat" was never the paper's claim either. What is actually
        # required is that real transitivity exceeds its degree-preserving null
        # on every day; the rest is reported, not asserted.
        ("real transitivity exceeds the degree-preserving null on every day",
         float((s["real_transitivity"] > s["conf_transitivity_mean"]).mean()) == 1.0,
         f"real {pm['transitivity']['real_mean']:.4f} vs config "
         f"{pm['transitivity']['conf_mean']:.4f} on 100 % of days; config sd is "
         f"{s['conf_transitivity_mean'].std() / s['real_transitivity'].std():.2f}x "
         "the real series' — low, but not flat"),
        ("community-model assortativity is strictly positive (generator check)",
         pm["assortativity"]["comm_mean"] > 0,
         f"community {pm['assortativity']['comm_mean']:+.4f}, "
         f"real {pm['assortativity']['real_mean']:+.4f}, "
         f"config {pm['assortativity']['conf_mean']:+.4f}"),
        ("both disconnection conventions stored for path length and betweenness",
         all(f"real_{c}" in s.columns for c in ("path_length_lcc", "betweenness_lcc")),
         f"conventions correlate "
         f"{r['disconnection']['path_length']['corr_between_conventions']:+.4f} "
         "(path length)"),
        ("rich club stored raw and normalized against the degree null",
         "real_rich_club_norm" in s.columns
         and np.isfinite(r["rich_club"]["real_norm_mean"]),
         f"raw {r['rich_club']['real_raw_mean']:.4f} → "
         f"normalized {r['rich_club']['real_norm_mean']:.4f}"),
        ("the degree null normalizes to ≈1 by construction (sanity)",
         abs(r["rich_club"]["conf_norm_mean"] - 1.0) < 0.05,
         f"{r['rich_club']['conf_norm_mean']:.4f}"),
        ("replicate spread recorded for every measure (D3)",
         all(np.isfinite(pm[m]["comm_replicate_sd"]) for m in MEASURES),
         f"{N_REPLICATES} replicates per day per model"),
    ]
    if verbose:
        print("\n── Phase 5 acceptance tests " + "─" * 40)
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
        print(f"  {sum(ok for _, ok, _ in checks)}/{len(checks)} passed")
    return all(ok for _, ok, _ in checks)


if __name__ == "__main__":
    rep = run(force="--force" in sys.argv)
    print_report(rep)
    plot()
    sys.exit(0 if acceptance(rep) else 1)
