"""
Paper-only replication of Silva et al. (2015).

This module exists to answer one question and nothing else: **run exactly what
the paper says it ran, and see whether the numbers come out the same.**

It is deliberately separate from `exp1_data.py` ... `exp9_report.py`, which
carry the 26-item methodological audit. Nothing in here is an addition to the
paper. No overlap correction, no normalized MSE, no noise control, no crisis
detector, no bootstrap intervals, no alternative conventions. Where the paper
is silent, one reading is chosen, stated in a comment, and used throughout.

What the paper does (Sec. II and III):

  1. log returns from daily closes
  2. Pearson correlation over a sliding 30-return window
  3. keep the top 10 % of pairs each day  -> one network per business day
  4. Louvain communities each day
  5. two null models per day, ONE realization each:
       community  - stochastic blockmodel from the community sizes and the
                    mixing matrix Pi (Eq. 2)
       degree     - configuration model, same degree distribution
  6. eight topological measures on all three networks
  7. average squared difference and Pearson correlation between the real
     series and each model series, split at 2002

Two things to know about the measure definitions, because the paper's own
plotted axes pin them down:

  * average shortest path is averaged over reachable pairs (igraph's default).
    Unreachable pairs are skipped, not counted as infinite.
  * the matching index is Costa et al. (2007)'s edge form, averaged over edges,
    which is the reference the paper cites. Averaging the same quantity over
    *all* pairs instead lands at ~0.065, far outside the 0.10-0.40 range the
    paper plots; the edge form lands at ~0.19, inside it.

Run:
    /opt/anaconda3/bin/python experiments/exp_paper_only.py
    /opt/anaconda3/bin/python experiments/exp_paper_only.py --force
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

import igraph as ig  # noqa: E402

from net.communities import MixingStore, graph_from_pairs  # noqa: E402
from net.measures import adjacency, rich_club_curve  # noqa: E402
from net.nullmodels import community_graph, configuration_graph  # noqa: E402
from utils.parallel import map_windows, stack  # noqa: E402

# ── configuration ──────────────────────────────────────────────────────────
SPLIT_YEAR = 2002          # the paper splits its tables at 2002
SEED = 42                  # every random draw is seeded from (SEED, window)
N_NODES = 360

RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(RESULTS, "paper_only.json")
SERIES = os.path.join(RESULTS, "series", "paper_only.parquet")
CHUNKS = os.path.join(RESULTS, "_chunks_paper_only")
EDGES_NPY = os.path.join(ROOT, "data", "edges.npy")
MIXING_NPZ = os.path.join(ROOT, "data", "mixing.npz")
COMM_SERIES = os.path.join(RESULTS, "series", "communities.parquet")

MEASURES = ("modularity", "path_length", "assortativity", "transitivity",
            "betweenness", "clique_number", "rich_club", "matching_index")
MODELS = {"comm": "Community", "conf": "Degree dist."}

#: The paper's published values, read off Fig. 6 and Fig. S3.
#: [measure][model] -> {"chi2": (<2002, >=2002), "rho": (<2002, >=2002)}
PAPER = {
    "modularity":     {"comm": {"chi2": (0.0003, 0.0002), "rho": (0.99, 0.96)},
                       "conf": {"chi2": (0.0295, 0.0120), "rho": (0.07, 0.71)}},
    "path_length":    {"comm": {"chi2": (0.0189, 0.1152), "rho": (0.89, 0.72)},
                       "conf": {"chi2": (0.0333, 0.0688), "rho": (0.81, 0.76)}},
    "assortativity":  {"comm": {"chi2": (0.0120, 0.0259), "rho": (0.76, 0.53)},
                       "conf": {"chi2": (0.1012, 0.0730), "rho": (-0.47, 0.78)}},
    "transitivity":   {"comm": {"chi2": (0.0188, 0.1184), "rho": (0.97, 0.80)},
                       "conf": {"chi2": (0.0161, 0.0110), "rho": (0.92, 0.92)}},
    "betweenness":    {"comm": {"chi2": (419.0, 2522.0), "rho": (0.93, 0.95)},
                       "conf": {"chi2": (691.0, 1095.0), "rho": (0.77, 0.94)}},
    "clique_number":  {"comm": {"chi2": (191.0, 552.0), "rho": (0.80, 0.38)},
                       "conf": {"chi2": (203.0, 259.0), "rho": (0.74, 0.85)}},
    "rich_club":      {"comm": {"chi2": (0.017, 0.183), "rho": (0.86, 0.41)},
                       "conf": {"chi2": (0.014, 0.011), "rho": (0.93, 0.96)}},
    "matching_index": {"comm": {"chi2": (0.005, 0.020), "rho": (0.97, 0.80)},
                       "conf": {"chi2": (0.007, 0.005), "rho": (0.90, 0.94)}},
}


# ── the eight measures, as the paper defines them ──────────────────────────

def matching_index(A: np.ndarray) -> float:
    """Costa et al. (2007), averaged over edges.

        mu_ij = (number of neighbours i and j share) / (d_i + d_j - 2 a_ij)

    Averaged over the edges of the graph, which is where the measure is
    defined. Returns 0 for an edgeless graph.
    """
    Af = A.astype(np.float64)
    shared = Af @ Af                      # no diagonal in A, so i and j drop out
    d = Af.sum(axis=1)
    denom = d[:, None] + d[None, :] - 2.0 * Af
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(denom > 0, shared / np.maximum(denom, 1e-12), 0.0)
    return float(M[A].mean()) if A.any() else 0.0


def measure_all(g: ig.Graph, seed: int, k_max: int) -> dict:
    """The paper's eight measurements on one graph."""
    A = adjacency(g)
    part = np.asarray(g.community_multilevel().membership)
    phi = rich_club_curve(A, k_max=k_max)
    return {
        "modularity": float(g.modularity(part)),
        # averaged over reachable pairs; unreachable pairs are skipped
        "path_length": float(g.average_path_length(unconn=True)),
        "assortativity": float(g.assortativity_degree()),
        "transitivity": float(g.transitivity_undirected(mode="zero")),
        "betweenness": float(np.mean(g.betweenness())),
        "clique_number": float(g.clique_number()),
        "rich_club": float(np.nanmean(phi)) if np.isfinite(phi).any() else np.nan,
        "matching_index": matching_index(A),
    }


# ── per-day work ───────────────────────────────────────────────────────────

_G = {}


def _seed(window: int) -> int:
    return SEED + 1_000_003 * int(window)


def _init(edges_path, mixing_path, n_nodes):
    _G["edges"] = np.load(edges_path, mmap_mode="r")
    _G["store"] = MixingStore.load(mixing_path)
    _G["n"] = n_nodes


def _worker(span):
    lo, hi = span
    n = _G["n"]
    rows = hi - lo
    out = {f"{fam}_{m}": np.full(rows, np.nan)
           for fam in ("real", "comm", "conf") for m in MEASURES}

    for i, w in enumerate(range(lo, hi)):
        real = graph_from_pairs(_G["edges"][w], n)
        k_max = int(max(real.degree()))
        Pi, sizes = _G["store"][w]
        s = _seed(w)

        # igraph's multilevel and rewire both draw from Python's `random`
        ig.set_random_number_generator(__import__("random"))

        m_real = measure_all(real, seed=s, k_max=k_max)

        # ONE realization of each null model per day, as in the paper
        rng = np.random.default_rng(s)
        g_comm, _ = community_graph(sizes, Pi, rng, n)
        m_comm = measure_all(g_comm, seed=s, k_max=k_max)

        g_conf, _ = configuration_graph(real, s + 7)
        m_conf = measure_all(g_conf, seed=s, k_max=k_max)

        for m in MEASURES:
            out[f"real_{m}"][i] = m_real[m]
            out[f"comm_{m}"][i] = m_comm[m]
            out[f"conf_{m}"][i] = m_conf[m]
    return out


# ── scoring, exactly the two numbers the paper reports ─────────────────────

def chi_square(a, b):
    """The paper's 'average squared difference'."""
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.mean((a[ok] - b[ok]) ** 2))


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def score(s: pd.DataFrame) -> pd.DataFrame:
    """One row per (measure, model, era), ours beside the paper's."""
    eras = {"pre": s.index.year < SPLIT_YEAR, "post": s.index.year >= SPLIT_YEAR}
    rows = []
    for m in MEASURES:
        real = s[f"real_{m}"].to_numpy()
        for mod in MODELS:
            gen = s[f"{mod}_{m}"].to_numpy()
            for j, (era, mask) in enumerate(eras.items()):
                p = PAPER[m][mod]
                rows.append({
                    "measure": m, "model": mod, "era": era, "n": int(mask.sum()),
                    "chi2": chi_square(real[mask], gen[mask]),
                    "paper_chi2": p["chi2"][j],
                    "rho": pearson(real[mask], gen[mask]),
                    "paper_rho": p["rho"][j],
                })
    t = pd.DataFrame(rows)
    t["rho_delta"] = t["rho"] - t["paper_rho"]
    return t


def run(force: bool = False, verbose: bool = True, n_workers: int = None) -> dict:
    if os.path.exists(CACHE) and os.path.exists(SERIES) and not force:
        with open(CACHE) as f:
            report = json.load(f)
        if verbose:
            print(f"[cached] {CACHE} — pass force=True to rebuild.")
        return report

    comm = pd.read_parquet(COMM_SERIES)
    idx = comm.index
    n_w = len(idx)
    t0 = time.time()
    if verbose:
        print(f"[paper-only] {n_w} networks x (1 real + 1 community + 1 degree) "
              f"= {3 * n_w:,} graph evaluations")

    chunks = map_windows(_worker, n_w, initializer=_init,
                         initargs=(EDGES_NPY, MIXING_NPZ, N_NODES),
                         n_workers=n_workers, chunk_size=48, verbose=verbose,
                         label="measures", cache_dir=CHUNKS)
    data = {k: stack(chunks, k) for k in chunks[0]}
    s = pd.DataFrame(data, index=idx)
    os.makedirs(os.path.dirname(SERIES), exist_ok=True)
    s.to_parquet(SERIES)

    t = score(s)
    d_rho = (t["rho"] - t["paper_rho"]).abs()
    d_chi = np.abs(np.log10(np.maximum(t["chi2"], 1e-12))
                   - np.log10(np.maximum(t["paper_chi2"], 1e-12)))
    report = {
        "n_windows": n_w, "n_nodes": N_NODES, "split_year": SPLIT_YEAR,
        "n_replicates": 1, "seed": SEED,
        "elapsed_sec": round(time.time() - t0, 1),
        "rho": {
            "mae_vs_paper": float(d_rho.mean()),
            "within_0_10": float((d_rho <= 0.10).mean()),
            "within_0_20": float((d_rho <= 0.20).mean()),
            "within_0_30": float((d_rho <= 0.30).mean()),
            "sign_agreement": float((np.sign(t["rho"]) == np.sign(t["paper_rho"])).mean()),
        },
        "chi2": {
            "median_abs_log10_ratio": float(np.median(d_chi)),
            "within_one_order": float((d_chi <= 1.0).mean()),
        },
        "head_to_head": {
            "ours_community_wins_chi2": int((t.pivot_table(index=["measure", "era"],
                columns="model", values="chi2").eval("comm < conf")).sum()),
            "paper_community_wins_chi2": int((t.pivot_table(index=["measure", "era"],
                columns="model", values="paper_chi2").eval("comm < conf")).sum()),
            "ours_community_wins_rho": int((t.pivot_table(index=["measure", "era"],
                columns="model", values="rho").eval("comm > conf")).sum()),
            "paper_community_wins_rho": int((t.pivot_table(index=["measure", "era"],
                columns="model", values="paper_rho").eval("comm > conf")).sum()),
            "of": 16,
        },
        "worst_cells": [
            {"measure": r["measure"], "model": r["model"], "era": r["era"],
             "ours": float(r["rho"]), "paper": float(r["paper_rho"]),
             "delta": float(r["rho_delta"])}
            for _, r in t.reindex(d_rho.sort_values(ascending=False).index).head(6).iterrows()
        ],
    }
    with open(CACHE, "w") as f:
        json.dump(report, f, indent=2)
    t.to_csv(os.path.join(RESULTS, "tables", "paper_only_scoring.csv"), index=False)
    return report


def load_series() -> pd.DataFrame:
    return pd.read_parquet(SERIES)


def load_table() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, "tables", "paper_only_scoring.csv"))


if __name__ == "__main__":
    rep = run(force="--force" in sys.argv)
    t = load_table()
    print(f"\n{'measure':<16}{'model':<6}{'era':<6}"
          f"{'chi2 ours':>12}{'chi2 paper':>12}{'rho ours':>10}{'rho paper':>11}{'delta':>8}")
    for _, x in t.iterrows():
        print(f"{x['measure']:<16}{x['model']:<6}{x['era']:<6}"
              f"{x['chi2']:>12.4g}{x['paper_chi2']:>12.4g}"
              f"{x['rho']:>10.3f}{x['paper_rho']:>11.2f}{x['rho_delta']:>+8.2f}")
    r = rep["rho"]
    print(f"\nrho: MAE {r['mae_vs_paper']:.3f}, {r['within_0_10']:.0%} within 0.10, "
          f"{r['within_0_30']:.0%} within 0.30, sign agrees {r['sign_agreement']:.0%}")
    h = rep["head_to_head"]
    print(f"community model closer on chi2 -- ours {h['ours_community_wins_chi2']}/16, "
          f"paper {h['paper_community_wins_chi2']}/16")
