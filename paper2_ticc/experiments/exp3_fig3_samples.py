"""
Experiment 3 -- TICC paper Figure 3: clustering accuracy vs. number of samples.

Vary the number of samples per segment (100..500) on the "1,2,3,4,1,2,3,4" sequence
and plot macro-F1 for TICC and the baselines. Reproduces the paper's key finding:
TICC needs far fewer samples than the other methods -- it clears 0.9 by ~200 samples,
while the distance-based methods never capture the structure.

Cached to results/fig3_samples.json; figure written to figures/fig3_samples.png.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic import generate_series
from solver.ticc import fit_ticc, stack_windows
from utils.metrics import macro_f1
from utils import baselines as B

from exp1_table1_f1 import N, W, LAM, BETA, N_RESTARTS, DATA_SEED, FIT_SEED, DTW_MAX_SAMPLES, _subsample

SEQUENCE = [1, 2, 3, 4, 1, 2, 3, 4]
SAMPLE_SIZES = [100, 200, 300, 400, 500]
METHODS = ["TICC", "TICC b=0", "GMM", "EEV", "DTW-GAK", "DTW-Euclidean",
           "Neural Gas", "K-means"]

CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "fig3_samples.json")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures", "fig3_samples.png")


def run(force=False, verbose=True):
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as f:
            return json.load(f)

    K = len(set(SEQUENCE))
    curves = {m: [] for m in METHODS}
    for seg in SAMPLE_SIZES:
        X, tl, _ = generate_series(SEQUENCE, seg_len=seg, n=N, w=W, seed=DATA_SEED)
        Xst = stack_windows(X, W)
        tv = tl[W - 1:]
        Xsub, tvsub = _subsample(Xst, tv, DTW_MAX_SAMPLES)

        lab, _, _ = fit_ticc(X, K=K, w=W, lam=LAM, beta=BETA, n_restarts=N_RESTARTS, seed=FIT_SEED)
        lab0, _, _ = fit_ticc(X, K=K, w=W, lam=LAM, beta=0.0, n_restarts=N_RESTARTS, seed=FIT_SEED)
        vals = {
            "TICC": macro_f1(tv, lab),
            "TICC b=0": macro_f1(tv, lab0),
            "GMM": macro_f1(tv, B.gmm_labels(Xst, K)),
            "EEV": macro_f1(tv, B.eev_labels(Xst, K)),
            "DTW-GAK": macro_f1(tvsub, B.dtw_gak_labels(Xsub, K, N, W)),
            "DTW-Euclidean": macro_f1(tvsub, B.dtw_euclidean_labels(Xsub, K, N, W)),
            "Neural Gas": macro_f1(tv, B.neural_gas_labels(Xst, K)),
            "K-means": macro_f1(tv, B.kmeans_labels(Xst, K)),
        }
        for m, v in vals.items():
            curves[m].append(float(v))
        if verbose:
            print(f"seg={seg}: TICC={vals['TICC']:.3f}")

    out = {"sample_sizes": SAMPLE_SIZES, "curves": curves}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(out, f, indent=2)
    return out


def plot(data, save=FIG):
    xs = data["sample_sizes"]
    curves = data["curves"]
    plt.figure(figsize=(7, 5))
    styles = {"TICC": dict(color="C0", lw=2.5, marker="o"),
              "TICC b=0": dict(color="C1", lw=2, marker="s")}
    for m in METHODS:
        st = styles.get(m, dict(lw=1.3, marker=".", alpha=0.8))
        plt.plot(xs, curves[m], label=m, **st)
    plt.xlabel("Number of samples per segment")
    plt.ylabel("Macro-F1 score")
    plt.title("Clustering accuracy vs. sample size (TICC Fig 3)")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="center right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=130)
    plt.close("all")
    return save


if __name__ == "__main__":
    data = run(force="--force" in sys.argv)
    path = plot(data)
    print("saved", path)
