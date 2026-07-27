"""
Experiment 1 -- TICC paper Table 1: macro-F1 clustering accuracy.

Compare TICC against baselines on four synthetic temporal sequences. Each method
clusters the same data; accuracy is the Hungarian-aligned macro-F1 vs ground truth.
Results are cached to results/table1_f1.json; delete it (or set FORCE=True) to recompute.

Paper reference: TICC averages 0.95 (0.90-0.98); GMM (best baseline) averages 0.67;
distance-based methods score much lower.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic import generate_series
from solver.ticc import fit_ticc, stack_windows
from utils.metrics import macro_f1
from utils import baselines as B

# ---- hyperparameters (paper Section 6) ----
N, W = 5, 5
SEG_LEN = 400
LAM = 0.07
BETA = 60
N_RESTARTS = 20
DATA_SEED = 7
FIT_SEED = 42
DTW_MAX_SAMPLES = 1500          # subsample cap for the (slow) DTW baselines

SEQUENCES = {
    "1,2,1": [1, 2, 1],
    "1,2,3,2,1": [1, 2, 3, 2, 1],
    "1,2,3,4,1,2,3,4": [1, 2, 3, 4, 1, 2, 3, 4],
    "1,2,2,1,3,3,3,1": [1, 2, 2, 1, 3, 3, 3, 1],
}
METHODS = ["TICC", "TICC b=0", "GMM", "EEV", "DTW-GAK", "DTW-Euclidean",
           "Neural Gas", "K-means"]

CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "table1_f1.json")


def _subsample(Xst, tv, cap, seed=0):
    if Xst.shape[0] <= cap:
        return Xst, tv
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(Xst.shape[0], cap, replace=False))
    return Xst[idx], tv[idx]


def run(force=False, verbose=True):
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as f:
            return json.load(f)

    results = {m: {} for m in METHODS}
    for name, seq in SEQUENCES.items():
        K = len(set(seq))
        X, tl, _ = generate_series(seq, seg_len=SEG_LEN, n=N, w=W, seed=DATA_SEED)
        Xst = stack_windows(X, W)
        tv = tl[W - 1:]
        Xsub, tvsub = _subsample(Xst, tv, DTW_MAX_SAMPLES)

        t0 = time.time()
        lab, _, _ = fit_ticc(X, K=K, w=W, lam=LAM, beta=BETA,
                             n_restarts=N_RESTARTS, seed=FIT_SEED)
        lab0, _, _ = fit_ticc(X, K=K, w=W, lam=LAM, beta=0.0,
                              n_restarts=N_RESTARTS, seed=FIT_SEED)
        scores = {
            "TICC": macro_f1(tv, lab),
            "TICC b=0": macro_f1(tv, lab0),
            "GMM": macro_f1(tv, B.gmm_labels(Xst, K)),
            "EEV": macro_f1(tv, B.eev_labels(Xst, K)),
            "DTW-GAK": macro_f1(tvsub, B.dtw_gak_labels(Xsub, K, N, W)),
            "DTW-Euclidean": macro_f1(tvsub, B.dtw_euclidean_labels(Xsub, K, N, W)),
            "Neural Gas": macro_f1(tv, B.neural_gas_labels(Xst, K)),
            "K-means": macro_f1(tv, B.kmeans_labels(Xst, K)),
        }
        for m, v in scores.items():
            results[m][name] = float(v)
        if verbose:
            print(f"[{name}] K={K} done in {time.time() - t0:.0f}s  TICC={scores['TICC']:.3f}")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(results, f, indent=2)
    return results


def print_table(results):
    seqs = list(SEQUENCES)
    header = f"{'Method':16s} " + " ".join(f"{s:>16s}" for s in seqs) + f" {'avg':>6s}"
    print(header)
    print("-" * len(header))
    for m in METHODS:
        row = [results[m][s] for s in seqs]
        avg = np.mean(row)
        print(f"{m:16s} " + " ".join(f"{v:>16.2f}" for v in row) + f" {avg:>6.2f}")


if __name__ == "__main__":
    res = run(force="--force" in sys.argv)
    print()
    print_table(res)
