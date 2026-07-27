"""
Experiment 2 -- TICC paper Table 2: MRF network edge-recovery F1.

TICC models each cluster as a Markov random field (the nonzero pattern of its
block-Toeplitz inverse covariance). Here we measure how well the recovered edge
structure matches the ground-truth network, averaged over clusters, for the same
four sequences. Cached to results/table2_recovery.json.

Paper reference: recovery F1 between 0.79 and 0.90.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic import generate_series
from solver.ticc import fit_ticc
from utils.metrics import macro_f1, network_recovery_f1, align_labels

from exp1_table1_f1 import (N, W, SEG_LEN, LAM, BETA, N_RESTARTS, DATA_SEED,
                            FIT_SEED, SEQUENCES)

CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "table2_recovery.json")


def run(force=False, verbose=True):
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as f:
            return json.load(f)

    results = {}
    for name, seq in SEQUENCES.items():
        K = len(set(seq))
        X, tl, thetas_true = generate_series(seq, seg_len=SEG_LEN, n=N, w=W, seed=DATA_SEED)
        tv = tl[W - 1:]
        labels, thetas_est, _ = fit_ticc(X, K=K, w=W, lam=LAM, beta=BETA,
                                         n_restarts=N_RESTARTS, seed=FIT_SEED)
        _, true_to_pred = align_labels(tv, labels)
        rec = network_recovery_f1(thetas_est, thetas_true, true_to_pred, N, W)
        clus = macro_f1(tv, labels)
        results[name] = {"recovery_f1": float(rec), "cluster_f1": float(clus)}
        if verbose:
            print(f"[{name}] K={K} recovery-F1={rec:.3f} (cluster-F1={clus:.3f})")

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(results, f, indent=2)
    return results


def print_table(results):
    print(f"{'Sequence':18s} {'Recovery-F1':>12s}")
    print("-" * 31)
    for name, r in results.items():
        print(f"{name:18s} {r['recovery_f1']:>12.2f}")


if __name__ == "__main__":
    res = run(force="--force" in sys.argv)
    print()
    print_table(res)
