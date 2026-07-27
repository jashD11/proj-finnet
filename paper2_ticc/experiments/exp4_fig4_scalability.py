"""
Experiment 4 -- TICC paper Figure 4: per-iteration scalability.

One TICC iteration = one M-step (solve the Toeplitz graphical lasso for each of the
K clusters via ADMM) plus one E-step (compute the T x K log-likelihoods and run the
dynamic-programming assignment). The ADMM cost is independent of the number of
observations T (it depends only on nw), so it contributes a constant offset, while
the E-step is linear in T. We therefore expect the per-iteration runtime to grow
linearly with T at large T -- the paper's result.

Setup mirrors the paper: observations in R^50, K=5 clusters, window size w=3
(so nw = 150). Runtime is structure-independent, so we time on random data and a
random initial labeling. Cached to results/fig4_scalability.json; figure to
figures/fig4_scalability.png.
"""

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.ticc import stack_windows
from solver.toeplitz_admm import solve_toeplitz_graphical_lasso, _toeplitz_groups
from solver.dp_assign import gaussian_neg_ll, assign_points

N, W, K = 50, 3, 5
LAM, BETA = 0.1, 60
T_VALUES = [1000, 3000, 10000, 30000, 100000]
SEED = 0

CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "fig4_scalability.json")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures", "fig4_scalability.png")


def _one_iteration(Xst, labels, n, w, gid, penalize):
    """Time a single M-step (K ADMM solves) + E-step (nll + DP)."""
    K_ = labels.max() + 1
    t0 = time.time()
    thetas = []
    for k in range(K_):
        idx = labels == k
        Sk = Xst[idx].T @ Xst[idx] / max(idx.sum(), 1)
        thetas.append(solve_toeplitz_graphical_lasso(
            Sk, LAM, n, w, gid=gid, penalize=penalize, max_iter=200))
    nll = gaussian_neg_ll(Xst, thetas)
    assign_points(nll, BETA)
    return time.time() - t0


def run(force=False, verbose=True):
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as f:
            return json.load(f)

    rng = np.random.default_rng(SEED)
    gid, penalize = _toeplitz_groups(N, W)
    times = []
    for T in T_VALUES:
        X = rng.standard_normal((T, N))
        Xst = stack_windows(X, W)
        labels = rng.integers(0, K, size=Xst.shape[0])
        _one_iteration(Xst, labels, N, W, gid, penalize)          # warm-up
        secs = np.median([_one_iteration(Xst, labels, N, W, gid, penalize) for _ in range(3)])
        times.append(float(secs))
        if verbose:
            print(f"T={T:>7d}  per-iteration={secs:.2f}s")

    out = {"T": T_VALUES, "times": times}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(out, f, indent=2)
    return out


def plot(data, save=FIG):
    T = np.array(data["T"], dtype=float)
    times = np.array(data["times"], dtype=float)
    plt.figure(figsize=(7, 5))
    plt.loglog(T, times, "o-", color="C0", lw=2, label="measured")
    # linear-scaling reference anchored at the largest T
    ref = times[-1] * (T / T[-1])
    plt.loglog(T, ref, "--", color="gray", alpha=0.7, label="linear in T (slope 1)")
    plt.xlabel("Number of observations T")
    plt.ylabel("Per-iteration runtime (s)")
    plt.title("TICC scalability (Fig 4): observations in R$^{50}$, K=5, w=3")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=130)
    plt.close("all")
    return save


if __name__ == "__main__":
    data = run(force="--force" in sys.argv)
    path = plot(data)
    print("saved", path)
