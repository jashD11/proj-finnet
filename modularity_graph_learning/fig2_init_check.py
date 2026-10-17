"""
Why solver/algorithm1.py starts from the Problem 1 estimate.

From the uniform complete graph the first V-step sees a (p-1)-fold eigenvalue,
so the basis eigh returns, and hence the components Algorithm 1 forces, are set
by floating-point round-off. Changing nothing but the number of BLAS threads
changes the answer. From the Problem 1 start the answer is the same either way.

Usage:  python fig2_init_check.py {uniform|problem1} {1|default}
Writes  cache_alg1_<start>_<threads>.npy (about 8 minutes each), read by
report_numbers.py. Paper 1's own data/alg1_L.npy is the (problem1, default) run.
"""

import os
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from common import HERE, K, load_sp

from solver.algorithm1 import algorithm1  # noqa: E402  (Paper 1)

ETA, RHO = 300.0, 100.0  # exp2_kcomponent.py's values


def cache_path(start, threads):
    return os.path.join(HERE, f"cache_alg1_{start}_{threads}.npy")


def main():
    start, threads = sys.argv[1], sys.argv[2]
    _, S, _ = load_sp()
    p = S.shape[0]
    L0 = np.eye(p) * (p - 1) / p - np.ones((p, p)) / p if start == "uniform" else None
    limit = 1 if threads == "1" else None
    with threadpool_limits(limits=limit):
        L = algorithm1(S, k=K, beta=0.0, eta=ETA, rho_degree=RHO, max_outer=80, L0=L0)
    np.save(cache_path(start, threads), L)
    print(f"Saved {cache_path(start, threads)}")


if __name__ == "__main__":
    main()
