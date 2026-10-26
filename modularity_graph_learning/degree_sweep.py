"""
Part I, loosening the degree rule: Algorithm 1 at a weaker degree penalty.

Part I's identity B = I - L - 11'/p needs every degree to be 1. Algorithm 1
only enforces diag(L) = 1 through the penalty (rho/2) ||diag(L) - 1||^2, so
lowering rho lets the degrees spread while the rank constraint still forces
K components. report_numbers.py (section G) measures how far the match between
L and B drifts as they do.

Usage:  python degree_sweep.py RHO [RHO ...]
Writes  cache_alg1_rho<RHO>.npy (about 8 minutes each). rho = 100 is not rerun:
it is Paper 1's own data/alg1_L.npy (the repaired Fig 2 graph).
"""

import os
import sys

import numpy as np

from common import HERE, K, load_sp

from solver.algorithm1 import algorithm1  # noqa: E402  (Paper 1)

ETA = 300.0  # exp2_kcomponent.py's value
RHOS = (100.0, 30.0, 10.0, 3.0, 1.0, 0.3)


def cache_path(rho):
    return os.path.join(HERE, f"cache_alg1_rho{rho:g}.npy")


def main():
    _, S, _ = load_sp()
    for rho in map(float, sys.argv[1:]):
        L = algorithm1(S, k=K, beta=0.0, eta=ETA, rho_degree=rho, max_outer=80)
        np.save(cache_path(rho), L)
        print(f"Saved {cache_path(rho)}", flush=True)


if __name__ == "__main__":
    main()
