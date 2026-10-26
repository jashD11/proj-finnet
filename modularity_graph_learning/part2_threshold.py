"""
Part II: where does the implicit null start to cluster by beta?

Cross pattern only (betas cut across sectors), delta from 0.40 to 0.60 in steps of
0.025, 5 seeds, inputs S / S-mode / oracle, both learners. Each learned L is
cached in part2_cache/threshold/ (synthetic Algorithm 1 runs take 1 to 2 s).

Usage:  python part2_threshold.py
"""

import os

import numpy as np

from part2 import CACHE, learn, synthetic_inputs, synthetic_returns

TCACHE = os.path.join(CACHE, "threshold")
FINE_DELTAS = tuple(np.round(np.arange(0.40, 0.601, 0.025), 3))
FINE_SEEDS = tuple(range(5))


def tpath(delta, seed, name, learner):
    return os.path.join(TCACHE, f"cross_{delta:g}_seed{seed}_{name}_{learner}.npy")


def main():
    os.makedirs(TCACHE, exist_ok=True)
    for delta in FINE_DELTAS:
        for seed in FINE_SEEDS:
            X, f, _, _ = synthetic_returns("cross", delta, seed)
            for name, S in synthetic_inputs(X, f).items():
                for learner in ("alg1", "prob1"):
                    path = tpath(delta, seed, name, learner)
                    if not os.path.exists(path):
                        np.save(path, learn(S, learner))
        print(f"delta {delta:g} done", flush=True)
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
