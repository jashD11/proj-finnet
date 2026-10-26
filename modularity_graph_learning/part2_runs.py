"""
Run the Part II graph-learning jobs; each learned L is cached in part2_cache/.

Jobs:
  real/<input>/<learner>                         6 inputs x 2 learners
  syn/<pattern>/<delta>/seed<s>/<input>/<learner>  9 configs x 2 seeds x 3 inputs x 2 learners

Usage:  python part2_runs.py WORKER
Any number of workers share one queue: each claims the next job that is neither
cached nor claimed (an O_EXCL '.claim' file), so the load balances itself and the
run can be stopped and restarted (delete stale *.claim files first, with no
worker running). Meant to be launched single-threaded and niced (see the README)
so it can share the machine.
"""

import os
import sys
import time
import traceback

import numpy as np

from common import load_sp
from part2 import (CACHE, cache_file, learn, load_index, real_inputs,
                   synthetic_configs, synthetic_inputs, synthetic_returns)

LEARNERS = ("alg1", "prob1")
SEEDS = (0, 1)


def jobs():
    out = [f"real/{name}/{lr}" for name in
           ("S", "JSJ", "S-mode", "resid-SPX", "resid-EW", "RMT") for lr in LEARNERS]
    for pattern, delta in synthetic_configs():
        for seed in SEEDS:
            for name in ("S", "S-mode", "oracle"):
                for lr in LEARNERS:
                    out.append(f"syn/{pattern}/{delta:g}/seed{seed}/{name}/{lr}")
    return out


def matrix_for(key, real):
    parts = key.split("/")
    if parts[0] == "real":
        return real[parts[1]]
    pattern, delta, seed, name = parts[1], float(parts[2]), int(parts[3][4:]), parts[4]
    X, f, _, _ = synthetic_returns(pattern, delta, seed)
    return synthetic_inputs(X, f)[name]


def main():
    w = int(sys.argv[1])
    os.makedirs(CACHE, exist_ok=True)
    X, _, _ = load_sp()
    real = real_inputs(X.values, load_index().values)
    todo = jobs()
    print(f"worker {w}: queue of {len(todo)} jobs", flush=True)
    for i, key in enumerate(todo, 1):
        path = cache_file(key)
        if os.path.exists(path):
            continue
        try:
            os.close(os.open(path + ".claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        t = time.time()
        print(f"START [{i}/{len(todo)}] {key}", flush=True)
        try:
            L = learn(matrix_for(key, real), key.split("/")[-1])
            np.save(path, L)
            print(f"DONE [{i}/{len(todo)}] {key} {time.time() - t:.0f}s", flush=True)
        except Exception:
            print(f"FAIL [{i}/{len(todo)}] {key}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
    print(f"worker {w} FINISHED", flush=True)


if __name__ == "__main__":
    main()
