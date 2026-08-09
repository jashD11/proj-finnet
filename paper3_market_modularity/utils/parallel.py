"""
A small process pool for the per-day work in Phases 4 and 5.

Every day of the sample is independent given the stored artifacts, so the whole
computation is embarrassingly parallel over windows. The only thing worth being
careful about is that each worker must open the big arrays once, not once per
day, so they are loaded in an initializer and kept in module globals.

Determinism does not depend on the schedule: every random draw is seeded from
(FIT_SEED, window, replicate), so a run with any number of workers, in any
order, produces bit-identical output.
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np

DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def chunk_ranges(n_items: int, chunk_size: int):
    return [(a, min(a + chunk_size, n_items)) for a in range(0, n_items, chunk_size)]


def map_windows(worker, n_windows: int, initializer=None, initargs=(),
                n_workers: int = None, chunk_size: int = 64,
                verbose: bool = True, label: str = "work"):
    """Run `worker((lo, hi))` over all windows, in order. Returns the list of
    per-chunk results.

    `worker` and `initializer` must be importable module-level callables --
    macOS spawns fresh interpreters rather than forking.
    """
    n_workers = n_workers or DEFAULT_WORKERS
    chunks = chunk_ranges(n_windows, chunk_size)
    t0 = time.time()

    if n_workers == 1:
        if initializer is not None:
            initializer(*initargs)
        out = []
        for i, c in enumerate(chunks):
            out.append(worker(c))
            _tick(verbose, label, i + 1, len(chunks), t0)
        return out

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=initializer, initargs=initargs) as pool:
        out = []
        for i, res in enumerate(pool.imap(worker, chunks)):
            out.append(res)
            _tick(verbose, label, i + 1, len(chunks), t0)
    return out


def _tick(verbose, label, done, total, t0):
    if not verbose or done % 10 and done != total:
        return
    el = time.time() - t0
    left = el / done * (total - done)
    sys.stdout.write(f"\r  [{label}] chunk {done}/{total}  "
                     f"{el:6.1f}s elapsed, ~{left:6.1f}s left    ")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def stack(chunk_results, key):
    """Concatenate one field across chunk results, preserving window order."""
    return np.concatenate([r[key] for r in chunk_results], axis=0)
