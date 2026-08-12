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


def _chunk_path(cache_dir, span):
    return os.path.join(cache_dir, f"chunk_{span[0]:06d}_{span[1]:06d}.npz")


def _save_chunk(cache_dir, span, result):
    """Write one chunk atomically, so a kill mid-write cannot leave a torn file."""
    path = _chunk_path(cache_dir, span)
    tmp = path + ".tmp"
    # Written through a file handle on purpose: given a *path* without a .npz
    # suffix, np.savez silently appends one, so the rename target would not
    # exist. Passing an open file leaves the name alone.
    with open(tmp, "wb") as fh:
        np.savez(fh, **{k: np.asarray(v) for k, v in result.items()})
    os.replace(tmp, path)


def _load_chunk(cache_dir, span):
    with np.load(_chunk_path(cache_dir, span)) as z:
        return {k: z[k] for k in z.files}


def map_windows(worker, n_windows: int, initializer=None, initargs=(),
                n_workers: int = None, chunk_size: int = 64,
                verbose: bool = True, label: str = "work", cache_dir=None):
    """Run `worker((lo, hi))` over all windows, in order. Returns the list of
    per-chunk results.

    `worker` and `initializer` must be importable module-level callables --
    macOS spawns fresh interpreters rather than forking.

    With `cache_dir`, every completed chunk is written to disk as it lands and
    re-used on a later call. A 6,315-day run that dies at 95 % then costs five
    minutes to finish instead of starting over -- which is the difference
    between an interrupted job and a lost one.
    """
    n_workers = n_workers or DEFAULT_WORKERS
    chunks = chunk_ranges(n_windows, chunk_size)

    todo = chunks
    n_cached = 0
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        todo = [c for c in chunks if not os.path.exists(_chunk_path(cache_dir, c))]
        n_cached = len(chunks) - len(todo)
        if verbose and n_cached:
            print(f"  [{label}] resuming: {n_cached}/{len(chunks)} chunks already "
                  f"on disk in {cache_dir}")
    t0 = time.time()

    def _finish(span, result):
        if cache_dir:
            _save_chunk(cache_dir, span, result)

    def _collect():
        if not cache_dir:
            return None
        return [_load_chunk(cache_dir, c) for c in chunks]

    if n_workers == 1:
        if initializer is not None:
            initializer(*initargs)
        out = {}
        for i, c in enumerate(todo):
            out[c] = worker(c)
            _finish(c, out[c])
            _tick(verbose, label, i + 1, len(todo), t0)
        return _collect() or [out[c] for c in chunks]

    # Pin each worker's BLAS to one thread. numpy's matmul (the 360x360 product
    # in the matching index) otherwise opens a full thread pool *per process*:
    # 7 workers x 16 threads on 8 cores drove the load average past 200 and made
    # the pool ~2x slower than the single-process benchmark predicted. This has
    # to happen before the children spawn, because OpenBLAS reads it once at
    # import time -- setting it inside the initializer is already too late.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    ctx = mp.get_context("spawn")
    got = {}
    with ctx.Pool(n_workers, initializer=initializer, initargs=initargs) as pool:
        for i, (span, res) in enumerate(zip(todo, pool.imap(worker, todo))):
            got[span] = res
            _finish(span, res)
            _tick(verbose, label, i + 1, len(todo), t0)
    return _collect() or [got[c] for c in chunks]


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
