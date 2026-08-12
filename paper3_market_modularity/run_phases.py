"""
Run a contiguous range of phases end to end, stopping at the first failure.

    /opt/anaconda3/bin/python run_phases.py 5 9
    /opt/anaconda3/bin/python run_phases.py 5 9 --force

Each phase runs its own `run()`, prints its report, draws its figures and runs
its acceptance tests, exactly as invoking the experiment directly would. A phase
whose acceptance tests fail stops the sequence -- later phases consume its
output, so continuing past a failure would silently propagate bad numbers.

Phase 5 checkpoints per chunk, so re-running after an interruption resumes.
"""

import importlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

MODULES = {
    1: "exp1_data", 2: "exp2_networks", 3: "exp3_communities",
    4: "exp4_nulls", 5: "exp5_measures", 6: "exp6_scoring",
    7: "exp7_pca", 8: "exp8_extensions", 9: "exp9_report",
}


def main(lo: int, hi: int, force: bool = False) -> int:
    t_all = time.time()
    results = []
    for n in range(lo, hi + 1):
        name = MODULES[n]
        print(f"\n{'=' * 74}\n=== Phase {n} — {name}\n{'=' * 74}")
        mod = importlib.import_module(name)
        t0 = time.time()
        rep = mod.run(force=force)
        if hasattr(mod, "print_report"):
            mod.print_report(rep)
        if hasattr(mod, "plot"):
            mod.plot()
        ok = mod.acceptance(rep)
        dt = time.time() - t0
        results.append((n, ok, dt))
        print(f"\n=== Phase {n}: {'PASS' if ok else 'FAIL'} in {dt / 60:.1f} min")
        if not ok:
            print(f"\nStopping: phase {n} failed its acceptance tests. Later "
                  f"phases consume its output.")
            break

    print(f"\n{'=' * 74}\nSummary ({(time.time() - t_all) / 60:.1f} min total)")
    for n, ok, dt in results:
        print(f"  Phase {n} {MODULES[n]:<18} {'PASS' if ok else 'FAIL':<5} "
              f"{dt / 60:6.1f} min")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lo = int(args[0]) if args else 5
    hi = int(args[1]) if len(args) > 1 else 9
    sys.exit(main(lo, hi, force="--force" in sys.argv))
