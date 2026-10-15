"""
How much of the detection result is real, given ONE crash?

The 24 COVID days are contiguous, so an iid bootstrap over days badly
overstates precision. Two tests:

  A. STRATIFIED BOOTSTRAP over days -> an optimistic CI on AUC, and on the
     paired difference AUC(candidate) - AUC(lambda2) on the same resample.
  B. SLIDING-EVENT NULL (the honest one) -> slide a 24-day contiguous window
     across the whole series and recompute AUC. If the signal is specifically
     about COVID rather than generically spiky, the true window should sit in
     the far right tail of that distribution.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate import HERE, TAU, COVID_START, COVID_END, load
from robustness import trailing_z

N_BOOT = 5000
RNG = np.random.default_rng(7)
FOCUS = ["lam2", "muN", "gap12", "mu1", "mu_ratio", "ipr1", "mu1_norm",
         "frob2", "energy", "trace_pos", "q_bound"]


def main():
    df = load()
    dts = pd.DatetimeIndex(df.index)
    lam2 = df["lam2"].values
    crash = np.asarray((dts >= COVID_START) & (dts <= COVID_END))
    width = int(crash.sum())

    zs = {}
    for c in FOCUS:
        x = df[c].values.astype(float)
        s = 1.0 if np.corrcoef(x, lam2)[0, 1] >= 0 else -1.0
        zs[c] = trailing_z(s * x)

    ok = np.isfinite(zs["lam2"])
    for v in zs.values():
        ok &= np.isfinite(v)
    y = crash[ok].astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    print(f"Scored days: {ok.sum()} ({y.sum()} crash, {len(y)-y.sum()} non-crash)\n")

    # ── A. stratified bootstrap ──────────────────────────────────────────────
    base = zs["lam2"][ok]
    print("=" * 88)
    print("A. Stratified bootstrap CI (optimistic - ignores that crash days are contiguous)")
    print("=" * 88)
    print(f"  {'quantity':<12} {'AUC':>7} {'95% CI':>18} {'dAUC vs lam2':>14} "
          f"{'95% CI':>18} {'P(>lam2)':>9}")
    boot_idx = [(RNG.choice(pos, len(pos), True), RNG.choice(neg, len(neg), True))
                for _ in range(N_BOOT)]
    rows = []
    for c in FOCUS:
        v = zs[c][ok]
        auc = roc_auc_score(y, v)
        bs, ds = [], []
        for bp, bn in boot_idx:
            ii = np.concatenate([bp, bn])
            yy = np.concatenate([np.ones(len(bp)), np.zeros(len(bn))])
            a = roc_auc_score(yy, v[ii])
            bs.append(a)
            ds.append(a - roc_auc_score(yy, base[ii]))
        bs, ds = np.array(bs), np.array(ds)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        dlo, dhi = np.percentile(ds, [2.5, 97.5])
        pw = (ds > 0).mean()
        rows.append(dict(quantity=c, auc=auc, lo=lo, hi=hi,
                         d_auc=ds.mean(), d_lo=dlo, d_hi=dhi, p_better=pw))
        print(f"  {c:<12} {auc:>7.3f} [{lo:>6.3f},{hi:>6.3f}] {ds.mean():>+14.3f} "
              f"[{dlo:>+6.3f},{dhi:>+6.3f}] {pw:>9.3f}")

    # ── B. sliding-event null ────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(f"B. Sliding-event null: AUC for every contiguous {width}-day window "
          f"(the honest test)")
    print("=" * 88)
    print(f"  {'quantity':<12} {'AUC(COVID)':>11} {'null med':>9} {'null 95%':>9} "
          f"{'null max':>9} {'pctile':>7} {'p':>6}")
    n = int(ok.sum())
    starts = [s for s in range(n - width + 1)]
    for r in rows:
        c = r["quantity"]
        v = zs[c][ok]
        null = []
        for s in starts:
            lab = np.zeros(n, dtype=int)
            lab[s:s + width] = 1
            null.append(roc_auc_score(lab, v))
        null = np.array(null)
        pct = (null < r["auc"]).mean()
        p = (null >= r["auc"]).mean()
        print(f"  {c:<12} {r['auc']:>11.3f} {np.median(null):>9.3f} "
              f"{np.percentile(null,95):>9.3f} {null.max():>9.3f} "
              f"{pct:>7.3f} {p:>6.3f}")
        r.update(null_med=np.median(null), null_p95=np.percentile(null, 95),
                 null_max=null.max(), slide_pctile=pct, slide_p=p)

    pd.DataFrame(rows).set_index("quantity").to_csv(
        os.path.join(HERE, "scores_significance.csv"), float_format="%.6g")

    # ── redundancy: is muN just lambda2 again? ───────────────────────────────
    print("\n" + "=" * 88)
    print("Redundancy of the leading candidates against lambda2 (z-scored detectors)")
    print("=" * 88)
    for c in FOCUS[1:]:
        r = np.corrcoef(zs[c][ok], base)[0, 1]
        print(f"  corr(z[{c}], z[lam2]) = {r:+.3f}")
    print("\nWrote scores_significance.csv")


if __name__ == "__main__":
    main()
