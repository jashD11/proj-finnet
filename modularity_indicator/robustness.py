"""
Two corrections to evaluate.py's raw table.

(1) DETECTOR FORM. Raw-level AUC is confounded: FAAMUNG's autumn-2019 regime is
    elevated, so a signal that is genuinely high during COVID still loses to
    every quiet day scored against it. Paper 3's convention is a strictly-prior
    trailing z-score; apply the same here (60-day baseline, the longest that
    fits a 201-day window) and re-score.

(2) MULTIPLE COMPARISONS. 28 candidates, one crash. Quantify how easy it is to
    beat S1 by luck at the same invest rate, using
      - a uniform random-gate null, and
      - a circular-shift null that preserves each signal's autocorrelation.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate import (HERE, TAU, COVID_START, COVID_END, REFERENCE,
                      LAPLACIAN_RESTATEMENTS, load, backtest)

BASELINE = 60
N_PERM = 20000
RNG = np.random.default_rng(42)


def trailing_z(x, baseline=BASELINE):
    """Strictly-prior rolling z-score (paper 3's _detector_scores convention)."""
    s = pd.Series(x)
    m = s.shift(1).rolling(baseline, min_periods=baseline // 2).mean()
    sd = s.shift(1).rolling(baseline, min_periods=baseline // 2).std()
    return ((s - m) / sd).values


def main():
    df = load()
    dts = pd.DatetimeIndex(df.index)
    inv_dts = pd.DatetimeIndex(df["invest_date"])
    lam2 = df["lam2"].values
    s1_ret = df["s1_ret"].values
    s1_final = s1_ret.sum()
    lam2_gate = lam2 < TAU
    n_invest = int(lam2_gate.sum())
    crash = (dts >= COVID_START) & (dts <= COVID_END)

    cols = [c for c in df.columns
            if c not in ("day_index", "s1_ret", "invest_date", "fwd_ret", "fwd_dd")]

    # ── (1) z-scored detector AUC ────────────────────────────────────────────
    rows = []
    for c in cols:
        x = df[c].values.astype(float)
        if np.std(x) < 1e-12:
            continue
        s = 1.0 if np.corrcoef(x, lam2)[0, 1] >= 0 else -1.0
        z = trailing_z(s * x)
        ok = np.isfinite(z)
        rows.append(dict(
            quantity=c, sign=int(s),
            auc_raw=roc_auc_score(crash.astype(int), s * x),
            auc_z=roc_auc_score(crash[ok].astype(int), z[ok]),
            z_peak_crash=np.nanmax(z[crash]),
            z_mean_crash=np.nanmean(z[crash]),
        ))
    Z = pd.DataFrame(rows).set_index("quantity")
    Z["family"] = ["laplacian ref" if q in REFERENCE else
                   ("L restatement" if q in LAPLACIAN_RESTATEMENTS else "modularity")
                   for q in Z.index]
    Z = Z[["family", "sign", "auc_raw", "auc_z", "z_mean_crash", "z_peak_crash"]]
    Z.to_csv(os.path.join(HERE, "scores_zdetector.csv"), float_format="%.6g")

    print(f"Crash days in window: {int(crash.sum())}/{len(df)}; "
          f"z-detector baseline = {BASELINE} days (first {BASELINE//2} days undefined)\n")
    print("=" * 96)
    print("COVID AUC: raw level vs strictly-prior 60-day z-score (the fair detector form)")
    print("=" * 96)
    print(Z.sort_values("auc_z", ascending=False).to_string(
        float_format=lambda v: f"{v:+.3f}"))

    # ── (2) how easy is it to beat S1 by luck? ───────────────────────────────
    print("\n" + "=" * 96)
    print(f"NULLS for the backtest (invest {n_invest}/{len(df)} days; S1 = {s1_final:+.4f})")
    print("=" * 96)

    # uniform random gate
    rand = np.empty(N_PERM)
    for i in range(N_PERM):
        pick = RNG.choice(len(s1_ret), size=n_invest, replace=False)
        rand[i] = s1_ret[pick].sum()
    print(f"  uniform random gate: mean {rand.mean():+.4f}, sd {rand.std():.4f}, "
          f"P(beat S1) = {(rand > s1_final).mean():.3f}, "
          f"95th pct = {np.percentile(rand, 95):+.4f}")

    # circular-shift null, per candidate: preserves the signal's own autocorrelation
    print("\n  circular-shift null (preserves each signal's autocorrelation):")
    print(f"  {'quantity':<16} {'S2 actual':>10} {'null mean':>10} "
          f"{'null 95%':>10} {'p-value':>8}")
    shift_rows = []
    n = len(s1_ret)
    for c in cols:
        x = df[c].values.astype(float)
        if np.std(x) < 1e-12:
            continue
        s = 1.0 if np.corrcoef(x, lam2)[0, 1] >= 0 else -1.0
        z = s * x
        actual = backtest(z, n_invest, s1_ret, inv_dts)["final"]
        null = np.array([backtest(np.roll(z, k), n_invest, s1_ret, inv_dts)["final"]
                         for k in range(1, n)])
        p = float((null >= actual).mean())
        shift_rows.append(dict(quantity=c, s2=actual, null_mean=null.mean(),
                               null_p95=np.percentile(null, 95), p_value=p))
    S = pd.DataFrame(shift_rows).set_index("quantity").sort_values("s2", ascending=False)
    for q, r in S.iterrows():
        print(f"  {q:<16} {r.s2:>+10.4f} {r.null_mean:>+10.4f} "
              f"{r.null_p95:>+10.4f} {r.p_value:>8.3f}")
    S.to_csv(os.path.join(HERE, "scores_nulls.csv"), float_format="%.6g")

    n_beat = int((S["s2"] > s1_final).sum())
    print(f"\n  {n_beat}/{len(S)} candidates beat S1. Under the uniform null a single "
          f"random gate beats S1 with prob {(rand > s1_final).mean():.3f},")
    print(f"  so ~{(rand > s1_final).mean()*len(S):.1f} of {len(S)} would be expected "
          f"to beat it by chance alone.")
    print("\nWrote scores_zdetector.csv, scores_nulls.csv")


if __name__ == "__main__":
    main()
