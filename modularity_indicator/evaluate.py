"""
Score every modularity-matrix quantity as a substitute for Paper 1's lambda2.

The question is equivalence: does the candidate call the same days that
lambda2 calls? So each candidate is judged on the four things Paper 1
actually uses lambda2 for, on Paper 1's own FAAMUNG data.

  1. AGREEMENT   correlation with lambda2 (levels, ranks, first differences)
  2. GATE        does its trading gate select the same days as lambda2 < tau=1.0
  3. MECHANISM   does it rise into the COVID crash off a local pre-crash baseline
  4. BACKTEST    S1 vs S2 when it replaces lambda2 in exp4's strategy

Direction convention: each candidate's "stress direction" s is fixed by the
sign of its correlation with lambda2, NOT by what scores best. s*x is then
always a high-means-stress signal, directly comparable to lambda2.

Everything is run on exp4's evaluation window (signals whose invest day falls
on or before 2020-05-01), so the lambda2 row reproduces Paper 1's published
numbers exactly.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(os.path.dirname(HERE), "paper1_gmrf_laplacian")

# ── exp4's constants ──────────────────────────────────────────────────────────
TAU = 1.0
END_DATE = pd.Timestamp("2020-05-01")
COVID_START = pd.Timestamp("2020-02-19")
COVID_END = pd.Timestamp("2020-03-23")
BASELINE_DAYS = 60
FWD = 5
# ──────────────────────────────────────────────────────────────────────────────

REFERENCE = ["lam2", "lam_max", "lam2_sym", "dbar", "two_m", "deg_cv"]
LAPLACIAN_RESTATEMENTS = {"one_minus_mu1n", "dbar_minus_mu1", "dual_resid"}


def load():
    df = pd.read_parquet(os.path.join(HERE, "series.parquet"))
    rets = pd.read_csv(os.path.join(P1, "data", "faamung.csv"),
                       index_col=0, parse_dates=True)
    port = rets.values.mean(axis=1)          # equal-weight daily log return
    dates = rets.index
    T = len(dates)

    # exp4 alignment: signal from window ending day t -> invest on day t+1
    keep = [i for i, t in enumerate(df["day_index"].values)
            if t + 1 < T and dates[t + 1] <= END_DATE]
    df = df.iloc[keep].copy()
    idx = df["day_index"].values.astype(int)
    df["s1_ret"] = port[idx + 1]
    df["invest_date"] = dates[idx + 1]

    # forward-window diagnostics from the signal date
    f_ret, f_dd = [], []
    for t in idx:
        w = port[t + 1: t + 1 + FWD]
        f_ret.append(w.sum() if len(w) else np.nan)
        f_dd.append(w.min() if len(w) else np.nan)
    df["fwd_ret"] = f_ret
    df["fwd_dd"] = f_dd
    return df


def backtest(stress, n_invest, s1_ret, invest_dates):
    """Invest on the n_invest lowest-stress days; cash otherwise."""
    thr = np.sort(stress)[n_invest - 1]
    inv = stress <= thr
    if inv.sum() != n_invest:                      # ties: trim deterministically
        order = np.argsort(stress, kind="stable")
        inv = np.zeros_like(stress, dtype=bool)
        inv[order[:n_invest]] = True
    s2 = np.where(inv, s1_ret, 0.0)
    cum = np.cumsum(s2)
    crash = (invest_dates >= COVID_START) & (invest_dates <= COVID_END)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    dd = (np.concatenate([[0.0], cum]) - peak).min()
    return dict(final=cum[-1], crash_pnl=s2[crash].sum(), max_dd=dd, invest=inv)


def main():
    df = load()
    dts = pd.DatetimeIndex(df.index)
    inv_dts = pd.DatetimeIndex(df["invest_date"])
    lam2 = df["lam2"].values
    s1_ret = df["s1_ret"].values
    s1_final = s1_ret.sum()

    lam2_gate = lam2 < TAU
    n_invest = int(lam2_gate.sum())
    print(f"Evaluation window: {len(df)} signal days, "
          f"{dts[0].date()} -> {inv_dts[-1].date()}")
    print(f"lambda2 gate (tau=1.0) invests {n_invest}/{len(df)} days "
          f"({100*n_invest/len(df):.1f}%) -- every candidate is matched to this rate.\n")

    crash_mask = (dts >= COVID_START) & (dts <= COVID_END)
    pre = np.where(dts < COVID_START)[0][-BASELINE_DAYS:]

    cols = [c for c in df.columns
            if c not in ("day_index", "s1_ret", "invest_date", "fwd_ret", "fwd_dd")]

    rows = []
    for c in cols:
        x = df[c].values.astype(float)
        if np.std(x) < 1e-12:
            continue
        r_lvl = np.corrcoef(x, lam2)[0, 1]
        s = 1.0 if r_lvl >= 0 else -1.0        # stress direction fixed by lambda2
        z = s * x                               # high = stress, like lambda2

        r_rank = spearmanr(z, lam2).statistic
        dz, dl = np.diff(z), np.diff(lam2)
        r_diff = np.corrcoef(dz, dl)[0, 1]

        bt = backtest(z, n_invest, s1_ret, inv_dts)
        agree = float((bt["invest"] == lam2_gate).mean())
        jac = float((bt["invest"] & lam2_gate).sum() /
                    max((bt["invest"] | lam2_gate).sum(), 1))

        mech = z[crash_mask].mean() - z[pre].mean()
        mech_sd = mech / max(z[pre].std(), 1e-12)   # baseline standard deviations

        auc = roc_auc_score(crash_mask.astype(int), z)
        ok = np.isfinite(df["fwd_ret"].values)
        r_fwd = np.corrcoef(z[ok], df["fwd_ret"].values[ok])[0, 1]
        r_dd = np.corrcoef(z[ok], df["fwd_dd"].values[ok])[0, 1]

        rows.append(dict(
            quantity=c, sign=int(s),
            r_lam2=abs(r_lvl), r_rank=r_rank, r_diff=r_diff,
            gate_agree=agree, gate_jaccard=jac,
            mech_sd=mech_sd, auc_covid=auc,
            r_fwd=r_fwd, r_dd=r_dd,
            s2_final=bt["final"], s2_crash=bt["crash_pnl"], s2_maxdd=bt["max_dd"],
        ))

    R = pd.DataFrame(rows).set_index("quantity")
    R["family"] = ["laplacian ref" if q in REFERENCE else
                   ("L restatement" if q in LAPLACIAN_RESTATEMENTS else "modularity")
                   for q in R.index]
    R.to_csv(os.path.join(HERE, "scores.csv"), float_format="%.6g")

    pd.set_option("display.width", 200)
    fmt = dict(float_format=lambda v: f"{v:+.3f}")

    print(f"S1 (buy & hold) final cumulative log-return: {s1_final:+.4f}")
    print(f"lambda2's own S2 (tau=1.0):                  "
          f"{backtest(lam2, n_invest, s1_ret, inv_dts)['final']:+.4f}\n")

    print("=" * 118)
    print("AGREEMENT WITH lambda2  (sign = stress direction; +1 means high value = stress)")
    print("=" * 118)
    a = R[["family", "sign", "r_lam2", "r_rank", "r_diff", "gate_agree", "gate_jaccard"]]
    print(a.sort_values("r_lam2", ascending=False).to_string(**fmt))

    print("\n" + "=" * 118)
    print("CRASH BEHAVIOUR  (mech_sd = rise into COVID in pre-crash baseline SDs; "
          "auc_covid = separates the 24 crash days)")
    print("=" * 118)
    b = R[["family", "sign", "mech_sd", "auc_covid", "r_fwd", "r_dd"]]
    print(b.sort_values("auc_covid", ascending=False).to_string(**fmt))

    print("\n" + "=" * 118)
    print(f"BACKTEST  (S2 at lambda2's invest rate {n_invest}/{len(df)}; "
          f"S1 = {s1_final:+.4f})")
    print("=" * 118)
    c = R[["family", "sign", "s2_final", "s2_crash", "s2_maxdd", "gate_agree"]]
    print(c.sort_values("s2_final", ascending=False).to_string(**fmt))

    print("\nWrote scores.csv")


if __name__ == "__main__":
    main()
