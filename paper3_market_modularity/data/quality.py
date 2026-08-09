"""
Return-series quality gates (audit A2/A3).

The paper's completeness filter asks only "does this ticker have a price on
every day". It does not ask whether those prices are *informative*. Two defects
survive it, and both would corrupt results in ways that look like findings:

  1. FROZEN PRICES. Several tickers carry runs of a literally unchanged price -
     RRC has 743 consecutive days of it, SU 497. A stock whose price does not
     move for an entire Delta_t window has zero variance, so its correlation
     with everything is undefined. Whatever convention we adopt, it becomes a
     guaranteed isolated node - which would manufacture exactly the isolated-node
     spike that Fig. 3B treats as the signature of a crisis.

  2. UNADJUSTED CORPORATE ACTIONS. A split, reorganisation or share exchange
     that the adjustment factors missed appears as one enormous log return, and
     poisons the Delta_t consecutive windows that contain it.

Both are ticker-level defects: one bad day makes the series untrustworthy for
30 networks, so the whole ticker goes rather than the single observation.

Deliberately NOT gated: large single-day moves that are real. AIG fell 60.8 % on
the day Lehman filed; State Street fell 59.0 % on 2009-01-20. Those are the
signal, not noise, and a threshold that removed them would be removing the
crisis from a paper about crises.
"""

import numpy as np
import pandas as pd

# A stock whose price is unchanged for a whole window has an undefined
# correlation in that window. Tying the gate to delta_t rather than to a round
# number is what makes it principled: at delta_t = 30, requiring the longest
# frozen run to be shorter than 30 guarantees every window is well defined.
# (Passed in explicitly; there is no default, on purpose.)

# A one-day log return of +/-1.0 means the price e-folded or fell by 63 %. In
# this panel the empirical gap is wide and unambiguous: the largest artifact is
# 1.179 and the largest genuine move is 0.943 (WMB during the July 2002
# liquidity crisis), so the boundary is not a tuned parameter.
EXTREME_RETURN = 1.0

# A move that the very next day almost exactly undoes is a bad print, not a
# price. 0.7 is generous; the one case in this panel reverses at 1.05.
REVERSAL_FRACTION = 0.7
LARGE_RETURN = 0.5


def longest_zero_run(row: np.ndarray) -> int:
    """Longest run of exactly-zero returns, i.e. of an unchanged price."""
    best = cur = 0
    for v in row == 0.0:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def quality_gate(returns: np.ndarray, tickers, delta_t: int,
                 extreme: float = EXTREME_RETURN,
                 reversal: float = REVERSAL_FRACTION,
                 verbose: bool = True):
    """Return (keep_mask, report) for a (N, T-1) log-return matrix.

    Three gates, each tied to a stated failure mode rather than to a
    hand-tuned cutoff:

        G1  longest frozen-price run >= delta_t   -> undefined correlation
        G2  a large move the next day undoes      -> bad print
        G3  |Y| >= extreme                        -> corporate-action artifact
    """
    n = returns.shape[0]
    tickers = np.asarray(tickers)

    runs = np.array([longest_zero_run(returns[i]) for i in range(n)])
    g1 = runs >= delta_t

    g2 = np.zeros(n, dtype=bool)
    g3 = np.zeros(n, dtype=bool)
    ti, di = np.where(np.abs(returns) > LARGE_RETURN)
    for i, d in zip(ti, di):
        y = returns[i, d]
        nxt = returns[i, d + 1] if d + 1 < returns.shape[1] else 0.0
        if -nxt / y > reversal:
            g2[i] = True
        if abs(y) >= extreme:
            g3[i] = True

    drop = g1 | g2 | g3
    keep = ~drop

    report = {
        "n_before": int(n),
        "n_after": int(keep.sum()),
        "delta_t": int(delta_t),
        "extreme_return": extreme,
        "reversal_fraction": reversal,
        "g1_frozen_run": sorted(tickers[g1].tolist()),
        "g2_reversing_spike": sorted(tickers[g2].tolist()),
        "g3_extreme_return": sorted(tickers[g3].tolist()),
        "n_dropped": int(drop.sum()),
        "max_frozen_run_before": int(runs.max()),
        "max_frozen_run_after": int(runs[keep].max()) if keep.any() else 0,
        "zero_return_share_before": float((returns == 0).mean()),
        "zero_return_share_after": float((returns[keep] == 0).mean()),
    }
    if verbose:
        print(f"[quality] G1 frozen run >= {delta_t}d : {g1.sum():>3}"
              f"  {sorted(tickers[g1].tolist())}")
        print(f"[quality] G2 reversing spike     : {g2.sum():>3}"
              f"  {sorted(tickers[g2].tolist())}")
        print(f"[quality] G3 |Y| >= {extreme}          : {g3.sum():>3}"
              f"  {sorted(tickers[g3].tolist())}")
        print(f"[quality] N {n} -> {keep.sum()}"
              f"   longest frozen run {runs.max()} -> {report['max_frozen_run_after']}"
              f"   zero returns {report['zero_return_share_before']:.2%}"
              f" -> {report['zero_return_share_after']:.2%}")
    return keep, report


def surviving_large_returns(returns: np.ndarray, tickers, dates,
                            threshold: float = LARGE_RETURN) -> pd.DataFrame:
    """Every |Y| > threshold left after the gate, with its trailing volatility.

    These are retained on purpose. The table is printed and written to disk so a
    reader can check them individually rather than take "no unexplained outliers"
    on trust.
    """
    ti, di = np.where(np.abs(returns) > threshold)
    rows = []
    for i, d in zip(ti, di):
        lo = max(0, d - 20)
        trail = float(np.std(np.delete(returns[i, lo:d + 1], -1))) if d - lo > 2 else np.nan
        rows.append({
            "ticker": str(np.asarray(tickers)[i]),
            "date": pd.Timestamp(dates[d]),
            "log_return": float(returns[i, d]),
            "pct_move": float(np.expm1(returns[i, d])),
            "trailing_vol_20d": trail,
            "sigmas": float(abs(returns[i, d]) / trail) if trail else np.nan,
        })
    return pd.DataFrame(rows).sort_values("log_return").reset_index(drop=True)
