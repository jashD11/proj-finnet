"""
Experiment 4 — Trading strategy (Fig 4).

Uses the algebraic connectivity λ₂ from Algorithm 2 (exp3) to gate investment:
  S1: buy & hold (equal-weight FAAMUNG, every day)
  S2: invest when λ₂(t) < τ; else hold cash (0 return that day)
      — paper direction: high λ₂ signals stress/high correlation → exit

Signal is causal: λ₂ computed from window ending day t → invest on day t+1.
Cumulative PnL is plotted as cumsum of log-returns.

DIAGNOSTIC NOTE (uncontrolled Laplacian, exact temporal penalty — dev_log §18):
  Backtest truncated at END_DATE = 2020-05-01, the paper's evaluation window.
  λ₂ ∈ [0.66, 1.50] over the period (median 1.10) — τ=1.0 gates 79/201 days.
  mean λ₂ INSIDE COVID crash = 1.085 vs local 60-day pre-crash baseline 0.903
  → λ₂ rises entering the crash (paper mechanism, matches paper Fig 3b shape).
  Global outside-mean is higher only due to the elevated autumn-2019 regime,
  which the paper's series also shows.
  Rule aligned to paper: invest when λ₂ < τ=1.0, cash when λ₂ ≥ τ.
  Honest outcome (to 2020-05-01): S2 dodges the crash (flat ≈ −0.01 while S1
  falls to −0.21) but is in cash from late Feb onward (λ₂ stays ≥ 1.0 through
  April) → S2 = −0.011 vs S1 = +0.078. Crash-avoidance replicates; the paper's
  "S2 beats S1" does not.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))

from solver.algorithm2 import algorithm2
from utils.graph_metrics import algebraic_connectivity

# ── Hyperparameters ────────────────────────────────────────────────────────────
WINDOW = 30
K = 1
ETA = 0.0
DELTA = 100.0
DEGREE_CONTROL = False  # Algorithm 2 uses uncontrolled Laplacians (degree_control is Algorithm 1 only)
# Paper direction: invest when λ₂ < TAU (exit when connectivity is high / stressed).
# Without degree normalisation, λ₂ scale matches the paper's τ=1.0 (uncontrolled Laplacian).
TAU = 1.0
# The paper's evaluation window ends 2020-05-01 (its Fig 4 x-axis); truncate there.
END_DATE = pd.Timestamp("2020-05-01")
LAM2_CACHE = os.path.join("data", "lam2_series.npy")
IDX_CACHE = os.path.join("data", "day_indices.npy")
# ──────────────────────────────────────────────────────────────────────────────


def load_or_compute(returns):
    if os.path.exists(LAM2_CACHE) and os.path.exists(IDX_CACHE):
        lam2 = np.load(LAM2_CACHE)
        day_indices = list(np.load(IDX_CACHE).astype(int))
        print(f"Loaded {len(lam2)} λ₂ values from cache.")
        return lam2, day_indices
    print("Running Algorithm 2 (no cache found) …")
    graphs, day_indices = algorithm2(
        returns, window=WINDOW, k=K, eta=ETA, beta=0.0, delta=DELTA,
        degree_control=DEGREE_CONTROL, verbose=True
    )
    lam2 = np.array([algebraic_connectivity(L) for L in graphs])
    np.save(LAM2_CACHE, lam2)
    np.save(IDX_CACHE, np.array(day_indices))
    return lam2, day_indices


def diagnose_signal(lam2, day_indices, dates, port_ret):
    """
    Print diagnostic: COVID-window λ₂ comparison and forward-return correlations.
    Returns (verdict_str) indicating which direction the data supports.
    """
    end_dates = pd.DatetimeIndex([dates[i] for i in day_indices])
    covid_start = pd.Timestamp("2020-02-19")
    covid_end   = pd.Timestamp("2020-03-23")
    inside  = (end_dates >= covid_start) & (end_dates <= covid_end)
    outside = ~inside

    mean_in  = lam2[inside].mean()
    mean_out = lam2[outside].mean()

    # Local pre-crash baseline: the 60 signal dates immediately before the
    # crash window. The global outside-mean is confounded by the elevated
    # autumn-2019 regime (impeachment episode — also elevated in the paper's
    # Fig 3b), so the mechanism test is the local rise into the crash.
    pre_mask = end_dates < covid_start
    pre_idx = np.where(pre_mask)[0][-60:]
    mean_base = lam2[pre_idx].mean()

    # Forward 5-day portfolio return from each signal date
    fwd5_ret, fwd5_dd, valid_idx = [], [], []
    for i, t in enumerate(day_indices):
        if t + 6 > len(port_ret):
            continue
        fwd = port_ret[t + 1: t + 6]
        fwd5_ret.append(fwd.sum())
        fwd5_dd.append(fwd.min())
        valid_idx.append(i)

    lam2_v   = lam2[valid_idx]
    fwd5_ret = np.array(fwd5_ret)
    fwd5_dd  = np.array(fwd5_dd)
    corr_ret = np.corrcoef(lam2_v, fwd5_ret)[0, 1]
    corr_dd  = np.corrcoef(lam2_v, fwd5_dd)[0, 1]

    print("\n=== Fix 1 Diagnostic: λ₂ signal direction ===")
    print(f"  mean(λ₂ INSIDE  COVID 2020-02-19→2020-03-23) = {mean_in:.4f}  (n={inside.sum()})")
    print(f"  mean(λ₂ OUTSIDE COVID window, global)         = {mean_out:.4f}  (n={outside.sum()})")
    print(f"  mean(λ₂ 60 days PRE-crash, local baseline)    = {mean_base:.4f}")
    print(f"  Corr(λ₂, fwd-5-day portfolio return)          = {corr_ret:+.4f}")
    print(f"  Corr(λ₂, fwd-5-day worst-day return)          = {corr_dd:+.4f}")

    if mean_in > mean_base:
        verdict = ("PAPER DIRECTION (local): λ₂ RISES entering the crash "
                   f"({mean_base:.3f} → {mean_in:.3f}) → invest when λ₂ < τ")
    else:
        verdict = ("OPPOSITE DIRECTION (local): λ₂ FALLS entering the crash "
                   f"({mean_base:.3f} → {mean_in:.3f}) → invest when λ₂ ≥ τ")
    print(f"  VERDICT: {verdict}")
    if mean_in <= mean_out:
        print("  (Global outside-mean exceeds the crash mean only because the")
        print("   autumn-2019 regime is elevated — same shape as paper Fig 3b.)")
    print("=" * 55)
    return verdict


def compute_s2(lam2, s1_rets, invest_high=False, tau=0.50):
    """
    Compute S2 returns.
    invest_high=False → paper direction: invest when λ₂ < tau
    invest_high=True  → current-code direction: invest when λ₂ ≥ tau
    """
    n = len(s1_rets)
    s2 = np.array([
        s1_rets[i] if (lam2[i] >= tau if invest_high else lam2[i] < tau) else 0.0
        for i in range(n)
        if i < len(lam2)
    ])
    return s2


def print_table(lam2, s1_rets, s1_cum):
    """Print both-directions × both-τ performance table."""
    tau_fixed  = 1.0
    tau_median = float(np.median(lam2))
    taus = [(tau_fixed, f"τ=1.0 (paper)"), (tau_median, f"τ=median={tau_median:.4f} (in-sample)")]
    directions = [
        (False, "invest when λ₂ < τ  [PAPER direction]"),
        (True,  "invest when λ₂ ≥ τ  [opposite direction]"),
    ]

    print("\n=== Both-directions × both-τ table ===")
    print(f"  S1 (buy & hold) final cumulative log-return: {s1_cum[-1]:.4f}")
    print()
    header = f"  {'Rule':<40}  {'τ=1.0 (paper)':>18}  {'τ=median (in-sample)':>22}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for invest_high, label in directions:
        row = f"  {label:<40}"
        for tau, _ in taus:
            s2 = compute_s2(lam2, s1_rets, invest_high=invest_high, tau=tau)
            final = np.concatenate([[0.0], np.cumsum(s2)])[-1]
            beats = "✓" if final > s1_cum[-1] else "✗"
            row += f"  {final:>+8.4f} {beats}         "
        print(row)
    print("  ★ figure uses: invest when λ₂ < τ, τ=1.0 (paper direction)")
    print("  (in-sample τ=median is explicitly tuned to this dataset)")
    print()


def main():
    df = pd.read_csv(
        os.path.join("data", "faamung.csv"), index_col=0, parse_dates=True
    )
    returns = df.values  # (T, 7)
    dates_idx = df.index
    T = returns.shape[0]

    lam2, day_indices = load_or_compute(returns)

    # Truncate to the paper's evaluation period: keep signals whose invest
    # day (t+1, causal) falls on or before END_DATE.
    keep = [
        i for i, t in enumerate(day_indices)
        if t + 1 < T and dates_idx[t + 1] <= END_DATE
    ]
    lam2 = lam2[keep]
    day_indices = [day_indices[i] for i in keep]
    print(f"Truncated to paper period: {len(day_indices)} signal days "
          f"(last invest day {dates_idx[day_indices[-1] + 1].date()}).")

    # Equal-weight portfolio daily log-return
    port_ret = returns.mean(axis=1)  # (T,)

    # Diagnostic: confirm signal direction before applying rule
    diagnose_signal(lam2, day_indices, dates_idx, port_ret)

    # Align: signal on day t (end of window) → invest on day t+1 (causal)
    invest_dates = []
    s1_rets = []

    for i, t in enumerate(day_indices):
        next_t = t + 1
        if next_t >= T:
            break
        invest_dates.append(dates_idx[next_t])
        s1_rets.append(port_ret[next_t])

    s1_rets = np.array(s1_rets)
    invest_dates = pd.DatetimeIndex(invest_dates)

    # S2: paper direction — invest when λ₂ < TAU (exit when high = stressed market)
    s2_rets = compute_s2(lam2, s1_rets, invest_high=False, tau=TAU)

    # Cumulative log-return (starts at 0)
    s1_cum = np.concatenate([[0.0], np.cumsum(s1_rets)])
    s2_cum = np.concatenate([[0.0], np.cumsum(s2_rets)])
    plot_dates = pd.DatetimeIndex([dates_idx[day_indices[0]]] + list(invest_dates))

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"\nλ₂  min={lam2.min():.4f}  median={np.median(lam2):.4f}  max={lam2.max():.4f}")
    invest_days = int((lam2[: len(s1_rets)] < TAU).sum())
    total_days = len(s1_rets)
    print(f"S2 invests {invest_days}/{total_days} days ({100*invest_days/total_days:.1f}%)")
    print(f"S1 final cumulative log-return: {s1_cum[-1]:.4f}")
    print(f"S2 final cumulative log-return: {s2_cum[-1]:.4f}")
    print(f"S2 beats S1: {s2_cum[-1] > s1_cum[-1]}")

    # Print the both-directions × both-τ table
    print_table(lam2, s1_rets, s1_cum)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    # S2 coincides exactly with S1 when τ never gates, so draw S1 solid and
    # S2 dashed on top — the red shows through the dash gaps.
    ax.plot(plot_dates, s1_cum, color="#d62728", linewidth=2.4,
            label="S1: Buy & Hold (equal-weight FAAMUNG)")
    s2_label = rf"S2: Invest when $\lambda_2 < \tau={TAU:.2f}$ (paper direction: cash when stressed)"
    if np.array_equal(s1_cum, s2_cum):
        s2_label += " — coincides with S1 ($\\tau$ never gates)"
    ax.plot(plot_dates, s2_cum, color="#1f77b4", linewidth=1.6,
            linestyle=(0, (4, 3)), label=s2_label)

    # Shade COVID crash
    ax.axvspan(
        pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"),
        alpha=0.12, color="red", label="COVID crash (approx.)"
    )
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative log-return", fontsize=11)
    ax.set_title(
        f"Trading strategy: S1 (buy & hold) vs S2 (connectivity-gated, $\\lambda_2 < \\tau={TAU:.2f}$)"
        " — Jun 2019 to May 2020 (paper period)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Annotate final values
    ax.annotate(
        f"S1: {s1_cum[-1]:.3f}",
        xy=(plot_dates[-1], s1_cum[-1]),
        xytext=(-55, 8),
        textcoords="offset points",
        fontsize=9,
        color="#d62728",
    )
    ax.annotate(
        f"S2: {s2_cum[-1]:.3f}",
        xy=(plot_dates[-1], s2_cum[-1]),
        xytext=(-55, -14),
        textcoords="offset points",
        fontsize=9,
        color="#1f77b4",
    )

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig4_trading.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nSaved figures/fig4_trading.png")


if __name__ == "__main__":
    main()
