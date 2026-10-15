"""
Figures for the modularity-indicator exploration.

Palette: slots 1-3 of the dataviz reference categorical palette
(blue/orange/aqua), which that reference documents as validated all-pairs in
both modes. Light-mode research PNGs; one measure per axis, never dual-axis.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import roc_auc_score

from evaluate import HERE, TAU, COVID_START, COVID_END, load, backtest
from robustness import trailing_z

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b2"
SURFACE = "#fcfcfb"
FIG = os.path.join(HERE, "figures")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10.5, "legend.frameon": False,
    "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
})


def shade_covid(ax):
    ax.axvspan(COVID_START, COVID_END, color=MUTED, alpha=0.28, lw=0,
               zorder=0)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load()
    dts = pd.DatetimeIndex(df.index)
    lam2 = df["lam2"].values
    n_invest = int((lam2 < TAU).sum())
    crash = np.asarray((dts >= COVID_START) & (dts <= COVID_END))

    # ── Fig A: do modularity quantities track lambda2? (z-scored, one axis) ──
    series = [("$\\lambda_2(L)$  — Paper 1's indicator", "lam2", BLUE),
              ("$\\bar{d}-\\mu_1(B)$", "dbar_minus_mu1", ORANGE),
              ("$\\mu_N(B)$", "muN", AQUA)]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    shade_covid(ax)
    ax.grid(axis="y", zorder=0)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
    for label, col, c in series:
        x = df[col].values.astype(float)
        s = 1.0 if np.corrcoef(x, lam2)[0, 1] >= 0 else -1.0
        z = trailing_z(s * x)
        ax.plot(dts, z, color=c, lw=2.0, label=label, zorder=3)
    ax.set_ylabel("stress signal (60-day trailing z-score)")
    ax.set_title("All three rise into the COVID crash — the modularity quantities "
                 "carry $\\lambda_2$'s signal", loc="left")
    ax.annotate("COVID crash\n19 Feb – 23 Mar 2020",
                xy=(COVID_START + (COVID_END - COVID_START) / 2, ax.get_ylim()[0]),
                xytext=(0, 6), textcoords="offset points", ha="center",
                color=INK2, fontsize=8)
    ax.legend(loc="upper left", ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "A_timeseries.png"), dpi=160)
    plt.close(fig)

    # ── Fig B: the equivalence scatter + gate quadrants ─────────────────────
    y = df["dbar_minus_mu1"].values
    thr = np.sort(y)[n_invest - 1]
    lam_gate, y_gate = lam2 < TAU, y <= thr
    agree = lam_gate == y_gate
    r = np.corrcoef(lam2, y)[0, 1]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.grid(zorder=0)
    ax.axvline(TAU, color=MUTED, lw=1.4, zorder=1)
    ax.axhline(thr, color=MUTED, lw=1.4, zorder=1)
    ax.scatter(lam2[agree], y[agree], s=26, color=BLUE, alpha=0.85,
               lw=0.8, edgecolor=SURFACE, zorder=3, label="gates agree")
    ax.scatter(lam2[~agree], y[~agree], s=30, color=ORANGE, alpha=0.95,
               lw=0.8, edgecolor=SURFACE, zorder=4, label="gates disagree")
    ax.set_xlabel("$\\lambda_2(L)$   (Paper 1; invest to the left of $\\tau=1.0$)")
    ax.set_ylabel("$\\bar{d}-\\mu_1(B)$   (invest below the matched threshold)")
    ax.set_title(f"The modularity gate reproduces {agree.mean():.0%} of "
                 f"$\\lambda_2$'s daily calls\nPearson r = {r:+.3f}, "
                 f"n = {len(lam2)} days, invest rate matched at "
                 f"{n_invest}/{len(lam2)}", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "B_gate_agreement.png"), dpi=160)
    plt.close(fig)

    # ── Fig C: ranked agreement, all quantities ─────────────────────────────
    sc = pd.read_csv(os.path.join(HERE, "scores.csv"), index_col=0)
    sc = sc.drop(index=["lam2"]).sort_values("r_lam2")
    cmap = {"modularity": BLUE, "laplacian ref": MUTED, "L restatement": ORANGE}
    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    ax.grid(axis="x", zorder=0)
    cols = [cmap[f] for f in sc["family"]]
    ax.hlines(range(len(sc)), 0, sc["r_lam2"], color=cols, lw=2.0, zorder=3)
    ax.scatter(sc["r_lam2"], range(len(sc)), s=46, color=cols, zorder=4,
               lw=0.8, edgecolor=SURFACE)
    ax.set_yticks(range(len(sc)))
    ax.set_yticklabels(sc.index, fontsize=8.5)
    ax.set_xlabel("|Pearson r| with $\\lambda_2(L)$   (201 days)")
    ax.set_xlim(0, 1)
    handles = [plt.Line2D([], [], color=v, lw=2.6, label=k) for k, v in cmap.items()]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("How closely each quantity tracks the Fiedler value", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "C_agreement_ranking.png"), dpi=160)
    plt.close(fig)

    # ── Fig D: the honesty panel — sliding-event null ───────────────────────
    picks = [("lam2", "$\\lambda_2(L)$", BLUE), ("muN", "$\\mu_N(B)$", AQUA)]
    width = int(crash.sum())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, (col, label, c) in zip(axes, picks):
        x = df[col].values.astype(float)
        s = 1.0 if np.corrcoef(x, lam2)[0, 1] >= 0 else -1.0
        z = trailing_z(s * x)
        ok = np.isfinite(z)
        v, n = z[ok], int(ok.sum())
        obs = roc_auc_score(crash[ok].astype(int), v)
        null = []
        for st in range(n - width + 1):
            lab = np.zeros(n, dtype=int)
            lab[st:st + width] = 1
            null.append(roc_auc_score(lab, v))
        null = np.array(null)
        ax.grid(axis="y", zorder=0)
        ax.hist(null, bins=24, color=MUTED, alpha=0.75, zorder=2,
                label=f"any {width}-day window")
        ax.axvline(obs, color=c, lw=2.4, zorder=4,
                   label=f"actual COVID window ({obs:.3f})")
        ax.set_xlabel("AUC")
        ax.set_title(f"{label} — COVID sits at the "
                     f"{(null < obs).mean():.0%} percentile", loc="left")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("number of 24-day windows")
    fig.suptitle("Neither indicator separates COVID from an arbitrary 24-day "
                 "stretch — one crash is not enough to certify either",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "D_sliding_null.png"), dpi=160)
    plt.close(fig)

    print("Wrote:", ", ".join(sorted(os.listdir(FIG))))


if __name__ == "__main__":
    main()
