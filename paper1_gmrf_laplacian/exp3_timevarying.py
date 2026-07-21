"""
Experiment 3 — Time-varying graph estimation (Fig 3).

Runs Algorithm 2 (rolling 30-day causal windows) on FAAMUNG log-returns,
extracts the algebraic connectivity λ₂(L_t) for each window, and plots:
  - Top panel: λ₂ time series with τ threshold line and COVID-crash shading
  - Bottom panels: three network snapshots (high / low / high connectivity)
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
from utils.graph_viz import plot_laplacian_graph

# ── Hyperparameters ────────────────────────────────────────────────────────────
WINDOW = 30
K = 1         # k=1: connected graph so λ₂ is a meaningful connectivity signal
ETA = 0.0
DELTA = 100.0
DEGREE_CONTROL = False  # Algorithm 2 uses uncontrolled Laplacians (degree_control is Algorithm 1 only)
TAU = 1.0     # paper's threshold for uncontrolled Laplacian (CLAUDE.md §2; dev_log §10)
FAAMUNG = ["META", "AAPL", "AMZN", "MSFT", "UBER", "NFLX", "GOOGL"]
LAM2_CACHE = os.path.join("data", "lam2_series.npy")
IDX_CACHE = os.path.join("data", "day_indices.npy")
# ──────────────────────────────────────────────────────────────────────────────


def compute_or_load(returns):
    if os.path.exists(LAM2_CACHE) and os.path.exists(IDX_CACHE):
        lam2 = np.load(LAM2_CACHE)
        day_indices = list(np.load(IDX_CACHE).astype(int))
        print(f"Loaded {len(lam2)} λ₂ values from cache.")
        return lam2, day_indices

    print(f"Running Algorithm 2 on {returns.shape[0]} days × {returns.shape[1]} stocks …")
    graphs, day_indices = algorithm2(
        returns,
        window=WINDOW,
        k=K,
        eta=ETA,
        beta=0.0,
        delta=DELTA,
        degree_control=DEGREE_CONTROL,
        verbose=True,
    )
    print(f"Computed {len(graphs)} rolling graphs.")

    lam2 = np.array([algebraic_connectivity(L) for L in graphs])
    np.save(LAM2_CACHE, lam2)
    np.save(IDX_CACHE, np.array(day_indices))
    print("Cached λ₂ series and graph indices.")

    # Keep graphs in memory so caller can make snapshots without re-running
    compute_or_load._graphs = graphs
    return lam2, day_indices


def get_graphs(returns, day_indices):
    """Return graphs — re-run if not already in memory from compute_or_load."""
    if hasattr(compute_or_load, "_graphs"):
        return compute_or_load._graphs
    print("Re-running Algorithm 2 for network snapshots …")
    graphs, _ = algorithm2(
        returns, window=WINDOW, k=K, eta=ETA, beta=0.0, delta=DELTA,
        degree_control=DEGREE_CONTROL, verbose=False,
    )
    return graphs


def main():
    df = pd.read_csv(
        os.path.join("data", "faamung.csv"), index_col=0, parse_dates=True
    )
    returns = df.values  # (T, 7)
    dates_idx = df.index

    lam2, day_indices = compute_or_load(returns)
    graphs = get_graphs(returns, day_indices)
    graph_dates = dates_idx[day_indices]

    print(f"\nλ₂ stats: min={lam2.min():.4f}  median={np.median(lam2):.4f}  max={lam2.max():.4f}")
    below_tau = (lam2 < TAU).sum()
    print(f"Windows with λ₂ < τ={TAU}: {below_tau}/{len(lam2)} ({100*below_tau/len(lam2):.1f}%)")

    # ── Choose 3 snapshot indices by target date ─────────────────────────────
    # Snapshots are selected by date, not by global extremum, so that the
    # "COVID crash" panel is actually from the crash period.
    snapshot_targets = [
        pd.Timestamp("2019-11-01"),   # autumn-2019 regime (elevated λ₂, cf. paper Fig 3b)
        pd.Timestamp("2020-03-16"),   # COVID crash
        pd.Timestamp("2020-05-22"),   # recovery
    ]
    snap_labels = ["Pre-COVID\n(autumn regime)", "COVID crash", "Recovery"]
    snap_indices = [
        int(np.argmin(np.abs(graph_dates - t))) for t in snapshot_targets
    ]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── Top: full-width λ₂ time series ────────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(graph_dates, lam2, color="#2166ac", linewidth=1.2,
                label=r"$\lambda_2(L_t)$")
    ax_top.axhline(TAU, color="red", linestyle="--", linewidth=1.0,
                   label=rf"$\tau = {TAU}$")

    # Mark the 3 snapshot times
    for si, sl in zip(snap_indices, snap_labels):
        ax_top.axvline(graph_dates[si], color="grey", linewidth=0.8, linestyle=":")

    # Shade COVID crash (approximate peak→trough)
    ax_top.axvspan(
        pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"),
        alpha=0.15, color="red", label="COVID crash (approx.)"
    )

    ax_top.set_ylabel(r"Algebraic connectivity $\lambda_2$", fontsize=11)
    ax_top.set_title(
        "Time-varying FAAMUNG graph: algebraic connectivity (window = 30 days)",
        fontsize=12,
    )
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_top.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_top.get_xticklabels(), rotation=30, ha="right")

    # ── Bottom: 3 network snapshots ───────────────────────────────────────────
    for col, (si, sl) in enumerate(zip(snap_indices, snap_labels)):
        ax = fig.add_subplot(gs[1, col])
        d = graph_dates[si]
        plot_laplacian_graph(
            graphs[si],
            node_labels=FAAMUNG,
            ax=ax,
            title=f"{sl}\n{d.strftime('%Y-%m-%d')}  (λ₂ = {lam2[si]:.3f})",
            layout="spring",
            seed=42,
        )

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig3_timevarying.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nSaved figures/fig3_timevarying.png")


if __name__ == "__main__":
    main()
