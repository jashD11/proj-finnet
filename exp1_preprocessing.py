"""
Experiment 1 — Preprocessing effects (Fig 1).

Runs the GMRF Laplacian solver with β=10 on 4 preprocessing variants:
  (a) Raw covariance, market in
  (b) Raw covariance, market out
  (c) Correlation, market in   ← paper's best (scaling implicitly handles market)
  (d) Correlation, market out

Data: S&P 500, 3 sectors (~97 stocks, 2016–2019 log returns).
Saves figures/fig1_preprocessing.png.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))

from solver.admm_graph import learn_laplacian
from utils.preprocessing import build_similarity, remove_market_factor
from utils.graph_viz import laplacian_to_graph
from utils.graph_metrics import check_laplacian, effective_rank
from data.tickers import SP500_SECTORS

# ── Hyperparameters ────────────────────────────────────────────────────────────
BETA      = 10.0   # sparsity regulariser (same β as Fig 2)
TOP_EDGES = 200    # show the top-K edges by weight per panel (equal density)
EDGE_SCALE = 6.0   # edge width multiplier (applied to weight / panel-max)
SEED       = 42
# ──────────────────────────────────────────────────────────────────────────────

SECTOR_COLORS = {
    "Industrials":      "#4C72B0",
    "ConsumerStaples":  "#DD8452",
    "Consumer Staples": "#DD8452",
    "Energy":           "#55A868",
}
DEFAULT_COLOR = "#AAAAAA"

CACHE_PATHS = [
    os.path.join("data", "L_a_raw_cov_mkt_in.npy"),
    os.path.join("data", "L_b_raw_cov_mkt_out.npy"),
    os.path.join("data", "L_c_corr_mkt_in.npy"),
    os.path.join("data", "L_d_corr_mkt_out.npy"),
]
VARIANT_LABELS = [
    "(a) Raw covariance\nMarket in",
    "(b) Raw covariance\nMarket out",
    "(c) Correlation\nMarket in",
    "(d) Correlation\nMarket out",
]


def build_variants(X):
    """Return 4 (S, label) pairs covering the preprocessing space."""
    X_mkt_out = remove_market_factor(X)
    return [
        build_similarity(X,         use_correlation=False),   # (a)
        build_similarity(X_mkt_out, use_correlation=False),   # (b)
        build_similarity(X,         use_correlation=True),    # (c) best
        build_similarity(X_mkt_out, use_correlation=True),    # (d)
    ]


def load_or_compute_all(S_list):
    Ls = []
    for i, (S, cache) in enumerate(zip(S_list, CACHE_PATHS)):
        label = VARIANT_LABELS[i].replace("\n", " ")
        if os.path.exists(cache):
            L = np.load(cache)
            print(f"Loaded {label} from cache.")
        else:
            print(f"Running GMRF solver on variant {label} … (p={S.shape[0]}, β={BETA})")
            L = learn_laplacian(S, beta=BETA, max_iter=2000, tol=1e-6)
            np.save(cache, L)
            print(f"  Done. Cached to {cache}")
        Ls.append(L)
    return Ls


def draw_panel(ax, L, tickers, title, seed=SEED):
    p = L.shape[0]
    node_color_map = {
        t: SECTOR_COLORS.get(SP500_SECTORS.get(t, ""), DEFAULT_COLOR)
        for t in tickers
    }

    # Build full graph, pick top-K edges by weight for visual clarity
    G_full = laplacian_to_graph(L, node_labels=tickers, threshold=0.0)
    all_edges = sorted(G_full.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)
    top_edges = all_edges[:TOP_EDGES]
    w_max = top_edges[0][2]["weight"] if top_edges else 1.0

    G = nx.Graph()
    G.add_nodes_from(tickers)
    for u, v, d in top_edges:
        G.add_edge(u, v, weight=d["weight"])

    pos = nx.spring_layout(G, seed=seed, k=0.5, iterations=200, weight="weight")

    colors = [node_color_map.get(n, DEFAULT_COLOR) for n in G.nodes()]
    edge_data  = list(G.edges(data=True))
    edge_pairs = [(u, v) for u, v, _ in edge_data]
    # Normalise widths by panel max weight so all panels use comparable visual scale
    widths = [d["weight"] / w_max * EDGE_SCALE for _, _, d in edge_data]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=90, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=4, font_color="white")
    if edge_pairs:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=edge_pairs, width=widths, alpha=0.55, edge_color="#555555"
        )

    # Sector-purity of top edges
    n_same = sum(
        1 for u, v, _ in top_edges
        if SP500_SECTORS.get(u, "?") == SP500_SECTORS.get(v, "?")
    )
    chk = check_laplacian(L)
    rk  = effective_rank(L)
    subtitle = (
        f"rank={rk}  max|L_ij|={w_max:.4f}  "
        f"within-sector edges: {n_same}/{len(top_edges)}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=8.5, pad=6)
    ax.axis("off")


def main():
    df = pd.read_csv(
        os.path.join("data", "sp500_3sectors.csv"), index_col=0, parse_dates=True
    )
    returns = df.values
    tickers = list(df.columns)
    p       = len(tickers)
    print(f"Loaded {p} stocks × {returns.shape[0]} trading days")

    S_list = build_variants(returns)
    for i, S in enumerate(S_list):
        diag = S.diagonal()
        print(f"  Variant {chr(ord('a')+i)}: S diag range [{diag.min():.4f}, {diag.max():.4f}], "
              f"off-diag max={S[~np.eye(p, dtype=bool)].max():.4f}")

    Ls = load_or_compute_all(S_list)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\n── Laplacian diagnostics ──")
    for i, L in enumerate(Ls):
        chk = check_laplacian(L)
        rk  = effective_rank(L)
        deg = np.diag(L)
        n_iso = int(np.sum(deg < 1e-6))
        print(f"  {chr(ord('a')+i)}: rank={rk}, isolated={n_iso}, "
              f"P1={chk['P1_L1_eq_0']}, P2={chk['P2_offdiag_leq_0']}, PSD={chk['P3_psd']}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.flatten()

    for i, (L, label) in enumerate(zip(Ls, VARIANT_LABELS)):
        draw_panel(axes[i], L, tickers, label, seed=SEED)

    # Sector legend
    legend_handles = [
        mpatches.Patch(color="#4C72B0", label="Industrials"),
        mpatches.Patch(color="#DD8452", label="Consumer Staples"),
        mpatches.Patch(color="#55A868", label="Energy"),
        mpatches.Patch(color=DEFAULT_COLOR, label="Unknown"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        fontsize=11, framealpha=0.8, bbox_to_anchor=(0.5, 0.01)
    )

    fig.suptitle(
        "Fig 1: Effect of Preprocessing on Learned Graph\n"
        f"S&P 500 ({p} stocks, 3 sectors, 2016–2019)  |  β={BETA}",
        fontsize=14, y=0.99
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig1_preprocessing.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nSaved figures/fig1_preprocessing.png")


if __name__ == "__main__":
    main()
