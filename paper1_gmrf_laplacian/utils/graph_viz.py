"""
NetworkX-based Laplacian graph visualisation.
Nodes coloured by sector; edge width ∝ |L_ij|.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# Sector colour palette (matches paper's sector groupings)
SECTOR_COLORS = {
    "Industrials": "#4C72B0",
    "Consumer Staples": "#DD8452",
    "Energy": "#55A868",
    # FAAMUNG — treat as a single group
    "Tech": "#C44E52",
}
DEFAULT_COLOR = "#8C8C8C"


def laplacian_to_graph(L, node_labels=None, threshold=0.0):
    """
    Build a networkx Graph from Laplacian L.
    Edges added for off-diagonal entries with |L_ij| > threshold.
    """
    p = L.shape[0]
    if node_labels is None:
        node_labels = [str(i) for i in range(p)]

    G = nx.Graph()
    G.add_nodes_from(node_labels)

    for i in range(p):
        for j in range(i + 1, p):
            w = -L[i, j]  # weight = |L_ij| (L_ij <= 0 for valid Laplacian)
            if w > threshold:
                G.add_edge(node_labels[i], node_labels[j], weight=float(w))
    return G


def plot_laplacian_graph(
    L,
    node_labels=None,
    node_sectors=None,
    ax=None,
    title="",
    edge_scale=5.0,
    threshold=0.01,
    layout="spring",
    seed=42,
):
    """
    Draw the graph encoded by Laplacian L.

    Parameters
    ----------
    L            : (p, p) Laplacian
    node_labels  : list of p ticker strings
    node_sectors : dict {label: sector_name} for colouring
    ax           : matplotlib Axes (creates one if None)
    title        : subplot title
    edge_scale   : multiplier for edge width
    threshold    : skip edges with |L_ij| below this
    layout       : 'spring' | 'circular' | 'kamada'
    seed         : random seed for layout reproducibility
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    G = laplacian_to_graph(L, node_labels=node_labels, threshold=threshold)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    # Node colours
    if node_sectors is not None:
        node_colors = [
            SECTOR_COLORS.get(node_sectors.get(n, ""), DEFAULT_COLOR)
            for n in G.nodes()
        ]
    else:
        node_colors = [DEFAULT_COLOR] * len(G.nodes())

    # Edge widths
    edges = G.edges(data=True)
    edge_list = list(edges)
    edge_widths = [d["weight"] * edge_scale for _, _, d in edge_list]
    edge_tuples = [(u, v) for u, v, _ in edge_list]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=200)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
    if edge_tuples:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=edge_tuples, width=edge_widths, alpha=0.7
        )

    ax.set_title(title, fontsize=9)
    ax.axis("off")


def sector_legend(ax, sectors=None):
    """Add a colour legend for sectors to the given Axes."""
    if sectors is None:
        sectors = list(SECTOR_COLORS.keys())
    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=SECTOR_COLORS.get(s, DEFAULT_COLOR),
            markersize=8,
            label=s,
        )
        for s in sectors
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)
