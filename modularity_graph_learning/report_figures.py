"""
Figures for reports/2026-10-27_parts1_2_report.tex (Parts I and II of the proposal).

Every learned graph drawn here is read from the committed caches (Paper 1's
alg1_L.npy, the degree-sweep caches, part2_cache/); the toy graphs are built
inline. Nothing is re-learned except the Problem 1 graphs of the bound figure
(seconds), exactly as report_numbers.py section L does.

Usage:  python report_figures.py            (writes reports/figures/parts1_2/*.pdf)
        python report_figures.py fig06      (only figures whose name contains fig06)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import null_space
from scipy.sparse.csgraph import connected_components

from common import ROOT, P1, K, adjacency, laplacian, load_sp, modularity_matrix

OUT = os.path.join(ROOT, "reports", "figures", "parts1_2")

SECTOR_COL = {"Industrials": "#1F4E79", "ConsumerStaples": "#2F8F5B", "Energy": "#C55A11"}
SECTOR_NAME = {"Industrials": "Industrials", "ConsumerStaples": "Consumer Staples",
               "Energy": "Energy"}
SYN_COL = ["#1F4E79", "#2F8F5B", "#C55A11"]
RED, GREY, BLUE = "#B3261E", "#8A939C", "#1F4E79"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "serif",
    "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


def eig(M):
    return np.linalg.eigvalsh((M + M.T) / 2.0)


# ---------------------------------------------------------------- drawing
def layout(W, seed=1, k=None, order_key=None, groups=None):
    """Spring layout per connected component (edges above 1e-4), components scaled by
    sqrt(size) and placed side by side, largest first (or by order_key(component))."""
    Wt = np.where(W > 1e-4, W, 0.0)
    if groups is None:
        n, lab = connected_components(Wt > 0)
    else:
        lab = np.asarray(groups)
        n = lab.max() + 1
    comps = [np.where(lab == c)[0] for c in range(n)]
    comps.sort(key=(order_key or (lambda c: -len(c))))
    pos, x0 = {}, 0.0
    for c in comps:
        r = np.sqrt(len(c))
        if len(c) == 1:
            pos[c[0]] = np.array([x0 + 0.5, 0.0])
            x0 += 1.5
            continue
        G = nx.from_numpy_array(Wt[np.ix_(c, c)])
        P = nx.spring_layout(G, weight="weight", seed=seed, k=k, iterations=300)
        A = np.array([P[i] for i in range(len(c))])
        A = A - A.mean(axis=0)
        A = A / (np.abs(A).max() + 1e-12) * r
        for t, i in enumerate(c):
            pos[i] = A[t] + np.array([x0 + r, 0.0])
        x0 += 2 * r + 1.2
    return pos


def draw_graph(ax, W, pos, colors, top=None, sizes=None, shapes=None, outline=None,
               edge_col="#9AA3AD", wscale=2.2, labels=None):
    """Edges: the `top` strongest (all if None), width by weight. Nodes coloured."""
    p = W.shape[0]
    iu = np.triu_indices(p, 1)
    w = W[iu]
    order = np.argsort(w)[::-1]
    keep = order[: (top or int(np.sum(w > 1e-4)))]
    keep = keep[w[keep] > 1e-4]
    wmax = w[keep].max() if len(keep) else 1.0
    for e in keep[::-1]:
        i, j = iu[0][e], iu[1][e]
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=edge_col,
                lw=0.25 + wscale * w[e] / wmax, alpha=0.35 + 0.5 * w[e] / wmax,
                zorder=1, solid_capstyle="round")
    sizes = np.full(p, 26.0) if sizes is None else sizes
    shapes = np.array(["o"] * p) if shapes is None else np.asarray(shapes)
    for m in np.unique(shapes):
        idx = np.where(shapes == m)[0]
        ec = ["k" if (outline is not None and outline[i]) else "white" for i in idx]
        lw = [1.4 if (outline is not None and outline[i]) else 0.5 for i in idx]
        ax.scatter([pos[i][0] for i in idx], [pos[i][1] for i in idx], s=sizes[idx],
                   c=[colors[i] for i in idx], marker=m, edgecolors=ec, linewidths=lw,
                   zorder=3)
    if labels:
        for i, t in labels.items():
            ax.annotate(t, pos[i], xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_axis_off()
    ax.set_aspect("equal")


def sector_legend(ax, loc="lower center", ncol=3, **kw):
    h = [Line2D([], [], marker="o", ls="", color=c, mec="white", ms=7, label=SECTOR_NAME[s])
         for s, c in SECTOR_COL.items()]
    ax.legend(handles=h, loc=loc, ncol=ncol, frameon=False, **kw)


# ---------------------------------------------------------------- toy graphs
def two_triangles(bridge=0.2):
    W = np.zeros((6, 6))
    for a, b in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        W[a, b] = W[b, a] = 1.0
    W[2, 3] = W[3, 2] = bridge
    return W


TOY_POS = {0: (-1.6, 0.6), 1: (-1.6, -0.6), 2: (-0.6, 0.0), 3: (0.6, 0.0), 4: (1.6, 0.6),
           5: (1.6, -0.6)}


def draw_toy(ax, W, values, cmap="coolwarm", vmin=-1, vmax=1, pos=TOY_POS, text=None,
             edge_labels=True, size=230):
    p = W.shape[0]
    for i in range(p):
        for j in range(i + 1, p):
            if W[i, j] > 0:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="#6B7580",
                        lw=0.6 + 2.2 * W[i, j] / W.max(), zorder=1)
                if edge_labels:
                    mx, my = (pos[i][0] + pos[j][0]) / 2, (pos[i][1] + pos[j][1]) / 2
                    ax.text(mx, my, f"{W[i, j]:g}", fontsize=7, ha="center", va="center",
                            color="#444", bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                                    ec="none"), zorder=2)
    sc = ax.scatter([pos[i][0] for i in range(p)], [pos[i][1] for i in range(p)], s=size,
                    c=values, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="k",
                    linewidths=0.6, zorder=3)
    if text is not None:
        for i in range(p):
            ax.text(pos[i][0], pos[i][1], text[i], ha="center", va="center", fontsize=7,
                    zorder=4)
    xs = np.array([pos[i][0] for i in range(p)]); ys = np.array([pos[i][1] for i in range(p)])
    ax.set_xlim(xs.min() - 0.45, xs.max() + 0.45)
    ax.set_ylim(ys.min() - 0.45, ys.max() + 0.45)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return sc


def fig02_laplacian():
    W = two_triangles(0.2)
    L = laplacian(W)
    lam, V = np.linalg.eigh(L)
    smooth = np.array([1, 1, 1, -1, -1, -1], float)
    rough = np.array([1, -1, 1, -1, 1, -1], float)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4),
                             gridspec_kw=dict(width_ratios=[1, 1, 0.95], wspace=0.25))
    for ax, x, t in [(axes[0], smooth, "(a) signal that follows the groups"),
                     (axes[1], rough, "(b) signal that ignores them")]:
        draw_toy(ax, W, x, text=[f"{v:+g}" for v in x])
        ax.set_title(f"{t}\n" + r"$\mathbf{x}^\top\mathbf{L}\mathbf{x}=$"
                     + f"{x @ L @ x:.1f}", fontsize=8.5)
    ax = axes[2]
    ax.bar(np.arange(1, 7), lam, color=[RED, RED] + [GREY] * 4, width=0.6)
    for i, v in enumerate(lam):
        ax.text(i + 1, v + 0.08, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"eigenvalue $\lambda_i(\mathbf{L})$")
    ax.set_title("(c) eigenvalues: one small $\\lambda_2$,\none weak cut", fontsize=8.5)
    save(fig, "fig02_laplacian")


def fig03_components():
    def triangles(eps):
        W = np.zeros((9, 9))
        for g in range(3):
            a, b, c = 3 * g, 3 * g + 1, 3 * g + 2
            for u, v in [(a, b), (a, c), (b, c)]:
                W[u, v] = W[v, u] = 1.0
        for u, v in [(2, 3), (5, 6), (8, 0)]:
            W[u, v] = W[v, u] = eps
        return W
    ang = np.array([90, 210, 330]) * np.pi / 180
    pos = {}
    for g in range(3):
        cx, cy = 1.6 * np.cos(ang[g]), 1.6 * np.sin(ang[g])
        for t in range(3):
            a2 = ang[g] + np.pi + (t - 1) * 2 * np.pi / 3
            pos[3 * g + t] = (cx + 0.55 * np.cos(a2), cy + 0.55 * np.sin(a2))
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.0),
                             gridspec_kw=dict(width_ratios=[1, 1.1, 1, 1.1]))
    for (ax_g, ax_s), eps, title in [((axes[0], axes[1]), 0.0, "(a) three components"),
                                     ((axes[2], axes[3]), 0.15, "(b) weakly joined")]:
        W = triangles(eps)
        col = np.repeat([0.0, 0.5, 1.0], 3)
        draw_toy(ax_g, W, col, cmap=matplotlib.colors.ListedColormap(SYN_COL),
                 vmin=0, vmax=1, pos=pos, edge_labels=False, size=70)
        for u, v in [(2, 3), (5, 6), (8, 0)]:
            if False:
                ax_g.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#6B7580",
                          lw=0.7, ls=(0, (2, 1.5)), zorder=1)
        ax_g.set_title(title, fontsize=8.5)
        lam = eig(laplacian(W))
        ax_s.bar(np.arange(1, 10), lam, color=[RED] * 3 + [GREY] * 6, width=0.6)
        ax_s.scatter(np.arange(1, 4), lam[:3], color=RED, s=12, zorder=3, clip_on=False)
        ax_s.set_xticks(range(1, 10))
        ax_s.set_ylim(0, 4.2)
        ax_s.set_xlabel(r"$i$")
        ax_s.set_ylabel(r"$\lambda_i(\mathbf{L})$")
        ax_s.set_title("three zeros" if eps == 0 else
                       f"one zero, then {lam[1]:.2f}, {lam[2]:.2f}", fontsize=8.5)
    save(fig, "fig03_components")


def fig04_modularity():
    W = two_triangles(0.2)
    W[0, 4] = W[4, 0] = 0.6          # an extra cross edge so B has negative entries to show
    d = W.sum(axis=1)
    E = np.outer(d, d) / d.sum()
    B = W - E
    mu, U = np.linalg.eigh(B)
    u = U[:, -1] * np.sign(U[0, -1])
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.05),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
    vmax = np.abs(np.concatenate([W.ravel(), E.ravel(), B.ravel()])).max()
    for ax, M, t in [(axes[0], W, r"(a) observed $W_{ij}$"),
                     (axes[1], E, r"(b) expected $d_i d_j/2m$"),
                     (axes[2], B, r"(c) $B_{ij}=W_{ij}-d_id_j/2m$")]:
        ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(6), [str(i + 1) for i in range(6)], fontsize=7)
        ax.set_yticks(range(6), [str(i + 1) for i in range(6)], fontsize=7)
        ax.set_title(t, fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(2.5, color="k", lw=0.5)
        ax.axvline(2.5, color="k", lw=0.5)
    pos = {i: (x * 0.8, y) for i, (x, y) in TOY_POS.items()}
    draw_toy(axes[3], W, u / np.abs(u).max(), pos=pos, text=[str(i + 1) for i in range(6)],
             edge_labels=False)
    axes[3].set_title("(d) top eigenvector of $\\mathbf{B}$\n"
                      f"($\\mu_1={mu[-1]:.2f}>0$: worth splitting)", fontsize=8.5)
    save(fig, "fig04_modularity")


# ---------------------------------------------------------------- data
def fig05_data(X, S, sector):
    order = np.argsort([list(SECTOR_COL).index(s) for s in sector], kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.45))
    ax = axes[0]
    im = ax.imshow(S[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    cuts = np.cumsum([np.sum(sector == s) for s in SECTOR_COL])[:-1]
    for c in cuts:
        ax.axhline(c - 0.5, color="k", lw=0.6)
        ax.axvline(c - 0.5, color="k", lw=0.6)
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(sector)]])
    ax.set_xticks((starts + ends) / 2 - 0.5, ["Industrials\n(52)", "Staples\n(28)", "Energy\n(17)"],
                  fontsize=7)
    ax.set_yticks([])
    ax.set_title("(a) correlation matrix $\\mathbf{S}$\n(stocks sorted by sector)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax = axes[1]
    iu = np.triu_indices(len(sector), 1)
    same = sector[iu[0]] == sector[iu[1]]
    bins = np.linspace(-0.2, 1.0, 49)
    ax.hist(S[iu][~same], bins=bins, color=GREY, alpha=0.8, density=True,
            label="pairs in different sectors")
    ax.hist(S[iu][same], bins=bins, color=BLUE, alpha=0.6, density=True,
            label="pairs in the same sector")
    ax.set_xlabel("correlation of daily returns")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(b) almost every pair is positively\ncorrelated, same-sector pairs more so")
    save(fig, "fig05_data")


# ---------------------------------------------------------------- Part I
def fig06_fig2_graph(S, sector):
    L = np.load(os.path.join(P1, "data", "alg1_L.npy"))
    W = adjacency(L)
    pos = layout(W, seed=3)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9), gridspec_kw=dict(width_ratios=[2.1, 1]))
    draw_graph(axes[0], W, pos, [SECTOR_COL[s] for s in sector], top=450)
    n, lab = connected_components(W > 1e-4)
    sizes = sorted(np.bincount(lab), reverse=True)
    axes[0].set_title(f"(a) learned graph: {n} components of {sizes[0]}, {sizes[1]} and "
                      f"{sizes[2]} stocks", fontsize=8.5)
    sector_legend(axes[0], loc="upper center", bbox_to_anchor=(0.5, 0.02))
    ax = axes[1]
    lam = eig(laplacian(W))[:10]
    ax.bar(np.arange(1, 11), lam, color=[RED] * 3 + [GREY] * 7, width=0.6)
    ax.scatter(np.arange(1, 4), lam[:3], color=RED, s=12, zorder=3, clip_on=False)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("$i$")
    ax.set_ylabel(r"$\lambda_i(\mathbf{L})$")
    ax.set_title("(b) its ten smallest eigenvalues", fontsize=8.5)
    d = W.sum(axis=1)
    ax.text(0.03, 0.97, f"degrees in [{d.min():.4f}, {d.max():.4f}]\n"
            f"$\\lambda_4={lam[3]:.4f}$", transform=ax.transAxes, va="top", fontsize=7.5)
    save(fig, "fig06_fig2_graph")


def fig07_reflection():
    L = np.load(os.path.join(P1, "data", "alg1_L.npy"))
    W = adjacency(L)
    p = W.shape[0]
    Q = null_space(np.ones((1, p)))
    lam = eig(Q.T @ laplacian(W) @ Q)
    mu = eig(Q.T @ modularity_matrix(W) @ Q)[::-1]
    dbar = W.sum(axis=1).mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    xs = np.linspace(-0.05, 1.55, 10)
    ax.plot(xs, 1 - xs, color=GREY, lw=1, label=r"Part I: $\mu=1-\lambda$")
    ax.scatter(lam, mu, s=10, color=RED, zorder=3, label="learned graph (96 pairs)")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(1, color="k", lw=0.4, ls=":")
    ax.set_xlabel(r"Laplacian eigenvalue $\lambda_i$ (increasing)")
    ax.set_ylabel(r"modularity eigenvalue $\mu_i$ (decreasing)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("(a) $\\mathbf{B}$'s spectrum is $\\mathbf{L}$'s, reflected", fontsize=8.5)
    ax.annotate("the 2 cuts that cost nothing\n($\\lambda=0$) are splits worth 1", (0.0, 1.0),
                xytext=(0.0, -0.35), fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax = axes[1]
    one_minus_mu = np.sort(1 - mu)
    dev = one_minus_mu - lam
    ax.axhline(0, color="k", lw=0.4)
    ax.scatter(lam, dev, s=9, color=RED, label="reflection about 1")
    ax.scatter(lam, dev + dbar - 1, s=9, color="#2F8F5B",
               label=f"reflection about mean degree {dbar:.4f}")
    ax.set_xlabel(r"$\lambda_i$")
    ax.set_ylabel(r"deviation from prediction")
    ax.legend(frameon=False, loc="center", fontsize=7.5)
    ax.set_title("(b) misfit: the degrees are 1.005, not 1", fontsize=8.5)
    ax.ticklabel_format(axis="y", style="plain")
    save(fig, "fig07_reflection")


def fan_sides(W, k):
    lam, mu = eig(laplacian(W)), eig(modularity_matrix(W))[::-1]
    return lam[:k].sum(), (k - 1) - mu[:k - 1].sum()


def fig08_counterexamples():
    K4 = (np.ones((4, 4)) - np.eye(4)) / 3
    C6 = np.zeros((6, 6))
    for i in range(6):
        C6[i, (i + 1) % 6] = C6[(i + 1) % 6, i] = 0.5
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[0.8, 0.9, 1.4]))
    posK = {i: (np.cos(np.pi / 4 + i * np.pi / 2), np.sin(np.pi / 4 + i * np.pi / 2)) for i in range(4)}
    posC = {i: (np.cos(i * np.pi / 3), np.sin(i * np.pi / 3)) for i in range(6)}
    draw_toy(axes[0], K4, np.zeros(4), pos=posK, edge_labels=False, cmap="Greys", vmin=-1, vmax=4)
    axes[0].set_title("(a) $K_4$, weights $1/3$\nevery degree 1", fontsize=8.5)
    draw_toy(axes[1], C6, np.zeros(6), pos=posC, edge_labels=False, cmap="Greys", vmin=-1, vmax=4)
    axes[1].set_title("(b) 6-cycle, weights $1/2$\nevery degree 1", fontsize=8.5)
    ax = axes[2]
    cases = [("$K_4$\n$k=2$", K4, 2), ("cycle\n$k=2$", C6, 2), ("cycle\n$k=3$", C6, 3),
             ("cycle\n$k=4$", C6, 4)]
    x = np.arange(len(cases))
    lhs = [fan_sides(W, k)[0] for _, W, k in cases]
    rhs = [fan_sides(W, k)[1] for _, W, k in cases]
    ax.bar(x - 0.18, lhs, 0.36, color=BLUE, label="Fan side")
    ax.bar(x + 0.18, rhs, 0.36, color="#E3A869", label="modularity side")
    for xi, a, b in zip(x, lhs, rhs):
        if not np.isclose(a, b):
            ax.text(xi, max(a, b) + 0.1, f"gap {a - b:.2f}", ha="center", fontsize=7.5, color=RED)
    ax.set_xticks(x, [c[0] for c in cases], fontsize=7.5)
    ax.set_ylim(0, 3.1)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5, bbox_to_anchor=(0.0, 1.0))
    ax.set_title("(c) the two sides of claim (b)", fontsize=8.5)
    save(fig, "fig08_counterexamples")


def fig09_degrees():
    T = np.zeros((6, 6))
    for a, b in [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (2, 5)]:
        T[a, b] = T[b, a] = 1.0
    d = T.sum(axis=1)
    pos = {0: (-1.2, 0.7), 1: (-1.2, -0.7), 2: (0, 0), 3: (1.1, 0.5), 4: (2.2, 0.9), 5: (1.0, -0.8)}
    L, B = laplacian(T), modularity_matrix(T)
    Dm = np.diag(d ** -0.5)
    Ls, Bn = np.eye(6) - Dm @ T @ Dm, Dm @ B @ Dm
    v = np.linalg.eigh(L)[1][:, 1]
    ub = np.linalg.eigh(B)[1][:, -1]
    us = np.linalg.eigh(Ls)[1][:, 1]
    ubn = np.linalg.eigh(Bn)[1][:, -1]
    fix = lambda z: z * np.sign(z[0]) / np.abs(z).max()
    res = lambda M, z: np.linalg.norm(M @ z - (z @ M @ z) * z)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.0))
    for ax, z, t in [(axes[0], v, "Fiedler vector of $\\mathbf{L}$"),
                     (axes[1], ub, "top eigenvector of $\\mathbf{B}$"),
                     (axes[2], us, "Fiedler vector of $\\mathbf{L}_{\\mathrm{sym}}$"),
                     (axes[3], ubn, "top eigenvector of $\\mathbf{B}_n$")]:
        draw_toy(ax, T, fix(z), pos=pos, text=[f"{int(x)}" for x in d], edge_labels=False)
        ax.set_title(t, fontsize=8.5)
    axes[0].text(0.5, -0.08, f"not an eigenvector of $\\mathbf{{B}}$\n(residual {res(B, v):.2f})",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=7.5, color=RED)
    axes[2].text(0.5, -0.08, f"also an eigenvector of $\\mathbf{{B}}_n$\n(residual {res(Bn, us):.0e})",
                 transform=axes[2].transAxes, ha="center", va="top", fontsize=7.5, color="#2F8F5B")
    save(fig, "fig09_degrees")


def fig10_sweep_graphs(sector, tickers):
    from degree_sweep import cache_path
    runs = [(100, os.path.join(P1, "data", "alg1_L.npy")), (1, cache_path(1.0)),
            (0.3, cache_path(0.3))]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.4))
    for ax, (rho, path) in zip(axes, runs):
        W = adjacency(np.load(path))
        d = W.sum(axis=1)
        pos = layout(W, seed=3)
        n, lab = connected_components(W > 1e-4)
        major = {c: max(set(sector[lab == c]), key=list(sector[lab == c]).count)
                 for c in np.unique(lab)}
        moved = np.array([sector[i] != major[lab[i]] for i in range(len(lab))])
        labels = {i: tickers[i] for i in np.where(moved)[0]}
        draw_graph(ax, W, pos, [SECTOR_COL[s] for s in sector], top=450,
                   sizes=10 + 110 * (d - 1.0), outline=moved, labels=labels)
        ax.set_title(f"degree penalty $\\rho={rho:g}$: degrees from {d.min():.3f} to {d.max():.3f}"
                     + (f"; {int(moved.sum())} stocks outside their sector's component"
                        if moved.any() else ""), fontsize=8.5)
    sector_legend(axes[2], loc="upper center", bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig10_sweep_graphs")


def fig11_sweep_lines(sector):
    from degree_sweep import RHOS, cache_path
    rows = []
    for rho in RHOS:
        path = os.path.join(P1, "data", "alg1_L.npy") if rho == 100 else cache_path(rho)
        W = adjacency(np.load(path))
        p = W.shape[0]
        d = W.sum(axis=1)
        L, B = laplacian(W), modularity_matrix(W)
        R = B - (np.eye(p) - L - np.ones((p, p)) / p)
        Dm = np.diag(1 / np.sqrt(d))
        rows.append((rho, d.std() / d.mean(), np.linalg.norm(R) / np.linalg.norm(B),
                     eig(B)[-1], eig(B)[-2], eig(Dm @ B @ Dm)[-1], eig(Dm @ B @ Dm)[-2]))
    r = np.array(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    ax.plot(r[:, 1], r[:, 2], "o-", color=RED, ms=4)
    for row in r:
        ax.annotate(f"$\\rho={row[0]:g}$", (row[1], row[2]), xytext=(4, -9),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("degree spread (coefficient of variation)")
    ax.set_ylabel(r"$\|\mathbf{B}-(\mathbf{I}-\mathbf{L}-\mathbf{11}^\top/p)\|_F/\|\mathbf{B}\|_F$")
    ax.set_title("(a) the Part I identity degrades", fontsize=8.5)
    ax = axes[1]
    ax.plot(r[:, 1], r[:, 3], "o-", color=RED, ms=4, label=r"$\mu_1(\mathbf{B})$")
    ax.plot(r[:, 1], r[:, 4], "s--", color=RED, ms=3.5, mfc="white", label=r"$\mu_2(\mathbf{B})$")
    ax.plot(r[:, 1], r[:, 5], "o-", color="#2F8F5B", ms=4, label=r"$\mu_1,\mu_2$ of $\mathbf{B}_n$")
    ax.axhline(1, color="k", lw=0.4, ls=":")
    ax.set_xlabel("degree spread (coefficient of variation)")
    ax.set_ylabel("top eigenvalues")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(b) Part I predicts exactly 1; the normalised pair keeps it", fontsize=8.5)
    save(fig, "fig11_sweep_lines")


# ---------------------------------------------------------------- Part II
def fig12_market_real(X, S, sector):
    from part2 import load_index, loadings, sin2theta
    l = loadings(X.values, load_index().values)
    p = S.shape[0]
    J = np.eye(p) - np.ones((p, p)) / p
    order = np.argsort([list(SECTOR_COL).index(s) for s in sector], kind="stable")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.3], wspace=0.35))
    for ax, M, t in [(axes[0], S, "(a) $\\mathbf{S}$"),
                     (axes[1], J @ S @ J, "(b) $\\mathbf{JSJ}$: daily average removed")]:
        ax.imshow(M[np.ix_(order, order)], cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        cuts = np.cumsum([np.sum(sector == s) for s in SECTOR_COL])[:-1]
        for c in cuts:
            ax.axhline(c - 0.5, color="k", lw=0.5)
            ax.axvline(c - 0.5, color="k", lw=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(t, fontsize=8.5)
    ax = axes[2]
    rng = np.random.default_rng(0)
    for g, s in enumerate(SECTOR_COL):
        y = l[sector == s]
        ax.scatter(g + rng.uniform(-0.18, 0.18, len(y)), y, s=12, color=SECTOR_COL[s], alpha=0.8)
        ax.hlines(y.mean(), g - 0.3, g + 0.3, color="k", lw=1.2)
    ax.axhline(l.mean(), color=GREY, lw=0.6, ls="--")
    ax.set_xticks(range(3), ["Industrials", "Staples", "Energy"])
    ax.set_ylabel("corr(stock, S&P 500)")
    ax.set_title("(c) market loading of each stock", fontsize=8.5)
    save(fig, "fig12_market_real")


def fig13_real_graphs(sector, tickers):
    from part2 import cache_file, clusters
    from sklearn.metrics import adjusted_rand_score as ari
    names = [("S", "$\\mathbf{S}$ (implicit null)"), ("S-mode", "S-mode"),
             ("resid-SPX", "resid-SPX"), ("RMT", "RMT")]
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 7.4))
    axes = axes.ravel()
    for ax, (name, title) in zip(axes, names):
        L = np.load(cache_file(f"real/{name}/alg1"))
        W = adjacency(L)
        lab = clusters(L, "alg1")
        major = {c: max(set(sector[lab == c]), key=list(sector[lab == c]).count)
                 for c in np.unique(lab)}
        wrong = np.array([sector[i] != major[lab[i]] for i in range(len(lab))])
        rank = {s: i for i, s in enumerate(SECTOR_COL)}
        pos = layout(W, seed=3, groups=lab, order_key=lambda c: rank[major[lab[c[0]]]])
        draw_graph(ax, W, pos, [SECTOR_COL[s] for s in sector], top=450, outline=wrong,
                   sizes=np.where(wrong, 40.0, 20.0),
                   labels={i: tickers[i] for i in np.where(wrong)[0]})
        ncomp = connected_components(W > 1e-4)[0]
        ax.set_title(f"{title}: {ncomp} component{'s' if ncomp > 1 else ''}, "
                     f"ARI {ari(sector, lab):.3f}, {int(wrong.sum())} stocks placed outside "
                     "their sector", fontsize=8.5)
    sector_legend(axes[3], loc="upper center", bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig13_real_graphs")


def fig14_synthetic_design():
    from part2 import synthetic_betas
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.9), sharey=True)
    for ax, pattern, t in [(axes[0], "aligned", "(a) aligned: one beta per sector"),
                           (axes[1], "cross", "(b) cross: high and low beta inside every sector")]:
        beta, sec = synthetic_betas(pattern, 0.5)
        ax.bar(np.arange(60), beta, color=[SYN_COL[g] for g in sec], width=0.85)
        ax.axhline(1, color="k", lw=0.4, ls=":")
        ax.set_xlabel("stock (three sectors of 20)")
        ax.set_title(t + r", $\delta=0.5$", fontsize=8.5)
        ax.set_xticks([10, 30, 50], ["sector 1", "sector 2", "sector 3"])
    axes[0].set_ylabel(r"beta $b_i$")
    save(fig, "fig14_synthetic_design")


def fig15_synthetic_graphs():
    from part2 import cache_file, clusters, synthetic_betas
    from sklearn.metrics import adjusted_rand_score as ari
    panels = [("cross", 0.4, "S", "(a) $\\delta=0.4$, from $\\mathbf{S}$"),
              ("cross", 0.6, "S", "(b) $\\delta=0.6$, from $\\mathbf{S}$"),
              ("cross", 0.6, "S-mode", "(c) $\\delta=0.6$, from S-mode")]
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.0))
    for ax, (pat, dl, name, title) in zip(axes, panels):
        beta, sec = synthetic_betas(pat, dl)
        L = np.load(cache_file(f"syn/{pat}/{dl:g}/seed0/{name}/alg1"))
        W = adjacency(L)
        lab = clusters(L, "alg1")
        pos = layout(W, seed=2)
        shapes = np.where(beta > 1, "^", "v")
        draw_graph(ax, W, pos, [SYN_COL[g] for g in sec], top=300, shapes=shapes,
                   sizes=np.full(60, 34.0))
        ax.set_title(f"{title}: ARI against sectors {ari(sec, lab):.2f}", fontsize=8.5)
    h = [Line2D([], [], marker="o", ls="", color=c, ms=6, label=f"sector {g + 1}")
         for g, c in enumerate(SYN_COL)]
    h += [Line2D([], [], marker="^", ls="", color="k", ms=6, label=r"high beta $1+\delta$"),
          Line2D([], [], marker="v", ls="", color="k", ms=6, label=r"low beta $1-\delta$")]
    axes[2].legend(handles=h, loc="upper center", ncol=5, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig15_synthetic_graphs")


def fig16_threshold():
    from part2 import metrics, sin2theta, synthetic_betas
    from part2_bound import population_loadings
    from part2_threshold import FINE_DELTAS, FINE_SEEDS, tpath
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    styles = {("S", "alg1"): (BLUE, "s", "Algorithm 1 on $\\mathbf{S}$"),
              ("S", "prob1"): (RED, "o", "Problem (1) on $\\mathbf{S}$")}
    rng = np.random.default_rng(1)
    for (name, lr), (c, m, lab) in styles.items():
        xs, means = [], []
        for delta in FINE_DELTAS:
            beta, sec = synthetic_betas("cross", delta)
            s2 = sin2theta(population_loadings(beta))
            vals = [metrics(np.load(tpath(delta, s, name, lr)), lr, sec, beta)["ari_sector"]
                    for s in FINE_SEEDS]
            ax.scatter(s2 + rng.uniform(-0.0012, 0.0012, len(vals)), vals, s=9, color=c,
                       alpha=0.35, marker=m)
            xs.append(s2)
            means.append(np.mean(vals))
        ax.plot(xs, means, color=c, marker=m, ms=4, lw=1.4, label=lab + " (mean of 5 seeds)")
    ax.plot([xs[0], xs[-1]], [1, 1], color="#2F8F5B", ls="--", lw=1.4,
            label="S-mode and oracle, both learners (all 180 runs)")
    ax.set_xlabel(r"beta inequality $\sin^2\theta$")
    ax.set_ylabel("ARI against sectors")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig16_threshold")


def fig17_bound(X, S, sector):
    from part2 import load_index, loadings, sin2theta, synthetic_betas, synthetic_returns
    from part2_bound import DELTAS, SEEDS, bound_row, population_loadings
    rows = []
    for delta in DELTAS[1:]:
        beta, sec = synthetic_betas("cross", delta)
        lp = population_loadings(beta)
        rr = [bound_row(np.corrcoef(synthetic_returns("cross", delta, s)[0].T), lp, sec)
              for s in SEEDS]
        rows.append((sin2theta(lp), *[np.mean([r[k] for r in rr]) for k in
                                       ("sin", "sin_dk", "sin_full", "dL", "gap", "ari_obs")]))
    r = np.array(rows)
    l = loadings(X.values, load_index().values)
    real = bound_row(S, l, sector)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.7))
    ax = axes[0]
    ax.semilogy(r[:, 0], r[:, 3], "o-", color="#E3A869", ms=4, label="Steps 1 + 2 chained")
    ax.semilogy(r[:, 0], r[:, 2], "s-", color=BLUE, ms=4, label="Step 2 with the true $\\Delta\\mathbf{L}$")
    ax.semilogy(r[:, 0], r[:, 1], "o-", color=RED, ms=4, label="actual movement")
    ax.axhline(np.sqrt(K), color="k", lw=0.8, ls="--", label=r"largest possible value $\sqrt{3}$")
    ax.set_xlabel(r"$\sin^2\theta$")
    ax.set_ylabel(r"$\|\sin\Theta\|_F$ (log scale)")
    ax.set_title("(a) how far the cluster eigenvectors move", fontsize=8.5)
    ax.legend(frameon=False, loc="lower right", fontsize=7.3)
    ax = axes[1]
    ratio = r[:, 4] / r[:, 5]
    sc = ax.scatter(r[:, 0], ratio, c=r[:, 6], cmap="RdYlGn", vmin=0, vmax=1, s=40,
                    edgecolors="k", linewidths=0.5, zorder=3, label="synthetic (colour = ARI)")
    ax.plot(r[:, 0], ratio, color=GREY, lw=0.8, zorder=2)
    ax.axhspan(1.067, 1.263, color=RED, alpha=0.12, lw=0)
    ax.text(0.33, 1.12, "switch happens here", fontsize=7.5, color=RED, ha="right")
    ax.scatter([sin2theta(l)], [real["dL"] / real["gap"]], marker="*", s=120, color="k",
               zorder=4, label=f"S&P panel ({real['dL'] / real['gap']:.2f})")
    ax.set_xlabel(r"$\sin^2\theta$")
    ax.set_ylabel(r"$\|\Delta\mathbf{L}\|_2\,/\,\mathrm{eigengap}$")
    ax.set_title("(b) the size of the change relative to the gap", fontsize=8.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.3)
    fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, label="ARI vs sectors")
    save(fig, "fig17_bound")


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    X, S, sector = load_sp()
    tickers = list(X.columns)
    figs = [("fig02", lambda: fig02_laplacian()), ("fig03", lambda: fig03_components()),
            ("fig04", lambda: fig04_modularity()), ("fig05", lambda: fig05_data(X, S, sector)),
            ("fig06", lambda: fig06_fig2_graph(S, sector)), ("fig07", lambda: fig07_reflection()),
            ("fig08", lambda: fig08_counterexamples()), ("fig09", lambda: fig09_degrees()),
            ("fig10", lambda: fig10_sweep_graphs(sector, tickers)),
            ("fig11", lambda: fig11_sweep_lines(sector)),
            ("fig12", lambda: fig12_market_real(X, S, sector)),
            ("fig13", lambda: fig13_real_graphs(sector, tickers)),
            ("fig14", lambda: fig14_synthetic_design()),
            ("fig15", lambda: fig15_synthetic_graphs()), ("fig16", lambda: fig16_threshold()),
            ("fig17", lambda: fig17_bound(X, S, sector))]
    for name, f in figs:
        if want in name:
            f()


if __name__ == "__main__":
    main()
