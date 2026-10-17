"""
Re-derive every number quoted in the proposal reports:
    reports/2026-10-18_proposal_issues.tex     (sections A, B, C)

Section C needs the fig2_init_check.py caches (run it for uniform/problem1 x 1/default).

Same discipline as modularity_indicator/report_numbers.py: the reports
transcribe this script's output and nothing else. Read-only apart from a cache
of the Problem 1 graph in this folder (rebuilt if deleted).

Run:  python report_numbers.py
"""

import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

from common import (HERE, ROOT, P1, K, load_sp, adjacency, laplacian,
                    modularity_matrix, top_eig, sector_block_graph)

from solver.laplacian_solver import learn_laplacian, log_gdet  # noqa: E402  (Paper 1)
from fig2_init_check import cache_path  # noqa: E402

P1_CACHE = os.path.join(HERE, "cache_problem1_L.npy")
OLD_REV = "427eea9"   # main before the Fig 2 repair: 1,005-day data, uniform start


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def problem1_graph(S):
    """Problem 1 (beta = 0, no degree or rank constraint) on the 753-day data."""
    if os.path.exists(P1_CACHE):
        return np.load(P1_CACHE)
    L = learn_laplacian(S, beta=0.0, max_iter=2000, tol=1e-7)
    np.save(P1_CACHE, L)
    return L


def ray(L0, S, k=K):
    """Along W -> cW the Part III objective is c(a - eta b) - r log c + const.

    a = tr(L0 S); b = max over orthonormal V (p x k-1) of tr(V' B(W0) V), the
    sum of the k-1 largest eigenvalues of B(W0) (scale-free eigenvectors, so the
    optimal V does not move with c); r = rank(L0).
    """
    W0 = adjacency(L0)
    a = float(np.trace(laplacian(W0) @ S))
    b = float(top_eig(modularity_matrix(W0), k - 1)[0].sum())
    r = int(np.sum(np.linalg.eigvalsh(laplacian(W0)) > 1e-10))
    return a, b, r


def objective_on_ray(L0, S, eta, c, k=K):
    W = c * adjacency(L0)
    L = laplacian(W)
    mu, _ = top_eig(modularity_matrix(W), k - 1)
    return float(np.trace(L @ S) - log_gdet(L) - eta * mu.sum())


def main():
    X, S, sector = load_sp()
    p = S.shape[0]
    rule(f"DATA  S&P three-sector panel: p = {p}, T = {len(X)}, "
         f"{X.index[0].date()} to {X.index[-1].date()}")
    print("  sector counts:", pd.Series(sector).value_counts().to_dict())

    # ------------------------------------------------------------------ A
    rule("A. Part III objective  tr(LS) - log gdet(L) - eta tr(V'B(W)V):  boundedness")
    iu = np.triu_indices(p, 1)
    Z = np.add.outer(np.diag(S), np.diag(S)) - 2 * S
    i, j = iu[0][np.argmin(Z[iu])], iu[1][np.argmin(Z[iu])]
    zmin = Z[i, j]
    print(f"  max off-diagonal correlation {S[iu].max():.3f}  "
          f"({X.columns[i]}/{X.columns[j]}), so Z_min = 2(1 - rho_max) = {zmin:.3f}")
    print(f"  BOUNDED for every eta < Z_min/(k-1) = {zmin / (K - 1):.3f}"
          "   [tr(LS) >= Z_min m and sum_{k-1} mu(B) <= (k-1) d_max <= (k-1) m]")

    # Worst case found: all weight on the K most-correlated disjoint pairs. B then
    # has eigenvalue w with multiplicity K-1, so b = (K-1) w while a = w sum Z.
    order = np.argsort(Z[iu])
    pairs, used = [], set()
    for t in order:
        a_, b_ = iu[0][t], iu[1][t]
        if a_ not in used and b_ not in used:
            pairs.append((a_, b_)); used |= {a_, b_}
        if len(pairs) == K:
            break
    W_pairs = np.zeros((p, p))
    for a_, b_ in pairs:
        W_pairs[a_, b_] = W_pairs[b_, a_] = 1.0
    print(f"  {K} most-correlated disjoint pairs: "
          + ", ".join(f"{X.columns[a_]}/{X.columns[b_]} (Z={Z[a_, b_]:.3f})" for a_, b_ in pairs))
    W_conn = 1e-6 * (np.ones((p, p)) - np.eye(p))   # keeps the graph connected

    rays = {
        "sector block graph (unit degree)": laplacian(sector_block_graph(sector)),
        "Problem 1 graph (beta = 0)": problem1_graph(S),
        f"{K} disjoint pairs (disconnected)": laplacian(W_pairs),
        f"{K} disjoint pairs + 1e-6 complete graph (connected)": laplacian(W_pairs + W_conn),
    }
    for name, L0 in rays.items():
        a, b, r = ray(L0, S)
        eta_ray = a / b
        print(f"\n  ray: {name}")
        print(f"    a = tr(L0 S) = {a:.3f}   b = sum top-{K-1} mu(B) = {b:.4f}   "
              f"rank {r}")
        print(f"    UNBOUNDED for every eta > a/b = {eta_ray:.2f}")
        eta = 2 * eta_ray
        vals = [objective_on_ray(L0, S, eta, c) for c in (1, 10, 100, 1000)]
        print(f"    objective at eta = 2 a/b along c = 1, 10, 100, 1000: "
              + ", ".join(f"{v:.1f}" for v in vals))

    if "--figdata" in sys.argv:
        L0 = rays[f"{K} disjoint pairs + 1e-6 complete graph (connected)"]
        print("\n  FIGDATA objective along the connected disjoint-pairs ray, x = log10 c")
        for eta in (0.1, 1.0):
            pts = " ".join(f"({np.log10(c):.2f},{objective_on_ray(L0, S, eta, c):.1f})"
                           for c in np.geomspace(1e-1, 1e4, 21))
            print(f"  eta = {eta}: {pts}")

    rule("A'. Identity  -tr(V'BV) = tr(V'LV) - sum_i d_i ||v_i - c_d||^2")
    W = adjacency(rays["Problem 1 graph (beta = 0)"])
    d = W.sum(axis=1)
    rng = np.random.default_rng(0)
    V, _ = np.linalg.qr(rng.standard_normal((p, K - 1)))
    lhs = -np.trace(V.T @ modularity_matrix(W) @ V)
    c_d = (d @ V) / d.sum()
    rhs = np.trace(V.T @ laplacian(W) @ V) - np.sum(d * np.sum((V - c_d) ** 2, axis=1))
    print(f"  random orthonormal V on the Problem 1 graph: |lhs - rhs| = {abs(lhs - rhs):.1e}")

    # ------------------------------------------------------------------ B
    rule("B. Part I(c) on the paper's time-varying graphs (Problem 11, FAAMUNG)")
    ser = pd.read_parquet(os.path.join(ROOT, "modularity_indicator", "series.parquet"))
    print(f"  {len(ser)} learned graphs (degree_control=False, as in Problem 11)")
    print(f"  mean degree d_bar: mean {ser.dbar.mean():.2f}, "
          f"range [{ser.dbar.min():.2f}, {ser.dbar.max():.2f}]   (Part I needs every d_i = 1)")
    print(f"  degree CV: mean {ser.deg_cv.mean():.3f}, "
          f"range [{ser.deg_cv.min():.3f}, {ser.deg_cv.max():.3f}]   (0 = regular)")
    above = (ser.lam2 >= 1.0).mean()
    print(f"  windows with lambda_2 >= 1 (where max(0, 1 - lambda_2) censors): "
          f"{100 * above:.0f}%  ({int((ser.lam2 >= 1.0).sum())}/{len(ser)})")
    print("  eigenvector overlap B vs L (unnormalised): see "
          "modularity_indicator/report_numbers.py")

    # ------------------------------------------------------------------ C
    rule("C. Paper 1 Fig 2: Algorithm 1 components against the sector labels")

    def describe(name, L):
        W = adjacency(L)
        n, lab = connected_components(W > 1e-4)
        sizes = sorted(np.bincount(lab).tolist(), reverse=True)
        d = W.sum(axis=1)
        print(f"  {name}")
        print(f"    {n} components, sizes {sizes[:6]}, ARI vs sectors {ari(sector, lab):.3f}, "
              f"degree range [{d.min():.3f}, {d.max():.3f}], isolated {int(np.sum(d < 1e-6))}")

    blob = subprocess.run(
        ["git", "show", f"{OLD_REV}:paper1_gmrf_laplacian/data/alg1_L.npy"],
        cwd=ROOT, capture_output=True, check=True).stdout
    describe(f"before: uniform start, old 1,005-day window (git {OLD_REV})",
             np.load(io.BytesIO(blob)))
    runs = {}
    for start, threads, path in [
            ("uniform", "1", cache_path("uniform", "1")),
            ("uniform", "default", cache_path("uniform", "default")),
            ("problem1", "1", cache_path("problem1", "1")),
            ("problem1", "default", os.path.join(P1, "data", "alg1_L.npy"))]:
        runs[start, threads] = np.load(path)
        describe(f"{start} start, 753-day window, BLAS threads = {threads}", runs[start, threads])
    for start in ("uniform", "problem1"):
        diff = np.abs(runs[start, "1"] - runs[start, "default"]).max()
        print(f"  {start} start: max |L(1 thread) - L(default threads)| = {diff:.1e}")
    _, U = top_eig(S, K)
    km = KMeans(K, n_init=20, random_state=0).fit_predict(U)
    print(f"  reference: k-means on the top {K} eigenvectors of the correlation matrix, "
          f"ARI vs sectors {ari(sector, km):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
