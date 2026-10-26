"""
Re-derive every number quoted in the proposal reports:
    reports/2026-10-18_proposal_issues.tex     (sections A, B, C)
    reports/2026-10-27_part1_lemma.tex         (sections D, E, F, G)

Section C needs the fig2_init_check.py caches (run it for uniform/problem1 x 1/default).
Section G needs the degree_sweep.py caches.

Same discipline as modularity_indicator/report_numbers.py: the reports
transcribe this script's output and nothing else. Read-only apart from a cache
of the Problem 1 graph in this folder (rebuilt if deleted).

Run:  python report_numbers.py [--part1] [--figdata] [--figdata-part1]
      (--part1 prints sections D to G only)
"""

import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.linalg import null_space, subspace_angles
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

from common import (HERE, ROOT, P1, K, load_sp, adjacency, laplacian,
                    modularity_matrix, top_eig, sector_block_graph)

from solver.laplacian_solver import learn_laplacian, log_gdet  # noqa: E402  (Paper 1)
from fig2_init_check import cache_path  # noqa: E402
from degree_sweep import RHOS, cache_path as rho_cache  # noqa: E402

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


def eig(M):
    return np.linalg.eigvalsh((M + M.T) / 2.0)


def fan_sides(W, k):
    """Part I(b): sum of the k smallest eig(L), and (k-1) - sum of the k-1 largest eig(B)."""
    lam, mu = eig(laplacian(W)), eig(modularity_matrix(W))[::-1]
    return lam[:k].sum(), (k - 1) - mu[:k - 1].sum()


def part1_metrics(L, sector, k=K):
    """How closely a learned graph obeys Part I's B = I - L - 11'/p, and what that buys."""
    W = adjacency(L)
    p = W.shape[0]
    L, B = laplacian(W), modularity_matrix(W)
    d = W.sum(axis=1)
    R = B - (np.eye(p) - L - np.ones((p, p)) / p)
    n, lab = connected_components(W > 1e-4)
    lhs, rhs = fan_sides(W, k)
    _, VL = np.linalg.eigh(L)
    _, UB = top_eig(B, k - 1)
    span_B = np.column_stack([np.ones(p) / np.sqrt(p), UB])
    angle = subspace_angles(VL[:, :k], span_B).max()
    km = KMeans(k, n_init=20, random_state=0).fit_predict(UB)
    Dm = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    return dict(mu_n=eig(Dm @ B @ Dm)[::-1][:k], dmin=d.min(), dmax=d.max(), cv=d.std() / d.mean(), n=n,
                sizes=sorted(np.bincount(lab).tolist(), reverse=True),
                isolated=int(np.sum(d < 1e-6)), res=np.linalg.norm(R),
                rel=np.linalg.norm(R) / np.linalg.norm(B), mu=eig(B)[::-1][:k + 1],
                lam=eig(L)[:k + 1], lhs=lhs, rhs=rhs, sin=np.sin(angle),
                ari_comp=ari(lab, km), ari_sec=ari(sector, km), ari_lab=ari(sector, lab),
                lab=lab)


def part1(S, sector, tickers):
    p = S.shape[0]

    # ------------------------------------------------------------------ D
    rule("D. Part I on the repaired Fig 2 graph (Algorithm 1, eta = 300, rho = 100)")
    L = np.load(os.path.join(P1, "data", "alg1_L.npy"))
    m = part1_metrics(L, sector)
    print(f"  degrees in [{m['dmin']:.4f}, {m['dmax']:.4f}], CV {m['cv']:.5f}   "
          "(Part I needs every d_i = 1)")
    print(f"  {m['n']} components, sizes {m['sizes']}, ARI vs sectors {m['ari_lab']:.3f}")
    print(f"  ||B - (I - L - 11'/p)||_F = {m['res']:.4f}   relative to ||B||_F: {m['rel']:.4f}")
    print("  smallest eig(L): " + ", ".join(f"{x:.1e}" if abs(x) < 1e-6 else f"{x:.4f}"
                                            for x in m["lam"]))
    print("  largest  eig(B): " + ", ".join(f"{x:.4f}" for x in m["mu"])
          + f"   (a) predicts exactly 1 with multiplicity k-1 = {K - 1}")
    print(f"  (b) Fan identity: sum of {K} smallest eig(L) = {m['lhs']:.1e},   "
          f"{K - 1} - sum of {K - 1} largest eig(B) = {m['rhs']:.4f}")
    print(f"  sin of largest angle, bottom-{K} eigenspace of L vs span(1, top-{K - 1} of B): "
          f"{m['sin']:.4f}")
    print(f"  k-means on the top {K - 1} eigenvectors of B: ARI vs components "
          f"{m['ari_comp']:.3f}, ARI vs sectors {m['ari_sec']:.3f}")

    # Spectra on 1-perp, where Part I predicts eig(B) = 1 - eig(L) one for one.
    Q = null_space(np.ones((1, p)))
    W = adjacency(L)
    lam_perp = eig(Q.T @ laplacian(W) @ Q)
    one_minus_mu = np.sort(1.0 - eig(Q.T @ modularity_matrix(W) @ Q))
    gap = np.abs(lam_perp - one_minus_mu)
    print(f"  on 1-perp ({p - 1} pairs): max |eig_i(L) - (1 - eig_i(B))| = {gap.max():.4f}, "
          f"mean {gap.mean():.4f};  eig(L) range [{lam_perp.min():.1e}, {lam_perp.max():.3f}]")
    dbar = W.sum(axis=1).mean()
    gap_c = np.abs(lam_perp - (one_minus_mu + dbar - 1.0))
    print(f"  mean degree c = {dbar:.4f}; reflecting about c instead of 1 (the c-regular form): "
          f"max |eig_i(L) - (c - eig_i(B))| = {gap_c.max():.4f}, mean {gap_c.mean():.4f}")
    if "--figdata-part1" in sys.argv:
        print("  FIGDATA (eig_i(L), 1 - eig_i(B)) on 1-perp:")
        print("  " + " ".join(f"({a:.6f},{b:.6f})" for a, b in zip(lam_perp, one_minus_mu)))

    # ------------------------------------------------------------------ E
    rule("E. The role of lambda_k <= 1: two unit-degree counterexamples to (b)")
    q = 4
    Kq = (np.ones((q, q)) - np.eye(q)) / (q - 1)
    lhs, rhs = fan_sides(Kq, 2)
    print(f"  complete graph K_{q}, weights 1/{q - 1}: eig(L) = "
          + ", ".join(f"{x:.4f}" for x in eig(laplacian(Kq)))
          + f";  k = 2: Fan side {lhs:.4f}, modularity side {rhs:.4f}")
    C6 = np.zeros((6, 6))
    for i in range(6):
        C6[i, (i + 1) % 6] = C6[(i + 1) % 6, i] = 0.5
    print("  6-cycle, weights 1/2: eig(L) = " + ", ".join(f"{x:.4f}" for x in eig(laplacian(C6)))
          + ";  eig(B) = " + ", ".join(f"{x:.4f}" for x in eig(modularity_matrix(C6))[::-1]))
    for k in (2, 3, 4):
        lhs, rhs = fan_sides(C6, k)
        print(f"    k = {k}: lambda_k = {eig(laplacian(C6))[k - 1]:.4f},  Fan side {lhs:.4f}, "
              f"modularity side {rhs:.4f}  {'match' if np.isclose(lhs, rhs) else 'MISMATCH'}")

    # ------------------------------------------------------------------ F
    rule("F. When do L and B share eigenvectors?")
    C6u = 2 * C6                                 # plain 6-cycle, every degree c = 2
    c = 2.0
    print(f"  c-regular (6-cycle, c = 2): ||B - (cI - L - (c/p)11')|| = "
          f"{np.linalg.norm(modularity_matrix(C6u) - (c * np.eye(6) - laplacian(C6u) - c / 6 * np.ones((6, 6)))):.1e}")
    T = np.zeros((6, 6))
    for a_, b_ in [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (2, 5)]:
        T[a_, b_] = T[b_, a_] = 1.0
    d = T.sum(axis=1)
    LT, BT = laplacian(T), modularity_matrix(T)
    v = np.linalg.eigh(LT)[1][:, 1]
    print(f"  triangle plus tails, d = {d.astype(int).tolist()}:")
    print(f"    combinatorial: Fiedler vector of L against B, ||Bv - (v'Bv)v|| = "
          f"{np.linalg.norm(BT @ v - (v @ BT @ v) * v):.2f}   (0 = eigenvector)")
    Dm = np.diag(d ** -0.5)
    Ls, Bn = np.eye(6) - Dm @ T @ Dm, Dm @ BT @ Dm
    u = np.linalg.eigh(Ls)[1][:, 1]
    print(f"    normalised: Fiedler vector of L_sym against B_n, residual "
          f"{np.linalg.norm(Bn @ u - (u @ Bn @ u) * u):.1e}")
    print("    eig(L_sym) = " + ", ".join(f"{x:.3f}" for x in eig(Ls)))
    print("    eig(B_n)   = " + ", ".join(f"{x:.3f}" for x in eig(Bn)[::-1]))
    _, Ub = top_eig(BT, 1)
    same = abs(np.sign(v) @ np.sign(Ub[:, 0])) == len(v)
    print(f"    sign split of L's Fiedler vector vs B's top eigenvector: "
          f"{'same partition' if same else 'different partitions'}")

    # ------------------------------------------------------------------ G
    rule("G. Loosening the degree rule: Algorithm 1 (eta = 300) at weaker penalties rho")
    print(f"  {'rho':>5} {'deg range':>17} {'deg CV':>7} {'comp':>5} {'sizes':>14} {'iso':>4} "
          f"{'rel res':>8} {'mu_1, mu_2 of B':>16} {'Fan rhs':>8} {'sin':>6} "
          f"{'ARI comp':>8} {'ARI sec':>8} {'mu_1, mu_2 of B_n':>18}")
    for rho in RHOS:
        path = os.path.join(P1, "data", "alg1_L.npy") if rho == 100 else rho_cache(rho)
        m = part1_metrics(np.load(path), sector)
        print(f"  {rho:>5g} [{m['dmin']:7.4f},{m['dmax']:7.4f}] {m['cv']:7.4f} {m['n']:5d} "
              f"{str(m['sizes'][:3]):>14} {m['isolated']:4d} {m['rel']:8.4f} "
              f"{m['mu'][0]:7.4f},{m['mu'][1]:7.4f} {m['rhs']:8.4f} {m['sin']:6.4f} "
              f"{m['ari_comp']:8.3f} {m['ari_sec']:8.3f}   {m['mu_n'][0]:.6f},{m['mu_n'][1]:.6f}")
        if m["ari_lab"] < 1.0:
            lab = m["lab"]
            major = {c: pd.Series(sector[lab == c]).mode()[0] for c in np.unique(lab)}
            moved = [f"{tickers[i]} ({sector[i]} -> {major[lab[i]]})"
                     for i in range(len(lab)) if sector[i] != major[lab[i]]]
            print(f"        not with their sector: {', '.join(moved)}")
    print("  Fan rhs = (k-1) - sum of top k-1 eig(B); the Fan lhs is ~0 whenever the rank "
          "constraint holds. sin = largest principal angle between the bottom-k eigenspace "
          "of L and span(1, top k-1 of B). B_n = D^-1/2 B D^-1/2.")


def main():
    X, S, sector = load_sp()
    p = S.shape[0]
    rule(f"DATA  S&P three-sector panel: p = {p}, T = {len(X)}, "
         f"{X.index[0].date()} to {X.index[-1].date()}")
    print("  sector counts:", pd.Series(sector).value_counts().to_dict())

    if "--part1" in sys.argv:
        part1(S, sector, list(X.columns))
        print("\nDone.")
        return

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

    part1(S, sector, list(X.columns))
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
