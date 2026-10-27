"""
Part II perturbation bound for Problem 1, term by term, against what happens.

In edge weights, Problem 1 is  min_w <w, z(S)> + sparsity sum(w) - log gdet L(w),
with z_ij(S) = S_ii + S_jj - 2 S_ij. A market term l l' in S adds exactly
dz_ij = (l_i - l_j)^2 to z, so the "clean" problem learns from S - l l'.

Step 1 (strong convexity). If every Laplacian on the segment between the clean and
the observed solution has lambda_max <= M, then -log gdet is mu-strongly convex
there with mu = 2 / M^2 (since ||L(dw)||_F^2 >= 2 ||dw||^2), so
    ||w_obs - w_clean|| <= ||dz|| / mu,   ||L_obs - L_clean||_F <= sqrt(2p) ||w_obs - w_clean||.
Step 2 (Davis-Kahan, Yu-Wang-Samworth form). With gap = lambda_{k+1} - lambda_k of
L_clean, the bottom-k eigenspaces satisfy  ||sin Theta||_F <= 2 sqrt(k) ||dL||_2 / gap.

Usage:  python part2_bound.py      (prints one row per synthetic run, then the real panel)
"""

import numpy as np
from scipy.linalg import subspace_angles

from common import K, adjacency, load_sp
from part2 import (SYN_SIG, SYN_P, learn, load_index, loadings, sin2theta,
                   synthetic_betas, synthetic_returns, clusters)
from sklearn.metrics import adjusted_rand_score as ari

SEEDS = range(3)
DELTAS = (0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8)


def edge_vec(L):
    iu = np.triu_indices(L.shape[0], 1)
    return -L[iu]


def dz(l):
    iu = np.triu_indices(len(l), 1)
    return (l[iu[0]] - l[iu[1]]) ** 2


def bound_row(S, l, sector):
    p = S.shape[0]
    L_obs = learn(S, "prob1")
    L_cln = learn(S - np.outer(l, l), "prob1")
    d = dz(l)
    M = max(np.linalg.eigvalsh(L_obs)[-1], np.linalg.eigvalsh(L_cln)[-1])
    mu = 2.0 / M ** 2
    dw_bound = np.linalg.norm(d) / mu
    dw = np.linalg.norm(edge_vec(L_obs) - edge_vec(L_cln))
    dL = L_obs - L_cln
    dL_bound = np.sqrt(2 * p) * dw_bound
    lam = np.linalg.eigvalsh(L_cln)
    gap = lam[K] - lam[K - 1]
    _, Vo = np.linalg.eigh(L_obs)
    _, Vc = np.linalg.eigh(L_cln)
    sin_actual = np.linalg.norm(np.sin(subspace_angles(Vo[:, :K], Vc[:, :K])))
    sin_dk_actual_dL = 2 * np.sqrt(K) * np.linalg.norm(dL, 2) / gap
    sin_dk_full = 2 * np.sqrt(K) * dL_bound / gap
    return dict(dz=np.linalg.norm(d), M=M, dw=dw, dw_bound=dw_bound,
                dL=np.linalg.norm(dL, 2), dLF=np.linalg.norm(dL), dL_bound=dL_bound, gap=gap,
                sin=sin_actual, sin_dk=sin_dk_actual_dL, sin_full=sin_dk_full,
                ari_obs=ari(sector, clusters(L_obs, "prob1")),
                ari_cln=ari(sector, clusters(L_cln, "prob1")))


def population_loadings(beta):
    s2 = SYN_SIG
    return beta * s2["m"] / np.sqrt(beta ** 2 * s2["m"] ** 2 + s2["s"] ** 2 + s2["e"] ** 2)


def main():
    head = (f"{'delta':>5} {'sin2':>6} {'||dz||':>7} {'M':>6} {'|dw|':>7} {'bound':>8} "
            f"{'|dL|2':>7} {'bound':>8} {'gap':>7} {'sinTh':>6} {'DK(dL)':>7} {'DK full':>8} "
            f"{'ARI obs':>7} {'ARI cln':>7}")
    print("SYNTHETIC, cross pattern (betas cut across sectors), p = 60, population loadings")
    print(head)
    for delta in DELTAS:
        beta, sector = synthetic_betas("cross", delta)
        l = population_loadings(beta)
        rows = [bound_row(np.corrcoef(synthetic_returns("cross", delta, s)[0].T), l, sector)
                for s in SEEDS]
        r = {k: np.mean([row[k] for row in rows]) for k in rows[0]}
        print(f"{delta:5.2f} {sin2theta(l):6.3f} {r['dz']:7.3f} {r['M']:6.3f} {r['dw']:7.3f} "
              f"{r['dw_bound']:8.3f} {r['dL']:7.3f} {r['dL_bound']:8.2f} {r['gap']:7.4f} "
              f"{r['sin']:6.3f} {r['sin_dk']:7.2f} {r['sin_full']:8.1f} "
              f"{r['ari_obs']:7.2f} {r['ari_cln']:7.2f}")

    X, S, sector = load_sp()
    l = loadings(X.values, load_index().values)
    r = bound_row(S, l, sector)
    print(f"\nREAL S&P panel (p = 97), loadings corr(x_i, SPX), sin2 = {sin2theta(l):.3f}")
    print(head)
    print(f"{'':>5} {sin2theta(l):6.3f} {r['dz']:7.3f} {r['M']:6.3f} {r['dw']:7.3f} "
          f"{r['dw_bound']:8.3f} {r['dL']:7.3f} {r['dL_bound']:8.2f} {r['gap']:7.4f} "
          f"{r['sin']:6.3f} {r['sin_dk']:7.2f} {r['sin_full']:8.1f} "
          f"{r['ari_obs']:7.2f} {r['ari_cln']:7.2f}")


if __name__ == "__main__":
    main()
