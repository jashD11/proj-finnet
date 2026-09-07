"""
Build the modularity-matrix quantity menu on Paper 1's FAAMUNG dataset.

Re-runs Algorithm 2 (the exact exp3/exp4 configuration) to recover the 243
learned Laplacians L_t that Paper 1 discards after extracting lambda2, then
recovers the weighted adjacency A_t = -offdiag(L_t) and computes every
modularity-matrix-derived quantity we want to test as a lambda2 substitute.

Nothing in paper1_gmrf_laplacian is modified; this reads its data and solver.

Output: series.parquet / series.csv, indexed by the window-close date.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(os.path.dirname(HERE), "paper1_gmrf_laplacian")
sys.path.insert(0, P1)

from solver.algorithm2 import algorithm2  # noqa: E402

# ── Paper 1's exp3/exp4 constants, copied verbatim ────────────────────────────
WINDOW = 30
K = 1
ETA = 0.0
BETA = 0.0
DELTA = 100.0
DEGREE_CONTROL = False
# ──────────────────────────────────────────────────────────────────────────────

EPS = 1e-12


def adjacency_from_laplacian(L):
    """A = -offdiag(L), symmetrized and clipped at 0 (P2 guarantees offdiag<=0)."""
    L = (L + L.T) / 2.0
    A = -L.copy()
    np.fill_diagonal(A, 0.0)
    return np.clip(A, 0.0, None)


def spectral_quantities(A):
    """Every candidate indicator for one window."""
    n = A.shape[0]
    d = A.sum(axis=1)
    two_m = d.sum()
    dbar = d.mean()

    # --- Laplacian baselines (what Paper 1 uses) ---
    L = np.diag(d) - A
    evl = np.linalg.eigvalsh((L + L.T) / 2.0)
    lam2 = evl[1]
    lam_max = evl[-1]

    # normalized Laplacian
    dinv = 1.0 / np.sqrt(np.maximum(d, EPS))
    Dm = np.diag(dinv)
    Lsym = np.eye(n) - Dm @ A @ Dm
    evls = np.linalg.eigvalsh((Lsym + Lsym.T) / 2.0)
    lam2_sym = evls[1]

    # --- modularity matrix B = A - dd'/2m ---
    B = A - np.outer(d, d) / two_m
    B = (B + B.T) / 2.0
    evb, vecb = np.linalg.eigh(B)

    # --- normalized modularity matrix B_n = D^-1/2 B D^-1/2 ---
    Bn = Dm @ B @ Dm
    Bn = (Bn + Bn.T) / 2.0
    evbn, vecbn = np.linalg.eigh(Bn)

    pos = evb[evb > 0]
    neg = evb[evb < 0]
    absev = np.abs(evb)
    p = absev / max(absev.sum(), EPS)
    p = p[p > 0]

    v1 = vecb[:, -1]
    vN = vecb[:, 0]

    out = {
        # ---- Laplacian reference block ----
        "lam2": lam2,                       # Paper 1's indicator
        "lam_max": lam_max,
        "lam2_sym": lam2_sym,
        "dbar": dbar,
        "two_m": two_m,
        "deg_cv": d.std() / max(dbar, EPS),

        # ---- modularity matrix, eigenvalue extremes ----
        "mu1": evb[-1],                     # leading modularity eigenvalue
        "mu2": evb[-2],
        "muN": evb[0],                      # most negative ("anti-community")
        "mu1_norm": evbn[-1],               # leading eigenvalue of B_n
        "muN_norm": evbn[0],

        # ---- gaps and spreads ----
        "gap12": evb[-1] - evb[-2],
        "spread": evb[-1] - evb[0],
        "mu_ratio": abs(evb[0]) / max(evb[-1], EPS),
        "absmax": max(evb[-1], abs(evb[0])),

        # ---- bulk / aggregate summaries ----
        "n_pos": float((evb > 1e-9).sum()),
        "trace_pos": pos.sum() if pos.size else 0.0,
        "trace_neg": neg.sum() if neg.size else 0.0,
        "energy": absev.sum(),                       # modularity "energy"
        "frob2": float((B * B).sum()),               # tr(B^2), closed form
        "spec_entropy": float(np.exp(-(p * np.log(p)).sum())),
        "q_bound": (pos.sum() if pos.size else 0.0) / max(two_m, EPS),

        # ---- eigenvector (localization) quantities ----
        "ipr1": float((v1 ** 4).sum()),              # leading-vector localization
        "iprN": float((vN ** 4).sum()),
        "v1_deg_align": float(abs(v1 @ (np.sqrt(np.maximum(d, EPS)) /
                                        np.linalg.norm(np.sqrt(np.maximum(d, EPS)))))),

        # ---- explicit Fiedler proxies read off B ----
        "dbar_minus_mu1": dbar - evb[-1],            # = lam2 exactly if regular
        "one_minus_mu1n": 1.0 - evbn[-1],            # = lam2_sym exactly, always
        "dual_resid": dbar - (lam2 + evb[-1]),       # regular-graph duality gap
    }
    return out


def main():
    df = pd.read_csv(os.path.join(P1, "data", "faamung.csv"),
                     index_col=0, parse_dates=True)
    returns = df.values
    dates = df.index
    print(f"FAAMUNG: {returns.shape[0]} days x {returns.shape[1]} stocks "
          f"({dates[0].date()} -> {dates[-1].date()})")

    graphs, day_indices = algorithm2(
        returns, window=WINDOW, k=K, eta=ETA, beta=BETA, delta=DELTA,
        degree_control=DEGREE_CONTROL, verbose=False,
    )
    print(f"Algorithm 2 produced {len(graphs)} rolling graphs.")

    rows = [spectral_quantities(adjacency_from_laplacian(L)) for L in graphs]
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(dates[day_indices], name="date"))
    out.insert(0, "day_index", np.asarray(day_indices, dtype=int))

    # --- verification against Paper 1's cached artifact ---
    cached = np.load(os.path.join(P1, "data", "lam2_series.npy"))
    err = np.abs(out["lam2"].values - cached).max()
    print(f"CHECK  max |recomputed lam2 - data/lam2_series.npy| = {err:.3e}")
    assert err < 1e-8, "lambda2 does not reproduce Paper 1's cached series"

    # --- verification of the exact normalized identity ---
    ident = np.abs(out["mu1_norm"].values - (1.0 - out["lam2_sym"].values)).max()
    print(f"CHECK  max |mu1(B_n) - (1 - lam2(L_sym))|          = {ident:.3e}")

    out.to_parquet(os.path.join(HERE, "series.parquet"))
    out.to_csv(os.path.join(HERE, "series.csv"), float_format="%.10g")
    print(f"\nWrote series.parquet / series.csv  ({out.shape[0]} rows x "
          f"{out.shape[1]} cols)")
    print(out.describe().T[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
