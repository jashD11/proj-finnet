"""Check the Laplacian/modularity near-duality and the fragmentation confound.

For a connected d-regular graph:  lambda_2(L) = d - mu_2(A)  and  lambda_1(B) = mu_2(A),
so  lambda_2(L) + lambda_1(B) = d  exactly. The Silva construction fixes MEAN degree
(f(N-1) = 35.9) but not the degree sequence, so the residual measures heterogeneity.

Also recompute a fragmentation-free Fiedler value on the largest connected component.
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT = _os.path.join(_os.path.dirname(_os.path.dirname(HERE)), "paper3_market_modularity")
OUT = HERE
sys.path.insert(0, ROOT)
from utils.crisis_dates import crisis_mask                      # noqa: E402
from net.construct import decode_pairs                          # noqa: E402

df = pd.read_parquet(os.path.join(OUT, "spectral_probe.parquet"))
comm = pd.read_parquet(os.path.join(ROOT, "results/series/communities.parquet"))
net = pd.read_parquet(os.path.join(ROOT, "results/series/networks.parquet"))
edges = np.load(os.path.join(ROOT, "data/edges.npy"), mmap_mode="r")
N, n_w = 360, edges.shape[0]
d_bar = 2 * edges.shape[1] / N
print(f"mean degree f(N-1) = {d_bar:.3f}")

print("\n--- pairwise correlations of the spectral series ---")
sub = df[["lap_lambda2", "lap_maxgap", "lap_lambda_max", "lap_n_zero",
          "mod_lambda1", "mod_gap12", "mod_n_pos", "mod_lambda_min",
          "corr_lambda1_frac", "corr_spec_entropy"]].copy()
sub["q_dyn"] = comm["q_dyn"].to_numpy()[:len(df)]
sub["tau"] = net["tau"].to_numpy()[:len(df)]
print(sub.corr().round(3).to_string())

print("\n--- duality residual ---")
resid = d_bar - (df["lap_lambda2"] + df["mod_lambda1"])
print(f"d_bar - (lambda2(L) + lambda1(B)):  mean {resid.mean():.3f}  sd {resid.std():.3f}  "
      f"min {resid.min():.3f}  max {resid.max():.3f}")
conn = df["lap_n_zero"] == 1
print(f"connected days: {conn.sum()} / {len(df)} ({100*conn.mean():.1f}%)")
print(f"  residual on connected days: mean {resid[conn].mean():.3f}  sd {resid[conn].std():.3f}")
print(f"  corr(lambda2(L), lambda1(B))          all days: {df['lap_lambda2'].corr(df['mod_lambda1']):.3f}")
print(f"  corr(lambda2(L), lambda1(B))    connected only: {df.loc[conn,'lap_lambda2'].corr(df.loc[conn,'mod_lambda1']):.3f}")

# ── Fiedler value of the largest connected component (fragmentation removed) ──
print("\n--- recomputing LCC Fiedler value ---", flush=True)
lcc_l2 = np.full(n_w, np.nan)
lcc_frac = np.full(n_w, np.nan)
lcc_b1 = np.full(n_w, np.nan)
A = np.zeros((N, N))
for w in range(n_w):
    A[:] = 0.0
    pairs = decode_pairs(np.asarray(edges[w]), N)
    A[pairs[:, 0], pairs[:, 1]] = 1.0
    A[pairs[:, 1], pairs[:, 0]] = 1.0
    ncomp, lab = connected_components(csr_matrix(A), directed=False)
    big = np.argmax(np.bincount(lab))
    m = lab == big
    lcc_frac[w] = m.sum() / N
    As = A[np.ix_(m, m)]
    ds = As.sum(axis=1)
    ev = np.clip(eigvalsh(np.diag(ds) - As), 0.0, None)
    lcc_l2[w] = ev[1] if len(ev) > 1 else np.nan
    Bs = As - np.outer(ds, ds) / ds.sum()
    lcc_b1[w] = eigvalsh(Bs)[-1]
    if (w + 1) % 1000 == 0:
        print(f"  {w+1}/{n_w}", flush=True)

df["lcc_lambda2"] = lcc_l2
df["lcc_frac"] = lcc_frac
df["lcc_mod_lambda1"] = lcc_b1
df["lcc_dbar"] = 2 * edges.shape[1] / (lcc_frac * N)
df.to_parquet(os.path.join(OUT, "spectral_probe.parquet"))

r2 = df["lcc_dbar"] - (df["lcc_lambda2"] + df["lcc_mod_lambda1"])
print(f"\nLCC: corr(lambda2, lambda1(B)) = {df['lcc_lambda2'].corr(df['lcc_mod_lambda1']):.3f}")
print(f"LCC duality residual: mean {r2.mean():.3f}  sd {r2.std():.3f}")
print(f"lcc_lambda2: mean {np.nanmean(lcc_l2):.3f} sd {np.nanstd(lcc_l2):.3f} "
      f"min {np.nanmin(lcc_l2):.3f} max {np.nanmax(lcc_l2):.3f}")

# ── rescore ──
from sklearn.metrics import roc_auc_score  # noqa: E402
def z(x, sign, baseline=250):
    s = pd.Series(np.asarray(x, float))
    mu = s.shift(1).rolling(baseline, min_periods=baseline // 2).mean()
    sd = s.shift(1).rolling(baseline, min_periods=baseline // 2).std()
    return (sign * (s - mu) / sd).to_numpy()

sharp = np.asarray(crisis_mask(df.index, tiers=("sharp",))).astype(int)
out = {}
for c in ["lcc_lambda2", "lcc_mod_lambda1", "lcc_frac"]:
    best = -1; bs = None
    for sg in (+1.0, -1.0):
        s = z(df[c].to_numpy(), sg); ok = np.isfinite(s)
        a = float(roc_auc_score(sharp[ok], s[ok]))
        if a > best: best, bs = a, ("up" if sg > 0 else "down")
    out[c] = (bs, best)
    print(f"{c:<20}{bs:>6}{best:>10.3f}")
json.dump({k: {"sign": v[0], "auc_sharp": v[1]} for k, v in out.items()},
          open(os.path.join(OUT, "duality_auc.json"), "w"), indent=2)
