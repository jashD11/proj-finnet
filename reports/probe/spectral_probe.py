"""
Feasibility probe: Laplacian spectrum vs modularity-matrix spectrum as crisis
signals, on the paper3 (Silva et al.) daily network sequence already built.

Reuses:  data/edges.npy, data/returns.npy, results/series/networks.parquet,
         results/series/communities.parquet, utils/crisis_dates.py
Protocol: identical to experiments/exp8_extensions.py (trailing 250-day z-score,
          ROC-AUC against the Phase-0 pre-registered sharp-tier windows).

Nothing here is written back into the repo; output goes to the scratchpad.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh

import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT = _os.path.join(_os.path.dirname(_os.path.dirname(HERE)), "paper3_market_modularity")
OUT = HERE
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from utils.crisis_dates import crisis_mask                      # noqa: E402
from net.construct import decode_pairs                          # noqa: E402

BASELINE = 250
DELTA_T = 30

net = pd.read_parquet(os.path.join(ROOT, "results/series/networks.parquet"))
comm = pd.read_parquet(os.path.join(ROOT, "results/series/communities.parquet"))
idx = comm.index
edges = np.load(os.path.join(ROOT, "data/edges.npy"), mmap_mode="r")
returns = np.load(os.path.join(ROOT, "data/returns.npy"))
N = returns.shape[0]
n_w = edges.shape[0]
print(f"N={N}  n_windows={n_w}  edges/day={edges.shape[1]}  index={len(idx)}", flush=True)

# ── Marchenko-Pastur edge for the correlation spectrum ────────────────────
def mp_edges(n, t):
    q = np.sqrt(n / t)
    return (1 - q) ** 2, (1 + q) ** 2

lam_minus_30, lam_plus_30 = mp_edges(N, DELTA_T)
print(f"MP band at dt=30, N={N}: [{lam_minus_30:.2f}, {lam_plus_30:.2f}]", flush=True)

cols = {k: np.full(n_w, np.nan) for k in [
    "lap_lambda2", "lap_maxgap", "lap_lambda_max", "lap_n_zero",
    "mod_lambda1", "mod_gap12", "mod_n_pos", "mod_lambda_min", "mod_trace_pos",
    "corr_lambda1_frac", "corr_n_group_modes", "corr_spec_entropy",
]}

def standardize(block):
    c = block - block.mean(axis=1, keepdims=True)
    nrm = np.sqrt((c ** 2).sum(axis=1))
    nrm = np.where(nrm == 0, 1.0, nrm)
    return c / nrm[:, None]

A = np.zeros((N, N))
for w in range(n_w):
    A[:] = 0.0
    pairs = decode_pairs(np.asarray(edges[w]), N)
    A[pairs[:, 0], pairs[:, 1]] = 1.0
    A[pairs[:, 1], pairs[:, 0]] = 1.0
    deg = A.sum(axis=1)
    two_m = deg.sum()

    # Laplacian
    L = np.diag(deg) - A
    ev = eigvalsh(L)
    ev = np.clip(ev, 0.0, None)
    cols["lap_lambda2"][w] = ev[1]
    cols["lap_lambda_max"][w] = ev[-1]
    cols["lap_n_zero"][w] = int((ev < 1e-8).sum())
    gaps = np.diff(ev[1:])                      # k = 2 .. N-1, per Kang et al.
    cols["lap_maxgap"][w] = gaps.max()

    # Modularity matrix  B = A - d d^T / 2m
    B = A - np.outer(deg, deg) / two_m
    evb = eigvalsh(B)
    cols["mod_lambda1"][w] = evb[-1]
    cols["mod_gap12"][w] = evb[-1] - evb[-2]
    cols["mod_n_pos"][w] = int((evb > 1e-9).sum())
    cols["mod_lambda_min"][w] = evb[0]
    cols["mod_trace_pos"][w] = evb[evb > 0].sum()

    # Correlation spectrum on the same window
    Z = standardize(returns[:, w:w + DELTA_T])
    C = Z @ Z.T
    evc = np.clip(eigvalsh(C), 0.0, None)
    cols["corr_lambda1_frac"][w] = evc[-1] / N
    cols["corr_n_group_modes"][w] = int(((evc > lam_plus_30) & (evc < evc[-1] - 1e-12)).sum())
    p = evc / evc.sum()
    p = p[p > 0]
    cols["corr_spec_entropy"][w] = float(np.exp(-(p * np.log(p)).sum()))

    if (w + 1) % 500 == 0:
        print(f"  {w+1}/{n_w}", flush=True)

df = pd.DataFrame(cols, index=idx[:n_w] if len(idx) >= n_w else idx)
df.to_parquet(os.path.join(OUT, "spectral_probe.parquet"))
print("spectra done", flush=True)

# ── Detector scoring, exp8 protocol ───────────────────────────────────────
from sklearn.metrics import roc_auc_score  # noqa: E402

def z(x, sign):
    s = pd.Series(np.asarray(x, float))
    mu = s.shift(1).rolling(BASELINE, min_periods=BASELINE // 2).mean()
    sd = s.shift(1).rolling(BASELINE, min_periods=BASELINE // 2).std()
    return (sign * (s - mu) / sd).to_numpy()

sharp = np.asarray(crisis_mask(df.index, tiers=("sharp",))).astype(int)
anyc = np.asarray(crisis_mask(df.index)).astype(int)

signals = {c: df[c].to_numpy() for c in df.columns}
signals["q_dyn"] = comm["q_dyn"].to_numpy()[:len(df)]
signals["tau"] = net["tau"].to_numpy()[:len(df)]

res = {}
for name, x in signals.items():
    row = {}
    for sign, tag in ((+1.0, "up"), (-1.0, "down")):
        s = z(x, sign)
        ok = np.isfinite(s)
        row[tag] = {
            "auc_sharp": float(roc_auc_score(sharp[ok], s[ok])),
            "auc_any": float(roc_auc_score(anyc[ok], s[ok])),
        }
    best = max(row, key=lambda k: row[k]["auc_sharp"])
    res[name] = {"best_sign": best, **row[best], "both": row}

with open(os.path.join(OUT, "spectral_probe_auc.json"), "w") as f:
    json.dump({"base_rate_sharp": float(sharp.mean()),
               "base_rate_any": float(anyc.mean()),
               "mp_lambda_plus": lam_plus_30, "detectors": res}, f, indent=2)

print(f"\nbase rate sharp = {sharp.mean():.4f}   any = {anyc.mean():.4f}")
print(f"{'signal':<24}{'sign':>6}{'AUC sharp':>12}{'AUC any':>10}")
for n, r in sorted(res.items(), key=lambda kv: -kv[1]["auc_sharp"]):
    print(f"{n:<24}{r['best_sign']:>6}{r['auc_sharp']:>12.3f}{r['auc_any']:>10.3f}")

print("\n--- descriptive ---")
print(df.describe().T[["mean", "std", "min", "max"]].to_string())
