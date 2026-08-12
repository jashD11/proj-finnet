"""Does the modularity-matrix spectrum add anything beyond the market mode?

Market-mode family:  tau, lambda_1(C)/N, spectral entropy of C  (all pairwise |rho| > 0.99)
Modularity family:   lambda_1(B), Q_dyn                         (corr 0.761 with each other,
                                                                 ~0.3 with the market mode)
Test: logistic detector on the market-mode signal alone vs. market mode + modularity signal.
Reported both in-sample (upper bound) and with a strict chronological split.
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT = _os.path.join(_os.path.dirname(_os.path.dirname(HERE)), "paper3_market_modularity")
OUT = HERE
sys.path.insert(0, ROOT)
from utils.crisis_dates import crisis_mask  # noqa: E402

df = pd.read_parquet(os.path.join(OUT, "spectral_probe.parquet"))
comm = pd.read_parquet(os.path.join(ROOT, "results/series/communities.parquet"))
df["q_dyn"] = comm["q_dyn"].to_numpy()[:len(df)]

def z(x, sign, baseline=250):
    s = pd.Series(np.asarray(x, float))
    mu = s.shift(1).rolling(baseline, min_periods=baseline // 2).mean()
    sd = s.shift(1).rolling(baseline, min_periods=baseline // 2).std()
    return (sign * (s - mu) / sd).to_numpy()

Z = pd.DataFrame({
    "market_mode":  z(df["corr_lambda1_frac"], +1),
    "mod_lambda1":  z(df["mod_lambda1"],       +1),
    "mod_lambda_min": z(df["mod_lambda_min"],  -1),
    "q_dyn":        z(df["q_dyn"],             -1),
    "lcc_lambda2":  z(df["lcc_lambda2"],       -1),
}, index=df.index)
y = np.asarray(crisis_mask(df.index, tiers=("sharp",))).astype(int)

ok = Z.notna().all(axis=1).to_numpy()
Zc, yc, idxc = Z[ok], y[ok], df.index[ok]
print(f"usable days {ok.sum()} / {len(df)}; sharp base rate {yc.mean():.4f}")

def fit_auc(cols, tr, te):
    m = LogisticRegression(max_iter=2000, class_weight="balanced")
    m.fit(Zc.loc[tr, cols], yc[tr])
    return float(roc_auc_score(yc[te], m.predict_proba(Zc.loc[te, cols])[:, 1]))

allmask = np.ones(len(yc), bool)
split = idxc < pd.Timestamp("2002-01-01")
print(f"train (pre-2002) {split.sum()} days, {yc[split].sum()} sharp; "
      f"test (post-2002) {(~split).sum()} days, {yc[~split].sum()} sharp")

sets = {
    "market_mode only":                 ["market_mode"],
    "+ lambda_1(B)":                    ["market_mode", "mod_lambda1"],
    "+ lambda_min(B)":                  ["market_mode", "mod_lambda_min"],
    "+ Q_dyn (Louvain)":                ["market_mode", "q_dyn"],
    "+ lambda_2(L) on LCC":             ["market_mode", "lcc_lambda2"],
    "market + all modularity spectrum": ["market_mode", "mod_lambda1", "mod_lambda_min"],
    "modularity spectrum only":         ["mod_lambda1", "mod_lambda_min"],
    "Q_dyn only":                       ["q_dyn"],
}
rows = {}
print(f"\n{'feature set':<36}{'in-sample':>11}{'OOS post-2002':>15}")
for name, cols in sets.items():
    ins = fit_auc(cols, allmask, allmask)
    oos = fit_auc(cols, split, ~split)
    rows[name] = {"in_sample": ins, "oos_post2002": oos, "features": cols}
    print(f"{name:<36}{ins:>11.3f}{oos:>15.3f}")

json.dump(rows, open(os.path.join(OUT, "incremental_auc.json"), "w"), indent=2)

print("\n--- correlation between the two families (z-scored detectors) ---")
print(Zc.corr().round(3).to_string())
