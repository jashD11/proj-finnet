"""
Phase 7 — PCA over the eight-measure space (Silva et al. 2015, Fig. 7).

The paper's closing evidence is a scatter plot. Each day becomes a point in the
8-dimensional measure space, PCA projects to two dimensions, and the three
families -- real networks, community null, configuration null -- are plotted as
three clouds. The text reads the picture as showing that the community model
agrees with the real networks.

Three things are missing from that reading, and this phase supplies them:

E7  The paper asserts "PCA1 is essentially time" and shows no loadings and no
    explained-variance ratios. Both are reported here, and "PCA1 is time" is
    turned into a number by correlating the PCA1 score against the date index.

E8  The three clouds visibly occupy *different* territory, and that is
    described as agreement. Eyeballing is replaced with two measurements:
    the distance between cloud centroids in units of the real cloud's own
    spread, and the cross-validated accuracy of a classifier trying to tell the
    families apart. A model indistinguishable from the real networks should be
    at chance -- 33 % for three classes.

All three families are projected into ONE space, fitted on the stacked matrix,
because projecting each family through its own PCA would guarantee overlapping
clouds regardless of whether the models agree with anything.

Run:
    /opt/anaconda3/bin/python experiments/exp7_pca.py
    /opt/anaconda3/bin/python experiments/exp7_pca.py --force
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp1_data import FIT_SEED, RESULTS  # noqa: E402
from net.measures import MEASURES  # noqa: E402

CACHE = os.path.join(RESULTS, "phase7_pca.json")
MEASURES_SERIES = os.path.join(RESULTS, "series", "measures.parquet")
TABLES = os.path.join(RESULTS, "tables")
SCORES_PARQUET = os.path.join(RESULTS, "series", "pca_scores.parquet")

FAMILIES = ("real", "comm", "conf")
FAMILY_LABEL = {"real": "inferred network", "comm": "community model",
                "conf": "configuration model"}


def _family_matrix(s, fam):
    cols = [f"real_{m}" if fam == "real" else f"{fam}_{m}_mean" for m in MEASURES]
    return s[cols].to_numpy(dtype=float)


def run(force: bool = False, verbose: bool = True) -> dict:
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as f:
            report = json.load(f)
        if verbose:
            print(f"[skip] {CACHE} exists — pass force=True to rebuild.")
        return report

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    s = pd.read_parquet(MEASURES_SERIES)
    mats = {f: _family_matrix(s, f) for f in FAMILIES}
    X = np.vstack([mats[f] for f in FAMILIES])
    y = np.concatenate([np.full(len(mats[f]), i) for i, f in enumerate(FAMILIES)])
    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    if verbose:
        print(f"[pca] {X.shape[0]:,} day-family points x {X.shape[1]} measures "
              f"({len(FAMILIES)} families x {len(s)} days)")

    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    pca = PCA(n_components=len(MEASURES), random_state=FIT_SEED).fit(Z)
    scores = pca.transform(Z)

    per_family = {}
    offset = 0
    score_frames = []
    for i, f in enumerate(FAMILIES):
        n = int((y == i).sum())
        sc = scores[y == i]
        per_family[f] = {
            "n": n,
            "centroid": [float(v) for v in sc[:, :2].mean(axis=0)],
            "spread_pc1": float(sc[:, 0].std()),
            "spread_pc2": float(sc[:, 1].std()),
        }
        score_frames.append(pd.DataFrame(
            {"family": f, "pc1": sc[:, 0], "pc2": sc[:, 1]}))
        offset += n
    pd.concat(score_frames, ignore_index=True).to_parquet(SCORES_PARQUET)

    # E7: "PCA1 is essentially time", as a number rather than an assertion.
    dates = s.index[ok[:len(s)]] if ok[:len(s)].all() else s.index
    t = np.arange(len(s), dtype=float)
    real_scores = scores[y == 0]
    n_real = min(len(t), len(real_scores))
    pc1_time = float(np.corrcoef(t[:n_real], real_scores[:n_real, 0])[0, 1])
    pc2_time = float(np.corrcoef(t[:n_real], real_scores[:n_real, 1])[0, 1])

    # E8: separation, measured two ways instead of eyeballed.
    c_real = np.array(per_family["real"]["centroid"])
    spread = np.array([per_family["real"]["spread_pc1"],
                       per_family["real"]["spread_pc2"]])
    separation = {}
    for f in ("comm", "conf"):
        d = np.array(per_family[f]["centroid"]) - c_real
        separation[f] = {
            "centroid_distance": float(np.hypot(*d)),
            "distance_in_real_sd": float(np.hypot(*(d / spread))),
        }

    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    acc = cross_val_score(clf, Z, y, cv=5, scoring="accuracy")
    clf2 = LogisticRegression(max_iter=2000)
    sel = y != 2                      # real vs community model only
    acc_rc = cross_val_score(clf2, Z[sel], y[sel], cv=5, scoring="accuracy")

    loadings = pd.DataFrame(pca.components_[:2].T, index=list(MEASURES),
                            columns=["PC1", "PC2"])
    os.makedirs(TABLES, exist_ok=True)
    loadings.to_csv(os.path.join(TABLES, "pca_loadings.csv"))

    report = {
        "n_points": int(X.shape[0]),
        "n_days": int(len(s)),
        "families": list(FAMILIES),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cumulative_variance": [float(v) for v in
                                np.cumsum(pca.explained_variance_ratio_)],
        "loadings_pc1": {m: float(v) for m, v in zip(MEASURES, pca.components_[0])},
        "loadings_pc2": {m: float(v) for m, v in zip(MEASURES, pca.components_[1])},
        "pc1_vs_time_corr": pc1_time,
        "pc2_vs_time_corr": pc2_time,
        "per_family": per_family,
        "separation": separation,
        "classifier": {
            "three_class_accuracy": float(acc.mean()),
            "three_class_sd": float(acc.std()),
            "chance_three_class": 1.0 / 3.0,
            "real_vs_community_accuracy": float(acc_rc.mean()),
            "real_vs_community_sd": float(acc_rc.std()),
            "chance_two_class": 0.5,
        },
    }
    with open(CACHE, "w") as f:
        json.dump(report, f, indent=2)
    return report


def plot(save_dir: str = None, verbose: bool = True):
    from utils.plots import plot_pca
    save_dir = save_dir or os.path.join(ROOT, "figures")
    os.makedirs(save_dir, exist_ok=True)
    sc = pd.read_parquet(SCORES_PARQUET)
    with open(CACHE) as f:
        r = json.load(f)
    loadings = pd.read_csv(os.path.join(TABLES, "pca_loadings.csv"), index_col=0)
    plot_pca(sc, r, loadings, save=os.path.join(save_dir, "fig9_pca.png"))


def print_report(r: dict) -> None:
    print("\n── Phase 7 PCA " + "─" * 53)
    ev = r["explained_variance_ratio"]
    print(f"  {r['n_points']:,} points ({len(r['families'])} families × "
          f"{r['n_days']} days) in {len(ev)} dimensions")
    print("  explained variance : " + "  ".join(f"PC{i+1} {v:.1%}"
                                                for i, v in enumerate(ev[:4])))
    print(f"     cumulative      : PC1+PC2 = {r['cumulative_variance'][1]:.1%}")
    print("  ── E7: the paper says PCA1 is essentially time. Measured: ──")
    print(f"     corr(PC1, date) = {r['pc1_vs_time_corr']:+.3f}    "
          f"corr(PC2, date) = {r['pc2_vs_time_corr']:+.3f}")
    print("  PC1 loadings:")
    for m, v in sorted(r["loadings_pc1"].items(), key=lambda kv: -abs(kv[1])):
        print(f"     {m:<16}{v:+.3f}")
    print("  ── E8: separation, measured instead of eyeballed ──")
    for f, d in r["separation"].items():
        print(f"     {FAMILY_LABEL[f]:<22} centroid {d['centroid_distance']:.3f} "
              f"from the real cloud = {d['distance_in_real_sd']:.2f} "
              "of its own sd")
    c = r["classifier"]
    print(f"     3-class classifier : {c['three_class_accuracy']:.1%} "
          f"± {c['three_class_sd']:.1%}   (chance {c['chance_three_class']:.1%})")
    print(f"     real vs community  : {c['real_vs_community_accuracy']:.1%} "
          f"± {c['real_vs_community_sd']:.1%}   (chance {c['chance_two_class']:.0%})"
          "   <- a model that truly matched would sit at chance")


def acceptance(r: dict, verbose: bool = True) -> bool:
    ev = r["explained_variance_ratio"]
    c = r["classifier"]
    checks = [
        ("explained-variance ratios reported and sum to 1 (E7)",
         abs(sum(ev) - 1.0) < 1e-6 and len(ev) == len(MEASURES),
         f"PC1 {ev[0]:.1%}, PC1+PC2 {r['cumulative_variance'][1]:.1%}"),
        ("full PC1 and PC2 loadings reported for all eight measures (E7)",
         len(r["loadings_pc1"]) == len(MEASURES) == len(r["loadings_pc2"]),
         f"{len(r['loadings_pc1'])} loadings, written to tables/pca_loadings.csv"),
        ("'PCA1 is time' turned into a number (E7)",
         np.isfinite(r["pc1_vs_time_corr"]),
         f"corr(PC1, date) = {r['pc1_vs_time_corr']:+.3f}"),
        ("cloud separation quantified by centroid distance (E8)",
         all("distance_in_real_sd" in v for v in r["separation"].values()),
         f"community {r['separation']['comm']['distance_in_real_sd']:.2f} sd, "
         f"configuration {r['separation']['conf']['distance_in_real_sd']:.2f} sd"),
        ("cloud separation quantified by a classifier (E8)",
         np.isfinite(c["three_class_accuracy"]),
         f"{c['three_class_accuracy']:.1%} vs {c['chance_three_class']:.1%} chance"),
        ("all three families projected into one shared PCA space",
         r["n_points"] >= 3 * r["n_days"] - 10,
         f"{r['n_points']:,} points from {r['n_days']} days x 3 families"),
    ]
    if verbose:
        print("\n── Phase 7 acceptance tests " + "─" * 40)
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
        print(f"  {sum(ok for _, ok, _ in checks)}/{len(checks)} passed")
    return all(ok for _, ok, _ in checks)


if __name__ == "__main__":
    rep = run(force="--force" in sys.argv)
    print_report(rep)
    plot()
    sys.exit(0 if acceptance(rep) else 1)
