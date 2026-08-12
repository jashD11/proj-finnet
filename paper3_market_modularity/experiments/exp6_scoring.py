"""
Phase 6 — Scoring: the paper's numbers, then the honest ones.

This is where the paper's central claim is finally tested, and where the
replication's highest-value result lives.

The paper's evidence is Figs. 6 and S3: for each of eight measures and each of
two null models, an average squared difference and a Pearson correlation
between the real and generated series, split at 2002. Those 64 numbers are
reproduced here as-is, against the paper's own values extracted from the PDF.

Then four things the paper did not do:

  E2  normalize the squared differences by the variance of the real series,
      because 0.0003 and 2,522 are printed in the same table as if comparable;
  E1  recompute every correlation on non-overlapping windows. Consecutive
      networks share 29/30 of their data, so the 5,978 points behind each
      rho are roughly 200 independent observations. This is the primary
      evidence for the primary claim and it goes unadjusted in the paper;
  E4  put block-bootstrap confidence intervals on every rho, with block length
      at least the window length;
  E5  correlate the eight measures with each other, to see how many
      independent facts the "eight measurements" actually are;
  D1  check the compression arithmetic: k + k(k+1)/2 against N, overall and
      inside crisis windows, where B5 predicts it inverts.

Run:
    /opt/anaconda3/bin/python experiments/exp6_scoring.py
    /opt/anaconda3/bin/python experiments/exp6_scoring.py --force
"""

import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from exp1_data import (DELTA_T, FIT_SEED, N_BOOTSTRAP, RESULTS,  # noqa: E402
                       SPLIT_YEAR)
from net.measures import MEASURES  # noqa: E402
from utils.crisis_dates import crisis_mask  # noqa: E402
from utils.scoring import (block_bootstrap_ci, chi_square,  # noqa: E402
                           compression_cost, effective_sample_size,
                           inter_measure_correlation, nonoverlapping_slice,
                           normalized_mse, pearson)

CACHE = os.path.join(RESULTS, "phase6_scoring.json")
MEASURES_SERIES = os.path.join(RESULTS, "series", "measures.parquet")
COMM_SERIES = os.path.join(RESULTS, "series", "communities.parquet")
TABLES = os.path.join(RESULTS, "tables")

MODELS = {"comm": "Community", "conf": "Degree dist."}

#: The paper's own Fig. 6 / Fig. S3 values, transcribed from the PDF. Panel
#: order is the paper's own measurement list, confirmed against both captions.
#: Keyed [measure][model] -> {"chi2": (pre, post), "rho": (pre, post)}.
PAPER_TABLE = {
    "modularity":     {"comm": {"chi2": (0.0003, 0.0002), "rho": (0.99, 0.96)},
                       "conf": {"chi2": (0.0295, 0.0120), "rho": (0.07, 0.71)}},
    "path_length":    {"comm": {"chi2": (0.0189, 0.1152), "rho": (0.89, 0.72)},
                       "conf": {"chi2": (0.0333, 0.0688), "rho": (0.81, 0.76)}},
    "assortativity":  {"comm": {"chi2": (0.0120, 0.0259), "rho": (0.76, 0.53)},
                       "conf": {"chi2": (0.1012, 0.0730), "rho": (-0.47, 0.78)}},
    "transitivity":   {"comm": {"chi2": (0.0188, 0.1184), "rho": (0.97, 0.80)},
                       "conf": {"chi2": (0.0161, 0.0110), "rho": (0.92, 0.92)}},
    "betweenness":    {"comm": {"chi2": (419.0, 2522.0), "rho": (0.93, 0.95)},
                       "conf": {"chi2": (691.0, 1095.0), "rho": (0.77, 0.94)}},
    "clique_number":  {"comm": {"chi2": (191.0, 552.0), "rho": (0.80, 0.38)},
                       "conf": {"chi2": (203.0, 259.0), "rho": (0.74, 0.85)}},
    "rich_club":      {"comm": {"chi2": (0.017, 0.183), "rho": (0.86, 0.41)},
                       "conf": {"chi2": (0.014, 0.011), "rho": (0.93, 0.96)}},
    "matching_index": {"comm": {"chi2": (0.005, 0.020), "rho": (0.97, 0.80)},
                       "conf": {"chi2": (0.007, 0.005), "rho": (0.90, 0.94)}},
}


def _eras(index):
    pre = index.year < SPLIT_YEAR
    return {"pre": pre, "post": ~pre, "all": np.ones(len(index), dtype=bool)}


def run(force: bool = False, verbose: bool = True) -> dict:
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as f:
            report = json.load(f)
        if verbose:
            print(f"[skip] {CACHE} exists — pass force=True to rebuild.")
        return report

    s = pd.read_parquet(MEASURES_SERIES)
    comm = pd.read_parquet(COMM_SERIES)
    eras = _eras(s.index)
    n_w = len(s)
    non_ov = nonoverlapping_slice(n_w, DELTA_T)
    if verbose:
        print(f"[scoring] {len(MEASURES)} measures x {len(MODELS)} models x "
              f"{len(eras)} eras;  overlapping n = {n_w}, "
              f"non-overlapping n = {len(non_ov)} "
              f"(every {DELTA_T}th window)")

    rows = []
    for m in MEASURES:
        real = s[f"real_{m}"].to_numpy()
        for mod in MODELS:
            gen = s[f"{mod}_{m}_mean"].to_numpy()
            for era, mask in eras.items():
                r_e, g_e = real[mask], gen[mask]
                # E1: the same correlation on windows that share no data.
                sub = np.intersect1d(non_ov, np.flatnonzero(mask))
                rho_no = pearson(real[sub], gen[sub])
                # zlib.crc32, not hash(): Python salts string hashing per
                # process, so hash() here would silently make the bootstrap
                # irreproducible between runs.
                tag = zlib.crc32(f"{m}|{mod}|{era}".encode())
                boot, lo, hi = block_bootstrap_ci(
                    r_e, g_e, block=DELTA_T, n_boot=N_BOOTSTRAP,
                    seed=FIT_SEED + tag % 10_000)
                paper = PAPER_TABLE[m][mod]
                rows.append({
                    "measure": m, "model": mod, "era": era, "n": int(mask.sum()),
                    "n_independent": effective_sample_size(int(mask.sum()), DELTA_T),
                    "chi2": chi_square(r_e, g_e),
                    "nmse": normalized_mse(r_e, g_e),
                    "rho": pearson(r_e, g_e),
                    "rho_nonoverlapping": rho_no,
                    "rho_boot_lo": lo, "rho_boot_hi": hi,
                    "paper_chi2": paper["chi2"][0] if era == "pre"
                    else (paper["chi2"][1] if era == "post" else np.nan),
                    "paper_rho": paper["rho"][0] if era == "pre"
                    else (paper["rho"][1] if era == "post" else np.nan),
                })
    table = pd.DataFrame(rows)
    os.makedirs(TABLES, exist_ok=True)
    table.to_csv(os.path.join(TABLES, "fig6_scoring.csv"), index=False)

    # E5: how independent are the eight measures?
    names, C, e5 = inter_measure_correlation(
        {m: s[f"real_{m}"].to_numpy() for m in MEASURES})
    pd.DataFrame(C, index=names, columns=names).to_csv(
        os.path.join(TABLES, "inter_measure_correlation.csv"))

    # D1: the compression arithmetic.
    k = comm["k"].to_numpy()
    n_nodes = int(json.load(open(os.path.join(RESULTS, "phase2_networks.json")))["n_nodes"])
    cost, base, cheaper = compression_cost(k, n_nodes)
    crisis = np.asarray(crisis_mask(s.index, tiers=("sharp",)))
    any_crisis = np.asarray(crisis_mask(s.index))
    d1 = {
        "n_nodes": n_nodes,
        "community_cost_median": float(np.median(cost)),
        "community_cost_max": float(cost.max()),
        "config_cost": base,
        "frac_days_cheaper": float(cheaper.mean()),
        "frac_days_cheaper_sharp_crisis": float(cheaper[crisis].mean()),
        "frac_days_cheaper_any_crisis": float(cheaper[any_crisis].mean()),
        "break_even_k": float((-3 + np.sqrt(9 + 8 * n_nodes)) / 2),
        "k_median": float(np.median(k)), "k_max": int(k.max()),
    }

    report = {
        "n_windows": n_w, "delta_t": DELTA_T, "split_year": SPLIT_YEAR,
        "n_nonoverlapping": int(len(non_ov)),
        "n_bootstrap": N_BOOTSTRAP,
        "inter_measure": e5,
        "compression": d1,
        "overlap_correction": _overlap_summary(table),
        "vs_paper": _paper_agreement(table),
    }
    with open(CACHE, "w") as f:
        json.dump(report, f, indent=2)
    return report


def _overlap_summary(table) -> dict:
    """Audit E1, as one number and its worst cases."""
    t = table[table["era"] != "all"].copy()
    t["drop"] = t["rho"] - t["rho_nonoverlapping"]
    worst = t.loc[t["drop"].abs().nlargest(5).index]
    comm = t[t["model"] == "comm"]
    return {
        "mean_rho_overlapping": float(t["rho"].mean()),
        "mean_rho_nonoverlapping": float(t["rho_nonoverlapping"].mean()),
        "mean_drop": float(t["drop"].mean()),
        "max_drop": float(t["drop"].max()),
        "n_of_32_that_drop": int((t["drop"] > 0).sum()),
        "community_mean_rho": float(comm["rho"].mean()),
        "community_mean_rho_nonoverlapping": float(comm["rho_nonoverlapping"].mean()),
        "worst": [{"measure": r["measure"], "model": r["model"], "era": r["era"],
                   "rho": float(r["rho"]),
                   "rho_nonoverlapping": float(r["rho_nonoverlapping"]),
                   "drop": float(r["drop"])} for _, r in worst.iterrows()],
    }


#: A cell further than this from the paper's value is called not reproduced.
#: Deliberately generous -- our universe and period differ from the paper's, so
#: small gaps are expected; 0.30 in a correlation is not a small gap.
NOT_REPRODUCED_TOL = 0.30


def _paper_agreement(table) -> dict:
    t = table[table["era"] != "all"].dropna(subset=["paper_rho"]).copy()
    t["delta"] = t["rho"] - t["paper_rho"]
    d_rho = t["delta"].abs()
    bad = t[d_rho > NOT_REPRODUCED_TOL].sort_values("delta", key=abs, ascending=False)
    return {
        "n_cells": int(len(t)),
        "rho_mae_vs_paper": float(d_rho.mean()),
        "rho_within_0_10": float((d_rho <= 0.10).mean()),
        "rho_within_0_20": float((d_rho <= 0.20).mean()),
        "sign_agreement": float((np.sign(t["rho"]) == np.sign(t["paper_rho"])).mean()),
        "tolerance": NOT_REPRODUCED_TOL,
        "n_not_reproduced": int(len(bad)),
        "not_reproduced": [
            {"measure": r["measure"], "model": r["model"], "era": r["era"],
             "ours": float(r["rho"]), "paper": float(r["paper_rho"]),
             "delta": float(r["delta"])} for _, r in bad.iterrows()],
        # Our configuration null tracks the real networks more closely than the
        # paper's does on every measure. Recorded as a number so the pattern is
        # not left as an impression.
        "conf_mean_rho_ours": float(t[t["model"] == "conf"]["rho"].mean()),
        "conf_mean_rho_paper": float(t[t["model"] == "conf"]["paper_rho"].mean()),
    }


def load_table() -> pd.DataFrame:
    return pd.read_csv(os.path.join(TABLES, "fig6_scoring.csv"))


def plot(save_dir: str = None, verbose: bool = True):
    from utils.plots import (plot_compression, plot_inter_measure,
                             plot_overlap_correction)
    save_dir = save_dir or os.path.join(ROOT, "figures")
    os.makedirs(save_dir, exist_ok=True)
    t = load_table()
    plot_overlap_correction(t, save=os.path.join(save_dir, "fig8_overlap.png"))
    C = pd.read_csv(os.path.join(TABLES, "inter_measure_correlation.csv"), index_col=0)
    plot_inter_measure(C, save=os.path.join(save_dir, "fig8b_inter_measure.png"))
    comm = pd.read_parquet(COMM_SERIES)
    with open(CACHE) as f:
        d1 = json.load(f)["compression"]
    plot_compression(comm.index, comm["k"].to_numpy(), d1,
                     save=os.path.join(save_dir, "fig8c_compression.png"))


def print_report(r: dict) -> None:
    t = load_table()
    print("\n── Phase 6 scoring " + "─" * 49)
    print(f"  {r['n_windows']} overlapping windows → "
          f"{r['n_nonoverlapping']} non-overlapping (every {r['delta_t']}th); "
          f"split at {r['split_year']}; {r['n_bootstrap']} bootstrap resamples")

    print("\n  Fig. 6 / S3 reproduced — Pearson ρ, ours vs the paper's")
    print(f"  {'measure':<16}{'model':<7}{'era':<6}{'ours':>8}{'paper':>8}"
          f"{'  Δ':>7}{'   non-overlap':>15}{'   95% block CI':>20}")
    for _, x in t[t["era"] != "all"].iterrows():
        print(f"  {x['measure']:<16}{x['model']:<7}{x['era']:<6}"
              f"{x['rho']:>8.3f}{x['paper_rho']:>8.2f}"
              f"{x['rho'] - x['paper_rho']:>+7.2f}"
              f"{x['rho_nonoverlapping']:>15.3f}"
              f"     [{x['rho_boot_lo']:+.2f}, {x['rho_boot_hi']:+.2f}]")

    vp = r["vs_paper"]
    print(f"\n  agreement with the paper: MAE {vp['rho_mae_vs_paper']:.3f} over "
          f"{vp['n_cells']} cells; {vp['rho_within_0_10']:.0%} within 0.10, "
          f"{vp['rho_within_0_20']:.0%} within 0.20, "
          f"sign agrees {vp['sign_agreement']:.0%}")
    if vp.get("not_reproduced"):
        print(f"  NOT REPRODUCED — {vp['n_not_reproduced']} cells differ by more "
              f"than {vp['tolerance']:.2f}:")
        for c in vp["not_reproduced"]:
            print(f"     {c['measure']:<15}{c['model']:<6}{c['era']:<5}"
                  f"ours {c['ours']:+.3f}   paper {c['paper']:+.3f}   "
                  f"Δ {c['delta']:+.3f}")
    if "conf_mean_rho_ours" in vp:
        print(f"  our degree null tracks the real networks more closely than the "
              f"paper's: mean ρ {vp['conf_mean_rho_ours']:.3f} vs "
              f"{vp['conf_mean_rho_paper']:.3f} across all 16 configuration cells")

    print("\n  ── E2: the same table, normalized by the variance of the real series ──")
    print(f"  {'measure':<16}{'comm χ²':>12}{'comm NMSE':>11}"
          f"{'conf χ²':>12}{'conf NMSE':>11}")
    for m in MEASURES:
        c = t[(t["measure"] == m) & (t["model"] == "comm") & (t["era"] == "all")].iloc[0]
        f = t[(t["measure"] == m) & (t["model"] == "conf") & (t["era"] == "all")].iloc[0]
        print(f"  {m:<16}{c['chi2']:>12.4g}{c['nmse']:>11.3f}"
              f"{f['chi2']:>12.4g}{f['nmse']:>11.3f}")

    o = r["overlap_correction"]
    print("\n  ── E1: THE OVERLAP CORRECTION — the headline result ──")
    print(f"     every ρ above uses {r['n_windows']} points that share 29/30 of "
          f"their data.")
    print(f"     recomputed on the {r['n_nonoverlapping']} windows that share none:")
    print(f"       mean ρ over all 32 cells : {o['mean_rho_overlapping']:.3f} → "
          f"{o['mean_rho_nonoverlapping']:.3f}   ({o['mean_drop']:+.3f})")
    print(f"       community model only     : {o['community_mean_rho']:.3f} → "
          f"{o['community_mean_rho_nonoverlapping']:.3f}")
    print(f"       {o['n_of_32_that_drop']} of 32 cells fall; worst drop "
          f"{o['max_drop']:.3f}")
    for w in o["worst"][:3]:
        print(f"         {w['measure']:<15}{w['model']:<6}{w['era']:<5}"
              f"{w['rho']:.3f} → {w['rho_nonoverlapping']:.3f}")

    e5 = r["inter_measure"]
    print("\n  ── E5: how many independent facts are the eight measurements? ──")
    print(f"     mean |ρ| between measures {e5['mean_abs_offdiag']:.3f} "
          f"(max {e5['max_abs_offdiag']:.3f}); "
          f"{e5['n_components_for_90pct']} of 8 components carry 90 % of the variance")

    d1 = r["compression"]
    print("\n  ── D1: is the community summary actually a compression? ──")
    print(f"     community stores k + k(k+1)/2, median {d1['community_cost_median']:.0f} "
          f"numbers (max {d1['community_cost_max']:.0f}); "
          f"degree sequence stores {d1['config_cost']:.0f}")
    print(f"     break-even at k = {d1['break_even_k']:.1f}; "
          f"observed k median {d1['k_median']:.0f}, max {d1['k_max']}")
    print(f"     genuinely cheaper on {d1['frac_days_cheaper']:.1%} of all days, "
          f"{d1['frac_days_cheaper_sharp_crisis']:.1%} of sharp-crisis days, "
          f"{d1['frac_days_cheaper_any_crisis']:.1%} of any-crisis days")


def acceptance(r: dict, verbose: bool = True) -> bool:
    t = load_table()
    o, e5, d1 = r["overlap_correction"], r["inter_measure"], r["compression"]

    def cell(m, mod, era, col):
        sel = t[(t["measure"] == m) & (t["model"] == mod) & (t["era"] == era)]
        return float(sel.iloc[0][col])

    checks = [
        # These check that OUR pipeline behaves as it must. Agreement with the
        # paper is a *result*, recorded below and labelled in the report -- not
        # a pass/fail gate, because a genuine non-reproduction is a finding and
        # gating on it would only ever tempt us to bury it.
        ("community-model modularity ρ > 0.9 (the paper's headline reproduces)",
         cell("modularity", "comm", "pre", "rho") > 0.9,
         f"pre-{r['split_year']} ρ = {cell('modularity', 'comm', 'pre', 'rho'):.3f} "
         f"vs the paper's {PAPER_TABLE['modularity']['comm']['rho'][0]}"),
        ("the community null beats the degree null on modularity, as claimed",
         cell("modularity", "comm", "pre", "rho")
         > cell("modularity", "conf", "pre", "rho") + 0.3,
         f"community {cell('modularity', 'comm', 'pre', 'rho'):.3f} vs "
         f"configuration {cell('modularity', 'conf', 'pre', 'rho'):.3f}"),
        ("every cell that fails to reproduce the paper is enumerated, not hidden",
         "not_reproduced" in r["vs_paper"]
         and len(r["vs_paper"]["not_reproduced"]) == r["vs_paper"]["n_not_reproduced"],
         f"{r['vs_paper']['n_not_reproduced']} of {r['vs_paper']['n_cells']} cells "
         f"differ by more than {r['vs_paper']['tolerance']:.2f}"),
        ("overlapping AND non-overlapping ρ reported for every measure",
         int(t["rho"].notna().sum()) == len(t)
         and int(t["rho_nonoverlapping"].notna().sum()) == len(t),
         f"{len(t)} cells, both columns complete"),
        ("non-overlapping ρ really uses independent windows",
         r["n_nonoverlapping"] == int(np.ceil(r["n_windows"] / r["delta_t"])),
         f"{r['n_nonoverlapping']} windows, every {r['delta_t']}th"),
        ("block bootstrap uses blocks at least as long as the window",
         r["delta_t"] >= DELTA_T and int(t["rho_boot_lo"].notna().sum()) == len(t),
         f"block = {r['delta_t']}, {r['n_bootstrap']} resamples"),
        ("normalized MSE computed for every cell (E2)",
         int(t["nmse"].notna().sum()) == len(t),
         f"raw χ² spans {t['chi2'].min():.4g}–{t['chi2'].max():.4g}; "
         f"NMSE spans {t['nmse'].min():.3f}–{t['nmse'].max():.3f}"),
        ("inter-measure correlation matrix computed (E5)",
         e5["n_measures"] == len(MEASURES),
         f"mean |ρ| = {e5['mean_abs_offdiag']:.3f}, "
         f"{e5['n_components_for_90pct']}/8 components for 90 %"),
        ("compression check reports crisis days separately (D1/B5)",
         "frac_days_cheaper_sharp_crisis" in d1,
         f"{d1['frac_days_cheaper']:.1%} overall vs "
         f"{d1['frac_days_cheaper_sharp_crisis']:.1%} in sharp crises"),
    ]
    if verbose:
        print("\n── Phase 6 acceptance tests " + "─" * 40)
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
        print(f"  {sum(ok for _, ok, _ in checks)}/{len(checks)} passed")
    return all(ok for _, ok, _ in checks)


if __name__ == "__main__":
    rep = run(force="--force" in sys.argv)
    print_report(rep)
    plot()
    sys.exit(0 if acceptance(rep) else 1)
