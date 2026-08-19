#!/usr/bin/env python
"""Verify every number quoted in reports/discrepancy_explained.tex.

Each check does two things:

  1. recomputes the value from the stored artifacts (the probe's parquet and
     JSONs, and the Paper-3 replication's phase results), and
  2. confirms that the string as *printed* in the .tex matches that value at
     the precision it is printed to, and actually occurs in the source.

So a number cannot drift in either direction: editing the prose without
rerunning the probe fails, and rerunning the probe without updating the prose
fails too.

    /opt/anaconda3/bin/python reports/probe/check_numbers.py

Exit status is 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "reports" / "probe"
PAPER3 = ROOT / "paper3_market_modularity"
TEX = ROOT / "reports" / "discrepancy_explained.tex"

N_STOCKS = 360
DELTA_T = 30
EDGE_FRACTION = 0.10


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
def load():
    src = {}
    src["spec"] = pd.read_parquet(PROBE / "spectral_probe.parquet")
    src["auc"] = json.loads((PROBE / "spectral_probe_auc.json").read_text())
    src["dual"] = json.loads((PROBE / "duality_auc.json").read_text())
    src["incr"] = json.loads((PROBE / "incremental_auc.json").read_text())
    for name, rel in (("phase4", "results/phase4_nulls.json"),
                      ("phase8", "results/phase8_extensions.json")):
        path = PAPER3 / rel
        src[name] = json.loads(path.read_text()) if path.exists() else None
    src["tex"] = TEX.read_text()
    return src


def zscore(s: pd.Series, window: int = 250, min_periods: int = 125) -> pd.Series:
    """The probe's detector transform: trailing z on strictly prior data."""
    roll = s.shift(1).rolling(window, min_periods=min_periods)
    return (s - roll.mean()) / roll.std()


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def build_checks(src):
    """Return a list of (section, label, printed, actual, decimals)."""
    d = src["spec"]
    auc = src["auc"]
    dual = src["dual"]
    incr = src["incr"]
    checks = []

    def add(section, label, printed, actual, decimals):
        checks.append((section, label, printed, actual, decimals))

    # ---- construction constants (S2) -------------------------------------
    n_pairs = N_STOCKS * (N_STOCKS - 1) // 2
    n_edges = round(EDGE_FRACTION * n_pairs)
    add("S2", "pairwise correlations per day", "64,620", n_pairs, 0)
    add("S2", "edges per day", "6,462", n_edges, 0)
    add("S2", "2m", "12,924", 2 * n_edges, 0)
    add("S2", "mean degree", "35.90", 2 * n_edges / N_STOCKS, 2)
    add("S2", "daily networks", "6,315", len(d), 0)
    add("S2", "entries of B", "129,600", N_STOCKS ** 2, 0)

    # ---- the configuration-model worked examples (S3) --------------------
    add("S3", "expected edges, two degree-36 stocks", "0.100", 36 * 36 / (2 * n_edges), 3)
    add("S3", "expected edges, two degree-100 stocks", "0.774", 100 * 100 / (2 * n_edges), 3)

    # ---- Figure 3 medians -------------------------------------------------
    add("F3", "median # zero eigenvalues of L", "4", d["lap_n_zero"].median(), 0)
    add("F3", "median lambda_max(L)", "133", d["lap_lambda_max"].median(), 0)
    add("F3", "median mu_1(B)", "24.4", d["mod_lambda1"].median(), 1)
    add("F3", "median mu_N(B)", "-10.3", d["mod_lambda_min"].median(), 1)
    add("F3", "median #{mu>0}", "156", d["mod_n_pos"].median(), 0)

    # ---- the ruler (S4) ---------------------------------------------------
    add("S4", "sharp base rate", "2.68", 100 * auc["base_rate_sharp"], 2)
    add("S4", "sharp crisis days", "169", round(auc["base_rate_sharp"] * len(d)), 0)
    add("S4", "Q reproduced to 10 dp", "0.5380684370",
        auc["detectors"]["q_dyn"]["auc_sharp"], 10)
    add("S4", "tau reproduced to 10 dp", "0.8071316467",
        auc["detectors"]["tau"]["auc_sharp"], 10)

    # ---- the leaderboard (Table 3 / Figure 4) -----------------------------
    leaderboard = {
        "spectral entropy of C": ("corr_spec_entropy", "0.822"),
        "absorption ratio":      ("corr_lambda1_frac", "0.812"),
        "tau(t)":                ("tau", "0.807"),
        "mu_N(B)":               ("mod_lambda_min", "0.760"),
        "# components":          ("lap_n_zero", "0.756"),
        "sum of positive mu":    ("mod_trace_pos", "0.750"),
        "lambda_max(L)":         ("lap_lambda_max", "0.731"),
        "#{mu>0}":               ("mod_n_pos", "0.722"),
        "mu_1 - mu_2":           ("mod_gap12", "0.636"),
        "# modes above noise":   ("corr_n_group_modes", "0.619"),
        "lambda_2(L)":           ("lap_lambda2", "0.617"),
        "mu_1(B)":               ("mod_lambda1", "0.607"),
        "Q(t)":                  ("q_dyn", "0.538"),
        "max Laplacian gap":     ("lap_maxgap", "0.526"),
    }
    for label, (key, printed) in leaderboard.items():
        add("T3", f"AUC {label}", printed, auc["detectors"][key]["auc_sharp"], 3)
    add("T3", "AUC largest-piece share", "0.757", dual["lcc_frac"]["auc_sharp"], 3)
    add("T3", "AUC lambda_2 on largest piece", "0.652", dual["lcc_lambda2"]["auc_sharp"], 3)

    # ---- detectability (S6) ----------------------------------------------
    mp_plus = (1 + np.sqrt(N_STOCKS / DELTA_T)) ** 2
    mp_minus = (1 - np.sqrt(N_STOCKS / DELTA_T)) ** 2
    add("S6", "MP upper edge", "19.93", mp_plus, 2)
    add("S6", "MP lower edge", "6.07", mp_minus, 2)
    add("S6", "MP upper edge, as stored", "19.93", auc["mp_lambda_plus"], 2)
    add("S6", "N / Delta t", "12", N_STOCKS / DELTA_T, 0)
    add("S6", "max rank of C", "29", DELTA_T - 1, 0)
    add("S6", "forced-zero eigenvalues", "331", N_STOCKS - (DELTA_T - 1), 0)
    modes = d["corr_n_group_modes"]
    add("S6", "mean escaping modes", "0.70", modes.mean(), 2)
    add("S6", "max escaping modes", "3", modes.max(), 0)
    for k, printed in ((0, "2,668"), (1, "2891"), (2, "712"), (3, "44")):
        add("S6", f"days with {k} escaping modes", printed, (modes == k).sum(), 0)
    add("S6", "median lambda_1(C)", "70", (d["corr_lambda1_frac"] * N_STOCKS).median(), 0)

    if src["phase4"] is not None:
        ne = src["phase4"]["noise_excess"]
        add("S6", "market Q", "0.2215", src["phase4"]["real"]["q_mean"], 4)
        add("S6", "market's own null", "0.1065", src["phase4"]["configuration_model"]["q_mean"], 4)
        add("S6", "market excess", "0.1150", ne["real_excess"], 4)
        add("S6", "noise Q (phase 4 run)", "0.2253", ne["q_noise"], 4)
        add("S6", "noise's own null", "0.1332", ne["q_noise_null"], 4)
        add("S6", "noise excess", "0.0921", ne["noise_excess"], 4)
        add("S6", "share of excess explained", "80", 100 * ne["frac_of_real_excess_explained"], 0)
        add("S6", "noise transitivity", "0.219", ne["noise_transitivity"], 3)

    if src["phase8"] is not None:
        ls = src["phase8"]["louvain_stability"]
        add("S6", "NMI within day", "0.543", ls["nmi_within_day_mean"], 3)
        add("S6", "NMI consecutive days", "0.463", ls["nmi_consecutive_days_mean"], 3)
        add("S6", "NMI gap", "0.079", ls["gap"], 3)

    # ---- the duality (S7 / Table 4) --------------------------------------
    d_bar = 2 * n_edges / N_STOCKS
    total = d["lap_lambda2"] + d["mod_lambda1"]
    resid = d_bar - total
    connected = d[d["lap_n_zero"] == 1]
    resid_conn = d_bar - (connected["lap_lambda2"] + connected["mod_lambda1"])
    resid_lcc = d["lcc_dbar"] - (d["lcc_lambda2"] + d["lcc_mod_lambda1"])

    add("T4", "observed lambda_2 + mu_1", "26.411", total.mean(), 3)
    add("T4", "residual mean", "9.489", resid.mean(), 3)
    add("T4", "residual sd", "4.627", resid.std(), 3)
    add("T4", "residual mean, connected days", "9.210", resid_conn.mean(), 3)
    add("T4", "residual sd, connected days", "3.595", resid_conn.std(), 3)
    add("T4", "residual mean, largest component", "11.109", resid_lcc.mean(), 3)
    add("T4", "residual sd, largest component", "5.314", resid_lcc.std(), 3)
    add("T4", "corr(lambda_2, mu_1)", "-0.187", d["lap_lambda2"].corr(d["mod_lambda1"]), 3)
    add("T4", "corr, connected days", "-0.119",
        connected["lap_lambda2"].corr(connected["mod_lambda1"]), 3)
    add("T4", "corr, largest component", "-0.169",
        d["lcc_lambda2"].corr(d["lcc_mod_lambda1"]), 3)
    add("S7", "one-piece days", "2,342", int((d["lap_n_zero"] == 1).sum()), 0)
    add("S7", "fraction disconnected", "63", 100 * (d["lap_n_zero"] > 1).mean(), 0)
    add("S7", "residual as % of mean degree", "26", 100 * resid.mean() / d_bar, 0)

    # ---- out of sample (S8 / Table 5) -------------------------------------
    oos = {
        "market mode alone":   ("market_mode only", "0.812", "0.730"),
        "+ mu_1(B)":           ("+ lambda_1(B)", "0.822", "0.629"),
        "+ mu_N(B)":           ("+ lambda_min(B)", "0.810", "0.719"),
        "+ Q(t)":              ("+ Q_dyn (Louvain)", "0.831", "0.699"),
        "+ lambda_2 on LCC":   ("+ lambda_2(L) on LCC", "0.810", "0.700"),
        "+ whole spectrum":    ("market + all modularity spectrum", "0.823", "0.640"),
        "spectrum alone":      ("modularity spectrum only", "0.806", "0.685"),
        "Q(t) alone":          ("Q_dyn only", "0.538", "0.667"),
    }
    for label, (key, p_in, p_out) in oos.items():
        add("T5", f"in-sample, {label}", p_in, incr[key]["in_sample"], 3)
        add("T5", f"out-of-sample, {label}", p_out, incr[key]["oos_post2002"], 3)

    # ---- the market-mode correlations (S8.3) ------------------------------
    signs = {k: (1 if v["best_sign"] == "up" else -1)
             for k, v in auc["detectors"].items()}
    z = d.apply(zscore)
    for col, sign in signs.items():
        if col in z:
            z[col] = z[col] * sign
    market = "corr_lambda1_frac"
    add("S8", "corr(mu_N, market mode), z-scored", "0.680",
        z["mod_lambda_min"].corr(z[market]), 3)
    add("S8", "corr(mu_1, market mode), z-scored", "0.025",
        z["mod_lambda1"].corr(z[market]), 3)
    add("S8", "corr(mu_N, market mode), levels", "-0.819",
        d["mod_lambda_min"].corr(d[market]), 3)
    add("S8", "corr(mu_1, market mode), levels", "0.294",
        d["mod_lambda1"].corr(d[market]), 3)
    add("S8", "corr(entropy, absorption ratio), levels", "0.99",
        abs(d["corr_spec_entropy"].corr(d[market])), 2)

    return checks


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def tex_contains(tex: str, printed: str) -> bool:
    """Is `printed` present in the source, allowing LaTeX's own notations?"""
    variants = {
        printed,
        printed.replace(",", "{,}"),                       # 6{,}315
        printed.replace("-", "$-$"),
        printed.lstrip("-"),                               # signed values in tables
        printed.replace("-", r"\mathbf{-"),
    }
    stripped = printed.lstrip("-")
    variants.add(stripped.replace(",", "{,}"))
    return any(v in tex for v in variants)


def main() -> int:
    src = load()
    checks = build_checks(src)
    tex = src["tex"]

    width = max(len(label) for _, label, *_ in checks)
    failures = []
    missing = []
    section = None

    for sect, label, printed, actual, decimals in checks:
        if sect != section:
            print(f"\n  {sect}")
            section = sect
        try:
            expected = float(printed.replace(",", ""))
        except ValueError:
            expected = float("nan")
        got = round(float(actual), decimals)
        ok = abs(got - expected) < 0.5 * 10 ** (-decimals) + 1e-12
        in_tex = tex_contains(tex, printed)

        flag = "ok " if ok else "FAIL"
        note = "" if in_tex else "   [not found in .tex]"
        print(f"    {flag}  {label:<{width}}  printed {printed:>14}"
              f"   recomputed {got:>14}{note}")
        if not ok:
            failures.append((label, printed, got))
        if not in_tex:
            missing.append((label, printed))

    total = len(checks)
    print(f"\n  {total - len(failures)}/{total} values match the artifacts.")
    if missing:
        print(f"  {len(missing)} printed string(s) not located in the .tex:")
        for label, printed in missing:
            print(f"    - {label}: {printed}")
    if failures:
        print(f"  {len(failures)} MISMATCH(ES):")
        for label, printed, got in failures:
            print(f"    - {label}: document says {printed}, artifacts say {got}")
        return 1
    if missing:
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
