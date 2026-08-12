"""
Phase 2 — Network construction (Section II of Silva et al. 2015).

Builds the full daily network sequence: rolling 30-day Pearson correlation,
thresholded each day so that exactly f = 10 % of pairs survive.

Beyond the paper, this phase records three things the construction throws away
or takes on faith:
    tau(t)            the threshold itself (audit A7)
    n_below_neg_tau   pairs the sign convention makes invisible (audit A8)
    edge weight spread   what binarization costs (audit A9)
and adds a pure-noise reference run (audit A4), because with 30 observations
and 360 series a correlation matrix is mostly estimation error, and the paper
never asks how much of its network is noise.

Run:
    /opt/anaconda3/bin/python experiments/exp2_networks.py
    /opt/anaconda3/bin/python experiments/exp2_networks.py --force
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

from exp1_data import (DATA_SEED, DELTA_T, EDGE_FRACTION, PANEL_CLEAN,  # noqa: E402
                       RESULTS, RETURNS_NPY, TIMESTAMP_CONVENTION)
from net.construct import (build_network_sequence, decode_pairs,  # noqa: E402
                           n_windows, target_edge_count, threshold_window,
                           window_correlation, window_timestamps)
from utils.crisis_dates import crisis_frame  # noqa: E402

CACHE = os.path.join(RESULTS, "phase2_networks.json")
EDGES_NPY = os.path.join(ROOT, "data", "edges.npy")
SERIES_PARQUET = os.path.join(RESULTS, "series", "networks.parquet")

# Windows used for the crisis acceptance tests. Black Monday is the paper's
# showcase (Fig. 3); autumn 2008 is the other event our sample fully covers.
BLACK_MONDAY = ("1987-10-14", "1987-11-13")
AUTUMN_2008 = ("2008-09-15", "2008-10-31")
FIG3_PANEL = ("1987-08-01", "1988-03-31")     # Fig. 3's plotted range


def _load_returns():
    Y = np.load(RETURNS_NPY)
    panel = pd.read_parquet(PANEL_CLEAN)
    return Y, list(panel.columns), panel.index


def run(force: bool = False, verbose: bool = True) -> dict:
    if os.path.exists(CACHE) and os.path.exists(EDGES_NPY) and not force:
        with open(CACHE) as f:
            report = json.load(f)
        if verbose:
            print(f"[skip] {CACHE} exists — pass force=True to rebuild.")
        return report

    Y, tickers, dates = _load_returns()
    N = Y.shape[0]
    if verbose:
        print(f"[networks] {N} tickers, {Y.shape[1]} returns, "
              f"Δt = {DELTA_T}, f = {EDGE_FRACTION:.0%}")

    seq = build_network_sequence(Y, DELTA_T, EDGE_FRACTION, verbose=verbose)
    np.save(EDGES_NPY, seq["edges"])

    win_dates = window_timestamps(dates, DELTA_T, TIMESTAMP_CONVENTION)
    series = pd.DataFrame({
        "tau": seq["tau"],
        "n_isolated": seq["n_isolated"],
        "n_below_neg_tau": seq["n_below_neg_tau"],
        "edge_w_mean": seq["edge_w_mean"],
        "edge_w_min": seq["edge_w_min"],
        "edge_w_max": seq["edge_w_max"],
        "n_flat": seq["n_flat"],
    }, index=pd.DatetimeIndex(win_dates, name="date"))
    os.makedirs(os.path.dirname(SERIES_PARQUET), exist_ok=True)
    series.to_parquet(SERIES_PARQUET)

    noise = noise_reference(Y.shape, verbose=verbose)
    report = {
        "n_nodes": int(N),
        "n_windows": int(seq["n_windows"]),
        "n_edges_per_day": int(seq["n_edges"]),
        "delta_t": DELTA_T,
        "f": EDGE_FRACTION,
        "timestamp_convention": TIMESTAMP_CONVENTION,
        "mean_degree": 2 * seq["n_edges"] / N,
        "mean_degree_target": EDGE_FRACTION * (N - 1),
        "tau_mean": float(seq["tau"].mean()),
        "tau_min": float(seq["tau"].min()),
        "tau_max": float(seq["tau"].max()),
        "tau_argmax_date": str(pd.Timestamp(win_dates[int(seq["tau"].argmax())]).date()),
        "isolated_mean": float(seq["n_isolated"].mean()),
        "isolated_max": int(seq["n_isolated"].max()),
        "isolated_argmax_date": str(
            pd.Timestamp(win_dates[int(seq["n_isolated"].argmax())]).date()),
        "n_flat_total": int(seq["n_flat"].sum()),
        "edge_w_min_overall": float(seq["edge_w_min"].min()),
        "edge_w_max_overall": float(seq["edge_w_max"].max()),
        "n_below_neg_tau_total": int(seq["n_below_neg_tau"].sum()),
        "n_below_neg_tau_max": int(seq["n_below_neg_tau"].max()),
        "noise_reference": noise,
        "crisis_windows": crisis_tau_summary(series),
        "convention_fit": convention_fit(dates, series),
    }
    with open(CACHE, "w") as f:
        json.dump(report, f, indent=2)
    return report


def noise_reference(shape, verbose: bool = True) -> dict:
    """Audit A4: run the identical pipeline on pure i.i.d. Gaussian returns.

    With Delta_t = 30 observations and N series, a sample correlation has
    standard error ~1/sqrt(29) ~ 0.19 even when the true correlation is zero.
    Thresholding noise at the top 10 % therefore yields a perfectly full,
    perfectly plausible-looking network. The paper applies no random-matrix
    cleaning and never raises the issue; this quantifies what "no structure at
    all" looks like under its own construction, so later modularity and tau
    values can be read against it.

    Run on a 500-window slice: the reference is a distribution, not a series.
    """
    rng = np.random.default_rng(DATA_SEED)
    n_ref = 500
    fake = rng.standard_normal((shape[0], n_ref + DELTA_T))
    seq = build_network_sequence(fake, DELTA_T, EDGE_FRACTION, verbose=False)
    out = {
        "n_windows": int(seq["n_windows"]),
        "tau_mean": float(seq["tau"].mean()),
        "tau_std": float(seq["tau"].std()),
        "isolated_mean": float(seq["n_isolated"].mean()),
        "edge_w_mean": float(seq["edge_w_mean"].mean()),
        "seed": DATA_SEED,
    }
    if verbose:
        print(f"[noise A4] i.i.d. Gaussian, same shape: "
              f"τ = {out['tau_mean']:.3f} ± {out['tau_std']:.3f}, "
              f"isolated = {out['isolated_mean']:.1f}, "
              f"mean edge weight = {out['edge_w_mean']:.3f}")
    return out


def crisis_tau_summary(series: pd.DataFrame) -> dict:
    """Where tau and the isolate count sit during the two events we can test."""
    out = {}
    q = series["tau"].quantile
    for name, (a, b) in (("black_monday", BLACK_MONDAY), ("autumn_2008", AUTUMN_2008)):
        sel = series.loc[a:b]
        out[name] = {
            "tau_max": float(sel["tau"].max()),
            "tau_pctile": float((series["tau"] < sel["tau"].max()).mean()),
            "isolated_max": int(sel["n_isolated"].max()),
            "isolated_pctile": float(
                (series["n_isolated"] < sel["n_isolated"].max()).mean()),
        }
    out["tau_q95"] = float(q(0.95))
    out["isolated_q95"] = float(series["n_isolated"].quantile(0.95))
    return out


def convention_fit(price_dates, series: pd.DataFrame) -> dict:
    """Audit A10 tiebreaker: which stamp puts our peaks where Fig. 3 puts them.

    Fig. 3 plots Aug 1987 - Mar 1988 and shows modularity collapsing, with the
    graph shattering, at the crash. The isolate count is the sharpest proxy we
    have in Phase 2. Under each convention we ask on what calendar date our
    isolate peak lands, and compare against 1987-10-19.
    """
    from net.construct import window_timestamps as wts
    peak = int(series["n_isolated"].to_numpy().argmax())
    # Restrict to the Fig. 3 panel so a later crisis cannot win the argmax.
    fig3 = series.loc[FIG3_PANEL[0]:FIG3_PANEL[1]]
    local_peak_pos = int(series.index.get_indexer([fig3["n_isolated"].idxmax()])[0])
    out = {"global_isolate_peak_date": str(series.index[peak].date())}
    for conv in ("t1", "midpoint", "t2"):
        stamps = wts(price_dates, DELTA_T, conv)
        d = pd.Timestamp(stamps[local_peak_pos])
        out[conv] = {
            "isolate_peak_date": str(d.date()),
            "days_from_black_monday": int((d - pd.Timestamp("1987-10-19")).days),
        }
    return out


def plot(save_dir: str = None, verbose: bool = True):
    from utils.plots import plot_network_construction, plot_event_zoom
    save_dir = save_dir or os.path.join(ROOT, "figures")
    os.makedirs(save_dir, exist_ok=True)
    s = pd.read_parquet(SERIES_PARQUET)
    rho = pd.read_parquet(os.path.join(RESULTS, "series", "rho_bar.parquet"))
    plot_network_construction(
        s.index, s["tau"].to_numpy(), s["n_isolated"].to_numpy(),
        s["n_below_neg_tau"].to_numpy(), rho["rho_bar"].to_numpy(),
        stamp=TIMESTAMP_CONVENTION,
        save=os.path.join(save_dir, "fig2_networks.png"))
    plot_event_zoom(
        s.index,
        {"τ(t) — correlation threshold": s["tau"].to_numpy(),
         "isolated nodes": s["n_isolated"].to_numpy(),
         "mean pairwise correlation": rho["rho_bar"].to_numpy()},
        FIG3_PANEL[0], FIG3_PANEL[1],
        "Black Monday — the Fig. 3 window (Aug 1987 – Mar 1988)",
        onset="1987-10-19",
        save=os.path.join(save_dir, "fig2b_blackmonday.png"))


def print_report(r: dict) -> None:
    print("\n── Phase 2 networks " + "─" * 48)
    print(f"  nodes / windows       : {r['n_nodes']} / {r['n_windows']}")
    print(f"  edges per day         : {r['n_edges_per_day']} (constant by construction)")
    print(f"  mean degree           : {r['mean_degree']:.4f}"
          f"   target f(N-1) = {r['mean_degree_target']:.4f}"
          f"   (residual {r['mean_degree'] - r['mean_degree_target']:+.4f}, integer rounding)")
    print(f"  τ(t)                  : mean {r['tau_mean']:.3f}, "
          f"range {r['tau_min']:.3f}–{r['tau_max']:.3f}, peak {r['tau_argmax_date']}")
    print(f"  isolated nodes        : mean {r['isolated_mean']:.1f}, "
          f"max {r['isolated_max']} on {r['isolated_argmax_date']}")
    print(f"  edge weights kept     : {r['edge_w_min_overall']:.3f} … "
          f"{r['edge_w_max_overall']:.3f}   <- all become the same edge (A9)")
    print(f"  ρ < −τ pairs          : {r['n_below_neg_tau_total']:,} total, "
          f"max {r['n_below_neg_tau_max']} in a day   <- invisible by construction (A8)")
    n = r["noise_reference"]
    print(f"  A4 noise reference    : i.i.d. Gaussian gives τ = {n['tau_mean']:.3f}"
          f" ± {n['tau_std']:.3f}, {n['isolated_mean']:.1f} isolated nodes")
    c = r["crisis_windows"]
    for k in ("black_monday", "autumn_2008"):
        w = c[k]
        print(f"  {k:<21}: τ peaks {w['tau_max']:.3f} "
              f"({w['tau_pctile']:.1%} of days below it), "
              f"isolated {w['isolated_max']} ({w['isolated_pctile']:.1%} below)")
    cf = r["convention_fit"]
    print("  A10 convention fit (isolate peak in the Fig. 3 panel):")
    for conv in ("t1", "midpoint", "t2"):
        v = cf[conv]
        print(f"      {conv:<9}-> {v['isolate_peak_date']}"
              f"   {v['days_from_black_monday']:+d} days from 1987-10-19")


def acceptance(r: dict, verbose: bool = True) -> bool:
    Y, tickers, dates = _load_returns()
    N = Y.shape[0]

    # Edge count is invariant by construction; verify it on the stored array
    # rather than trusting the builder.
    edges = np.load(EDGES_NPY, mmap_mode="r")
    per_day_unique = len(set(np.unique(edges[i]).size for i in
                             range(0, edges.shape[0], 500)))

    # Independent spot-check of one window against scipy.
    from scipy.stats import pearsonr
    w = 1234
    C = window_correlation(Y, w, DELTA_T)
    errs = [abs(C[i, j] - pearsonr(Y[i, w:w + DELTA_T], Y[j, w:w + DELTA_T])[0])
            for i, j in ((0, 1), (7, 200), (N - 1, N - 2), (42, 99))]

    # And that the stored edge set really is the top-f% of that window.
    iu = np.triu_indices(N, 1)
    tau_w, pairs_w, _ = threshold_window(C, target_edge_count(N, EDGE_FRACTION), iu)
    stored = np.asarray(edges[w])
    c = r["crisis_windows"]

    checks = [
        ("edge count identical every day and equal to round(f·N(N−1)/2)",
         per_day_unique == 1 and edges.shape[1] == target_edge_count(N, EDGE_FRACTION),
         f"{edges.shape[1]} edges/day, mean degree {r['mean_degree']:.4f} "
         f"vs f(N−1) = {r['mean_degree_target']:.4f}"),
        ("N_w = C_p − Δt",
         r["n_windows"] == n_windows(len(dates), DELTA_T),
         f"{r['n_windows']}"),
        ("correlation matches scipy.stats.pearsonr to 1e-10",
         max(errs) < 1e-10, f"max error {max(errs):.2e}"),
        ("stored edge set reproduces an independent re-threshold of that window",
         np.array_equal(stored, pairs_w), f"window {w}, τ = {tau_w:.4f}"),
        ("τ spikes during Black Monday (top 5 % of all days)",
         c["black_monday"]["tau_pctile"] >= 0.95,
         f"τ = {c['black_monday']['tau_max']:.3f} at the "
         f"{c['black_monday']['tau_pctile']:.1%} percentile"),
        ("τ spikes during autumn 2008 (top 5 % of all days)",
         c["autumn_2008"]["tau_pctile"] >= 0.95,
         f"τ = {c['autumn_2008']['tau_max']:.3f} at the "
         f"{c['autumn_2008']['tau_pctile']:.1%} percentile"),
        ("isolated nodes spike in the Black Monday window (Fig. 3B signature)",
         c["black_monday"]["isolated_pctile"] >= 0.95,
         f"{c['black_monday']['isolated_max']} isolated, "
         f"{c['black_monday']['isolated_pctile']:.1%} percentile"),
        ("no window had a zero-variance series (Phase 1 gate held)",
         r["n_flat_total"] == 0, f"{r['n_flat_total']} flat rows"),
    ]
    if verbose:
        print("\n── Phase 2 acceptance tests " + "─" * 40)
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
        print(f"  {sum(ok for _, ok, _ in checks)}/{len(checks)} passed")
    return all(ok for _, ok, _ in checks)


if __name__ == "__main__":
    rep = run(force="--force" in sys.argv)
    print_report(rep)
    plot()
    sys.exit(0 if acceptance(rep) else 1)
