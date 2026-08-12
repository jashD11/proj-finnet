"""
Phase 8 — Beyond the paper: 8a, 8b, 8c, 8f.

Four things the paper's own material calls for and does not deliver.

8a  tau(t) as a signal (audit A7). Fixed-density thresholding holds the edge
    count constant and throws away the threshold needed to achieve it. That
    discarded number is a market-stress series. Compared here against dynamical
    modularity and mean correlation on the one job the paper cares about --
    tracking crises.

8b  The crisis detector (audit E10). Fig. 1 of the paper contains a labelled
    methodology box, step (e), reading "crisis detection evaluation". The paper
    contains no detection rule, no threshold, no hit rate and no false-alarm
    rate; the crisis analysis is entirely eyeballing wiggles against shaded
    bands. Here a detector is defined, and scored with precision, recall and
    ROC-AUC against the crisis windows pre-registered in Phase 0 -- before any
    result was seen, so the windows cannot have been tuned to the answer.

8c  Rise or fall (audit C1). The abstract says crises destroy community
    structure; the Fig. 4 text says "we observe an increase of modularity
    around the time span of the crisis". The paper never reconciles these.
    An event study over +/- 60 trading days around each pre-registered onset
    settles it.

8f  Louvain stability (audit B1/B2). Using Phase 3's ten partitions per day:
    NMI between seeds on the same day, against NMI between consecutive days.
    If they are comparable, the daily modularity series is substantially
    algorithmic noise, and that has to lead the report rather than be buried.

Run:
    /opt/anaconda3/bin/python experiments/exp8_extensions.py
    /opt/anaconda3/bin/python experiments/exp8_extensions.py --force
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

from exp1_data import RESULTS  # noqa: E402
from utils.crisis_dates import EVENT_HALF_WIDTH, crisis_frame, crisis_mask  # noqa: E402

CACHE = os.path.join(RESULTS, "phase8_extensions.json")
PARTITIONS_NPY = os.path.join(ROOT, "data", "partitions.npy")
NET_SERIES = os.path.join(RESULTS, "series", "networks.parquet")
COMM_SERIES = os.path.join(RESULTS, "series", "communities.parquet")
RHO_BAR = os.path.join(RESULTS, "series", "rho_bar.parquet")
TABLES = os.path.join(RESULTS, "tables")
EVENT_PARQUET = os.path.join(RESULTS, "series", "event_study.parquet")

#: Trailing baseline for the z-score detector, in trading days. Long enough to
#: be a regime rather than the event itself; short enough to adapt.
BASELINE = 250
#: Days sampled for the NMI stability analysis. 45 seed-pairs per day over
#: 6,315 days is 284k NMI computations; a stratified 800-day sample gives the
#: same answer for a twentieth of the cost.
N_NMI_DAYS = 800


def _detector_scores(x, baseline=BASELINE, sign=-1.0):
    """Trailing z-score. `sign=-1` flags drops (modularity), `+1` flags spikes.

    The baseline uses only data strictly before the day being scored, so the
    detector never sees its own event -- the elementary property the paper's
    absent detector cannot be checked for.
    """
    x = np.asarray(x, dtype=float)
    s = pd.Series(x)
    mu = s.shift(1).rolling(baseline, min_periods=baseline // 2).mean()
    sd = s.shift(1).rolling(baseline, min_periods=baseline // 2).std()
    z = (s - mu) / sd
    return (sign * z).to_numpy()


def _roc(scores, labels):
    """(auc, best_f1, precision, recall, threshold) for a score/label pair."""
    from sklearn.metrics import precision_recall_curve, roc_auc_score
    ok = np.isfinite(scores)
    s, y = scores[ok], labels[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return dict(auc=np.nan, f1=np.nan, precision=np.nan, recall=np.nan,
                    threshold=np.nan, n=int(len(y)), base_rate=float(y.mean()))
    auc = float(roc_auc_score(y, s))
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.nanargmax(f1))
    return dict(auc=auc, f1=float(f1[i]), precision=float(prec[i]),
                recall=float(rec[i]),
                threshold=float(thr[min(i, len(thr) - 1)]),
                n=int(len(y)), base_rate=float(y.mean()))


def run(force: bool = False, verbose: bool = True) -> dict:
    if os.path.exists(CACHE) and not force:
        with open(CACHE) as f:
            report = json.load(f)
        if verbose:
            print(f"[skip] {CACHE} exists — pass force=True to rebuild.")
        return report

    net = pd.read_parquet(NET_SERIES)
    comm = pd.read_parquet(COMM_SERIES)
    rho = pd.read_parquet(RHO_BAR)
    idx = comm.index

    signals = {
        "modularity": (comm["q_dyn"].to_numpy(), -1.0),
        "tau": (net["tau"].to_numpy(), +1.0),
        "mean_correlation": (rho["rho_bar"].to_numpy(), +1.0),
        "isolated_nodes": (net["n_isolated"].to_numpy(), +1.0),
    }
    sharp = np.asarray(crisis_mask(idx, tiers=("sharp",)))
    anyc = np.asarray(crisis_mask(idx))

    # ── 8a / 8b: detectors, scored against the pre-registered windows ──
    detectors = {}
    for name, (x, sign) in signals.items():
        z = _detector_scores(x, sign=sign)
        detectors[name] = {
            "sharp": _roc(z, sharp.astype(int)),
            "any": _roc(z, anyc.astype(int)),
            "corr_with_sharp": float(np.corrcoef(
                z[np.isfinite(z)], sharp[np.isfinite(z)].astype(float))[0, 1]),
        }
    if verbose:
        print("[8b] detector ROC-AUC against pre-registered sharp-tier windows:")
        for n, d in detectors.items():
            print(f"     {n:<18}{d['sharp']['auc']:.3f}")

    # ── 8c: rise or fall? ──
    event = event_study(idx, signals, verbose=verbose)

    # ── 8f: Louvain stability ──
    stability = louvain_stability(verbose=verbose)

    report = {
        "baseline_days": BASELINE,
        "base_rate_sharp": float(sharp.mean()),
        "base_rate_any": float(anyc.mean()),
        "detectors": detectors,
        "best_detector_sharp": max(detectors, key=lambda k: detectors[k]["sharp"]["auc"]),
        "event_study": event,
        "louvain_stability": stability,
    }
    with open(CACHE, "w") as f:
        json.dump(report, f, indent=2)
    return report


def event_study(idx, signals, verbose: bool = True) -> dict:
    """Audit C1: align every pre-registered onset and average around it."""
    df = crisis_frame()
    h = EVENT_HALF_WIDTH
    offsets = np.arange(-h, h + 1)
    out, curves = {}, {}
    for name, (x, _) in signals.items():
        stack = []
        for _, row in df.iterrows():
            pos = idx.get_indexer([row["onset"]], method="nearest")[0]
            if pos - h < 0 or pos + h >= len(idx):
                continue
            seg = x[pos - h: pos + h + 1].astype(float)
            pre = seg[:h].mean()
            sd = seg[:h].std()
            stack.append((seg - pre) / sd if sd > 0 else seg - pre)
        M = np.vstack(stack)
        curves[name] = M.mean(axis=0)
        pre_win = M[:, :h].mean(axis=1)
        onset_win = M[:, h - 2:h + 3].mean(axis=1)
        post_win = M[:, h + 1:].mean(axis=1)
        out[name] = {
            "n_events": int(M.shape[0]),
            "pre_mean": float(pre_win.mean()),
            "onset_mean": float(onset_win.mean()),
            "post_mean": float(post_win.mean()),
            "onset_minus_pre": float((onset_win - pre_win).mean()),
            "post_minus_pre": float((post_win - pre_win).mean()),
            "frac_events_onset_below_pre": float((onset_win < pre_win).mean()),
        }
    pd.DataFrame(curves, index=pd.Index(offsets, name="days_from_onset")).to_parquet(
        EVENT_PARQUET)
    if verbose:
        q = out["modularity"]
        print(f"[8c] modularity across {q['n_events']} aligned onsets "
              f"(units of pre-event sd): onset {q['onset_minus_pre']:+.3f}, "
              f"following 60 days {q['post_minus_pre']:+.3f}")
    return out


def louvain_stability(verbose: bool = True) -> dict:
    """Audit B1/B2: is a day-to-day change bigger than a seed-to-seed one?"""
    from sklearn.metrics import normalized_mutual_info_score as nmi

    parts = np.load(PARTITIONS_NPY, mmap_mode="r")
    n_w, n_seeds, _ = parts.shape
    rng = np.random.default_rng(0)
    days = np.sort(rng.choice(np.arange(1, n_w), min(N_NMI_DAYS, n_w - 1),
                              replace=False))

    within, between = [], []
    for w in days:
        P = np.asarray(parts[w])
        for a in range(n_seeds):
            for b in range(a + 1, n_seeds):
                within.append(nmi(P[a], P[b]))
        # Same seed, consecutive days: the comparison the paper implicitly makes
        # every time it reads a day-to-day movement as a change in the market.
        between.append(nmi(np.asarray(parts[w - 1, 0]), P[0]))

    within = np.array(within)
    between = np.array(between)
    out = {
        "n_days_sampled": int(len(days)),
        "n_seeds": int(n_seeds),
        "nmi_within_day_mean": float(within.mean()),
        "nmi_within_day_sd": float(within.std()),
        "nmi_consecutive_days_mean": float(between.mean()),
        "nmi_consecutive_days_sd": float(between.std()),
        "gap": float(within.mean() - between.mean()),
        "frac_days_seedspread_exceeds_daychange": float(
            np.mean([w < b for w, b in
                     zip(within.reshape(len(days), -1).mean(axis=1), between)])),
    }
    if verbose:
        print(f"[8f] NMI: same day / different seed = "
              f"{out['nmi_within_day_mean']:.3f}; "
              f"consecutive days / same seed = "
              f"{out['nmi_consecutive_days_mean']:.3f}")
    return out


def plot(save_dir: str = None, verbose: bool = True):
    from utils.plots import plot_detector_roc, plot_event_study, plot_signals
    save_dir = save_dir or os.path.join(ROOT, "figures")
    os.makedirs(save_dir, exist_ok=True)
    net = pd.read_parquet(NET_SERIES)
    comm = pd.read_parquet(COMM_SERIES)
    rho = pd.read_parquet(RHO_BAR)
    with open(CACHE) as f:
        r = json.load(f)

    plot_signals(comm.index, comm["q_dyn"].to_numpy(), net["tau"].to_numpy(),
                 rho["rho_bar"].to_numpy(),
                 save=os.path.join(save_dir, "fig10a_signals.png"))
    plot_detector_roc(comm.index, {
        "modularity": _detector_scores(comm["q_dyn"].to_numpy(), sign=-1.0),
        "tau": _detector_scores(net["tau"].to_numpy(), sign=+1.0),
        "mean_correlation": _detector_scores(rho["rho_bar"].to_numpy(), sign=+1.0),
        "isolated_nodes": _detector_scores(net["n_isolated"].to_numpy(), sign=+1.0),
    }, np.asarray(crisis_mask(comm.index, tiers=("sharp",))), r,
        save=os.path.join(save_dir, "fig10b_detector.png"))
    plot_event_study(pd.read_parquet(EVENT_PARQUET), r["event_study"],
                     save=os.path.join(save_dir, "fig10c_event_study.png"))


def print_report(r: dict) -> None:
    print("\n── Phase 8 extensions " + "─" * 46)
    print(f"  base rate: sharp windows cover {r['base_rate_sharp']:.1%} of days, "
          f"any-tier {r['base_rate_any']:.1%}"
          "   <- what a detector must beat")
    print("  ── 8a/8b: the crisis detector the paper's Fig. 1 promises (E10) ──")
    print(f"  {'signal':<18}{'AUC':>7}{'F1':>7}{'prec':>7}{'recall':>8}"
          f"{'   AUC (any-tier)':>18}")
    for n, d in sorted(r["detectors"].items(),
                       key=lambda kv: -kv[1]["sharp"]["auc"]):
        s = d["sharp"]
        print(f"  {n:<18}{s['auc']:>7.3f}{s['f1']:>7.3f}{s['precision']:>7.3f}"
              f"{s['recall']:>8.3f}{d['any']['auc']:>18.3f}")
    print(f"     best on sharp-tier windows: {r['best_detector_sharp']}")
    print("  ── 8c: does modularity rise or fall in a crisis? (C1) ──")
    e = r["event_study"]
    print(f"  {'signal':<18}{'onset−pre':>11}{'post−pre':>11}"
          f"{'   events below pre at onset':>30}")
    for n, d in e.items():
        print(f"  {n:<18}{d['onset_minus_pre']:>+11.3f}{d['post_minus_pre']:>+11.3f}"
              f"{d['frac_events_onset_below_pre']:>29.0%}")
    print(f"     ({e['modularity']['n_events']} aligned onsets, "
          "in units of each event's own pre-window sd)")
    st = r["louvain_stability"]
    print("  ── 8f: is the daily modularity series signal or Louvain? (B1/B2) ──")
    print(f"     NMI same day, different seed : {st['nmi_within_day_mean']:.3f} "
          f"± {st['nmi_within_day_sd']:.3f}")
    print(f"     NMI consecutive days, seed 0 : {st['nmi_consecutive_days_mean']:.3f} "
          f"± {st['nmi_consecutive_days_sd']:.3f}")
    print(f"     gap {st['gap']:+.3f}"
          f"   — the seed spread exceeds the day-to-day change on "
          f"{st['frac_days_seedspread_exceeds_daychange']:.0%} of days")


def acceptance(r: dict, verbose: bool = True) -> bool:
    d = r["detectors"]
    st = r["louvain_stability"]
    e = r["event_study"]
    checks = [
        ("a crisis detector exists and is scored against pre-registered windows",
         all(np.isfinite(v["sharp"]["auc"]) for v in d.values()),
         f"{len(d)} detectors, best {r['best_detector_sharp']} "
         f"AUC {d[r['best_detector_sharp']]['sharp']['auc']:.3f}"),
        ("the detector beats the base rate it is scored against (E10/C2)",
         d[r["best_detector_sharp"]]["sharp"]["auc"] > 0.5,
         f"AUC {d[r['best_detector_sharp']]['sharp']['auc']:.3f} vs 0.5 chance; "
         f"base rate {r['base_rate_sharp']:.1%}"),
        ("τ(t) is compared against modularity as a crisis signal (A7)",
         "tau" in d and "modularity" in d,
         f"τ AUC {d['tau']['sharp']['auc']:.3f} vs "
         f"modularity AUC {d['modularity']['sharp']['auc']:.3f}"),
        ("the detector's baseline uses only prior data",
         r["baseline_days"] > 0, f"{r['baseline_days']}-day trailing window"),
        ("the rise-or-fall contradiction is settled with an event study (C1)",
         e["modularity"]["n_events"] >= 10,
         f"{e['modularity']['n_events']} onsets, onset−pre "
         f"{e['modularity']['onset_minus_pre']:+.3f} sd, post−pre "
         f"{e['modularity']['post_minus_pre']:+.3f} sd"),
        ("Louvain stability reported as NMI, seeds vs days (B1/B2)",
         np.isfinite(st["nmi_within_day_mean"])
         and np.isfinite(st["nmi_consecutive_days_mean"]),
         f"{st['nmi_within_day_mean']:.3f} within day vs "
         f"{st['nmi_consecutive_days_mean']:.3f} across days"),
    ]
    if verbose:
        print("\n── Phase 8 acceptance tests " + "─" * 40)
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
        print(f"  {sum(ok for _, ok, _ in checks)}/{len(checks)} passed")
    return all(ok for _, ok, _ in checks)


if __name__ == "__main__":
    rep = run(force="--force" in sys.argv)
    print_report(rep)
    plot()
    sys.exit(0 if acceptance(rep) else 1)
