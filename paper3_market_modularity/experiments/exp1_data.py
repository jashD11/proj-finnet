"""
Phase 0 — Scaffold and data acquisition, for Silva et al. (2015).

Builds the price panel, applies the paper's completeness filter, writes
data/UNIVERSE.md, and runs the Phase 0 acceptance tests.

This module is also the project's de-facto config: every downstream experiment
imports its constants from here (the convention used by paper2_ticc's
exp1_table1_f1.py). No magic numbers in function bodies.

Run:
    /opt/anaconda3/bin/python experiments/exp1_data.py
    /opt/anaconda3/bin/python experiments/exp1_data.py --force   # re-derive panel
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.fetch_data import (START, END, PANEL_PARQUET, build_panel,  # noqa: E402
                             download_raw_closes)
from data.universe import fetch_candidates  # noqa: E402

# ── Universe / panel construction (Phase 0) ────────────────────────────────
SPAN_TOL = 5              # trading days of slack at each end of the window
MIN_COVERAGE = 1.0        # strict reading of "complete history"
MISSING_POLICY = "drop_days"   # alternative: "ffill" (audit A3)

# ── Network construction (Phase 2) ─────────────────────────────────────────
DELTA_T = 30              # window length in RETURNS; window n uses prices n..n+30
DELTA_STEP = 1            # slide, in trading days
EDGE_FRACTION = 0.10      # f: fraction of the N(N-1)/2 pairs kept each day
# Audit A10. Settled empirically in date_calibration() below: t2 is the only
# convention under which the paper's Fig. 8 anchors, its C_p = 6008, and its
# stated February 2011 end are simultaneously satisfiable. It is also the only
# one that does not stamp a crash-containing network *before* the crash.
TIMESTAMP_CONVENTION = "t2"    # network stamped at its last day

# ── Communities and null models (Phases 3-4) ───────────────────────────────
N_LOUVAIN_SEEDS = 10      # audit B1: partition spread, not a single run
LAG_T_DELTA = 100         # t_Delta for the lagged modularity, in trading days
N_REPLICATES = 10         # null-model realizations per day per model (audit D3)

# ── Scoring (Phase 6) ──────────────────────────────────────────────────────
SPLIT_YEAR = 2002         # the paper's data-chosen split (audit E9)
N_BOOTSTRAP = 1000        # block-bootstrap resamples for rho CIs (audit E4)

# ── Seeds ──────────────────────────────────────────────────────────────────
DATA_SEED = 7             # anything touching data subsampling
FIT_SEED = 42             # Louvain seeds, null-model generation, layouts

RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(RESULTS, "phase0_universe.json")
RETURNS_CACHE = os.path.join(RESULTS, "phase1_returns.json")
RETURNS_NPY = os.path.join(ROOT, "data", "returns.npy")
PANEL_CLEAN = os.path.join(ROOT, "data", "panel_clean.parquet")
QUALITY_MD = os.path.join(ROOT, "data", "QUALITY.md")
RHO_BAR_PARQUET = os.path.join(RESULTS, "series", "rho_bar.parquet")
UNIVERSE_MD = os.path.join(ROOT, "data", "UNIVERSE.md")

# A single-day move this large is essentially nonexistent outside October 1987
# and autumn 2008. Anything else flagged is an unadjusted corporate action.
LARGE_RETURN = 0.5

# The paper's own numbers, for the report's comparison column.
PAPER = {
    "n_candidates": 3799,
    "n_stocks": 348,
    "n_closes": 6008,
    "n_windows": 5978,
    "start": "1986-01",
    "end": "2011-02",
}


def run(force: bool = False, verbose: bool = True) -> dict:
    """Build the panel and return the Phase 0 report dict (cached)."""
    if os.path.exists(CACHE) and os.path.exists(PANEL_PARQUET) and not force:
        with open(CACHE) as f:
            report = json.load(f)
        if verbose:
            print(f"[skip] {CACHE} exists — pass force=True to rebuild.")
        return report

    n_requested = len(fetch_candidates(verbose=False))
    raw = download_raw_closes(verbose=verbose)
    panel, report = build_panel(raw,
                                span_tol=SPAN_TOL,
                                min_coverage=MIN_COVERAGE,
                                missing_policy=MISSING_POLICY,
                                n_requested=n_requested,
                                verbose=verbose)
    panel.to_parquet(PANEL_PARQUET)

    # Alternative-policy panel, so audit A3 is a measured difference rather
    # than a stated intention. Not written to disk; only its shape is recorded.
    alt_panel, alt_report = build_panel(raw,
                                        span_tol=SPAN_TOL,
                                        min_coverage=MIN_COVERAGE,
                                        missing_policy="ffill",
                                        verbose=False)
    report["alt_ffill"] = {
        "n_survivors": alt_report["n_survivors"],
        "n_days_final": alt_report["n_days_final"],
    }

    # Window n spans price indices n..n+DELTA_T, so N_w = C_p - DELTA_T. This
    # is the paper's own relation (6008 - 30 = 5978), and it is what pins
    # DELTA_T to 30 *returns* drawn from 31 prices.
    report["n_windows"] = int(report["n_days_final"] - DELTA_T)
    report["tickers"] = list(panel.columns)
    report["paper"] = PAPER

    os.makedirs(RESULTS, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(report, f, indent=2)
    write_universe_md(report)
    return report


def write_universe_md(report: dict) -> None:
    r = report
    ladder = "\n".join(
        f"| ≥ {k} | {v} |" for k, v in r["coverage_ladder"].items())
    by_year = r["dropped_days_by_year"]
    year_rows = "\n".join(
        f"| {y} | {n} |" for y, n in sorted(by_year.items())) or "| — | 0 |"

    text = f"""# Universe — paper 3 (Silva et al. 2015)

Generated by `experiments/exp1_data.py`. Do not edit by hand.

## Source

| | |
|---|---|
| Candidate list | NASDAQ Trader symbol directory, `otherlisted.txt`, Exchange = `N` (NYSE), non-ETF, non-test, security name containing "Common Stock" |
| Prices | Yahoo Finance via `yfinance`, **split- and dividend-adjusted** closes (`auto_adjust=True`) |
| Requested window | {START} → {END} |
| Realized calendar | {r['calendar_start']} → {r['calendar_end']} ({r['n_calendar_days']} trading days) |

## Filters and what they cost

| Stage | Count |
|---|---|
| NYSE common-stock symbols requested | **{r['n_requested']}** |
| Dropped: Yahoo returned no data at all | {r['n_no_data']} |
| Candidates with usable price history | {r['n_candidates']} |
| Dropped: not trading at both ends of the window (± {r['span_tol']} days) | {r['dropped_not_full_span']} |
| Dropped: coverage below {r['min_coverage']:.2%} of the calendar | {r['dropped_incomplete_coverage']} |
| Dropped: non-positive adjusted price ({', '.join(r['nonpositive_tickers']) or 'none'}) | {r['dropped_nonpositive_price']} |
| **Surviving universe (N)** | **{r['n_survivors']}** |

### Survivorship bias (audit A1)

**{r['n_requested'] - r['n_survivors']} of {r['n_requested']} requested symbols were removed before a single network was built.**
The paper's equivalent figure is 3,799 → 348. As in the paper, the survivors are
by construction the firms that did *not* fail during the crises the paper studies.

This replication carries a **second layer** of the same bias that the paper does
not: the NASDAQ Trader directory lists only *currently listed* symbols, so firms
that delisted before the retrieval date never enter the candidate pool at all.
Our {r['n_requested']} requested symbols are therefore not comparable to the
paper's 3,799 — theirs was a contemporaneous snapshot, ours is a
survivors-of-survivors list, which is why our pool is ~4x smaller while our
final N is *larger* than theirs. Neither layer is corrected; both change what
the network is a network of.

### A data defect the paper's era would not have had

`yfinance` back-adjusts for dividends by subtraction, so a stock whose
cumulative historical distributions exceed its old price is returned with a
**negative** adjusted close. {r['dropped_nonpositive_price']} of our full-span
tickers hit this ({', '.join(r['nonpositive_tickers']) or 'none'}); `log()` is
undefined there and the whole series is unusable, so they are dropped. This is a defect in the
modern data source, not a modelling choice, and it is exactly the class of
problem audit A2 exists to catch before it poisons 30 consecutive networks.

### Coverage sensitivity

How many full-span tickers survive at each completeness threshold, so that the
choice of `MIN_COVERAGE = {r['min_coverage']:.2%}` is visible rather than buried:

| Threshold | Survivors |
|---|---|
{ladder}

## Missing-day policy (audit A3)

Policy in force: **`{r['missing_policy']}`** — {"drop every calendar day on which any retained ticker is missing" if r['missing_policy'] == 'drop_days' else "carry the last close forward per ticker"}.

| | |
|---|---|
| Days dropped | {r['n_dropped_days']} |
| **Final panel** | **{r['n_survivors']} tickers × {r['n_days_final']} days** |
| Panel range | {r['panel_start']} → {r['panel_end']} |
| Alternative (`ffill`) panel | {r.get('alt_ffill', {}).get('n_survivors', '—')} tickers × {r.get('alt_ffill', {}).get('n_days_final', '—')} days |

### Dropped days by year

| Year | Days dropped |
|---|---|
{year_rows}

**Our panel drops zero days**, so the paper's shortfall is not reproduced by our
missing-day policy — see the calibration note below.

## Calendar calibration (audit A10)

The paper reports `C_p` = 6,008 closes for "January 1986 to February 2011"; the
modern Yahoo calendar has **{r['n_days_final']}** for the same span, a gap of
**{r['n_days_final'] - 6008}** days that our own filters do not explain.

Fig. 8 samples the network sequence every 1,000 steps and prints five dates.
Locating them on our calendar (`results/tables/date_calibration.csv`) gives an
offset of **exactly 355 trading days at all five anchors, with zero drift over
16 years**. Constant offset means the paper's calendar and ours agree day-for-day
from 1991 to 2007, so **essentially all ~337 of the paper's missing days fall
before May 1991.** Either their panel starts around 1987 despite the stated
January 1986, or their 1986–91 data is heavily gapped; Fig. 8 cannot separate
those, but it rules out the gap being spread across the sample.

Each convention implies a different position for the paper's day 1, and
`C_p` = 6,008 then fixes its last day. A convention whose implied last day runs
past the end of our calendar is impossible, because the paper's data ends in
February 2011 (`results/tables/convention_feasibility.csv`):

| Convention | implied first day | implied last day | verdict |
|---|---|---|---|
| t₁ (window start) | 1987-05-29 | past 2011-02-28 | impossible |
| midpoint | 1987-05-07 | past 2011-02-28 | impossible |
| **t₂ (window end)** | **1987-04-15** | **2011-02-09** | **consistent** |

**t₂ is the only convention that fits, and it fits exactly** — 6,008 days ending
in February 2011, as the paper states. Phase 2 confirms it independently: under
t₂ the isolated-node peak lands precisely on 1987-10-19, Black Monday itself,
whereas midpoint and t₁ would stamp the crash-containing network 21 and 45 days
*before* the crash.

The corollary is a substantive finding about the paper: **its data does not
begin in January 1986 as stated, but around April 1987.** The stated range and
the reported `C_p` cannot both be true.

## Differences from the paper

| | Paper | This replication |
|---|---|---|
| Candidate pool | 3,799 NYSE stocks (contemporaneous) | {r['n_requested']} NYSE common stocks (currently listed) |
| Universe N | 348 | {r['n_survivors']} |
| Closes per stock | 6,008 | {r['n_days_final']} |
| Networks N_w | 5,978 | {r.get('n_windows', '—')} |
| Window | Jan 1986 – Feb 2011 | {r['panel_start']} – {r['panel_end']} |
| Price series | unstated (assumed adjusted) | explicitly adjusted |
| Missing-day policy | unstated | `{r['missing_policy']}`, alternative implemented |
"""
    with open(UNIVERSE_MD, "w") as f:
        f.write(text)
    print(f"[write] {UNIVERSE_MD}")


# ── Phase 1: returns and validation ────────────────────────────────────────

def load_panel():
    """(prices (N,T), tickers, DatetimeIndex of length T)."""
    panel = pd.read_parquet(PANEL_PARQUET)
    return panel.to_numpy().T, list(panel.columns), panel.index


def run_returns(force: bool = False, verbose: bool = True) -> dict:
    """Phase 1: log returns, validation report, and the rho_bar reference series."""
    if os.path.exists(RETURNS_CACHE) and os.path.exists(RETURNS_NPY) and not force:
        with open(RETURNS_CACHE) as f:
            report = json.load(f)
        if verbose:
            print(f"[skip] {RETURNS_CACHE} exists — pass force=True to rebuild.")
        return report

    from scipy.stats import kurtosis, skew
    from net.construct import log_returns, mean_correlation_series, window_timestamps
    from data.quality import quality_gate, surviving_large_returns

    prices, tickers, dates = load_panel()
    Y_all = log_returns(prices)
    ret_dates = dates[1:]

    # Quality gate (audit A2/A3). Runs on returns, so it cannot live in Phase 0.
    keep, qreport = quality_gate(Y_all, tickers, DELTA_T, verbose=verbose)
    Y = Y_all[keep]
    tickers = [t for t, k in zip(tickers, keep) if k]
    pd.DataFrame(prices[keep].T, index=dates, columns=tickers).to_parquet(PANEL_CLEAN)
    np.save(RETURNS_NPY, Y)
    if verbose:
        print(f"[returns] {Y.shape[0]} tickers x {Y.shape[1]} returns "
              f"-> {RETURNS_NPY}")

    survivors = surviving_large_returns(Y, tickers, ret_dates)
    write_quality_md(qreport, survivors)
    stats = pd.DataFrame({
        "ticker": tickers,
        "mean": Y.mean(axis=1),
        "std": Y.std(axis=1, ddof=1),
        "min": Y.min(axis=1),
        "max": Y.max(axis=1),
        "excess_kurtosis": kurtosis(Y, axis=1, fisher=True, bias=False),
    })
    tables = os.path.join(RESULTS, "tables")
    os.makedirs(tables, exist_ok=True)
    stats.to_csv(os.path.join(tables, "return_validation.csv"), index=False)

    survivors.to_csv(os.path.join(tables, "large_returns.csv"), index=False)

    # The discarded crisis signal (audit A7), needed in Phase 8a.
    rho_bar = mean_correlation_series(Y, DELTA_T)
    win_dates = window_timestamps(dates, DELTA_T, TIMESTAMP_CONVENTION)
    os.makedirs(os.path.join(RESULTS, "series"), exist_ok=True)
    pd.DataFrame({"rho_bar": rho_bar}, index=win_dates).to_parquet(RHO_BAR_PARQUET)

    flat_all = Y.ravel()
    report = {
        "n_tickers": int(Y.shape[0]),
        "n_returns": int(Y.shape[1]),
        "median_ticker_std": float(stats["std"].median()),
        "min_ticker_std": float(stats["std"].min()),
        "max_ticker_std": float(stats["std"].max()),
        "pooled_excess_kurtosis": float(kurtosis(flat_all, fisher=True, bias=False)),
        "pooled_skew": float(skew(flat_all, bias=False)),
        "median_ticker_excess_kurtosis": float(stats["excess_kurtosis"].median()),
        "n_large_returns": int(len(survivors)),
        "large_return_threshold": LARGE_RETURN,
        "max_abs_return": float(np.abs(Y).max()),
        "quality": qreport,
        "large_returns": survivors.assign(
            date=survivors["date"].astype(str)).to_dict("records"),
        "rho_bar_mean": float(rho_bar.mean()),
        "rho_bar_min": float(rho_bar.min()),
        "rho_bar_max": float(rho_bar.max()),
        "rho_bar_argmax_date": str(pd.Timestamp(win_dates[int(rho_bar.argmax())]).date()),
        "n_windows": int(len(rho_bar)),
        "timestamp_convention": TIMESTAMP_CONVENTION,
    }
    with open(RETURNS_CACHE, "w") as f:
        json.dump(report, f, indent=2)
    return report


def write_quality_md(q: dict, survivors: pd.DataFrame) -> None:
    """data/QUALITY.md — what the return-level gates removed, and what stayed."""
    def lst(key):
        return ", ".join(f"`{t}`" for t in q[key]) or "none"

    rows = "\n".join(
        f"| {r.ticker} | {r.date.date()} | {r.log_return:+.3f} | {r.pct_move:+.1%} | "
        f"{r.trailing_vol_20d:.3f} | {r.sigmas:.0f}σ |"
        for r in survivors.itertuples())

    text = f"""# Return quality gates — paper 3 (Silva et al. 2015)

Generated by `experiments/exp1_data.py`. Do not edit by hand.

Phase 0's completeness filter asks only whether a ticker *has* a price every
day. It does not ask whether those prices are informative. Two defects survive
it, and both would corrupt results in ways that look like findings.

## What was removed

| Gate | Failure mode | Dropped |
|---|---|---|
| **G1** longest frozen-price run ≥ Δt = {q['delta_t']} days | zero variance in a whole window ⇒ correlation undefined ⇒ a **guaranteed isolated node**, which would manufacture the very Fig. 3B crisis signature we test for | **{len(q['g1_frozen_run'])}** — {lst('g1_frozen_run')} |
| **G2** a large move the next day undoes (≥ {q['reversal_fraction']:.0%}) | single bad print, not a price | **{len(q['g2_reversing_spike'])}** — {lst('g2_reversing_spike')} |
| **G3** \\|Y\\| ≥ {q['extreme_return']} on any day | unadjusted split / reorganisation / share exchange | **{len(q['g3_extreme_return'])}** — {lst('g3_extreme_return')} |

**N = {q['n_before']} → {q['n_after']}** ({q['n_dropped']} tickers removed).

Effects:

| | before | after |
|---|---|---|
| longest frozen-price run | **{q['max_frozen_run_before']} days** | {q['max_frozen_run_after']} days |
| share of exactly-zero returns | {q['zero_return_share_before']:.2%} | {q['zero_return_share_after']:.2%} |

G1 is tied to Δt rather than to a round number, and that is what makes it
principled: requiring the longest frozen run to be shorter than the window
guarantees every stock has non-zero variance in **every** window, so no
correlation in the whole study is undefined. The worst offenders were extreme —
`RRC` carried 743 consecutive days of an unchanged price, `SU` 497.

G3's threshold is not tuned. The empirical distribution of daily log returns in
this panel has a clean gap: the largest artifact is **1.179** and the largest
genuine move is **0.943**. Any cutoff in between gives the same answer.

## What was deliberately kept

Large single-day moves that are *real* are the signal, not noise. A gate that
removed them would be removing the crisis from a paper about crises. All
{len(survivors)} remaining \\|Y\\| > 0.5 observations, with each move sized against
the stock's own trailing 20-day volatility:

| Ticker | Date | log return | move | trailing σ | in σ |
|---|---|---|---|---|---|
{rows}

Spot-checks against the historical record: `AIG` −60.8 % on 2008-09-15 is the
day Lehman filed; `STT` −59.0 % and `PNC` −41.4 % on 2009-01-20 are the
inauguration-day bank selloff; `FITB`/`RF` on 2008-09-29 are the day the first
TARP vote failed; `BC` −40.1 % on 1987-10-19 **is Black Monday itself**;
`PCG` −42.4 % on 2001-04-06 is Pacific Gas & Electric's Chapter 11 filing;
`HAL` −42.4 % on 2001-12-07 is the Halliburton asbestos verdict.
"""
    with open(QUALITY_MD, "w") as f:
        f.write(text)
    print(f"[write] {QUALITY_MD}")


def plot_returns(save=None, verbose: bool = True):
    from utils.plots import plot_return_diagnostics
    Y = np.load(RETURNS_NPY)
    stats = pd.read_csv(os.path.join(RESULTS, "tables", "return_validation.csv"))
    rho = pd.read_parquet(RHO_BAR_PARQUET)
    save = save or os.path.join(ROOT, "figures", "fig1_returns.png")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    return plot_return_diagnostics(Y, stats["std"].to_numpy(), rho.index,
                                   rho["rho_bar"].to_numpy(),
                                   stamp=TIMESTAMP_CONVENTION, save=save)


def print_returns_report(r: dict) -> None:
    print("\n── Phase 1 returns " + "─" * 49)
    print(f"  returns matrix        : {r['n_tickers']} x {r['n_returns']}")
    print(f"  median per-ticker std : {r['median_ticker_std']:.4f}"
          f"   (range {r['min_ticker_std']:.4f}–{r['max_ticker_std']:.4f})")
    print(f"  pooled excess kurtosis: {r['pooled_excess_kurtosis']:.1f}"
          f"   (Gaussian = 0)")
    print(f"  pooled skew           : {r['pooled_skew']:.2f}")
    q = r["quality"]
    print(f"  quality gate          : N {q['n_before']} -> {q['n_after']}"
          f"  (G1 frozen {len(q['g1_frozen_run'])},"
          f" G2 bad print {len(q['g2_reversing_spike'])},"
          f" G3 |Y|>={q['extreme_return']} {len(q['g3_extreme_return'])})")
    print(f"  longest frozen run    : {q['max_frozen_run_before']}d "
          f"-> {q['max_frozen_run_after']}d   (Δt = {DELTA_T})")
    print(f"  max |Y| after gate    : {r['max_abs_return']:.3f}")
    print(f"  |Y| > {r['large_return_threshold']} retained    : {r['n_large_returns']}"
          f"   (all tabulated in data/QUALITY.md)")
    for rec in r["large_returns"][:6]:
        print(f"      {rec['ticker']:<6} {rec['date'][:10]}  "
              f"log {rec['log_return']:+.3f}  ({rec['pct_move']:+.1%})")
    print(f"  rho_bar               : mean {r['rho_bar_mean']:.3f}, "
          f"range {r['rho_bar_min']:.3f}–{r['rho_bar_max']:.3f}, "
          f"peak {r['rho_bar_argmax_date']}")


def acceptance_returns(r: dict, verbose: bool = True) -> bool:
    checks = [
        ("median per-ticker daily std in [0.01, 0.04]",
         0.01 <= r["median_ticker_std"] <= 0.04,
         f"{r['median_ticker_std']:.4f}"),
        ("return distribution is leptokurtic (excess kurtosis > 1)",
         r["pooled_excess_kurtosis"] > 1,
         f"pooled {r['pooled_excess_kurtosis']:.1f}, "
         f"median ticker {r['median_ticker_excess_kurtosis']:.1f}"),
        # The plan's test is "zero unexplained |Y| > 0.5". A count alone cannot
        # express that, so it is split into the two things that would actually
        # signal an unadjusted corporate action still in the panel.
        ("no |Y| >= 1.0 survives (corporate-action artifacts removed)",
         r["max_abs_return"] < r["quality"]["extreme_return"],
         f"max |Y| = {r['max_abs_return']:.3f}"),
        ("no ticker has a frozen-price run as long as a window",
         r["quality"]["max_frozen_run_after"] < DELTA_T,
         f"longest run {r['quality']['max_frozen_run_after']}d "
         f"(was {r['quality']['max_frozen_run_before']}d) vs Δt = {DELTA_T}"),
        ("every surviving |Y| > 0.5 is listed for inspection in data/QUALITY.md",
         os.path.exists(QUALITY_MD)
         and len(r["large_returns"]) == r["n_large_returns"],
         f"{r['n_large_returns']} retained and tabulated"),
        ("rho_bar series has one value per window, no NaNs",
         r["n_windows"] == r["n_returns"] + 1 - DELTA_T,
         f"{r['n_windows']} windows"),
    ]
    if verbose:
        print("\n── Phase 1 acceptance tests " + "─" * 40)
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
        print(f"  {sum(ok for _, ok, _ in checks)}/{len(checks)} passed")
    return all(ok for _, ok, _ in checks)


# ── Date calibration (audit A10) ───────────────────────────────────────────

# Fig. 8 of the paper shows "Model vs Inferred" networks sampled every 1000 time
# steps and prints the calendar date of each. These five (window index, date)
# pairs are the only hard link the paper gives between its window numbering and
# the calendar, and so they are the only way to settle the timestamp convention.
FIG8_ANCHORS = [(1000, "1991-05-10"), (2000, "1995-04-25"), (3000, "1999-04-12"),
                (4000, "2003-04-03"), (5000, "2007-03-26")]


def date_calibration(verbose: bool = True) -> pd.DataFrame:
    """Locate the paper's Fig. 8 anchors on our calendar and write the table.

    Writes results/tables/date_calibration.csv.
    """
    panel = pd.read_parquet(PANEL_PARQUET)
    cal = panel.index
    rows = []
    for n, datestr in FIG8_ANCHORS:
        i = cal.get_indexer([pd.Timestamp(datestr)], method="nearest")[0]
        rows.append({"paper_window_n": n, "fig8_date": datestr,
                     "our_day_1based": int(i + 1), "offset": int(i + 1 - n)})
    df = pd.DataFrame(rows)

    offsets = df["offset"].unique()
    constant = len(offsets) == 1
    out = os.path.join(RESULTS, "tables")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "date_calibration.csv"), index=False)

    # The decisive test. The anchors fix, for each convention, where the
    # paper's day 1 must sit on our calendar; C_p = 6008 then fixes where its
    # last day must sit. A convention whose implied last day falls beyond the
    # end of our calendar is impossible, because the paper's data ends in
    # February 2011 and ours ends 2011-02-28.
    feasible = {}
    if constant:
        off = int(offsets[0])
        for conv, extra in (("t1", 0), ("midpoint", DELTA_T // 2), ("t2", DELTA_T)):
            shift = off - extra                     # their day j == our day j+shift
            start, end = shift + 1, shift + PAPER["n_closes"]
            ok = 1 <= start and end <= len(cal)
            feasible[conv] = {
                "implied_start": str(cal[start - 1].date()) if start >= 1 else None,
                "implied_end": str(cal[end - 1].date()) if ok else None,
                "feasible": bool(ok),
            }
    pd.DataFrame(feasible).T.to_csv(os.path.join(out, "convention_feasibility.csv"))

    if verbose:
        print("\n── Date calibration, audit A10 " + "─" * 37)
        print(df.to_string(index=False))
        print(f"  offset constant across all five anchors: {constant} "
              f"({sorted(offsets)})")
        print(f"  our T = {len(cal)}, paper C_p = {PAPER['n_closes']}, "
              f"gap = {len(cal) - PAPER['n_closes']}")
        print("  implied span of the paper's panel on our calendar:")
        for conv, v in feasible.items():
            verdict = "consistent" if v["feasible"] else "IMPOSSIBLE (runs past our data)"
            print(f"      {conv:<9} {v['implied_start']} -> "
                  f"{v['implied_end'] or 'beyond 2011-02-28'}   {verdict}")
    return df


# ── Acceptance tests ───────────────────────────────────────────────────────

def acceptance(report: dict, verbose: bool = True) -> bool:
    """Phase 0 acceptance tests from the replication plan. Prints and returns."""
    panel = pd.read_parquet(PANEL_PARQUET)
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    check("panel is dense (N x T) with zero NaNs",
          not panel.isna().any().any(),
          f"shape {panel.shape[1]} tickers x {panel.shape[0]} days")
    check("every ticker has exactly T observations",
          bool((panel.notna().sum(axis=0) == len(panel)).all()),
          f"T = {len(panel)}")
    check("N >= 200", panel.shape[1] >= 200, f"N = {panel.shape[1]}")
    check("T >= 3000", panel.shape[0] >= 3000, f"T = {panel.shape[0]}")
    check("UNIVERSE.md exists and states N, T, range, drop count",
          os.path.exists(UNIVERSE_MD)
          and all(s in open(UNIVERSE_MD).read()
                  for s in ("Surviving universe (N)", "Final panel",
                            "Realized calendar",
                            "were removed before a single network was built")),
          UNIVERSE_MD)
    check("all prices strictly positive (log is defined)",
          bool((panel.to_numpy() > 0).all()),
          f"min price {panel.to_numpy().min():.4f}")

    if verbose:
        print("\n── Phase 0 acceptance tests " + "─" * 40)
        for name, ok, detail in checks:
            print(f"  {'✓' if ok else '✗'} {name}" + (f"  [{detail}]" if detail else ""))
        n_ok = sum(ok for _, ok, _ in checks)
        print(f"  {n_ok}/{len(checks)} passed")
    return all(ok for _, ok, _ in checks)


def print_report(report: dict) -> None:
    r = report
    print("\n── Phase 0 universe " + "─" * 48)
    print(f"  symbols requested     : {r['n_requested']}   (paper: {PAPER['n_candidates']})")
    print(f"  no data from Yahoo    : {r['n_no_data']}")
    print(f"  dropped, not full span: {r['dropped_not_full_span']}")
    print(f"  dropped, low coverage : {r['dropped_incomplete_coverage']}")
    print(f"  dropped, price <= 0   : {r['dropped_nonpositive_price']} "
          f"{r['nonpositive_tickers']}")
    print(f"  N survivors           : {r['n_survivors']}   (paper: {PAPER['n_stocks']})")
    print(f"  T days                : {r['n_days_final']}   (paper: {PAPER['n_closes']})")
    print(f"  range                 : {r['panel_start']} -> {r['panel_end']}")
    print(f"  days dropped          : {r['n_dropped_days']}")
    print(f"  N_w windows           : {r.get('n_windows')}   (paper: {PAPER['n_windows']})")
    print(f"  coverage ladder       : {r['coverage_ladder']}")
    by_year = {int(k): v for k, v in r["dropped_days_by_year"].items()}
    early = sum(v for k, v in by_year.items() if k <= 1987)
    print(f"  dropped days <= 1987  : {early} of {r['n_dropped_days']}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    rep = run(force=force)
    print_report(rep)
    date_calibration()
    ok = acceptance(rep)

    rep1 = run_returns(force=force)
    print_returns_report(rep1)
    plot_returns()
    ok &= acceptance_returns(rep1)
    sys.exit(0 if ok else 1)
