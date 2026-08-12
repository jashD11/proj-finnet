"""
Price-panel acquisition for Silva et al. (2015).

One-time fetch. Run once from the paper root:
    /opt/anaconda3/bin/python data/fetch_data.py

Downloads split/dividend-ADJUSTED daily closes (yfinance auto_adjust=True) for
every NYSE common-stock candidate over the paper's window, then applies the
paper's completeness filter to produce a dense (N x T) panel.

Adjusted, not raw, closes: on raw closes a 2-for-1 split is a fake -69% log
return that poisons 30 consecutive networks (audit A2). Phase 1's |Y| > 0.5
detector is the check that this actually held.

Writes:
    data/raw_closes.parquet     all candidates, ragged (NaNs kept)
    data/panel_prices.parquet   dense (T x N) survivors, zero NaNs
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.universe import fetch_candidates  # noqa: E402

RAW_PARQUET = os.path.join(HERE, "raw_closes.parquet")
PANEL_PARQUET = os.path.join(HERE, "panel_prices.parquet")

# ── Paper window ───────────────────────────────────────────────────────────
# "January 1986 to February 2011", C_p = 6008 closing prices per stock.
START = "1986-01-01"
END = "2011-03-01"          # yfinance end is exclusive; last close is 2011-02-28

BATCH_SIZE = 100            # yfinance is unreliable with 1500 tickers in one call
BATCH_PAUSE = 1.0           # seconds between batches, to stay polite


def download_raw_closes(symbols=None, force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Download adjusted closes for all candidates. Cached; ragged (NaNs kept)."""
    if os.path.exists(RAW_PARQUET) and not force:
        df = pd.read_parquet(RAW_PARQUET)
        if verbose:
            print(f"[skip] {RAW_PARQUET} exists ({df.shape[0]} days x "
                  f"{df.shape[1]} symbols) — delete or pass force=True to re-fetch.")
        return df

    import yfinance as yf

    if symbols is None:
        symbols = fetch_candidates(verbose=verbose)["symbol"].tolist()

    if verbose:
        print(f"[fetch] {len(symbols)} symbols, {START} -> {END}, adjusted closes")

    frames = []
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        t0 = time.time()
        raw = yf.download(batch, start=START, end=END, auto_adjust=True,
                          progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:                                   # single surviving symbol in batch
            close = raw[["Close"]].rename(columns={"Close": batch[0]})
        frames.append(close)
        if verbose:
            got = int(close.notna().any(axis=0).sum())
            print(f"[fetch] batch {i // BATCH_SIZE + 1:>3}/"
                  f"{(len(symbols) - 1) // BATCH_SIZE + 1}: "
                  f"{got}/{len(batch)} symbols returned data "
                  f"({time.time() - t0:.1f}s)")
        time.sleep(BATCH_PAUSE)

    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df.sort_index()
    # Drop symbols that returned nothing at all.
    df = df.loc[:, df.notna().any(axis=0)]
    df.to_parquet(RAW_PARQUET)
    if verbose:
        print(f"[fetch] wrote {RAW_PARQUET}: {df.shape[0]} days x {df.shape[1]} symbols")
    return df


def build_panel(raw: pd.DataFrame,
                span_tol: int = 5,
                min_coverage: float = 1.0,
                missing_policy: str = "drop_days",
                n_requested: int = None,
                verbose: bool = True):
    """Apply the paper's completeness filter and return (panel, report).

    The paper says only "complete history from January 1986 to February 2011".
    That is two separate requirements, which we make explicit:

    1. FULL SPAN — the ticker must be trading at both ends of the window. A
       ticker whose first close is later than `span_tol` trading days into the
       window, or whose last close is more than `span_tol` days before the end,
       is dropped. This is the survivorship filter (audit A1).
    2. MISSING DAYS — among full-span tickers, isolated gaps (halts, missing
       ticks) remain. `min_coverage` is the fraction of the reference calendar
       a ticker must actually trade on; 1.0 is the strict reading.
    3. POSITIVE PRICES — yfinance back-adjusts for dividends by subtraction, so
       a stock whose cumulative historical distributions exceed its old price
       can be handed to us with a NEGATIVE adjusted close. log() is undefined
       there and the whole series is unusable. Such tickers are dropped and
       named in the report; this is a data defect, not a modelling choice.

    `missing_policy` then resolves whatever gaps survive (audit A3):
       "drop_days" — drop every calendar day on which ANY retained ticker is
                     missing. Dense panel, fewer days. DEFAULT.
       "ffill"     — carry the last close forward per ticker. Dense panel, all
                     days, but manufactures zero returns on halt days, which
                     deflates that stock's correlations.

    Both are implemented so Phase 1 can test that results do not flip.
    """
    if missing_policy not in ("drop_days", "ffill"):
        raise ValueError(f"unknown missing_policy: {missing_policy!r}")

    calendar = raw.index
    n_days = len(calendar)
    n_candidates = raw.shape[1]

    first_ok = calendar[span_tol]
    last_ok = calendar[-1 - span_tol]

    first_valid = raw.apply(lambda s: s.first_valid_index())
    last_valid = raw.apply(lambda s: s.last_valid_index())
    full_span = (first_valid <= first_ok) & (last_valid >= last_ok)

    dropped_span = int((~full_span).sum())
    spanning = raw.loc[:, full_span[full_span].index]

    coverage = spanning.notna().mean(axis=0)
    complete = coverage >= min_coverage
    dropped_coverage = int((~complete).sum())
    kept = spanning.loc[:, complete[complete].index]

    positive = kept.min(axis=0) > 0          # min() skips NaN, unlike (kept > 0).all()
    nonpositive_tickers = sorted(kept.columns[~positive].tolist())
    kept = kept.loc[:, positive[positive].index]

    # Coverage sensitivity — how many survivors at each threshold, so the
    # choice of min_coverage is visible rather than buried.
    coverage_ladder = {f"{thr:.4f}": int((coverage >= thr).sum())
                       for thr in (1.0, 0.999, 0.995, 0.99, 0.95)}

    if missing_policy == "ffill":
        panel = kept.ffill().bfill()
        dropped_days_idx = pd.DatetimeIndex([])
    else:
        good_day = kept.notna().all(axis=1)
        dropped_days_idx = calendar[~good_day]
        panel = kept.loc[good_day]

    dropped_days_by_year = (pd.Series(1, index=dropped_days_idx)
                            .groupby(dropped_days_idx.year).sum()
                            .to_dict() if len(dropped_days_idx) else {})

    report = {
        "n_requested": int(n_requested) if n_requested is not None else None,
        "n_candidates": n_candidates,
        "n_no_data": (int(n_requested) - n_candidates) if n_requested is not None else None,
        "n_calendar_days": n_days,
        "calendar_start": str(calendar[0].date()),
        "calendar_end": str(calendar[-1].date()),
        "span_tol": span_tol,
        "min_coverage": min_coverage,
        "missing_policy": missing_policy,
        "dropped_not_full_span": dropped_span,
        "dropped_incomplete_coverage": dropped_coverage,
        "dropped_nonpositive_price": len(nonpositive_tickers),
        "nonpositive_tickers": nonpositive_tickers,
        "coverage_ladder": coverage_ladder,
        "n_survivors": int(panel.shape[1]),
        "n_days_final": int(panel.shape[0]),
        "panel_start": str(panel.index[0].date()),
        "panel_end": str(panel.index[-1].date()),
        "n_dropped_days": int(len(dropped_days_idx)),
        "dropped_days_by_year": {int(k): int(v) for k, v in dropped_days_by_year.items()},
    }

    if verbose:
        print(f"[panel] {n_candidates} with data "
              f"-> {n_candidates - dropped_span} full-span "
              f"-> {n_candidates - dropped_span - dropped_coverage} complete "
              f"-> {report['n_survivors']} positive-priced")
        if nonpositive_tickers:
            print(f"[panel] dropped for non-positive adjusted price: "
                  f"{', '.join(nonpositive_tickers)}")
        print(f"[panel] policy={missing_policy}: {report['n_days_final']} days "
              f"({report['panel_start']} -> {report['panel_end']}), "
              f"{report['n_dropped_days']} days dropped")

    assert not panel.isna().any().any(), "panel still contains NaNs"
    assert (panel.to_numpy() > 0).all(), "panel contains non-positive prices"
    return panel, report


def main(force: bool = False):
    raw = download_raw_closes(force=force)
    panel, report = build_panel(raw)
    panel.to_parquet(PANEL_PARQUET)
    print(f"[panel] wrote {PANEL_PARQUET}: {panel.shape[0]} days x {panel.shape[1]} tickers")
    return panel, report


if __name__ == "__main__":
    main(force="--force" in sys.argv)
