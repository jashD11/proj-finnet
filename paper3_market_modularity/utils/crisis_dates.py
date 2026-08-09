"""
Pre-registered crisis windows for Silva et al. (2015).

WHY THIS FILE EXISTS, AND WHY IT IS WRITTEN IN PHASE 0
------------------------------------------------------
The paper names 17 crises (Fig. 4, sourced to Guhathakurta 2015 plus press
articles) but publishes NUMERIC DATES FOR EXACTLY ONE of them: Black Monday,
October 19, 1987. The other 16 exist only as shaded regions drawn on a figure.
The paper also concedes: "the dates specified by the source are approximate
and, in general, the duration of a crisis is not well-defined in the
literature."

That leaves the replicator free to choose windows, and choosing them AFTER
seeing the modularity series would let any detector be tuned into looking good.
Audit item E10 is that the paper's own Fig. 1 has a box labelled "crisis
detection evaluation" and the paper contains no detection rule, no threshold,
no hit rate and no false-alarm rate. Phase 8b builds that missing evaluation —
so these windows are fixed here, in Phase 0, before a single network exists.

DO NOT EDIT THESE DATES AFTER PHASE 2. If a window turns out to be wrong,
add a corrected entry alongside the original and report both.

Each event carries:
    onset  — the single day the event is conventionally dated from. Used for
             Phase 8c's event-window study (align on onset, average +/- 60 d).
    start/end — the band drawn on figures and used for detection scoring.
    tier   — "sharp"   : a dated market event, window under ~4 weeks. These are
                         the fair test of a detector.
              "diffuse": a multi-month regime whose boundaries are a judgement
                         call. Included for figure parity with the paper, but
                         scored separately, because 17 fuzzy bands over 25 years
                         make a large fraction of the timeline "crisis" and
                         visual alignment then proves little (audit C2).
"""

from datetime import date

import pandas as pd

# Order follows the paper's own Fig. 4 listing.
CRISES = [
    dict(name="Black Monday",
         onset="1987-10-19", start="1987-10-14", end="1987-11-13", tier="sharp",
         note="The only crisis with an explicit date in the paper (Sec. IV)."),
    dict(name="Friday the 13th mini-crash",
         onset="1989-10-13", start="1989-10-13", end="1989-10-31", tier="sharp",
         note="UAL buyout financing collapse; DJIA -6.9% intraday."),
    dict(name="1990-1991 Recession",
         onset="1990-07-01", start="1990-07-01", end="1991-03-31", tier="diffuse",
         note="NBER peak Jul 1990, trough Mar 1991."),
    dict(name="Japanese asset price bubble",
         onset="1990-01-04", start="1990-01-04", end="1992-08-31", tier="diffuse",
         note="Nikkei peaked 1989-12-29; the collapse ran into 1992. The "
              "fuzziest band in the list and the one least tied to NYSE."),
    dict(name="Black Wednesday",
         onset="1992-09-16", start="1992-09-16", end="1992-09-30", tier="sharp",
         note="Sterling forced out of the ERM."),
    dict(name="1997 Asian financial crisis",
         onset="1997-07-02", start="1997-07-02", end="1998-01-30", tier="diffuse",
         note="Thai baht float 1997-07-02."),
    dict(name="October 27, 1997 mini-crash",
         onset="1997-10-27", start="1997-10-27", end="1997-11-07", tier="sharp",
         note="NYSE circuit breakers halted trading; DJIA -7.2%."),
    dict(name="1998 Russian financial crisis",
         onset="1998-08-17", start="1998-08-17", end="1998-10-30", tier="sharp",
         note="Rouble devaluation/GKO default; LTCM rescue 1998-09-23."),
    dict(name="Boo.com collapses",
         onset="2000-05-18", start="2000-05-18", end="2000-06-16", tier="sharp",
         note="A dot-com marker rather than a market-wide event; expected to "
              "be a detector miss, and that is informative."),
    dict(name="Dot-com bubble burst",
         onset="2000-03-10", start="2000-03-10", end="2001-04-04", tier="diffuse",
         note="NASDAQ peak 2000-03-10 to trough 2001-04-04."),
    dict(name="September 11 attacks",
         onset="2001-09-11", start="2001-09-11", end="2001-09-28", tier="sharp",
         note="NYSE closed 09-11 to 09-14; reopened 09-17. Expect a calendar "
              "gap here, not just a price move."),
    dict(name="Stock market downturn of 2002",
         onset="2002-03-19", start="2002-03-19", end="2002-10-09", tier="diffuse",
         note="S&P 500 peak 2002-03-19 to trough 2002-10-09. Note this straddles "
              "the paper's own 2002 split point (audit E9)."),
    dict(name="Chinese stock bubble of 2007",
         onset="2007-02-27", start="2007-02-27", end="2007-03-16", tier="sharp",
         note="Shanghai -8.8% on 2007-02-27; DJIA -3.3% same day."),
    dict(name="United States bear market of 2007-2009",
         onset="2007-10-09", start="2007-10-09", end="2009-03-09", tier="diffuse",
         note="S&P 500 peak 2007-10-09 to trough 2009-03-09. Fully contains the "
              "next entry."),
    dict(name="Late-2000s financial crisis",
         onset="2008-09-15", start="2008-09-15", end="2009-03-31", tier="diffuse",
         note="Lehman filing 2008-09-15."),
    dict(name="2009 Dubai debt standstill",
         onset="2009-11-25", start="2009-11-25", end="2009-12-14", tier="sharp",
         note="Dubai World standstill request."),
    dict(name="European sovereign debt crisis",
         onset="2010-04-23", start="2010-04-23", end="2011-02-28", tier="diffuse",
         note="Greek bailout request 2010-04-23; band is truncated by the end "
              "of the paper's data window."),
]

# Phase 8c event-window half-width, in trading days.
EVENT_HALF_WIDTH = 60


def crisis_frame() -> pd.DataFrame:
    """The registry as a DataFrame with parsed timestamps."""
    df = pd.DataFrame(CRISES)
    for col in ("onset", "start", "end"):
        df[col] = pd.to_datetime(df[col])
    assert (df["start"] <= df["end"]).all(), "a crisis window ends before it starts"
    assert (df["onset"] >= df["start"]).all() and (df["onset"] <= df["end"]).all(), \
        "an onset falls outside its own window"
    return df


def crisis_mask(index: pd.DatetimeIndex, tiers=("sharp", "diffuse")) -> pd.Series:
    """Boolean 'is this day inside any crisis band' over a trading calendar."""
    df = crisis_frame()
    df = df[df["tier"].isin(tiers)]
    mask = pd.Series(False, index=index)
    for _, r in df.iterrows():
        mask |= (index >= r["start"]) & (index <= r["end"])
    return mask


def coverage(index: pd.DatetimeIndex) -> dict:
    """Fraction of the timeline labelled 'crisis'.

    This number is the honest framing of audit C2: if it is large, alignment of
    a wiggly series with the shaded bands is close to unfalsifiable, and any
    detector must be judged against that base rate rather than against zero.
    """
    return {
        "n_days": int(len(index)),
        "frac_any": float(crisis_mask(index).mean()),
        "frac_sharp": float(crisis_mask(index, tiers=("sharp",)).mean()),
        "frac_diffuse": float(crisis_mask(index, tiers=("diffuse",)).mean()),
        "n_events": len(CRISES),
        "n_sharp": sum(c["tier"] == "sharp" for c in CRISES),
        "n_diffuse": sum(c["tier"] == "diffuse" for c in CRISES),
    }


if __name__ == "__main__":
    df = crisis_frame()
    print(df[["name", "tier", "onset", "start", "end"]].to_string(index=False))
    cal = pd.bdate_range("1987-01-01", "2011-02-28")
    cov = coverage(cal)
    print(f"\nOver {cov['n_days']} business days 1987-2011:")
    print(f"  any crisis band     : {cov['frac_any']:.1%}")
    print(f"  sharp events only   : {cov['frac_sharp']:.1%}")
    print(f"  diffuse regimes only: {cov['frac_diffuse']:.1%}")
    print(f"  {cov['n_sharp']} sharp + {cov['n_diffuse']} diffuse = {cov['n_events']} events")
