"""
Sanity tests for the Silva et al. (2015) replication.

Run:
    /opt/anaconda3/bin/python -m pytest tests/

Tests are added phase by phase; everything here is Phase 0.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.fetch_data import build_panel  # noqa: E402
from data.universe import to_yahoo_symbol  # noqa: E402
from utils.crisis_dates import CRISES, coverage, crisis_frame, crisis_mask  # noqa: E402


# ── Phase 0: universe ──────────────────────────────────────────────────────

def test_symbol_translation():
    """NASDAQ Trader writes share classes with a dot; Yahoo wants a dash."""
    assert to_yahoo_symbol("BRK.B") == "BRK-B"
    assert to_yahoo_symbol("IBM") == "IBM"
    assert to_yahoo_symbol("  GE ") == "GE"


def _ragged_frame():
    """A 100-day, 4-ticker frame exercising every filter branch."""
    idx = pd.bdate_range("2000-01-03", periods=100)
    df = pd.DataFrame(100.0, index=idx, columns=["FULL", "LATE", "EARLY", "GAPPY"])
    df.loc[idx[:20], "LATE"] = np.nan       # starts late  -> fails full-span
    df.loc[idx[-20:], "EARLY"] = np.nan     # ends early   -> fails full-span
    df.loc[idx[50], "GAPPY"] = np.nan       # one missing day
    return df, idx


def test_full_span_filter_drops_late_starters_and_early_enders():
    df, _ = _ragged_frame()
    panel, rep = build_panel(df, span_tol=5, min_coverage=0.0,
                             missing_policy="drop_days", verbose=False)
    assert rep["dropped_not_full_span"] == 2
    assert set(panel.columns) == {"FULL", "GAPPY"}


def test_strict_coverage_drops_the_gappy_ticker():
    """min_coverage=1.0 is the strict reading of 'complete history'."""
    df, _ = _ragged_frame()
    panel, rep = build_panel(df, span_tol=5, min_coverage=1.0,
                             missing_policy="drop_days", verbose=False)
    assert rep["dropped_incomplete_coverage"] == 1
    assert list(panel.columns) == ["FULL"]
    assert rep["n_dropped_days"] == 0


def test_missing_day_policies_trade_days_against_fabricated_returns():
    """drop_days loses the day; ffill keeps it and manufactures a zero return.

    This is audit A3 made concrete: the two policies disagree, and the
    disagreement is exactly one trading day per gap.
    """
    df, _ = _ragged_frame()
    dropped, rep_d = build_panel(df, span_tol=5, min_coverage=0.99,
                                 missing_policy="drop_days", verbose=False)
    filled, rep_f = build_panel(df, span_tol=5, min_coverage=0.99,
                                missing_policy="ffill", verbose=False)
    assert set(dropped.columns) == set(filled.columns) == {"FULL", "GAPPY"}
    assert rep_d["n_days_final"] == rep_f["n_days_final"] - 1
    assert rep_d["n_dropped_days"] == 1
    assert not dropped.isna().any().any()
    assert not filled.isna().any().any()


def test_build_panel_rejects_unknown_policy():
    df, _ = _ragged_frame()
    with pytest.raises(ValueError):
        build_panel(df, missing_policy="interpolate", verbose=False)


# ── Phase 0: pre-registered crisis windows ─────────────────────────────────

def test_crisis_registry_is_well_formed():
    """crisis_frame() asserts internally; this pins the registry's shape."""
    df = crisis_frame()
    assert len(df) == 17, "the paper names exactly 17 crises in Fig. 4"
    assert set(df["tier"]) == {"sharp", "diffuse"}
    assert df["name"].is_unique
    # Black Monday is the one crisis the paper dates explicitly.
    bm = df[df["name"] == "Black Monday"].iloc[0]
    assert str(bm["onset"].date()) == "1987-10-19"


def test_crisis_mask_and_coverage_agree():
    idx = pd.bdate_range("1987-01-01", "2011-02-28")
    cov = coverage(idx)
    assert cov["frac_sharp"] < cov["frac_any"] <= 1.0
    assert cov["n_sharp"] + cov["n_diffuse"] == 17
    # The C2 headline: bands cover a large minority of the timeline, so a
    # detector must be scored against that base rate, not against zero.
    assert 0.25 < cov["frac_any"] < 0.45
    assert cov["frac_sharp"] < 0.06
    assert np.isclose(crisis_mask(idx).mean(), cov["frac_any"])


def test_black_monday_is_inside_its_own_band():
    idx = pd.bdate_range("1987-01-01", "1988-01-01")
    mask = crisis_mask(idx, tiers=("sharp",))
    assert mask.loc[pd.Timestamp("1987-10-19")]
    assert not mask.loc[pd.Timestamp("1987-06-01")]


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
