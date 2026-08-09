"""
Candidate universe construction for Silva et al. (2015).

The paper starts from "3799 stocks traded on the New York Stock Exchange"
(Yahoo! financial database) and keeps the N = 348 with a complete history from
January 1986 to February 2011. That candidate list is not published and the
Yahoo endpoint it came from no longer exists, so we rebuild an equivalent
candidate pool from the NASDAQ Trader symbol directory.

Source: https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt
  (pipe-delimited; "Exchange" == 'N' means NYSE)

IMPORTANT — this list is CURRENTLY-LISTED symbols only. Firms that delisted
before the retrieval date are absent entirely, which is a *second* layer of
survivorship bias stacked on top of the paper's own completeness filter
(audit A1). Both layers are documented in data/UNIVERSE.md; neither is fixable
within the paper's design.

Writes: data/nyse_candidates.csv
"""

import csv
import io
import os
import ssl
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_CSV = os.path.join(HERE, "nyse_candidates.csv")

SYMDIR_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
NYSE_EXCHANGE_CODE = "N"

# The directory tags every share class, warrant, unit, preferred and depositary
# receipt in the same file. "Common Stock" in the security name is the coarse
# filter that leaves ordinary equity; it is deliberately conservative.
COMMON_STOCK_MARKER = "Common Stock"


def _http_get(url: str, timeout: int = 30) -> str:
    """Fetch a URL as text. certifi is present in the anaconda env; fall back to
    an unverified context only if the local trust store is unusable."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except ssl.SSLError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read().decode("utf-8", errors="replace")


def to_yahoo_symbol(act_symbol: str) -> str:
    """NASDAQ Trader writes share classes as 'BRK.B'; Yahoo wants 'BRK-B'."""
    return act_symbol.strip().replace(".", "-")


def fetch_candidates(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Return the NYSE common-stock candidate universe, cached to CSV.

    Columns: symbol (Yahoo form), act_symbol (directory form), name.
    """
    if os.path.exists(CANDIDATES_CSV) and not force:
        df = pd.read_csv(CANDIDATES_CSV)
        if verbose:
            print(f"[skip] {CANDIDATES_CSV} exists ({len(df)} symbols) — "
                  f"pass force=True to re-fetch.")
        return df

    if verbose:
        print(f"[fetch] {SYMDIR_URL}")
    text = _http_get(SYMDIR_URL)

    rows = list(csv.DictReader(io.StringIO(text), delimiter="|"))
    n_raw = len(rows)

    keep = []
    for r in rows:
        if r.get("Exchange") != NYSE_EXCHANGE_CODE:
            continue
        if r.get("ETF") != "N" or r.get("Test Issue") != "N":
            continue
        name = r.get("Security Name") or ""
        if COMMON_STOCK_MARKER not in name:
            continue
        act = (r.get("ACT Symbol") or "").strip()
        if not act:
            continue
        keep.append({"symbol": to_yahoo_symbol(act), "act_symbol": act, "name": name})

    df = pd.DataFrame(keep).drop_duplicates("symbol").sort_values("symbol")
    df = df.reset_index(drop=True)

    if verbose:
        print(f"[universe] {n_raw} directory rows -> {len(df)} NYSE common stocks")
    df.to_csv(CANDIDATES_CSV, index=False)
    return df


if __name__ == "__main__":
    import sys
    fetch_candidates(force="--force" in sys.argv)
