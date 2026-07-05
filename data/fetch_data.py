"""
One-time data fetch. Run once from the project root:
    python data/fetch_data.py

Writes:
    data/sp500_3sectors.csv  — daily log-returns for ~130 S&P500 stocks, 2016-01-01 to 2019-12-31
    data/faamung.csv         — daily log-returns for FAAMUNG 7 stocks, 2019-06-01 to 2020-06-30
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from data.tickers import SP500_130, FAAMUNG

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fetch_log_returns(tickers: list[str], start: str, end: str, min_coverage: float = 0.9) -> pd.DataFrame:
    """Download adjusted close prices, compute log returns, drop thin columns."""
    print(f"  Downloading {len(tickers)} tickers {start} → {end} …")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]

    # yfinance may return a single column as a Series
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    # Drop tickers with too many missing prices
    threshold = int(min_coverage * len(raw))
    raw = raw.dropna(axis=1, thresh=threshold)

    # Log returns: ln(P_t / P_{t-1})
    log_ret = np.log(raw / raw.shift(1)).dropna()

    dropped = set(tickers) - set(log_ret.columns)
    if dropped:
        print(f"  Dropped {len(dropped)} tickers (insufficient data): {sorted(dropped)}")

    print(f"  Result: {log_ret.shape[1]} stocks × {log_ret.shape[0]} days")
    return log_ret


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)

    # --- 130-stock S&P500 dataset (Fig 1 & 2) ---
    sp500_path = out_dir / "sp500_3sectors.csv"
    if sp500_path.exists():
        print(f"[skip] {sp500_path} already exists — delete to re-fetch.")
    else:
        print("[fetch] S&P500 130-stock universe …")
        df_sp = fetch_log_returns(SP500_130, start="2016-01-01", end="2020-01-01")
        df_sp.to_csv(sp500_path)
        print(f"  Saved → {sp500_path}  ({df_sp.shape})")

    # --- FAAMUNG dataset (Fig 3 & 4) ---
    faamung_path = out_dir / "faamung.csv"
    if faamung_path.exists():
        print(f"[skip] {faamung_path} already exists — delete to re-fetch.")
    else:
        print("[fetch] FAAMUNG 7-stock dataset …")
        df_fa = fetch_log_returns(FAAMUNG, start="2019-06-01", end="2020-07-01")
        # Reorder columns to canonical FAAMUNG order (keep only those that downloaded)
        cols = [t for t in FAAMUNG if t in df_fa.columns]
        df_fa = df_fa[cols]
        df_fa.to_csv(faamung_path)
        print(f"  Saved → {faamung_path}  ({df_fa.shape})")

    print("\nDone. Both CSVs ready.")


if __name__ == "__main__":
    main()
