"""
Ticker lists for the two datasets used in the paper replication.

SP500_130: ~130 stocks from 3 S&P500 sectors (Industrials, Consumer Staples, Energy)
           as of ~2016 membership. Used for Figs 1 & 2.

FAAMUNG:   7 large-cap tech/growth stocks. Used for Figs 3 & 4.
           Note: FB -> META (use META for yfinance; historical data under META works).
           Note: UBER IPO was May 2019, so data starts Jun 2019.
"""

FAAMUNG = ["META", "AAPL", "AMZN", "MSFT", "UBER", "NFLX", "GOOGL"]

# Sector assignments for FAAMUNG (all tech/communication services — used for coloring)
FAAMUNG_SECTORS = {t: "Tech" for t in FAAMUNG}

# ---------------------------------------------------------------------------
# S&P 500 universe: Industrials (IND), Consumer Staples (CS), Energy (ENE)
# Membership reconstructed from ~2016 S&P 500 index components.
# ---------------------------------------------------------------------------

INDUSTRIALS = [
    "MMM", "AOS", "ALK", "ALLE", "AME", "ADP", "AIZ", "BA", "CHRW",
    "CAT", "CTAS", "CSX", "CMI", "DE", "DAL", "DOV", "ETN", "EMR",
    "EFX", "EXPD", "FAST", "FDX", "FLS", "FTV", "GD", "GE", "GWW",
    "HON", "HWM", "IEX", "ITW", "IR", "JCI", "LHX", "LMT", "MAS",
    "NOC", "PCAR", "PH", "PNR", "RTX", "ROK", "ROP", "RSG", "LUV",
    "SWK", "TXT", "TT", "UNP", "UAL", "UPS", "URI", "VRSK", "WM", "XYL",
]

CONSUMER_STAPLES = [
    "MO", "ADM", "BG", "CPB", "CHD", "CLX", "KO", "CL", "CAG", "STZ",
    "COST", "EL", "GIS", "HRL", "HSY", "HLN", "K", "KMB", "KR",
    "LW", "MKC", "MCD", "MDLZ", "PEP", "PM", "PG", "SJM", "SYY",
    "TAP", "TSN", "WBA", "WMT",
]

ENERGY = [
    "APA", "BKR", "CVX", "COP", "CTRA", "DVN", "FANG", "EOG",
    "MPC", "MRO", "OXY", "OKE", "PSX", "PXD", "SLB", "VLO",
    "WMB", "XOM", "HAL", "HES",
]

SP500_130 = INDUSTRIALS + CONSUMER_STAPLES + ENERGY

# Sector label map for every ticker (used for graph coloring)
SP500_SECTORS = (
    {t: "Industrials" for t in INDUSTRIALS}
    | {t: "ConsumerStaples" for t in CONSUMER_STAPLES}
    | {t: "Energy" for t in ENERGY}
)
