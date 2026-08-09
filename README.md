# Financial Networks Project

From-scratch replications of three papers on learning graph/network structure from multivariate
financial and time-series data. Each paper lives in its own self-contained folder with its own
solver, experiments, tests, and a single results notebook.

Replication status is tracked per paper below and in each folder's `CLAUDE.md`. Results are
labelled *reproduced*, *partial* or *not reproduced* against the source paper; deviations in
protocol or data generation are recorded rather than absorbed into the headline numbers.

| Folder | Paper | Method | Notebook |
|--------|-------|--------|----------|
| [`paper1_gmrf_laplacian/`](paper1_gmrf_laplacian/) | Cardoso & Palomar (2020), *Learning Undirected Graphs in Financial Markets* (`docs/2005.09958v4.pdf`) | GMRF graph-Laplacian (precision) estimation for clustering + a trading signal | `replication_notebook.ipynb` |
| [`paper2_ticc/`](paper2_ticc/) | Hallac, Vare, Boyd & Leskovec (KDD 2017), *Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data* (`docs/1706.03161v2.pdf`) | Subsequence clustering via block-Toeplitz inverse-covariance MRFs + dynamic-programming segmentation | `replication_notebook.ipynb` |
| [`paper3_market_modularity/`](paper3_market_modularity/) | Silva, Comin, Peron, Rodrigues, Ye, Wilson, Hancock & Costa (2015), *Modular Dynamics of Financial Market Networks* (`docs/1501.05040v3.pdf`) | Sliding-window correlation networks, Louvain communities, and a stochastic-blockmodel null that compresses the market to its community mixing matrix | `replication_notebook.ipynb` |

All three paper PDFs are in [`docs/`](docs/).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Paper 1 — GMRF graph Laplacians

Learns graph Laplacian (precision) matrices from stock returns under a Gaussian Markov Random
Field model, then uses the learned graphs for sector clustering and an algebraic-connectivity
trading signal. Four experiments (`exp1..4.py`) reproduce the paper's Figures 1–4. See
[`paper1_gmrf_laplacian/README`-equivalent docs](paper1_gmrf_laplacian/CLAUDE.md) for details.

```bash
cd paper1_gmrf_laplacian
python tests/test_solver.py      # sanity-check the solver (script-style checks)
python exp1_preprocessing.py     # → figures/fig1_preprocessing.png   (etc. for exp2..4)
jupyter notebook replication_notebook.ipynb
```

## Paper 2 — TICC (Toeplitz Inverse Covariance-Based Clustering)

Simultaneously segments and clusters multivariate time series: each cluster is a block-Toeplitz
Gaussian MRF (inverse covariance), fit via ADMM, and points are assigned by a dynamic-programming
(Viterbi) pass with a temporal-consistency penalty.

**Replication status — partial.** The paper's headline results reproduce: TICC substantially
outperforms every baseline on zero-mean structural clusters (Table 1), and reaches high accuracy
from far fewer samples (Fig 3). The solver is verified independently — DP matches brute force,
ADMM returns valid block-Toeplitz Θ, and there is no ground-truth leakage into the fit.

Not yet reproduced, and **not** to be cited as replicated:

- the `TICC, β=0` ablation is systematically too weak (avg 0.75 vs the paper's 0.88), which
  *overstates* the value of the temporal-consistency penalty;
- network recovery (Table 2) sits below the paper's 0.79–0.90 band;
- Fig 4 is a timing proxy over a narrower range, not an actual TICC run;
- window-size robustness, the micro-F1 cross-check and BIC selection of K are not implemented;
- the automobile-sensor case study uses proprietary data and is genuinely unavailable.

Committed tables are a single data seed with no error bars, and use 400 samples per segment rather
than the paper's `100·K`. Full accounting, including a 5-seed audit, in
[`paper2_ticc/CLAUDE.md`](paper2_ticc/CLAUDE.md) §5–6.

```bash
cd paper2_ticc
python -m pytest tests/
python experiments/exp1_table1_f1.py       # Table 1
python experiments/exp3_fig3_samples.py    # → figures/fig3_samples.png   (etc.)
jupyter notebook replication_notebook.ipynb
```

## Paper 3 — Modular dynamics of financial market networks

Builds a daily sequence of stock-correlation networks over 25 years, detects communities each day
with Louvain, then discards everything except the community sizes and the inter-community mixing
matrix and regenerates a random network from that summary alone. The claim is that this compressed
description reproduces most of the real network's topology, and that during crises the market
leaves a well-defined community structure for a much more uniform one.

The replication is organised around a 26-item methodological audit of the paper
([`PITFALLS.md`](paper3_market_modularity/PITFALLS.md)) as well as the reproduction itself. The
highest-value item: every correlation the paper reports is computed on 30-day sliding windows that
share 29/30 of their data, so ~200 effectively independent points are presented as 5,978. That
correction, and the crisis-detection evaluation the paper's own flowchart promises but never
performs, are the parts worth reading.

**Replication status — Phase 0 of 9 complete** (universe and price panel). N = 377 NYSE stocks ×
6,345 trading days, 1986-01-02 → 2011-02-28, against the paper's 348 × 6,008. Nothing is
reproduced yet; see [`paper3_market_modularity/CLAUDE.md`](paper3_market_modularity/CLAUDE.md) §8
for the phase checklist and §9 for known deviations.

Requires `python-igraph` (Louvain, exact clique number, betweenness) and, on this machine, the
`/opt/anaconda3` interpreter rather than `.venv`, which lacks `yfinance` and `pyarrow`.

```bash
cd paper3_market_modularity
python -m pytest tests/
python data/fetch_data.py                  # one-time price download (~10 min)
python experiments/exp1_data.py            # Phase 0 → data/UNIVERSE.md
jupyter notebook replication_notebook.ipynb
```
