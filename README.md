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

**Replication status — complete, Phases 0–9.** N = 360 NYSE stocks x 6,345 trading
days against the paper's 348 x 6,008; 6,315 daily networks, 132,615 graph evaluations.
The paper's headline reproduces: the community null tracks real modularity at rho = 0.98
(paper: 0.99). Across all 32 comparison cells the mean absolute difference from the
paper's own table is 0.155 and the sign agrees on 97 %.

Four findings the paper's own construction does not support:

- **Modularity's level is not evidence.** Louvain on i.i.d.-noise networks scores
  Q = 0.225 against the real market's 0.222, and reproduces 80 % of the market's
  excess over its degree-sequence null. Thresholded correlation matrices are
  transitive by construction, and no degree-based null controls for that.
- **The community model wins on one measure only.** Normalized by the variance of
  the real series, it beats the degree sequence on modularity — the quantity Pi
  directly encodes — and loses on six of the other seven.
- **The compression claim inverts during crises**, from cheaper on 69 % of all days
  to 34 % of crisis days, because the network shatters into singleton communities.
- **tau(t), which fixed-density thresholding discards, detects crises at AUC 0.807.
  Modularity manages 0.538**, barely above chance.

The audit's own central hypothesis — that correcting for 29/30 window overlap would
collapse the evidence — is **not supported**: mean rho moves 0.783 to 0.777. That
negative result is reported as prominently as the positive ones.

Full accounting in [`PITFALLS.md`](paper3_market_modularity/PITFALLS.md) and
[`results/REPLICATION_REPORT.md`](paper3_market_modularity/results/REPLICATION_REPORT.md).

Requires `python-igraph` (Louvain, exact clique number, betweenness) and, on this machine, the
`/opt/anaconda3` interpreter rather than `.venv`, which lacks `yfinance` and `pyarrow`.

```bash
cd paper3_market_modularity
python -m pytest tests/
python data/fetch_data.py                  # one-time price download (~10 min)
python run_phases.py 1 9                   # everything (~25 min; Phase 5 is resumable)
python experiments/exp6_scoring.py         # or one phase at a time
jupyter notebook replication_notebook.ipynb
```

## Original research on top of the replications

Since 12 August 2026 the project has been building an original comparison on top of
the three rebuilds: the **graph Laplacian** `L = D − A` and the **modularity matrix**
`B = A − ddᵀ/2m` are the operators behind the two crisis indicators in this
literature (Paper 1's Fiedler value `λ₂(L)` and Paper 3's modularity `Q`), and the
question is what their spectra actually encode.

| Folder | What it holds |
|--------|---------------|
| [`reports/`](reports/) | Write-ups for supervision, plus `probe/`, the spectral evidence over Paper 3's full 6,315-day network sequence. Start with [`reports/README.md`](reports/README.md). |
| [`modularity_indicator/`](modularity_indicator/) | The 31 Aug to 7 Sep study asking whether `B` yields an equivalent of the Fiedler value, tested on Paper 1's FAAMUNG data. |

**Where it stands.** Two results, one from each direction:

- **The modularity matrix beats the modularity scalar.** On Paper 3's networks and
  its own detector, Louvain's `Q(t)` scores AUC 0.538 against a 0.50 coin flip, while
  three readings of the matrix `Q` is derived from reach 0.72 to 0.76. The
  information was in `B` all along and the Louvain step is where it is lost.
- **`B` and `L` are the same operator up to degree normalisation, and that closes the
  obvious route.** Normalised, `B_n = I − L_sym − uuᵀ` with shared eigenvectors and
  eigenvalues paired as `μ = 1 − λ`, verified to 1.8e-15, so no normalised-`B`
  indicator can carry anything new. The unnormalised shortcut that appeared to work
  requires a regular graph and does not hold here. What survives is that `λ₂` blends
  structure with degree, and the two matrices differ only in how they combine them.

Both findings are negative or deflationary about the incumbent indicators, including
Paper 1's own: under its published setup, `λ₂` does not separate the COVID window
from an arbitrary volatile stretch of the same series (`p = 0.243`). The write-ups
say so plainly rather than reporting the favourable framing.

The current report is
[`reports/2026-09-08_modularity_equivalence.tex`](reports/2026-09-08_modularity_equivalence.tex);
[`reports/progress_report.tex`](reports/progress_report.tex) is the running log.
