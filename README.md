# Financial Networks Project

Faithful, from-scratch replications of two papers on learning graph/network structure from
multivariate financial and time-series data. Each paper lives in its own self-contained folder
with its own solver, experiments, tests, and a single results notebook.

| Folder | Paper | Method | Notebook |
|--------|-------|--------|----------|
| [`paper1_gmrf_laplacian/`](paper1_gmrf_laplacian/) | Cardoso & Palomar (2020), *Learning Undirected Graphs in Financial Markets* (`docs/2005.09958v4.pdf`) | GMRF graph-Laplacian (precision) estimation for clustering + a trading signal | `replication_notebook.ipynb` |
| [`paper2_ticc/`](paper2_ticc/) | Hallac, Vare, Boyd & Leskovec (KDD 2017), *Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data* (`docs/1706.03161v2.pdf`) | Subsequence clustering via block-Toeplitz inverse-covariance MRFs + dynamic-programming segmentation | `replication_notebook.ipynb` |

Both paper PDFs are in [`docs/`](docs/).

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
(Viterbi) pass with a temporal-consistency penalty. Reproduces the paper's quantitative results on
synthetic data (Table 1 clustering accuracy, Table 2 network recovery, Fig 3 sample sweep, Fig 4
scalability). The paper's automobile-sensor case study uses proprietary data and is documented as
non-reproducible. See [`paper2_ticc/CLAUDE.md`](paper2_ticc/CLAUDE.md).

```bash
cd paper2_ticc
python -m pytest tests/
python experiments/exp1_table1_f1.py       # Table 1
python experiments/exp3_fig3_samples.py    # → figures/fig3_samples.png   (etc.)
jupyter notebook replication_notebook.ipynb
```
