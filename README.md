# Financial Networks Project

Replication of Cardoso & Palomar (2020), *"Learning Undirected Graphs in Financial Markets"* (arXiv:2005.09958v4).

Learns graph Laplacian (precision) matrices from stock returns under a Gaussian Markov Random Field model via a custom ADMM solver, then uses the learned graphs for clustering and a trading strategy.

## Experiments

| Fig | Experiment | Data | Script |
|-----|-----------|------|--------|
| 1 | Preprocessing effects | 130 S&P 500 stocks (3 sectors) | `exp1_preprocessing.py` |
| 2 | Degree-controlled k-component graphs vs SGL | same 130 stocks | `exp2_kcomponent.py` |
| 3 | Time-varying graphs (rolling window) | FAAMUNG (7 stocks) | `exp3_timevarying.py` |
| 4 | Trading strategy (algebraic connectivity signal) | FAAMUNG (7 stocks) | `exp4_trading.py` |

## Setup

```bash
pip install -r requirements.txt
python data/fetch_data.py   # one-time data pull, cached to data/*.csv
python -m pytest tests/     # sanity-check the ADMM solver
```

Each `exp*.py` script is standalone: it loads the cached CSVs, runs the solver/algorithm, and writes a figure to `figures/`.

## Structure

- `solver/` — ADMM Laplacian solver (`admm_graph.py`), k-component wrapper (`algorithm1.py`), time-varying wrapper (`algorithm2.py`), SGL baseline (`sgl_benchmark.py`)
- `utils/` — preprocessing, graph metrics, visualization
- `data/` — ticker lists and cached log-return CSVs
- `tests/` — solver sanity checks
- `figures/` — output plots
