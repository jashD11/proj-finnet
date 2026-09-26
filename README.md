# Spectral Graph Learning for Financial Networks

Three network-inference papers rebuilt from scratch on real market data, then used to test
whether the crisis indicators they propose actually carry a signal.

Summer Undergraduate Research Programme 2026, supervised by Prof. Rajasekhar Anguluri.
11,623 lines of Python across 70 files, 51 tests, three replications and one original study.

The short version: the papers' mechanisms mostly reproduce, and their indicators mostly do not
survive the null models the papers omit. Those negative results are the substance of the work
and are reported here rather than buried.

## Headline results

| # | Result | Source |
|---|---|---|
| 1 | **Louvain discards the crisis signal.** Its modularity `Q(t)` detects crises at **AUC 0.538** against 17 pre-registered windows, barely above a coin flip. The matrix `Q` is computed from reaches **0.760** on the same days and the same detector. | [`reports/probe/spectral_probe_auc.json`](reports/probe/spectral_probe_auc.json) (`q_dyn`, `mod_lambda_min`) |
| 2 | **The level of modularity is not evidence of modular structure.** Louvain on i.i.d. Gaussian noise pushed through the identical pipeline scores **Q = 0.2246 ± 0.0067** against the real market's **0.2215 ± 0.0413**, and reproduces **80.1 %** of the market's excess over its degree-sequence null. Only the *variance* of `Q` is informative. | [`phase3_communities.json`](paper3_market_modularity/results/phase3_communities.json), [`phase4_nulls.json`](paper3_market_modularity/results/phase4_nulls.json) |
| 3 | **The discarded threshold beats the published indicator.** Fixed-density thresholding throws away `τ(t)`; recovered at zero cost it detects crises at **AUC 0.807**, against modularity's 0.538. Base rate 2.68 %. | [`phase8_extensions.json`](paper3_market_modularity/results/phase8_extensions.json) |
| 4 | **A dead Fiedler signal traced to one sign.** An inverted temporal-penalty sign confined `λ₂` to `[0.00, 0.78]` with **65 of 243 windows at exactly zero**, making the paper's `τ = 1.0` gate non-binding and its trading figure vacuous. An exact edge-weight reformulation revived all 243: `λ₂ ∈ [0.6557, 1.5011]`, none at zero. | [`ENGINEERING_NOTES.md` §4](paper1_gmrf_laplacian/ENGINEERING_NOTES.md), [`data/lam2_series.npy`](paper1_gmrf_laplacian/data/) |
| 5 | **Two operators, one operator.** After degree normalisation the modularity matrix *is* the Laplacian: `B_n = I − L_sym − uuᵀ`, shared eigenvectors, eigenvalues paired `μ = 1 − λ`, verified to **1.8e-15** on all 243 windows. No normalised-`B` indicator can carry anything new. | [`modularity_indicator/FINDINGS.md`](modularity_indicator/FINDINGS.md), reproduce with `report_numbers.py` |
| 6 | **34 methodological defects logged** in Silva et al. (2015), 33 identified before implementation. The largest: every correlation the paper reports is computed on 30-day windows slid by one day, so its 6,315 points carry **211** independent ones. | [`PITFALLS.md`](paper3_market_modularity/PITFALLS.md), [`phase6_scoring.json`](paper3_market_modularity/results/phase6_scoring.json) |

The replication itself reproduces: the community null tracks real modularity at **ρ = 0.9836**
pre-2002 and 0.9882 post (paper: 0.99, 0.96), and across all 32 comparison cells the mean absolute
difference from the paper's own table is **0.1554** with the sign agreeing on **96.9 %**
([`tables/fig6_scoring.csv`](paper3_market_modularity/results/tables/fig6_scoring.csv),
[`phase6_scoring.json`](paper3_market_modularity/results/phase6_scoring.json)).

## Limitations

- **No trading-cost model anywhere.** There is no Sharpe ratio, no transaction costs, no slippage
  and no turnover in this repository. Every P&L figure is a gross cumulative log-return, evaluated
  in a single pass rather than walk-forward. Paper 1's strategy is 7 stocks across one crash, and
  on the paper's own window the gated strategy **loses** to buy-and-hold (S2 = −0.0111 against
  S1 = +0.0779): crash-avoidance replicates, the paper's P&L ranking does not.
- **The detection claims are underpowered and labelled as such.** A sliding-event null puts COVID
  at the 76th percentile for `λ₂` (`p = 0.243`) and the 66th for `μ_N(B)` (`p = 0.345`), so neither
  indicator is certified on one event
  ([`scores_significance.csv`](modularity_indicator/scores_significance.csv)). Result 1 above rests
  on 17 crisis windows, which is stronger, but still one market.
- **Survivorship bias is present and uncorrected, in two layers.** Silva et al. keep 348 of 3,799
  NYSE stocks with a complete 1986–2011 history, excluding every firm that failed in the crises
  they study. Our universe adds a second layer: the candidate list is a *currently-listed*
  directory, so 1,575 symbols yield 377 survivors. Both layers are documented, neither is fixed.
- **The TICC replication is partial and one result points the wrong way.** Segment length is 400
  throughout instead of the paper's `100·K`, up to 2× the data; tables are a single seed with no
  error bars; the `β=0` ablation comes out too weak, which *overstates* the paper's own
  temporal-penalty contribution; and the scalability figure is a timing proxy that never calls
  `fit_ticc`. Details in [`paper2_ticc/CLAUDE.md`](paper2_ticc/CLAUDE.md) §5–6.
- **Some checks are recorded as not implemented rather than quietly dropped.** TICC window-size
  robustness, micro-F1 and BIC selection of K; Paper 3's Δt robustness (A6), market-mode removal
  (A5) and the 2002-break mechanism (E11). Market-mode removal in particular bounds Result 2 above:
  real returns carry a market factor that i.i.d. noise does not.

## Repo map

| Path | Contents |
|---|---|
| [`paper1_gmrf_laplacian/`](paper1_gmrf_laplacian/) | Cardoso & Palomar (2020), *Learning Undirected Graphs in Financial Markets*. GMRF graph-Laplacian estimation for sector clustering and an algebraic-connectivity trading signal. Solver, 4 experiments, [engineering notes](paper1_gmrf_laplacian/ENGINEERING_NOTES.md). |
| [`paper2_ticc/`](paper2_ticc/) | Hallac, Vare, Boyd & Leskovec (KDD 2017), *TICC*. Subsequence clustering by block-Toeplitz inverse covariance with dynamic-programming segmentation. Solver, 6 baselines, 4 experiments. |
| [`paper3_market_modularity/`](paper3_market_modularity/) | Silva et al. (2015), *Modular Dynamics of Financial Market Networks*. 6,315 daily correlation networks over 25 years, Louvain communities, a blockmodel null, and the [34-item audit](paper3_market_modularity/PITFALLS.md) the replication is organised around. |
| [`modularity_indicator/`](modularity_indicator/) | Original study: does the modularity matrix `B` yield an equivalent of the Fiedler value `λ₂(L)`? Answer, negative, in [`FINDINGS.md`](modularity_indicator/FINDINGS.md). |
| [`reports/`](reports/) | Write-ups, plus [`probe/`](reports/probe/) — spectral evidence over all 6,315 networks. Start at [`reports/README.md`](reports/README.md). |
| [`docs/`](docs/) | The three source papers as PDFs. |

Each paper folder is self-contained and shares no code with the others. Each has a `CLAUDE.md`
recording its own status, deviations and gaps, labelled *reproduced / partial / not reproduced /
not implemented*.

## Setup and reproduce

Python 3.12 (verified on 3.12.0 and 3.12.4).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements-lock.txt for exact versions
```

Tests first; all three suites together finish in under five seconds.

```bash
./run_tests.sh            # all three, one command
```

Or individually:

```bash
python paper1_gmrf_laplacian/tests/test_solver.py   # 4 assertion checks, ~0.6 s
cd paper2_ticc && python -m pytest tests/ -q        #  3 tests, ~2.1 s
cd paper3_market_modularity && python -m pytest tests/ -q   # 48 tests, ~0.9 s
```

A single root-level `pytest` does **not** work, and that is structural rather than a
bug: each paper folder is self-contained and they have identically named
subpackages (`solver/`, `utils/`, `tests/`, `data/`), so whichever is imported first
shadows the others. `run_tests.sh` runs them as three processes.

Which script regenerates which artifact:

| Command | Runtime | Writes |
|---|---|---|
| `cd paper1_gmrf_laplacian && python exp1_preprocessing.py` … `exp4_trading.py` | seconds each, `exp2` ~1 min | `figures/fig1..4*.png`; `exp3` caches `data/lam2_series.npy` |
| `cd paper2_ticc && python experiments/exp1_table1_f1.py` … `exp4_*.py` | ~1–3 min each | `results/table1_f1.json`, `table2_recovery.json`, `fig3_samples.json`, `fig4_scalability.json` |
| `cd paper3_market_modularity && python data/fetch_data.py` | ~10 min, network | `data/panel_prices.parquet` (already committed) |
| `cd paper3_market_modularity && python run_phases.py 1 9` | ~25 min total; Phase 3 217 s, Phase 4 194 s, Phase 5 1,289 s and resumable | `results/phase1..8_*.json`, `results/series/*.parquet`, `results/tables/*.csv`, `figures/*.png` |
| `cd paper3_market_modularity && python experiments/exp9_report.py` | seconds | `results/REPLICATION_REPORT.md`, and rebuilds `replication_notebook.ipynb` **without outputs** |
| `cd paper3_market_modularity && jupyter nbconvert --to notebook --execute --inplace replication_notebook.ipynb` | ~8 s | restores the notebook's stored outputs, so it renders on GitHub |
| `cd paper3_market_modularity && jupyter nbconvert --to html --output results/replication.html replication_notebook.ipynb` | seconds | `results/replication.html` |
| `cd modularity_indicator && python build_series.py && python evaluate.py && python robustness.py && python significance.py` | under 1 min total | `series.parquet`, `scores*.csv` |
| `python reports/probe/spectral_probe.py` then `duality.py` | ~6 min each | `reports/probe/spectral_probe.parquet`, `*_auc.json` |

Two scripts exist only to check that the prose matches the data, and exit non-zero on any mismatch:

```bash
python reports/probe/check_numbers.py        # all 91 values in the 19 Aug report
python modularity_indicator/report_numbers.py # every number in the 8 Sep report
```

Phase-level artifacts under `paper3_market_modularity/data/` (`edges.npy`, `returns.npy`,
`partitions.npy`, `mixing.npz`) are gitignored as rebuildable; `run_phases.py 1 3` regenerates
them. The price panel is committed so the results are reproducible without a fresh download.

**Data provenance.** Prices are Yahoo Finance adjusted closes fetched with `yfinance` for academic
replication, cached in `paper3_market_modularity/data/panel_prices.parquet` and the two CSVs under
`paper1_gmrf_laplacian/data/`. They are not redistributable as a dataset; rebuild them from source
with the `fetch_data.py` scripts. The figure crops in `paper3_market_modularity/docs/paper_figs/`
are from the Silva et al. preprint, reproduced for side-by-side comparison only.

## Reports

- [`reports/2026-09-08_modularity_equivalence.tex`](reports/2026-09-08_modularity_equivalence.tex)
  — current. Whether `B` yields an equivalent of the Fiedler value. The answer is negative in two
  parts, and the negative answer is the deliverable. 10 pages.
- [`reports/2026-08-19_discrepancy_explained.tex`](reports/2026-08-19_discrepancy_explained.tex)
  — the three-paper exploration built up from Silva's own definitions. 17 pages. Its §9 is
  superseded and says so; the measurements stand, the spine built on them does not.
- [`reports/progress_report.tex`](reports/progress_report.tex) — running dated working log.
- [`paper3_market_modularity/results/REPLICATION_REPORT.md`](paper3_market_modularity/results/REPLICATION_REPORT.md)
  — generated claim-by-claim comparison against the paper.

## Status

The three replications are complete to the extent recorded in each folder's `CLAUDE.md`; Paper 3
runs Phases 0–9, Paper 2 is explicitly partial. Active work is the spectral direction: `μ_N(B)`,
the most negative modularity eigenvalue, is the best modularity-derived crisis signal on both
constructions tested — 0.760 on Paper 3's 6,315 networks and 0.775 on Paper 1's FAAMUNG data — and
is significant on neither alone. The next tests are a third, unrelated network construction, then
the two robustness checks currently out of scope (A5 market-mode removal, A6 window length).
