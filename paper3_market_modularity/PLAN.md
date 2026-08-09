# Paper 3 — Silva et al. (2015), *Modular Dynamics of Financial Market Networks*

## Context

`docs/1501.05040v3.pdf` (arXiv 1501.05040v3, Silva, Comin, Peron, Rodrigues, Ye, Wilson, Hancock, Costa) was dropped into the repo untracked and has no code against it. It joins `paper1_gmrf_laplacian` (Cardoso & Palomar) and `paper2_ticc` (Hallac et al.) as the third from-scratch replication.

The paper builds a daily sequence of correlation networks over 348 NYSE stocks (1986–2011), detects Louvain communities each day, and argues that a stochastic blockmodel driven only by community sizes and the inter-community mixing matrix `Π` reproduces most of the real network's topology — including the loss of community structure during crises.

The user has read the paper and produced an audit of 26 methodological pitfalls (A1–E11), plus a 9-phase execution spec (`~/Downloads/silva2015_replication_plan.md`). Both are inputs to this build. **The replication is therefore not just "reproduce Fig. 6" — it is "reproduce it, then report what the paper's own construction hides."** The highest-value single result is E1: every ρ in the paper is computed on 30-day sliding windows sharing 29/30 of their data, so ~200 independent points are reported as 5,978.

### Decisions taken (user-confirmed)

| | |
|---|---|
| **Layout** | Repo convention (`paper2_ticc` style), not the spec doc's `config.py`/`src/`/`notebooks/` |
| **Scope** | Phases 0–9, with Phase 8 trimmed to **8a, 8b, 8c, 8f**; **8d (Δt robustness) and 8e (market-mode removal) are out of scope** — both require full Phase 2–6 reruns |
| **Universe** | NYSE survivors, paper-faithful; survivorship drop count reported as a number, not fixed |
| **Cadence** | **Stop after each phase**, run that phase's acceptance tests, show results, wait for go-ahead |

---

## Environment facts (verified this session, not assumed)

- **Interpreter: `/opt/anaconda3/bin/python` (3.12.4).** Has `yfinance 1.4.1`, `pandas 2.2.2`, `numpy 1.26.4`, `scipy`, `sklearn`, `pyarrow`, `certifi`, `tqdm`. `.venv/bin/python` has numpy/pandas/scipy/matplotlib/sklearn/pytest but **no** yfinance, pyarrow, or certifi (raw `urllib` fails SSL verification there). This matches the gotcha already recorded in `paper1_gmrf_laplacian/PIPELINE.md §8`.
- **`python-igraph` is missing from both environments.** It is a hard requirement (Louvain, exact clique number, betweenness, degree-sequence rewiring — all C-speed). **First action of Phase 0 is `pip install python-igraph` into the anaconda env, which requires user approval.** networkx fallback is ~20–50× slower and would push Phase 5 from minutes into many hours.
- **Data is live.** `yf.download("GE", start="1986-01-01", end="2011-03-01", auto_adjust=True)` → **6,345 rows, 1986-01-02 → 2011-02-28, zero NaNs**. `auto_adjust=True` gives split/dividend-adjusted closes, which resolves audit A2 directly.
- **Candidate universe source works.** `https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt` → 2,849 non-ETF NYSE symbols, **1,575 tagged "Common Stock"**. Paper had 3,799 candidates → 348 survivors.

## Paper ground truth (extracted from the PDF, for the report tables)

- N = 348 of 3,799 NYSE stocks; `C_p` = 6,008 closes; `N_w = C_p − Δt` = **5,978** networks. Δt = 30, δt = 1, f = 10% → 6,038 edges/day, mean degree ≈ 34.7.
- `N_w = C_p − Δt` pins the window convention: window *n* spans **price** indices `n … n+30` (31 prices, 30 returns) → `Y[n−1 : n−1+30]`.
- Three modularities: dynamical, fixed (`C(0)`), lagged (`t_Δ = 100`).
- Fig. 6 headline: modularity ρ = **0.99 / 0.96** (community model, pre/post-2002) vs **0.07 / 0.71** (configuration model); assortativity ρ = **−0.47** for the configuration model pre-2002. Full Fig. 6 / Fig. S3 χ² and ρ tables extracted and will be hard-coded into `results/REPLICATION_REPORT.md` as the paper column.
- Crisis bands: **17 named events, but only Black Monday (1987-10-19) has an explicit date in the text.** The other 16 exist only as drawn shading. We must supply ranges ourselves.

### The timestamp convention (audit A10) — already partly solved

Fig. 8 samples the sequence "every 1000 time steps" and prints five dates: 1991-05-10, 1995-04-25, 1999-04-12, 2003-04-03, 2007-03-26. On our Yahoo calendar these land at 1-based trading-day indices **1355, 2355, 3355, 4355, 5355** — *exactly* 1000 apart. So the paper's calendar spacing matches ours over 1991–2007, with a constant offset: **paper window `n` ↔ our trading day `n + 355`.**

Our 6,345 days vs the paper's 6,008 is a 337-day gap. If those dropped days were spread evenly the offset would be ~283, not 355 — so the drops must be concentrated in 1986–early 1987, i.e. **the paper's usable panel effectively starts around mid-1987** once every one of its 348 tickers is required to have a price. That is consistent with Fig. 4's x-axis starting at 1988 and Fig. 3's window opening in Aug 1987. Phase 0/1 will test this hypothesis by reporting the by-year distribution of dropped days under our own missing-data policy.

**Convention adopted:** every network is stamped at **`t2`, the last day of its window** (retrospective — the network is only knowable at window close). The `+355` offset is stored as a calibration diagnostic in `results/tables/date_calibration.csv`, and Black Monday alignment against Fig. 3 is a Phase 2 acceptance test.

---

## Directory layout

New folder `paper3_market_modularity/`, self-contained, sharing nothing with paper1/paper2 (paper2's stated convention).

```
paper3_market_modularity/
  CLAUDE.md                     source of truth, paper2 §-structure
  PITFALLS.md                   the A1–E11 audit  ← explicit user deliverable
  replication_notebook.ipynb    Phase 9
  data/
    __init__.py
    universe.py                 nasdaqtrader candidate list + filters
    fetch_data.py               yfinance panel, skip-guard cache
    UNIVERSE.md                 provenance, N, T, drop counts, deltas vs paper
    panel_prices.parquet        (N × T) adjusted closes, dense
    returns.npy                 (N × T−1) log returns
  net/
    __init__.py
    construct.py                rolling corr, fixed-density threshold, τ(t)
    communities.py              multi-seed Louvain, Π(t), three modularities
    nullmodels.py               blockmodel (Eq. 2), configuration model
    measures.py                 the 8 measures, both disconnection conventions
  utils/
    __init__.py
    scoring.py                  χ², normalized MSE, overlap-corrected ρ, block bootstrap
    crisis_dates.py             17 pre-registered crisis windows
    plots.py                    every figure as a reusable function
  experiments/
    exp1_data.py exp2_networks.py exp3_communities.py exp4_nulls.py
    exp5_measures.py exp6_scoring.py exp7_pca.py exp8_extensions.py
  tests/
    __init__.py  test_paper3.py
  results/
    series/*.parquet  tables/*.csv  *.json  REPLICATION_REPORT.md
  figures/
    fig3_blackmonday.png fig4_modularity_pathlen.png fig5_three_modularities.png
    fig6_measures.png figS3_measures.png fig7_pca.png fig8a_tau.png
    fig8b_roc.png fig8c_eventwindow.png fig8f_louvain_stability.png
```

Conventions carried over from `paper2_ticc` (see `paper2_ticc/CLAUDE.md`, `paper2_ticc/experiments/exp1_table1_f1.py`):

- Every experiment module exposes `run(force=False, verbose=True)` returning a dict and caching to `results/*.json`, plus `plot(data, save=...)`; `if __name__ == "__main__": run(force="--force" in sys.argv)`.
- `sys.path.insert(0, <paper root>)` header in every script and test; no `setup.py`.
- Hyperparameters as UPPERCASE constants at the top of `exp1_data.py`; downstream experiments import from it (paper2's de-facto config module — this satisfies the spec doc's "single config" rule without adding a `config.py` that breaks convention).
- `DATA_SEED` / `FIT_SEED` split; `matplotlib.use("Agg")`; `savefig(dpi=130)` then `plt.close("all")`; never `plt.show()` in scripts.
- Tests are pytest-collectable zero-arg functions with a `__main__` fallback, run via `python -m pytest tests/`.

---

## Phase-by-phase build

Each phase is one commit, ends with its acceptance tests **run and reported**, then stops for review. An acceptance test that fails gets debugged, never weakened.

### Phase 0 — `exp1_data.py` (part 1): universe and panel
Install `python-igraph`. Build candidate list from nasdaqtrader (Exchange `N`, non-ETF, non-test, "Common Stock"); download adjusted closes 1986-01-01 → 2011-03-01 in batches with a skip-guard cache; apply the completeness filter. Write `UNIVERSE.md` with N, T, date range, **and the drop count** — audit A1 made visible. Implement both missing-day policies (drop-day across all tickers = default; forward-fill = flag) and report the **by-year distribution of dropped days** to test the mid-1987 hypothesis above.
*Accept:* dense (N × T), zero NaNs, every ticker exactly T obs, N ≥ 200, T ≥ 3000, `UNIVERSE.md` states all four numbers.
Also written this phase (before any results are seen, for E10 honesty): `utils/crisis_dates.py` with all 17 events as explicit pre-registered `(start, end)` ranges, sourced and dated in comments.

### Phase 1 — `exp1_data.py` (part 2): returns and validation
`Y = diff(log(P))`. Per-ticker mean/std/min/max; return distribution plot. **Split detector: flag every `|Y| > 0.5` and investigate each hit** — with `auto_adjust=True` there should be near-zero, and any survivor is a real data defect. Store the cross-sectional mean correlation per 30-day window as `results/series/rho_bar.parquet` — the crisis signal the paper's construction discards (A7), needed in Phase 8a.
*Accept:* median per-ticker daily std ∈ [0.01, 0.04]; zero unexplained `|Y| > 0.5`; excess kurtosis > 1.

### Phase 2 — `exp2_networks.py` + `net/construct.py`
Rolling Pearson correlation on each (N × 30) slice; fixed-density threshold via `np.partition` on the upper triangle so exactly `f·N(N−1)/2` edges survive daily. **Store `tau[t]` for every day** (A7). Store edge lists packed as `uint16` (~150 MB for 5,978 days — acceptable; null-model graphs are *not* stored, they are regenerated from `Π` + recorded seed).
*Accept:* mean degree `= f·(N−1)` **exactly, every day**; `tau[t]` spikes Oct 1987 and Sep–Oct 2008; isolated-node count spikes in the Black Monday window (Fig. 3B signature) — **if this doesn't reproduce, stop and debug**; one window spot-checked against `scipy.stats.pearsonr`; date calibration table written and the `+355` offset reported.

### Phase 3 — `exp3_communities.py` + `net/communities.py`
`igraph.Graph.community_multilevel`, **10 seeds per day**, all retained (B1). Store `k[t]` and the full `Π[t]` per day (B4/D1 depend on it). Compute dynamical / fixed / lagged (`t_Δ = 100`) modularity, plus `Q_dyn − Q_lag`, which the paper never takes. Figs. 3, 4, 5.
*Accept:* dynamical Q roughly in [0.05, 0.55]; `Q_fixed ≤ Q_lagged ≤ Q_dyn` **every day** (a violation is a bug); Q drops sharply in the Black Monday window; **seed-to-seed spread in Q reported and compared against day-to-day movement — if comparable, say so loudly**; `k[t]` median and crisis-window value reported.

### Phase 4 — `exp4_nulls.py` + `net/nullmodels.py`
Eq. 2 blockmodel with **independent coin flips** (deliberately keeping D2's randomness rather than fixing it), and the configuration model via `Graph.Degree_Sequence` (record which method — `configuration` vs `vl` differ on multi-edges). **10 replicates/day/model** (D3). Record realized-vs-target edge counts per block to quantify D2.
*Accept:* blockmodel total edges within a few percent of real, daily; **blockmodel modularity against the planted partition ≈ real dynamical modularity** (the unit test for Eq. 2); configuration model reproduces the degree sequence (report multi-edge collapse loss); **configuration-model modularity ≈ 0.10 and flat — this is the noise floor (B6), drawn on every modularity figure from here on.**

### Phase 5 — `exp5_measures.py` + `net/measures.py`
8 measures × 3 families × ~5,978 days × 10 replicates ≈ 126k graph evaluations.
- Disconnection handled explicitly (E6): **both** largest-component-only and harmonic-mean/global-efficiency series stored for path length and betweenness.
- Rich-club: **both** raw φ(k) and φ normalized against the configuration model (D4).
- Aggregate replicates to mean ± sd per measure per day.
- **Compute-budget guard:** exact clique number is the one measure that can blow up (the paper reports ω reaching 40–80, which is enormous for n=348 at 10% density). Time it on ~20 sample days first; if the projected cost exceeds ~30 min, subsample clique number to every k-th day and **state the stride in the report** rather than silently substituting a heuristic.
*Accept:* every series length `N_w`, no undocumented NaNs; clique number reaches 40+ somewhere (Fig. S3); configuration-model transitivity low and flat; **blockmodel assortativity strictly positive** — this is the paper's key mechanism and confirms the generator.

### Phase 6 — `exp6_scoring.py` + `utils/scoring.py`
The paper's numbers, then the honest ones:
1. χ² (mean squared difference) and Pearson ρ per measure per model, split at 2002 → reproduces Fig. 6 / Fig. S3 tables.
2. **Normalized MSE** = χ² / var(real series) (E2) — the paper's raw values span 0.0188 to 2,522 and are not comparable.
3. **ρ recomputed on non-overlapping windows** (every Δt-th day, ~200 points) (E1). *This is the single highest-value result in the replication.* Both numbers side by side, difference stated explicitly.
4. Block-bootstrap CIs on every ρ, block length ≥ Δt (E4).
5. **Inter-measure correlation matrix** on the real networks (E5) — quantifies how few of the "eight facts" are independent.
6. **Compression check** (D1): plot `k + k(k+1)/2` vs `N` over time; report the fraction of days the community summary is genuinely cheaper, and separately **the fraction during crisis windows**, where B5 predicts it inverts.
*Accept:* community-model modularity ρ > 0.9; configuration-model modularity ρ low pre-2002; configuration-model assortativity ρ negative pre-2002; overlapping and non-overlapping ρ both reported for **every** measure.

### Phase 7 — `exp7_pca.py`
Standardize the 8 measures, day × 8 matrix for all three families, PCA → PCA1/PCA2 scatter (Fig. 7). **Report explained-variance ratios and full loadings** (E7). **Test "PCA1 is time" as a number** — correlate PCA1 scores against the date index. Quantify cloud separation (centroid distances + a simple classifier's accuracy) rather than eyeballing it (E8).
*Accept:* variance ratios reported and summing correctly; PCA1-vs-time correlation reported as a number.

### Phase 8 — `exp8_extensions.py` (8a, 8b, 8c, 8f only)
- **8a τ(t) signal (A7):** overlay τ(t), dynamical modularity, and mean correlation with crisis bands; state which tracks crises better.
- **8b the crisis detector the paper never built (E10):** Fig. 1 has a box labelled "crisis detection evaluation" and the paper contains no rule, threshold, or score. Define a z-score-vs-trailing-baseline rule on modularity; score precision / recall / ROC against the **pre-registered** windows from Phase 0; compare detectors on modularity vs τ(t) vs mean correlation.
- **8c rise-or-fall (C1):** the abstract says crisis destroys structure, the Fig. 4 text says modularity *increases* around the crisis. Event-window analysis: align all crisis onsets, average modularity over ±60 days. Settle whether the crash day is a trough inside an elevated bear-market period.
- **8f Louvain stability (B1/B2):** using Phase 3's 10 seeds/day, report NMI between seeds and NMI between consecutive days. **If seed-to-seed disagreement ≈ day-to-day disagreement, the daily modularity series is substantially algorithmic noise** — and that finding must lead the report, not be buried.
*(Out of scope by decision: 8d Δt = 20/60/120 robustness, 8e market-mode removal. Both are recorded in `PITFALLS.md` as untested audit items so nothing is silently dropped.)*

### Phase 9 — `replication_notebook.ipynb`
**Loads artifacts, never computes them.** Every cell reads `results/` and calls a `utils/plots.py` function. Fresh-kernel Run-All under 2 minutes. Every code cell sandwiched between a "what question and why it matters" markdown cell and a "what the output actually shows, with concrete numbers" cell. Every technical term defined inline on first use. Sections §0–§9 mirroring the paper's argument, with §6's overlap correction given the most space and §9 a claim-by-claim verdict table.

Follows **paper2's** notebook style (`import expN as e; res = e.run(verbose=False)`) rather than paper1's `%run`, since the experiment modules already expose `run()`/`plot()`. Absolute hardcoded `PROJECT_ROOT` + `os.chdir` in the §0 setup cell, matching both existing notebooks.
*Accept:* Run-All clean under 2 min; **zero code cells lacking both surrounding markdown cells, checked programmatically over the `.ipynb` JSON**; no cell recomputes a Phase 2–8 result; every figure has axis labels, units, title; term-definition grep passes; static HTML exported to `results/replication.html`.

---

## Repo-level changes outside `paper3_market_modularity/`

- `README.md` — "two papers" → "three"; new index-table row; new `## Paper 3 — Modular Dynamics of Financial Market Networks` section with description, an honest **Replication status** paragraph in paper2's style, and the run block.
- `requirements.txt` — append `python-igraph>=0.11  # paper 3 (Silva): Louvain, cliques, degree-sequence rewiring` and `pyarrow>=14  # paper 3: parquet series`.
- `git add docs/1501.05040v3.pdf` (currently the repo's only untracked file).
- New branch `add-silva-replication` off current HEAD. Note: `add-ticc-replication` is not yet merged into `main`.

## Deliverables

1. `paper3_market_modularity/PITFALLS.md` — the A1–E11 audit verbatim, each item tagged with the phase that addresses it, the result once known, and **explicitly marked UNTESTED for 8d/8e**.
2. `paper3_market_modularity/replication_notebook.ipynb` — the narrative walkthrough.
3. `paper3_market_modularity/results/REPLICATION_REPORT.md` — terse reference: per claim, the paper's number, our number, and whether the gap is *dataset*, *ambiguity resolved differently*, or *a genuine problem with the original analysis*.
4. `paper3_market_modularity/CLAUDE.md` — source of truth, paper2's section structure, including a "Known deviations" table.

## Verification

- `cd paper3_market_modularity && /opt/anaconda3/bin/python -m pytest tests/` — unit tests for the fixed-density thresholder (exact edge count), Eq. 2 normalization (blockmodel reproduces planted modularity), the configuration model (degree sequence preserved), the overlap-corrected ρ (recovers the known value on a synthetic AR series), and the harmonic-mean path length on a deliberately disconnected graph.
- Each `experiments/expN_*.py` runnable standalone, printing its own acceptance-test results with `✓`/`✗`.
- End-to-end: `for f in experiments/exp*.py; do python $f; done` from cold caches, then `jupyter nbconvert --execute replication_notebook.ipynb` clean.
- **The report is judged on honesty, not agreement.** Following the existing repo's editorial rule: label each result *reproduced / partial / not reproduced / not implemented*, and never round toward the paper. If the non-overlapping ρ collapses from 0.99 to 0.7, that is the headline.
