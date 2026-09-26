# RESUME_BRIEF.md

Factual brief, 2026-09-27, branch `add-silva-replication`. **V** = VERIFIED (machine-written results file,
or reproduced by a command this session). **R** = REPORTED (prose only).

## A. Summary

Three from-scratch replications of papers that learn network structure from multivariate financial time
series, plus original work on what the graph Laplacian `L = D − A` and modularity matrix `B = A − ddᵀ/2m`
spectra encode. Each replication is instrumented with a methodological audit rather than a pass/fail
verdict. The conclusion is deflationary: the headline mechanisms mostly reproduce, but the published crisis
indicators do not survive the null models the papers omit. Louvain modularity detects crises at AUC 0.538
against a coin flip; the threshold `τ(t)` the construction discards reaches 0.807 (V).

## B. Data (V)

| Dataset | Universe | Period | Freq |
|---|---|---|---|
| FAAMUNG (`paper1/data/faamung.csv`) | 7 US tech stocks | 2019-06-04→2020-06-30, 272 rows | daily log returns |
| S&P 500 3-sector (`sp500_3sectors.csv`) | **97 cols** (docs say 130) | 2016-01-05→2019-12-31, 1005 rows | daily log returns |
| NYSE panel (`phase0/1_*.json`) | 1,575 requested → 377 → **N = 360** after quality gates | 1986-01-02→2011-02-28, T = 6,345, 6,315 networks | daily |
| TICC | **synthetic only** (case-study data proprietary), 5-dim, 4 sequences | n/a | n/a |

yfinance adjusted closes throughout. **No train/validation/test split anywhere.** Paper 3's pre/post-2002
split is an *era* comparison (`exp1_data.py:53`). The only true out-of-sample test is `probe/incremental.py`,
chronological pre/post-2002: market-mode AUC 0.812 in-sample → 0.730 OOS. TICC fixes K to its true value.

## C. Methods

- **Paper 1**: GMRF Laplacian estimation by edge-weight reparameterization + L-BFGS-B,
  `L(w)=Kᵀdiag(w)K` (`solver/admm_graph.py`); Algorithm 1 alternating V/L-step; Algorithm 2 causal rolling
  window with exact Frobenius temporal penalty; SGL baseline.
- **Paper 2**: Toeplitz graphical lasso via ADMM (`toeplitz_admm.py`), exact Viterbi DP (`dp_assign.py`),
  EM with 20 contiguous restarts (`ticc.py`); baselines GMM, EEV, k-means, DTW-GAK, DTW-Euclidean,
  Neural Gas.
- **Paper 3**: rolling Pearson, fixed-density threshold with τ re-solved daily (`net/construct.py`);
  Louvain × 10 seeds + mixing matrix Π (`communities.py`); Eq. 2 blockmodel null + configuration model by
  degree-preserving edge swaps (`nullmodels.py`); 8 measures (modularity, path length, assortativity,
  transitivity, betweenness, clique number, rich club, matching index, `measures.py:37`).
- **Statistical tests**: block bootstrap CIs, non-overlapping resampling, χ²/normalized MSE, PCA
  (`utils/scoring.py`); AUC/F1 vs 17 pre-registered crisis windows (`crisis_dates.py`); circular-shift,
  random-gate and sliding-event nulls, paired ΔAUC bootstrap (`modularity_indicator/`).
- **NOT IMPLEMENTED** (in docs, absent from code): TICC window-size robustness, micro-F1, BIC selection of
  K, centrality cluster interpretation, automobile case study; Paper 3 Δt robustness (A6), market-mode
  removal (A5), 2002-break mechanism (E11). `exp4_fig4_scalability.py` **never calls `fit_ticc`** — it times
  a hand-rolled proxy (R).

## D. Evaluation

Metrics: macro-F1, network-recovery F1, Pearson ρ vs the paper, χ²/NMSE, AUC, NMI, cumulative log-return.
**No Sharpe, transaction cost, slippage, turnover or drawdown metric exists anywhere in the repo** (V,
repo-wide grep of `*.py`); only an equity-curve max-drawdown in `evaluate.py:86`. Every P&L number is gross,
single-pass, **not walk-forward**, on 7 stocks and one crash. Look-ahead guards: Algorithm 2 is causal
(window-t uses data to t only); alignment is signal t → invest t+1 (`exp4_trading.py:205`); detectors use a
strictly-prior 60-day trailing z-score with 250-day burn-in; crisis windows pre-registered in Phase 0 before
any detector was scored (R). Overfitting controls: candidate direction fixed by correlation with λ₂ not by
score, every gate matched to λ₂'s own 79/201 invest rate (R); in-sample τ=median rows printed but kept out
of the figure.

## E. Results

| Result | Config | Source | Label |
|---|---|---|---|
| S1 buy-hold **+0.0779** vs S2 gated **−0.0111** cum. log-return | FAAMUNG, τ=1.0, 79/201 days, ends 2020-05-01 | reproduced this session | **V** |
| S2 crash P&L −0.0878 vs S1 −0.3147; S1 trough −0.2123 | 2020-02-19→03-23 | reproduced this session | **V** |
| λ₂ ∈ [0.6557, 1.5011], median 1.1036 | 243 windows | `lam2_series.npy` | **V** |
| Paper's Fig 4: S2 ≈1.35 > S1 ≈1.12 | read off the figure | `paper1/CLAUDE.md` D1 | **R** |
| TICC macro-F1 avg **0.9935** (paper 0.95) | seed 7, SEG_LEN=400 | `table1_f1.json` | **V** |
| `TICC β=0` avg **0.8024** (paper 0.88); 0.75 on 5 seeds | — | `table1_f1.json` / §5.1 | V / **R** |
| Recovery F1 avg **0.8036** (paper 0.85); 5-seed 0.776 | — | `table2_recovery.json` / §5.1 | V / **R** |
| Community null tracks real Q at ρ = **0.9836** pre, 0.9882 post (paper 0.99 / 0.96) | N=360 | `fig6_scoring.csv` | **V** |
| 32-cell table vs paper: MAE **0.1554**, sign agrees **96.9%**, 4 cells failed | all config-model | `phase6_scoring.json` | **V** |
| Overlap correction: ρ **0.7829 → 0.7774**, falls on 12/32; 6,315 windows = **211** independent | — | `phase6_scoring.json` | **V** |
| Louvain on i.i.d. noise Q = **0.2246 ± 0.0067** vs real 0.2215 ± 0.0413; noise = **80.1%** of excess over the degree null | 360×30 Gaussian | `phase3/4_*.json` | **V** |
| Crisis AUC: mean-corr **0.8073**, τ 0.8071, isolates 0.7597, modularity **0.5381**; base rate 2.68% | 17 windows | `phase8_extensions.json` | **V** |
| Compression cheaper on **69.3%** of days, **34.3%** of crisis days | break-even k = 25.4 | `phase6_scoring.json` | **V** |
| 3 of 8 components carry 90% variance; corr(PC1,date) **+0.439**; classifier **98.6%** vs 33.3% chance | — | `phase6/7_*.json` | **V** |
| Q low **0.0897** on 1987-10-19 (25-y low), k → 156; τ max **0.8614** same day (global max) | — | `phase2/3_*.json` | **V** |
| `tr(B²)` S2 = **+0.1271** vs S1 +0.0779, circular-shift **p = 0.305** | matched invest rate | `scores_nulls.csv` | **V** |

**Doc vs artifact conflicts.** (1) **"130 S&P500 stocks"** (README, `paper1/CLAUDE.md` §1) vs **97** realized
columns and 97×97 Laplacians; ticker lists sum to 107 (V) — 130 is unsupported. (2) **"26-item audit"**
(README, `paper3/CLAUDE.md`) vs **34 items A1–E11** in the `PITFALLS.md` table (V, counted). (3) TICC β=0:
**0.8024** committed (V) vs "avg 0.75" in prose (the 5-seed paper-protocol mean) — different objects in the
same voice. (4) Noise Q: 0.2246 (`phase3`) vs 0.2253 (`phase4`, 101 days) vs "0.225" (README), all V.
(5) `dev_log` §18 quotes S2 +0.038 / S1 +0.214 on the *untruncated* series vs −0.011 / +0.078 on the paper's
window; both true, easily mixed. (6) `reports/README.md` calls the probe parquet "6,315 × 15"; it has 16
columns (V).

## F. Negative results

- **Spurious indicator withdrawn.** `d̄ − μ₁(B)` reproduced 91% of λ₂'s daily calls but rests on an identity
  requiring a *regular* graph; residual sd 0.208 = 89% of λ₂'s own sd, only 40% of its variance is λ₂ (R).
  This retracted the D1/D2 spine of the 19 August report.
- **Trading edge was luck.** 5 of 28 candidates beat buy-and-hold; a random gate at the same invest rate
  beats it 39.2% of the time, so ~11 were expected by chance (R); all circular-shift p ≥ 0.115 (V).
- **The audit's own central hypothesis failed.** E1 predicted 29/30 window overlap would collapse the
  paper's evidence; ρ moved 0.783 → 0.777 (V). Reported as prominently as the positives.
- **Correlation rejected as a novelty screen**: spectral entropy of `B_n` correlates −0.264 with λ₂ yet
  rebuilds from spec(`L_sym`) to 8.0e-15 (R).
- **TICC β=0 points away from the paper's own conclusion**, overstating the temporal penalty; undiagnosed.
- **Configuration-model gap unexplained**: ours ρ = 0.861 vs paper 0.721; two generators and 10× rewiring
  ruled out (V + R). **BLAS oversubscription**: 7 workers × 16 threads on 8 cores ran 64% *slower* than
  single-process; pinning gave 7.0× (R).

## G. Engineering

Pipeline: fetch → universe/quality gates → returns → networks → communities → nulls → measures → scoring →
PCA → extensions → report, as `exp1..exp9` each exposing `run(force=False) -> dict` with a JSON cache,
orchestrated by `run_phases.py`. Phase 5 resumable via chunked `.npz` (264 chunk files, V). Null graphs
regenerated from stored Π + seed rather than persisted (~700M edges avoided, R). Explicit logged seeds
(`DATA_SEED=7`, `FIT_SEED=42`). Generated docs written from artifacts; `probe/check_numbers.py` re-derives
all 91 numbers in the 19 Aug report and exits non-zero on drift (R).

**Size** (V, `wc -l`, excluding `.venv`/`__pycache__`): **11,607 lines of Python in 70 files** — paper3
6,994/28, paper1 1,764/17, modularity_indicator 1,151/6, paper2 1,087/15, reports 611/4. Four notebooks,
130 cells. ~2,600 lines LaTeX, ~5,800 lines Markdown. `cloc` not installed.

**Tests, all run this session** (V): paper3 `pytest` **48 passed in 1.52s**; paper2 `pytest` **3 passed in
3.60s**; paper1 `test_solver.py` is script-style, 4 assertions, **PASS in 0.67s** (row-sum 8.9e-16, min
eigenvalue 5.0e-16, rank 9/10). 51 pytest tests. Phase acceptance tests are claimed passing in §8 but are
not a re-runnable suite (R).

**Git** (V): 24 commits on HEAD, 27 across all refs, 2026-07-05 → 2026-09-08, single author (Jash Dalal
100%: 26 + 1 as GitHub noreply `jashD11`), 3 branches.

## H. Hardest problem solved

Paper 1's Fiedler-value signal was silently dead, and the failure looked like a modelling result rather than
a bug. Algorithm 2's temporal penalty `δ‖L_t − L_{t−1}‖²_F` had been approximated by augmenting the
similarity matrix as `S + (2δ/n)L_prev`, but the Frobenius cross-term is `−2δ·tr(L·L_prev)`, so the sign was
inverted: the consistency penalty actively *repelled* each window from the previous graph and shrank every
edge weight. The symptom was λ₂ crushed into [0.00, 0.78] with 65 of 243 windows exactly zero, which made
the paper's τ = 1.0 non-binding and the trading figure vacuous (S2 ≡ S1), while every unit test still passed
because the solver was returning perfectly valid Laplacians. The fix implemented the penalty exactly in
edge-weight space as `γ(2‖Δw‖² + ‖K_sqᵀΔw‖²)` with closed-form gradient plus a real warm start, verified by
finite-difference gradient check, exact-Frobenius equality check and a γ→∞ pin test; λ₂ recovered to
[0.6557, 1.5011] with zero disconnected windows, the gate became live at 79/201 days, and the honest answer
(crash-avoidance replicates, the paper's P&L ranking does not) became visible (`dev_log.md` §18; range V,
mechanism R).

## I. Six candidate resume bullets

≤105 chars, past-tense verb, ≤1 number, source tagged.

1. `Rebuilt 3 network-inference papers from scratch in 11.6k lines of Python [wc -l, V]` (82)
2. `Showed Louvain modularity detects crises at AUC 0.538, a coin flip [phase8_extensions, V]` (88)
3. `Proved the modularity and Laplacian operators coincide once normalised, to 1.8e-15 [FINDINGS, R]` (96)
4. `Found i.i.d. noise reproduces 80% of the market's modularity excess [phase4_nulls.json, V]` (89)
5. `Killed a 5-candidate trading edge with a circular-shift null: p=0.305 [scores_nulls.csv, V]` (90)
6. `Traced a sign bug zeroing 65 of 243 Fiedler estimates to an inverted penalty [dev_log §18, R]` (92)

Reserve: `Logged 34 methodological defects in a 2015 market-networks paper [PITFALLS.md, V]` (79).

## J. Open questions for the author

1. Is the S&P universe 97, 107 or 130 names? Artifacts say 97.
2. Is the audit 26 items or 34? Fix README and both `CLAUDE.md` files.
3. Which TICC β=0 number is the headline, 0.80 (committed, one seed) or 0.75 (5-seed, paper protocol)? §6
   already commits to re-running Tables 1/2 at `SEG_LEN = 100·K` with error bars.
4. No Sharpe, cost, turnover or walk-forward exists. Is any P&L number citable on a quant resume? The honest
   framing is "gross cumulative log-return, 7 stocks, one crash".
5. λ₂ fails its own sliding-event null at p = 0.243. Is Paper 1's indicator *unreplicated*, or only unpowered
   on this sample?
6. μ_N(B) wins on both constructions but is significant on neither. Next test, on what data?
7. Is the configuration-model gap a defect in the paper or in us?
8. Two bibliography entries in the 19 Aug report are flagged red and must not be cited as-is.

---

# REPO-SPECIFIC

## Replications

| | Paper 1 — Cardoso & Palomar 2020 | Paper 2 — TICC, Hallac et al. 2017 | Paper 3 — Silva et al. 2015 |
|---|---|---|---|
| **Data** | 97 S&P names 2016-19; FAAMUNG 7 names 2019-20 | synthetic only | 360 NYSE names, 1986-2011, 6,315 networks |
| **Repo's label** | Figs 1–3 **reproduced**; Fig 4 **partial** (D1) | **partial** | **complete, Phases 0–9** |
| **Reproduced** | Fig 2 rank 94 = p−3 at η=300; Fig 3 λ₂ tracks Fig 3b month-for-month; Fig 1 panel d cleanest | Table 1 TICC 0.9935 vs 0.95; Fig 3 0.85@100 → 0.99@200 | Q null ρ = 0.9836 (paper 0.99); config floor Q = 0.1065 (paper ≈0.10); Eq. 2 MAE 0.00139; mean degree exactly f(N−1) |
| **Failed** | S2 −0.0111 < S1 +0.0779; paper has S2 > S1 | β=0 0.80 vs 0.88; recovery 0.804 vs 0.85; Fig 4 a proxy, T 1e3–1e5 vs 1e4–1e7 | 4 of 32 cells off >0.30, all configuration-model |
| **Main deviation** | Algorithm 2 uses uncontrolled Laplacians; SGL cannot produce true isolated nodes | `SEG_LEN=400` not `100·K` (≤2× data); edge weights bounded off zero; single seed, no error bars | 377 of 1,575 vs 348 of 3,799 (pre-filtered by survival); T 6,345 vs C_p 6,008; t₂ timestamp |

## The ADMM failure

Two distinct things, easily conflated.

1. **ADMM was abandoned at design stage, not after a measured divergence** (`dev_log.md` §2, all R): the
   L-update has no closed form because of `−log gdet(L)`, making ADMM a two-level loop; the Z-update is not
   a true projection because the four Laplacian-cone constraints interact (zeroing off-diagonals after the
   PSD step destroys PSD); and "early experiments showed ADMM's dual variable U grows without bound near the
   boundary of the Laplacian cone". **No artifact, timing or iteration log for those experiments exists, and
   there are no ADMM-vs-L-BFGS-B comparison numbers** (V, searched). Treat the divergence claim as
   undocumented.
2. **The reparameterization is real and verified.** `L(w) = Kᵀdiag(w)K` on the complete graph's signed
   incidence matrix makes `L1 = 0`, `L_ij ≤ 0` and `L ⪰ 0` hold by construction, reducing the problem to
   `w ≥ 0`, a box constraint for L-BFGS-B. Verified this session: row-sum 8.9e-16, min eigenvalue 5.0e-16,
   rank 9 of 10, 0.67s.
3. **The measured failure-and-fix is a different bug** — the inverted temporal-penalty sign (§H). Its
   before/after is the only real comparison here: λ₂ [0.00, 0.78] with 65/243 zero windows →
   **[0.6557, 1.5011] with 0 zeros** (V); gate 0 → 79/201 days live (V).

Residual cost: because the parameterization forbids `w_ij = 0 ∀j` (log gdet → −∞), the SGL baseline cannot
produce true isolated nodes, so Paper 1's Fig 2 failure mode is a draw-threshold artifact (deviation D3, R).

## The audit

**Location** `paper3_market_modularity/PITFALLS.md`, 375 lines, written *before* implementation, with a
status column per item (`addressed` / `UNTESTED` / `pending`). Called "26-item"; the table carries **34**
(V). Five most consequential:

1. **A4** — at N/T ≈ 12 the correlation matrix is degenerate: i.i.d. noise through the identical pipeline
   gives τ = 0.241 and Q = 0.2246 vs the real market's 0.2215, so no claim about modularity's *level*
   survives, only about its variance (V).
2. **E1** — every reported ρ uses 30-day windows slid by 1 day, so 6,315 points carry 211 independent ones;
   correcting it moves mean ρ only 0.783 → 0.777, refuting the audit's own central hypothesis (V).
3. **E10** — Fig. 1 step (e) is a labelled "crisis detection evaluation" the paper never performs; built and
   scored here, modularity reaches AUC 0.538 while the discarded τ(t) reaches 0.807 (V).
4. **A7** — fixed-density thresholding deletes the primary crisis signal: τ(t) peaks at 0.8614 on Black
   Monday, its 25-year global maximum, and is free before any community is detected (V).
5. **A1** — survivorship bias: the paper keeps 348 of 3,799 names over 25 years, excluding every firm that
   failed in the crises it studies; this replication carries a *second* layer, its candidate list being a
   currently-listed directory (V, funnel 1,575 → 377).

## Original spectral work

**Question.** Paper 1's indicator is λ₂(L), Paper 3's is Louvain Q; both derive from one of two operators,
`L = D − A` and `B = A − ddᵀ/2m`. Does `B` yield an equivalent of the Fiedler value, and what do the two
spectra encode? (Supervisor question, 31 Aug 2026.)

**Method / data.** Two tracks. (i) `modularity_indicator/`: re-run Paper 1's Algorithm 2 with exp3/exp4's
exact constants to recover the 243 Laplacians Paper 1 discards after extracting λ₂, set `A = −offdiag(L)`,
compute a 28-quantity menu (`series.parquet`, 243 × 29 V), score on exp4's 201-day window behind two gates:
recomputed λ₂ matches the cache to 1.2e-10 and the published S1/S2 reproduce. (ii) `reports/probe/`: full
spectra of `L`, `B` and `C` for all 6,315 Paper 3 networks (`spectral_probe.parquet`, 6,315 × 16 V), scored
with the replication's own detector against the same 17 crises.

**Findings.** (a) *Negative and load-bearing*: normalised, `B_n = I − L_sym − uuᵀ` with shared eigenvectors
and eigenvalues paired `μ = 1 − λ`, verified to 1.8e-15 on all 243 windows (R) — so no normalised-`B`
indicator can be new, and this is a proof, not a measurement. (b) The unnormalised shortcut `d̄ − μ₁(B)` needs
a regular graph and is withdrawn (§F). (c) *The matrix beats the scalar*: on Paper 3, Q(t) scores AUC 0.5381
while `μ_N(B)` reaches 0.7597, `mod_trace_pos` 0.7500 and `mod_n_pos` 0.7221 (V) — the information is in `B`
and the Louvain step is where it is lost. (d) `μ_N(B)` also wins on FAAMUNG, z-scored AUC 0.7752 vs λ₂'s
0.7588 (V): the same quantity on two unrelated constructions. (e) *λ₂ is a blend*: the containment
`d_min·λ₂(L_sym) ≤ λ₂(L) ≤ d_max·λ₂(L_sym)` holds 243/243, and splitting detection gives structure-only
0.700, degree-only 0.537 (coin flip), full 0.759 — degree is a modifier, not a signal, and normalisation is
exactly what destroys it (V/R). (f) Neither indicator is certified: ΔAUC 95% CI [−0.115, +0.148]; the
sliding-event null puts COVID at p = 0.243 for λ₂ and 0.345 for μ_N (V). At Δt = 30, N = 360 the mean count
of correlation eigenvalues escaping the Marchenko-Pastur bulk, excluding the market mode, is **0.70** (R),
so there may be no resolvable community structure at this window length at all.

**Next steps** (as the repo states them): test `μ_N(B)` on a construction that is neither FAAMUNG nor Paper
3's; A5 market-mode removal, which bounds the noise-control result; A6 Δt robustness. Reports:
`reports/2026-09-08_modularity_equivalence.tex` (current), `2026-08-19_discrepancy_explained.tex` (§9
superseded), `progress_report.tex` (running log).
