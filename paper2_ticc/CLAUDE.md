# CLAUDE.md — paper2_ticc

> Replication of **TICC** — Hallac, Vare, Boyd & Leskovec,
> *"Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data"*,
> KDD 2017 (`../docs/1706.03161v2.pdf`).
> Self-contained; shares nothing with `paper1_gmrf_laplacian/`.

---

## 1. What TICC does

Simultaneously **segments and clusters** a multivariate time series. Each cluster is
a **block-Toeplitz Gaussian MRF** — the inverse covariance Θ of a length-`w` window of
observations. Points are assigned to clusters by a dynamic-programming pass that
balances log-likelihood against a temporal-consistency (switching) penalty. It is
solved by EM-style alternating minimisation.

```
Problem (1):  minimize_{Θ, P}  Σ_i Σ_{X_t ∈ P_i} [ -ll(X_t, Θ_i) + β·1{X_{t-1} ∉ P_i} ]
              + Σ_i ||λ ∘ Θ_i||_1     subject to  Θ_i block-Toeplitz
```

- **E-step (Algorithm 1):** fix Θ, assign points → exact Viterbi min-cost path, node
  cost = negative log-likelihood, edge cost = β on a cluster switch. O(TK).
- **M-step (Toeplitz graphical lasso, Problem 4):** fix P, update each Θ_i via ADMM
  (paper eqs 5–9): closed-form log-det prox (Θ-update) + block-Toeplitz group-average
  and soft-threshold (Z-update) + dual update.

## 2. Code map

```
solver/toeplitz_admm.py   solve_toeplitz_graphical_lasso(S, lam, n, w, ...) -> Theta   (Problem 4)
solver/dp_assign.py       gaussian_neg_ll(X, thetas); assign_points(nll, beta)          (Algorithm 1)
solver/ticc.py            fit_ticc(X, K, w, lam, beta, n_restarts, ...) -> labels,thetas,Xst  (Algorithm 2)
data/synthetic.py         generate_series(sequence, seg_len, n, w, ...); random_toeplitz_theta(...)   (Sec 6)
utils/metrics.py          macro_f1 (Table 1, Hungarian-aligned); network_recovery_f1 (Table 2)
utils/baselines.py        gmm / eev / kmeans / dtw_gak / dtw_euclidean / neural_gas
experiments/exp1..4.py    Table 1, Table 2, Fig 3, Fig 4  (cache to results/, figures to figures/)
tests/test_ticc.py        ADMM validity+recovery, DP vs brute force, end-to-end macro-F1
replication_notebook.ipynb single notebook rendering all four results
```

## 3. Hyperparameters (constants at top of `experiments/exp1_table1_f1.py`)

| name | value | meaning | matches paper? |
|------|-------|---------|----------------|
| `N, W` | 5, 5 | dimension, window size | yes (Sec 6) |
| `SEG_LEN` | 400 | observations per segment (Table 1/2) | **no** — paper uses `100·K`, i.e. 200/300/400/300 |
| `LAM` | 0.07 | Toeplitz-graphical-lasso l1 strength | not stated in paper |
| `BETA` | 60 | temporal switching penalty | not stated in paper |
| `N_RESTARTS` | 20 | random contiguous-init restarts, keep min-objective | our addition (paper: random init) |
| `DATA_SEED` | 7 | single realisation; no error bars are reported | — |

> **`SEG_LEN` is a real deviation, not a paper value.** Sec 6: *"Each segment ... has 100K
> observations in R^5, where K is the number of clusters (2, 3, 4, and 3, respectively)."*
> Using 400 everywhere gives up to **2× the paper's data** on the easier sequences. See §6.

## 4. Key implementation decisions

- **Init / restarts.** Clusters are zero-mean and differ only in second-order
  structure, so a Euclidean init (K-means/GMM on raw windows) can't see them and EM
  traps for K≥3. We instead restart from random **contiguous-segment** labelings and
  keep the lowest penalised objective; this reliably finds the global optimum (the
  true segmentation is the lowest-objective fixed point — verified).
- **Solver returns Z**, the exactly block-Toeplitz, symmetric, sparse consensus
  variable (PD-guarded), not the smooth Θ iterate. Better for support recovery.
- **Edge weights bounded away from 0** in the generator (random sign, magnitude in
  [0.5, 0.9]). Zero-mean like the paper, but pure N(0,1) weights put many edges below
  the noise floor and cap recovery ~0.6; bounding restores the paper's 0.79–0.90 band.

## 5. Results (vs paper)

| Result | Paper | This impl | Status |
|--------|-------|-----------|--------|
| Table 1 TICC macro-F1 (4 seqs) | 0.90–0.98, avg 0.95 | ~0.99 avg | reproduced |
| Table 1 `TICC, β=0` | 0.86–0.89, avg 0.88 | 0.68–0.94, avg 0.80 | **not reproduced** |
| Table 1 GMM (best baseline) | avg 0.67 | avg ~0.84 | inflated by `SEG_LEN` |
| Table 1 distance-based | low | low, same ordering | reproduced |
| Table 2 network recovery F1 | 0.79–0.90, avg 0.85 | 0.75–0.85, avg 0.80 | **partial** (below band) |
| Fig 3 TICC sample efficiency | clears 0.9 by ~200 | 0.85 @100 → 0.99 @200 | reproduced |
| Fig 3 `β=0` curve | rises, ~0.9 by 300+ | flat at ~0.68 | **not reproduced** |
| Fig 4 scalability | linear, T = 1e4–1e7 | linear, T = 1e3–1e5, proxy | **partial** |
| Sec 6 window-size robustness | F1 0.95–0.98 for w ∈ [4,15] | — | **not implemented** |
| Sec 6 micro-F1 cross-check | within 1–2% of macro-F1 | — | **not implemented** |
| Sec 7 BIC selection of K | K=5 by BIC | — | **not implemented** |
| Sec 7 case study (Table 3/Fig 5) | — | — | not reproducible (proprietary) |

### 5.1 Seed audit (5 data seeds: 7, 11, 23, 101, 202; fit seed 42, 20 restarts)

The committed tables are a **single** realisation (`DATA_SEED = 7`). Reran across seeds under both
protocols — mean ± sd of macro-F1 / recovery-F1:

| Sequence | protocol | TICC | `β=0` | GMM | recovery |
|---|---|---|---|---|---|
| 1,2,1 | paper (200) | 0.988 ± .006 | 0.858 ± .036 | 0.711 ± .201 | **0.673 ± .085** |
| 1,2,1 | repo (400) | 0.995 ± .003 | 0.917 ± .034 | 0.946 ± .017 | 0.740 ± .043 |
| 1,2,3,2,1 | paper (300) | 0.995 ± .002 | 0.692 ± .067 | 0.769 ± .156 | 0.762 ± .061 |
| 1,2,3,2,1 | repo (400) | 0.995 ± .002 | 0.724 ± .072 | 0.689 ± .175 | 0.769 ± .062 |
| 1,2,3,4,1,2,3,4 | both (400) | 0.995 ± .003 | 0.684 ± .035 | 0.665 ± .141 | 0.787 ± .050 |
| 1,2,2,1,3,3,3,1 | paper (300) | 0.994 ± .003 | 0.755 ± .068 | 0.889 ± .025 | 0.808 ± .045 |
| 1,2,2,1,3,3,3,1 | repo (400) | 0.997 ± .001 | 0.746 ± .077 | 0.890 ± .030 | 0.807 ± .061 |

Readings:
- **TICC is genuinely robust** (σ ≤ 0.006). That row is not a lucky seed.
- **Every committed Table 2 value is at or above its 5-seed mean** (reported avg 0.803 vs measured
  0.776). Seed 7 is favourable; three decimals from one run overstates precision.
- **`SEG_LEN` explains most of the GMM anomaly**: at the paper's length, GMM on `1,2,1` scores
  0.711 vs the paper's 0.68 — a near match. The 4-seq average drops 0.80 → 0.759.
- **`β=0` is low across all seeds**, so it is a real defect, not noise.

## 6. Known deviations / gaps

**Disclosed deviations that inflate results**

- **`SEG_LEN = 400` for all four sequences** instead of the paper's `100·K`. Up to 2× the data.
  Costs most on Table 2: `1,2,1` recovery falls 0.740 → **0.673** under the paper's protocol.
- **Edge weights bounded away from zero** in the generator (random sign, magnitude in [0.5, 0.9]).
  Zero-mean like the paper, but the paper only says "centered at 0". Unbounded N(0,1) weights put
  many edges below the noise floor and cap recovery near 0.6; bounding restores the 0.75–0.85 band.
  This is the single largest thumb on the scale for Table 2.
- **Single data seed, no error bars** anywhere in `results/`. See §5.1.
- **Asymmetric baseline tuning** — `utils/baselines.py` claims "apples-to-apples", but TICC gets 20
  restarts while `GaussianMixture` uses sklearn's default `n_init=1`, and the DTW rows use
  `max_iter=10, n_init=1` on a 1500-window subsample of ~3200. The distance-based conclusion
  survives (ours 0.30–0.51 vs paper 0.17–0.64), but the setup is not what the file says.

**Genuine non-reproductions**

- **`TICC, β=0` is systematically too weak** (avg 0.75 vs the paper's 0.88, σ ≈ 0.04–0.08). Because
  our β=0 is too low, this replication **overstates** the contribution of the temporal-consistency
  penalty — the one place our error points away from the paper's own conclusion.
- **Fig 3's `β=0` curve is flat** at 0.66–0.71 across 100→500 samples. The paper reports it rising
  rapidly, near-matching TICC by ~200, and hovering ~0.9 — and uses that overlap to argue the
  low-sample accuracy comes mainly from the **Toeplitz constraint**, not β. Our figure contradicts
  that argument. Cause undiagnosed; the restart-selection rule is a suspect, since with β=0 the
  objective reduces to total negative log-likelihood.
- **Fig 4 does not run TICC.** `exp4` times a hand-rolled M+E step on random N(0,1) data with a
  random labelling, ADMM capped at 200 iterations (real fits use 1000); it never calls `fit_ticc`.
  T range 1e3–1e5 vs the paper's 1e4–1e7. The paper's headline (10M points, ~25 min/iter) is
  untested.

**Not implemented**

- **Window-size robustness** (Sec 6, p.7): "any window size between 4 and 15 yields macro-F1
  0.95–0.98; recovery 0.87–0.89 for w ∈ [5,14]". Every experiment hardcodes `W = 5`. This is the
  cheapest of the five synthetic results to add and the one that would actually stress the model.
- **Micro-F1 cross-check** (Sec 6): paper reports it within 1–2% of macro-F1.
- **BIC selection of K** (Sec 7): K is fixed to the true value everywhere. Consistent with the
  paper's Table 1 protocol, but model selection is untested.
- **Betweenness-centrality cluster interpretation** (Table 3) — no centrality code exists, so the
  interpretability mechanism is supported in principle but not demonstrated.

**Approximations (acceptable)**

- **EEV / DTW / Neural Gas** stand in for the paper's exact toolchain (mclust / custom). They are
  the low-scoring rows and don't affect the headline.
- **Case study (Table 3 / Fig 5):** automobile-sensor data is proprietary — genuinely unavailable.

## 7. Status

Verified independently:

- [x] `solver/toeplitz_admm.py` — returns symmetric, PD, exactly block-Toeplitz Θ
- [x] `solver/dp_assign.py` — DP optimum matches brute force
- [x] `solver/ticc.py` — **no ground-truth leakage**; init is random contiguous segments, restart
      selection uses only the penalised objective
- [x] true segmentation confirmed to be the lowest-objective fixed point (EM from ground truth and
      the 20-restart search reach the same solution)
- [x] `tests/test_ticc.py` — 3/3 PASSING
- [x] exp1 Table 1, exp2 Table 2, exp3 Fig 3, exp4 Fig 4 (with the caveats in §6)
- [x] `replication_notebook.ipynb`

Outstanding (see §6):

- [ ] fix `SEG_LEN` to `100·K` and re-run Tables 1/2
- [ ] report seed-averaged results with error bars
- [ ] diagnose the `β=0` discrepancy (Table 1 row and Fig 3 curve)
- [ ] add the window-size robustness sweep
- [ ] equalise baseline tuning (GMM `n_init`, DTW iterations/subsampling)
- [ ] extend Fig 4 toward the paper's T range, or run it through `fit_ticc`
- [ ] micro-F1 cross-check; BIC-based selection of K
