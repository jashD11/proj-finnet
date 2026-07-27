# CLAUDE.md — Financial_networks_project

> Project guide for Claude Code. Read this first in every session.
> Goal: faithfully replicate the results of Cardoso & Palomar (2020),
> *"Learning Undirected Graphs in Financial Markets"* (arXiv:2005.09958v4).

---

## 0. How to use this file

- This is the source of truth for the project. Do **not** re-derive context that's already here.
- Each experiment section below is self-contained. When working on one experiment, read **only** that section + the Shared Solver section.
- Before writing any file, check the "Status" checkboxes at the bottom to know what already exists and works.
- After completing a step, update the Status checklist (tick the box) so future sessions don't redo work.

---

## 1. What we're building

Replicate the paper's 4 experiments. The paper learns **graph Laplacian matrices** (precision matrices) from stock returns under a Gaussian Markov Random Field (GMRF) model, then uses them for clustering and a trading strategy.

| Fig | Experiment | Data | Core method |
|-----|-----------|------|-------------|
| Fig 1 | Preprocessing effects | 130 S&P500 stocks (3 sectors) | CLR two-stage [Nie 2016] |
| Fig 2 | Degree control / k-component | same 130 stocks | Algorithm 1 (proposed) vs SGL |
| Fig 3 | Time-varying graphs | FAAMUNG (7 stocks) | Algorithm 2 (rolling Algorithm 1) |
| Fig 4 | Trading strategy | FAAMUNG (7 stocks) | algebraic connectivity signal |

**Everything depends on one shared engine: the GMRF Laplacian solver.** Build and verify it first.

---

## 2. The math (reference — don't skip)

### Core problem (Problem 1)
Penalized maximum-likelihood estimate of the Laplacian precision matrix:

```
minimize_L   tr(L S) - log gdet(L) + h_α(L)
subject to   L1 = 0,  L_ij = L_ji <= 0  (i != j),  L >= 0 (PSD)
```

- `S` = similarity matrix (sample **correlation** by default, see guidelines).
- `gdet(L)` = **pseudo-determinant** = product of the *positive* eigenvalues of L (L is singular, so plain det = 0).
- `h_α(L)` = regularizer (sparsity). β is its hyperparameter.
- Solve with **ADMM** or **MM** — NOT cvxpy (log gdet doesn't scale, not DCP-compliant).

### Laplacian properties to verify (P1, P2)
- **P1:** `L1 = 0` → graph-mean zero, null space contains the all-ones vector.
- **P2:** off-diagonals `L_ij <= 0` → only non-negative conditional correlations.
- Together → L is PSD. Conditional correlation between nodes i,j = `-L_ij / sqrt(L_ii L_jj)`.

### Preprocessing guidelines (critical for Fig 1)
- **Use correlation, not covariance:** `S̄ = diag(S)^-1/2 · S · diag(S)^-1/2`. Otherwise two perfectly correlated stocks with different variances look distant.
- **Market factor:** model `x_t = β·x_mkt,t + ε_t`. When normalizing each stock, the market factor is *automatically removed* in the squared-distance matrix Z. Paper's Fig 1 conclusion: **best result = scaled data + market NOT explicitly removed** (panel d).

### Algorithm 1 — k-component GMRF (Fig 2)
Relaxed Problem 8. Alternating minimization:

```
repeat until converged:
  (V-step)  V = k eigenvectors of L for the k SMALLEST eigenvalues   # Problem 9
  (L-step)  minimize_L  tr(L (S + η·V·Vᵀ)) - log gdet(L)             # Problem 10
            subject to L1=0, L_ij<=0, diag(L)=1
```

- The L-step is the convex GMRF solver from §3 (the shared engine), just with an augmented similarity matrix `S + η·V·Vᵀ`.
- `rank(L) = p - k` enforced via the spectral penalty (sum of k smallest eigenvalues → 0, Fan's theorem).
- `diag(L) = 1` is the **degree control** constraint — this is the paper's key fix vs SGL. Without it → isolated nodes.
- Paper uses **β = 10** for both methods, **k = 3** (three sectors).

### Algorithm 2 — Time-varying (Fig 3)
Rolling-window causal estimator. For each window t, solve Algorithm-1-style problem with a temporal-consistency penalty:

```
minimize  Σ n_t [tr(S_t L_t) - log gdet(L_t)] + δ Σ ||L_t - L_{t-1}||²_F
```

- **Causal:** to estimate L_t use only data up to time t (no look-ahead). Store L̂_t|t.
- δ = temporal smoothness hyperparameter. Paper uses **δ = 100**.
- Window = **30 days**, shift **1 day** at a time → **200 graphs** total.
- Indicator = **algebraic connectivity** = 2nd-smallest eigenvalue λ₂(L_t).

### Trading strategy (Fig 4)
- **S1:** uniformly invest 1 unit the whole period (buy & hold benchmark).
- **S2:** invest only when algebraic connectivity λ₂ < τ; exit (cash) when λ₂ >= τ — high connectivity = tightened conditional correlations = stress. **τ = 1.0** (verified against paper text + Fig 4; genuinely gates here: λ₂ ∈ [0.66, 1.50]).
- Plot cumulative PnL of both. Paper's result: S2 > S1 (dodges the COVID crash). **Partially replicates: crash-avoidance yes, final ranking no — see D1.**

---

## 3. Folder structure

```
Financial_networks_project/
├── CLAUDE.md                  # this file — project source of truth
├── requirements.txt           # pinned dependencies
├── README.md                  # short human-facing summary (optional)
│
├── data/
│   ├── fetch_data.py          # one-time yfinance pull → CSVs. Run once.
│   ├── sp500_3sectors.csv     # cached: 130 stocks, 2016-2019 log returns
│   ├── faamung.csv            # cached: 7 stocks, 2019-2020 log returns
│   └── tickers.py             # ticker lists (the 130 universe + FAAMUNG)
│
├── solver/
│   ├── admm_graph.py          # THE ENGINE: GMRF Laplacian ADMM solver (Problem 1 / 10)
│   ├── algorithm1.py          # k-component wrapper (alternating V-step + L-step)
│   ├── algorithm2.py          # time-varying rolling wrapper around algorithm1
│   └── sgl_benchmark.py       # SGL [Kumar 2016] baseline for Fig 2 comparison
│
├── utils/
│   ├── preprocessing.py       # correlation vs covariance, scaling, market removal
│   ├── graph_metrics.py       # algebraic connectivity, rank, PSD checks, λ extraction
│   └── graph_viz.py           # networkx plotting: nodes by sector, edge width by weight
│
├── tests/
│   └── test_solver.py         # synthetic GMRF sanity check (run after building solver)
│
├── exp1_preprocessing.py      # Fig 1 — 4-panel preprocessing comparison
├── exp2_kcomponent.py         # Fig 2 — Algorithm 1 vs SGL (k=3)
├── exp3_timevarying.py        # Fig 3 — FAAMUNG rolling λ₂ indicator + network snapshots
├── exp4_trading.py            # Fig 4 — S1 vs S2 cumulative PnL
│
└── figures/                   # all output PNGs land here
    ├── fig1_preprocessing.png
    ├── fig2_kcomponent.png
    ├── fig3_timevarying.png
    └── fig4_trading.png
```

### What each file does

**Root**
- `CLAUDE.md` — you're reading it. Project memory.
- `requirements.txt` — `numpy scipy pandas yfinance matplotlib networkx scikit-learn`. Pin versions once it runs.
- `README.md` — optional one-paragraph human summary + how to run.

**data/**
- `fetch_data.py` — pulls prices from yfinance, computes log returns, writes the two CSVs. **Run once, then never again** (caches to CSV so experiments don't re-hit the network or burn tokens re-fetching).
- `tickers.py` — hardcoded ticker lists. The 130-stock S&P500 universe (Industrials + Consumer Staples + Energy, as of ~2016) and FAAMUNG = `[META, AAPL, AMZN, MSFT, UBER, NFLX, GOOGL]`. (Note: FB→META; the paper's 2020 "Facebook" is today's META ticker.)
- `*.csv` — cached log-return matrices. Inputs to every experiment.

**solver/** — the heart of the project
- `admm_graph.py` — implements Problem 1 / Problem 10: convex GMRF Laplacian estimation via ADMM. Signature: `learn_laplacian(S, beta, max_iter, tol) -> L`. **Build and verify this before anything else.**
- `algorithm1.py` — wraps the solver in the alternating V-step/L-step loop to produce k-component graphs (Fig 2's proposed method).
- `algorithm2.py` — wraps algorithm1 in a causal rolling window with the δ temporal penalty (Fig 3).
- `sgl_benchmark.py` — the SGL baseline (spectral constraints, NO degree control) so Fig 2 can show the isolated-node failure mode.

**utils/**
- `preprocessing.py` — functions for the 4 preprocessing variants: `to_correlation()`, `scale_data()`, `remove_market_factor()`.
- `graph_metrics.py` — `algebraic_connectivity(L)`, `is_psd(L)`, `effective_rank(L)`, eigenvalue helpers, the P1/P2 validity checks.
- `graph_viz.py` — networkx rendering: layout, color nodes by sector, edge thickness ∝ weight. Reused by Figs 1, 2, 3.

**tests/**
- `test_solver.py` — generates a synthetic p=10 GMRF, runs the solver, asserts L1≈0, off-diagonals ≤0, PSD, rank=p−1. **The gate that must pass before any experiment.**

**experiments/** (root-level exp*.py) — each is a thin script: load CSV → preprocess → call solver/algorithm → plot → save PNG to `figures/`. Independent and runnable in isolation.

---

## 4. Conventions

- **Language:** Python 3.10+.
- **Style:** numpy-first, vectorized. Matrices are `np.ndarray`, shape comments on non-obvious ops.
- **No re-fetching:** experiments read CSVs from `data/`, never call yfinance directly.
- **Plots:** matplotlib, save to `figures/`, don't `plt.show()` in scripts (headless-friendly).
- **Determinism:** set `np.random.seed(42)` anywhere randomness appears (solver init, synthetic data).
- **Verify, don't assume:** after writing a solver/algorithm file, run it on a tiny input and print shape + the P1/P2 checks before moving on.
- **Hyperparameters live at the top of each exp script** as named constants (BETA=10, K=3, WINDOW=30, DELTA=100, TAU=1.0), not buried in function calls.

---

## 5. Build order (dependency-driven)

1. `requirements.txt` + `data/tickers.py` + `data/fetch_data.py` → run it → CSVs exist.
2. `solver/admm_graph.py` → `tests/test_solver.py` → **must pass.**
3. `utils/` (preprocessing, metrics, viz).
4. `solver/algorithm1.py` + `solver/sgl_benchmark.py`.
5. `exp3` + `exp4` (FAAMUNG — small, fast, vivid). Needs `algorithm2.py`.
6. `exp2` (k-component, needs algorithm1 solid).
7. `exp1` (130-stock preprocessing — messiest data, do last).

---

## 6. Solver strategy note

The inner GMRF ADMM is the only hard part. Two viable paths:
- **(A) Implement ADMM in Python from scratch** (~100–150 lines). Reference: Zhao et al. 2019 [ref 4] and Boyd et al. ADMM monograph [ref 14]. Preferred — keeps it pure-Python, no R dependency.
- **(B) Port the authors' R package** (`fingraph` / `spectralGraphTopology`) logic. Use only as a cross-check reference if (A) misbehaves.

Default to (A). If convergence is unstable, the usual culprits are the PSD projection step and the non-positivity projection on off-diagonals — debug those first.

---

## 7. Status checklist (update as you go)

- [x] requirements.txt
- [x] data/tickers.py
- [x] data/fetch_data.py + CSVs generated
- [x] solver/admm_graph.py
- [x] tests/test_solver.py PASSING
- [x] utils/preprocessing.py
- [x] utils/graph_metrics.py
- [x] utils/graph_viz.py
- [x] solver/algorithm1.py
- [x] solver/sgl_benchmark.py
- [x] solver/algorithm2.py
- [x] exp3_timevarying.py → fig3 reproduced
- [x] exp4_trading.py → fig4 produced honestly (paper-direction rule; S2 does NOT beat S1 — see D1)
- [x] exp2_kcomponent.py → fig2 reproduced (Alg1 has no isolated nodes; true k=3 at η=300, rank=94)
- [x] exp1_preprocessing.py → fig1 reproduced (panel d cleanest)
- [x] Fix 1 — trading signal direction diagnosed + aligned + τ table
- [x] Fix 2 — Fig 3 snapshots selected by date
- [x] Fix 3 — Fig 2 histogram-led + honest SGL caption
- [x] Fix 4 — notebook single-render (plt.close in exp scripts) + dev_log + CLAUDE.md

---

## 8. Known gotchas

- **FB ticker:** Facebook is now `META`. yfinance won't find `FB` for recent data, but historical 2019-2020 data may need `META` regardless. Verify the pull returns 7 columns.
- **Uber IPO:** UBER only went public May 2019 — fine for the Jun 2019 start, but no data before that. Don't extend the window earlier.
- **130-stock universe:** the paper doesn't list exact tickers. Reconstruct from S&P500 sector membership ~2016. Expect minor count differences; aim for ~the three clean sector clusters, exact node count is not critical.
- **pseudo-determinant:** compute via eigenvalues, dropping the (near-)zero one: `log_gdet = sum(log(eigvals[eigvals > tol]))`.
- **Causal estimation (Fig 3):** never use future data. L_t uses windows ending at t only.
- **τ calibration (Fig 3/4):** Algorithm 2 uses uncontrolled Laplacians (degree_control=False — verified against the paper: Problem 11 has no diag constraint). With the exact temporal penalty (dev_log §18) λ₂ ∈ [0.66, 1.50] for FAAMUNG, crossing the paper's τ=1.0 constantly — the gate is live. **Warning from history:** an earlier S_aug linearization with an inverted sign compressed λ₂ into [0, 0.78] (65 windows exactly 0 = disconnected estimates) and made τ=1.0 non-binding; if λ₂ ever collapses like that again, suspect the temporal penalty, not the threshold. Algorithm 2 uses k=1 (connected graph), not k=3 — k=3 forces three ~0 eigenvalues and kills the λ₂ signal.

---

## 9. Known deviations (post-fix)

These are documented, intentional divergences from the paper — not bugs.

| # | What diverges | Paper | This implementation | Reason |
|---|--------------|-------|---------------------|--------|
| D1 | Trading result (Fig 4) | invest when λ₂ < τ, τ=1.0; S2 exits during the crash AND re-enters in April to catch the rebound → S2 (~1.35) beats S1 (~1.12) [read off paper Fig 4] | Same rule, same τ, live gate, same evaluation window (backtest ends 2020-05-01 = END_DATE, matching the paper's Fig 4; λ₂ ∈ [0.66, 1.50], 79/201 days invested). S2 dodges the crash exactly like the paper (flat ≈ −0.01 while S1 falls to −0.21) and also sits out autumn 2019 (like paper's S2 in Oct–Nov). **But our λ₂ stays ≈1.3 through April, so S2 misses the early rebound: S2 = −0.011 < S1 = +0.078.** | Post-crash λ₂ decay speed differs from the paper's (their series collapses below 1.0 in April; ours doesn't until late May). Mechanism and crash-avoidance replicate; the final P&L ranking does not, due to re-entry timing. No threshold is tuned to fix this — τ=median tables are printed as in-sample reference only. |
| D2 | ~~k-components (Fig 2)~~ **RESOLVED** | rank=p−3 (3 components) | rank=94 = p−3 achieved with η=300 (η=10 and η=150 insufficient; caches alg1_L_eta150/300.npy) | No longer a deviation. Kept as a record: the spectral penalty needs η≈300 at p=97 to force k=3. |
| D3 | SGL isolated nodes (Fig 2) | True zero-degree nodes (L-space ADMM) | No true isolated nodes (edge-weight solver) | Edge-weight parameterisation makes w_ij=0 ∀j impossible (log gdet → −∞). SGL ring is a draw-threshold artifact, not true isolation. |
| D4 | ~~Temporal penalty (Fig 3)~~ **RESOLVED — was a bug, not a deviation** | Full Frobenius ‖L_t−L_{t-1}‖²_F | Now implemented EXACTLY in edge-weight space: γ(2‖Δw‖² + ‖K_sqᵀΔw‖²), γ=δ/n_t, plus true warm start (w0=w_prev) | The old "linear cross-term" S_aug = S_t **+** 2δ/n_t·L_prev had the sign inverted (the Frobenius cross-term is −2δ·tr(L·L_prev)) — it repelled each window from the previous graph and crushed λ₂ into [0, 0.78] with 65 disconnected windows. The old claim that the quadratic term "breaks the pseudoinverse gradient" was false: it is a plain quadratic in w. See dev_log §18. |