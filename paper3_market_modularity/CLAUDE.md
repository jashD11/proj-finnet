# CLAUDE.md — paper3_market_modularity

Replication of **Silva, Comin, Peron, Rodrigues, Ye, Wilson, Hancock & Costa (2015),
*Modular Dynamics of Financial Market Networks*** — arXiv:1501.05040v3, `docs/1501.05040v3.pdf`.

Self-contained; shares nothing with `paper1_gmrf_laplacian/` or `paper2_ticc/`.

## 0. How to use this file

Source of truth for the replication. Read §1–§4 plus the section for the phase
you are working on. Tick §8's checklist after each phase. Companion docs:

- `PLAN.md` — the approved phase-by-phase build plan.
- `PITFALLS.md` — the 26-item methodological audit (A1–E11) this replication is organised around.
- `data/UNIVERSE.md` — generated; the realized universe and every difference from the paper's.
- `results/REPLICATION_REPORT.md` — generated; claim-by-claim paper vs. us.

## 1. What the paper claims

Build a daily sequence of stock-correlation networks. Detect communities each
day. Keep **only** the community sizes and the inter-community mixing matrix Π,
throw the network away, and regenerate a random network from that summary alone.
The claim is that the regenerated network reproduces most of the real network's
topology — and that during crises the market leaves a well-defined community
structure for a much more uniform organisation.

This is **not prediction.** Communities are re-detected from the real network
every single day; the null model is a same-day compression, not a forecast.

| Paper artifact | Experiment | Method |
|---|---|---|
| Sec. II data + Eq. 1 | `exp1_data.py` | universe, adjusted closes, log returns |
| Sec. II networks | `exp2_networks.py` | rolling Pearson ρ, fixed-density threshold |
| Figs. 3, 4, 5 | `exp3_communities.py` | Louvain × 10 seeds, three modularities |
| Sec. III, Eq. 2 | `exp4_nulls.py` | blockmodel + configuration model |
| Figs. 6, S3 | `exp5_measures.py` | the 8 topological measures |
| Fig. 6/S3 tables | `exp6_scoring.py` | ⟨χ²⟩, ρ, **and the overlap correction** |
| Fig. 7 | `exp7_pca.py` | PCA over the 8-measure space |
| — (beyond the paper) | `exp8_extensions.py` | τ(t), crisis detector, event windows, Louvain stability |

## 2. The math

**Returns.** `Y_i(t) = ln P_i(t) − ln P_i(t−1)`, on adjusted closes.

**Similarity (Eq. 1).** Pearson ρ_ij over a Δt = 30-return window, slid by δt = 1.
Window *n* spans price indices `n … n+30` (31 prices, 30 returns), i.e.
`Y[n−1 : n−1+30]`. This is pinned by the paper's own `N_w = C_p − Δt = 5978`.

**Threshold.** `A_ij = Θ(ρ_ij − τ) − δ_ij`, with **τ re-solved every day** so that
exactly `f = 10%` of the `N(N−1)/2` pairs survive. Fixed density, not fixed
threshold — so mean degree is `f·(N−1)` on every single day, and τ(t) itself
becomes a discarded signal worth recovering (audit A7).

**Modularity.** `Q = Σ_c [ e_c/m − (d_c/2m)² ]`. Three variants, identical
formula, only the partition changes: **dynamical** `C(t)`, **fixed** `C(0)`,
**lagged** `C(t − t_Δ)` with `t_Δ = 100`.

**Community null model (Eq. 2).** From partition `C(t)`, build the mixing matrix
`Π_αβ(t)` and community sizes `|α|`, then

```
π_αβ = 2·Π_αα / (|α|·(|α|−1))    if α == β
π_αβ =   Π_αβ / (|α|·|β|)         if α != β
```

Generate by starting from `Σ|α|` isolated nodes, assigning them to communities of
the observed sizes, and connecting each pair `(i∈α, j∈β)` with **independent
probability** `π_αβ`. Not degree-corrected — deliberately, per the paper.

**Configuration model.** Preserves the daily degree sequence only. Its modularity
is the noise floor (≈ 0.10, flat) that every modularity figure must show.

## 3. Folder structure

```
data/      universe.py fetch_data.py UNIVERSE.md
           nyse_candidates.csv raw_closes.parquet panel_prices.parquet
net/       construct.py communities.py nullmodels.py measures.py
utils/     crisis_dates.py scoring.py plots.py
experiments/  exp1_data.py … exp8_extensions.py
tests/     test_paper3.py
results/   series/*.parquet  tables/*.csv  phase*.json  REPLICATION_REPORT.md
figures/   fig3_blackmonday.png … fig8f_louvain_stability.png
```

Null-model graphs are **never stored** — 126k graphs would be ~700M edges. They
are regenerated on demand from the stored `Π[t]`, sizes, and the recorded seed;
only their measures are persisted.

## 4. Conventions

- **Interpreter: `/opt/anaconda3/bin/python`** (3.12.4). `.venv` lacks `yfinance`,
  `pyarrow` and `certifi`. `python-igraph` was installed into the anaconda env
  for this paper and is a hard dependency (Louvain, exact clique number,
  betweenness, degree-sequence rewiring).
- **`exp1_data.py` is the config module.** Every constant (`DELTA_T`,
  `EDGE_FRACTION`, `N_LOUVAIN_SEEDS`, `LAG_T_DELTA`, `N_REPLICATES`,
  `SPLIT_YEAR`, `DATA_SEED`, `FIT_SEED`) lives at its top; downstream
  experiments import from it. No magic numbers in function bodies.
- **Every experiment exposes** `run(force=False, verbose=True) -> dict` with a
  JSON cache in `results/`, plus `plot(data, save=...)`, plus
  `if __name__ == "__main__"`. Never recompute an earlier phase to run a later one.
- **Every stochastic operation takes an explicit seed and logs it.**
  `DATA_SEED = 7` for data-side randomness, `FIT_SEED = 42` for fits, Louvain,
  null-model generation and layouts.
- **Plots:** `matplotlib.use("Agg")`, save to `figures/`, `dpi=130`,
  `plt.close("all")`, never `plt.show()` in a script.
- **Assertions over trust.** Every phase ends in hard acceptance tests that are
  run and reported. A failing test gets debugged, never weakened.
- **Where the paper is ambiguous, implement both readings behind a flag** and
  record which was used (missing-day policy, disconnection convention,
  configuration-model method).
- **Honest labelling.** Results are *reproduced / partial / not reproduced /
  not implemented*. Never round toward the paper. Where our error points away
  from the paper's own conclusion, say so first, not last.

## 5. Build order

Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9, one commit each, acceptance tests
run and reported at every boundary before the next begins.

## 6. Key resolved ambiguities

| Ambiguity | Resolution | Why |
|---|---|---|
| Adjusted vs. raw closes | **adjusted** (`auto_adjust=True`) | raw closes turn every split into a fake −69% return poisoning 30 networks (A2) |
| Missing days | **`drop_days`** default, `ffill` implemented | ffill manufactures zero returns on halt days, deflating that stock's correlations (A3) |
| Window timestamp | **t₂ (window close)** | Settled empirically. Fig. 8's anchors + `C_p` = 6,008 + a February 2011 end are jointly satisfiable *only* under t₂; and under t₂ our isolate peak lands on Black Monday to the day, where t₁/midpoint would stamp the shattered network 45/21 days *before* the crash (A10) |
| Non-positive adjusted prices | **drop the ticker** | yfinance back-adjusts dividends by subtraction and can return negative closes (`VHI`); `log()` is undefined (A2) |
| Δt = 30 meaning | **30 returns from 31 prices** | forced by the paper's own `N_w = C_p − Δt` |
| Crisis date ranges | **pre-registered in Phase 0** | the paper dates only Black Monday; choosing windows after seeing results would let any detector be tuned (C2/E10) |

## 7. Scope

Phases 0–9, with Phase 8 trimmed to **8a** (τ signal), **8b** (crisis detector),
**8c** (rise-or-fall event study), **8f** (Louvain stability).

Deliberately **out of scope**, recorded in `PITFALLS.md` so nothing is silently
dropped: **A6 / 8d** (Δt = 20/60/120 robustness), **A5 / 8e** (market-mode
removal), **E11** (mechanism for the 2002 break). The first two each require a
full Phase 2–6 rerun.

## 8. Status

- [x] Phase 0 — universe, panel, `UNIVERSE.md`, pre-registered crisis windows
      *N = 377 × T = 6,345 (1986-01-02 → 2011-02-28), 6/6 acceptance tests pass.*
- [x] Phase 1 — returns, quality gates, `rho_bar` reference series
      *N 377 → 360 after the frozen-price / bad-print / split gates; 6/6 pass.*
- [x] Phase 2 — networks, τ(t), A10 resolved, A4 noise reference
      *6,315 networks × 6,462 edges, mean degree exactly 35.9 = f(N−1); 8/8 pass.*
- [ ] Phase 3 — communities, `k[t]`, `Π[t]`, three modularities
- [ ] Phase 4 — null models, noise floor
- [ ] Phase 5 — the eight measures
- [ ] Phase 6 — scoring, **overlap correction**, compression check
- [ ] Phase 7 — PCA with the omitted diagnostics
- [ ] Phase 8 — extensions (8a, 8b, 8c, 8f)
- [ ] Phase 9 — notebook + `REPLICATION_REPORT.md`

## 9. Known deviations

Filled in as phases complete. See `PITFALLS.md` for the audit items each one maps to.

| # | What diverges | Paper | This implementation | Reason |
|---|---|---|---|---|
| D1 | Universe | 348 of 3,799 NYSE stocks | **377 of 1,575** | The paper's contemporaneous candidate list no longer exists. Ours comes from a *currently-listed* directory, so it is pre-filtered by survival — a smaller pool but a larger survivor set (A1). |
| D2 | Panel length | `C_p` = 6,008 | **T = 6,345** | The paper's own count is inconsistent with its stated Jan-1986–Feb-2011 range on any modern calendar; Fig. 8's anchors localise the 337-day shortfall to before May 1991 (A10). |
| D3 | Windows | `N_w` = 5,978 | **6,314** | Follows directly from D2. |
| D4 | `VHI` excluded | not applicable | dropped | `yfinance` returns negative dividend-adjusted closes for it; a 2015-era Yahoo pull would not have had this defect (A2). |
| D5 | Timestamp convention | unstated | **t₂ (window close)** | Only convention consistent with Fig. 8's anchors, `C_p` = 6,008 and a Feb-2011 end; independently confirmed by Black Monday alignment (A10). |
| D6 | Return quality gates | none | **17 further tickers dropped** | Frozen-price runs up to 743 days would have produced guaranteed isolated nodes, faking the Fig. 3B crisis signature; plus one bad print and unadjusted corporate actions. See `data/QUALITY.md` (A2/A3). |
| D7 | Analysis universe | N = 348 | **N = 360** | Phase 0's 377 completeness survivors minus the 17 removed by D6. |
