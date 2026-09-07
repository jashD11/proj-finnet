# A modularity-matrix equivalent of the Fiedler value

Exploration, 2026-09-07; Result 1 corrected 2026-09-08. Testbed: **Paper 1's
FAAMUNG data only** (`paper1_gmrf_laplacian/data/faamung.csv`), as requested.

Full write-up: `reports/2026-09-08_modularity_equivalence.tex`.
Every number here is printed by `report_numbers.py`.

## The question

Paper 1 (Cardoso & Palomar 2020) learns a GMRF graph Laplacian `L_t` on causal
30-day rolling windows of 7 tech stocks and uses the Fiedler value `λ₂(L_t)` as
its stress indicator: high λ₂ = tightened conditional correlations = stress, so
strategy S2 holds cash when λ₂ ≥ τ = 1.0.

Find a quantity derived from the **modularity matrix** `B = A − ddᵀ/2m` that
carries the same information, and test it on this same data.

## Setup

Algorithm 2 is re-run with exp3/exp4's exact constants (WINDOW=30, K=1, ETA=0,
BETA=0, DELTA=100, degree_control=False) to recover the 243 learned Laplacians
that Paper 1 discards after extracting λ₂. From each, `A_t = −offdiag(L_t)`,
then 22 modularity-matrix quantities plus 6 Laplacian reference quantities.

Two verification gates pass before anything else:

- recomputed λ₂ matches `data/lam2_series.npy` to **1.2e-10**
- the identity `μ₁(B_norm) = 1 − λ₂(L_sym)` holds to **1.3e-15** on all 243 windows

Scoring runs on exp4's evaluation window (201 signal days, ending 2020-05-01),
so the λ₂ row reproduces Paper 1's published S1 = +0.078 / S2 = −0.011.

Each candidate's **stress direction is fixed by its correlation with λ₂**, not by
what scores best, and every trading gate is **matched to λ₂'s own invest rate**
(79/201 days). Neither is tuned to the outcome.

## Result 1: the equivalence question has a negative answer

**Normalised `B` is the Laplacian.** With `u = D^{1/2}1/√(2m)`,

```
B_n = D^-1/2 B D^-1/2 = I − L_sym − uuᵀ,     u = L_sym's own null vector
spec(B_n) = {0} ∪ {1 − λ_i(L_sym)},  i ≥ 2,  with the SAME eigenvectors
```

Verified on all 243 windows: full-spectrum match to **1.8e-15**, eigenvector
agreement **1.000000** (min 1.000000). So every quantity derived from normalised
`B` is a Laplacian quantity relabelled, and this is a proof rather than a measurement. Unnormalised,
the two do *not* share a basis (overlap 0.871; the Fiedler vector and `B`'s
leading mode agree at only 0.039 on some windows), so anything new must live there.

**The candidate that topped the old table was spurious.** `d̄ − μ₁(B)` reproduced
91% of λ₂'s daily calls, but the identity `μ₁(B) = d − λ₂(L)` requires a
**regular** graph. Substituting the *mean* degree is not a weakened hypothesis:

| diagnostic | value |
|---|---|
| whole-spectrum error of the `d̄` substitution | **1.25** (5.3× λ₂'s sd of 0.235) |
| residual sd | 0.208, **89% of λ₂'s own sd** |
| share of `d̄ − μ₁`'s variance that is λ₂ | **40%** (32% residual, 28% covariance) |
| corr(residual, λ₂) | +0.395; error moves *with* the signal |
| corr(residual, degree CV) | +0.020; no correction available |
| residual in-crash vs out | 0.308 vs 0.512; error is event-dependent |
| degree CV of these graphs | 0.305 (0 = regular) |

And the `d̄` term does no work at all: controlling for scale, `μ₁(B)` alone gives
−0.851 against the proxy's +0.851. **This candidate is withdrawn.** What survives
is a regression fact with no theorem attached: *at fixed graph scale, `μ₁(B)`
tracks λ₂ at −0.85.*

**Much of the rest of the table was one quantity.** Correlation with `μ₁(B)`:
`spread` 0.966, `trace_pos` 0.951, `gap12` 0.935, `frob2` 0.898, `energy` 0.891,
`mu_ratio` −0.881, `q_bound` 0.793. `absmax` *is* `|μ_N|` (|μ_N| > μ₁ on 243/243).
The only plain-B candidate standing apart from the μ₁ family is **`μ_N(B)` at
−0.668**.

**Correlation was the wrong screen.** It admitted the spurious candidate above,
and it independently misses pure tautology: spectral entropy of `B_n` correlates
only **−0.264** with λ₂, which looks independent, yet rebuilds from the `L_sym`
spectrum alone to **8.0e-15**. Novelty must mean *not reconstructible from
spec(L)*, not *decorrelated from λ₂*.

## Result 2: crash detection (weakly powered, one event)

Raw levels are confounded: FAAMUNG's autumn-2019 regime is elevated, so raw-level
COVID AUC puts **λ₂ itself at 0.472, below chance**. Applying Paper 3's detector
convention (strictly-prior 60-day trailing z-score) fixes it:

| quantity | AUC raw | **AUC z-scored** | ΔAUC vs λ₂ (95% CI) |
|---|---|---|---|
| μ_N(B) | 0.791 | **0.775** | +0.015 [−0.115, +0.148] |
| λ₂(L) | 0.472 | 0.759 | n/a |
| μ₁(B) − μ₂(B) | 0.798 | 0.746 | −0.013 [−0.151, +0.127] |
| μ₁(B) | 0.716 | 0.726 | −0.034 [−0.153, +0.088] |
| IPR of B's leading eigenvector | 0.747 | 0.724 | −0.035 [−0.109, +0.042] |
| μ₁(B_norm) | 0.594 | 0.700 | −0.059 [−0.103, −0.016] |

μ_N(B), the most negative modularity eigenvalue and the "anti-community" mode,
edges out λ₂, and it is only weakly redundant with it (z-score correlation 0.51,
against −0.92 for μ₁(B_norm)). So it is plausibly carrying *different* information
rather than restating λ₂.

**But the difference is not significant, and neither indicator is certified.**
Sliding a 24-day window across the series and recomputing AUC (figure
`figures/D_sliding_null.png`, regenerated by `figures.py`) puts COVID at only the **76th percentile for λ₂ (p = 0.243)**
and the **66th for μ_N (p = 0.345)**. λ₂ clears a bootstrap over days (CI
[0.634, 0.875], excluding 0.5) but fails this harder null: it beats a random
*day*, not a random 24-day *stretch*.

## Result 3: the trading result is noise (null)

At λ₂'s own invest rate, five modularity quantities beat buy-and-hold, led by
`tr(B²)` at S2 = +0.127 against S1 = +0.078, with zero loss during the crash
window, apparently succeeding where Paper 1's own λ₂ replication failed
(S2 = −0.011).

It does not survive:

- a circular-shift null preserving each signal's autocorrelation gives every
  candidate **p ≥ 0.115**; `tr(B²)`'s own p is **0.305**
- a uniform random gate at the same invest rate beats S1 **39.2%** of the time,
  so **~11 of 28** candidates were expected to beat it by luck. Five did.

`tr(B²)` and `energy` also lose most of their apparent detection power once
z-scored (0.925 → 0.687 and 0.616): they were tracking the regime, not the crash.

## Result 4: what λ₂ is made of (the reframing)

λ₂ is read off the **unnormalised** L, so it blends structure with degree. The
containment is exact (`d_min·λ₂(L_sym) ≤ λ₂(L) ≤ d_max·λ₂(L_sym)` holds on
**243/243** windows), so the split is structural, not fitted. Splitting the
detection (z-scored, same detector):

| | AUC |
|---|---|
| λ₂(L), the full indicator | **0.759** |
| λ₂(L_sym), structure only | 0.700 |
| d̄, scale only | 0.537 (coin flip) |
| deg_cv, degree spread | 0.606 |
| μ_N(B) | **0.775** |

**Degree information is a modifier, not a signal**: worthless alone, yet it
improves detection when combined with structure, and it is exactly what
normalisation destroys. `L = D − A` combines degrees additively; `B = A − ddᵀ/2m`
combines them through a random-rewiring null. That difference is the only thing
`B` offers, and it explains rather than merely observes why μ_N(B) keeps winning.

## Reading

- **The equivalence claim fails, and the failure is the result.** Normalise and
  you get a tautology; skip the normalisation and the natural shortcut is unsound.
  That closes the whole μ₁-based family rather than one candidate.
- **The best match is not the best detector**, and neither target was well posed:
  λ₂'s own status on this data is not established.
- **FAAMUNG cannot settle any detection claim.** 7 stocks, 201 days, one crash.
  It cannot certify μ_N, and it cannot certify λ₂ either.

For context only, outside the requested testbed: `reports/probe/` already scored
these operators on Paper 3's 6,315-day, 360-stock sequence with 17 pre-registered
crises, and **μ_N(B) was the best modularity signal there too** (AUC 0.760 vs
λ₂'s 0.617). The same quantity winning on two independent constructions is
suggestive, and is the obvious next test if this direction is worth pursuing.

## Files

| file | what |
|---|---|
| `build_series.py` | re-runs Algorithm 2, computes the 28-quantity menu → `series.parquet` / `.csv` |
| `evaluate.py` | agreement with λ₂, gate agreement, crash mechanism, backtest → `scores.csv` |
| `robustness.py` | z-scored detector AUC, random-gate and circular-shift nulls → `scores_zdetector.csv`, `scores_nulls.csv` |
| `significance.py` | bootstrap AUC CIs, paired ΔAUC, sliding-event null → `scores_significance.csv` |
| `figures.py` | figures A–D into `figures/` (gitignored, regenerate on demand) |
| `report_numbers.py` | re-derives every number in `reports/2026-09-08_modularity_equivalence.tex` |

Run in order with `../.venv/bin/python`. Total runtime under a minute.
`series.parquet` and the four `scores_*.csv` are tracked, so every number above is
checkable without re-running the solver. `series.csv`, `figdata.txt` and `figures/`
are gitignored as redundant or regenerable.
Nothing in `paper1_gmrf_laplacian/` is modified; it is read only.
