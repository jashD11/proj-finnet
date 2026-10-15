# modularity_indicator/

Does the **modularity matrix** `B = A − ddᵀ/2m` carry a crash indicator equivalent
to the Fiedler value `λ₂(L)` that Paper 1 reads off the graph Laplacian?

Asked by the supervisor on 31 August 2026, and answered here on Paper 1's own
FAAMUNG data, as scoped. Findings in [`FINDINGS.md`](FINDINGS.md); the write-up for
supervision is [`reports/2026-09-08_modularity_equivalence.tex`](../reports/2026-09-08_modularity_equivalence.tex).

**Answer: no, and the negative answer is the result.** After degree normalisation
`B` and `L` are the same operator (`B_n = I − L_sym − uuᵀ`, shared eigenvectors,
paired eigenvalues `μ = 1 − λ`, verified to 1.8e-15 on all 243 windows), so no
normalised-`B` quantity can be new. Unnormalised, the natural shortcut `d̄ − μ₁(B)`
needs a regular graph and is unsound here. What survives is a reframing: `λ₂` is a
blend of structure and degree, and `L` and `B` differ only in how they combine the
two.

## Running it

```bash
cd modularity_indicator
../.venv/bin/python build_series.py     # re-runs Algorithm 2, builds the menu
../.venv/bin/python evaluate.py         # agreement with λ₂, gate, mechanism, backtest
../.venv/bin/python robustness.py       # z-scored detector, random-gate + shift nulls
../.venv/bin/python significance.py     # bootstrap CIs, sliding-event null
../.venv/bin/python figures.py          # figures A to D
../.venv/bin/python report_numbers.py   # every number quoted in the report
```

Runs in order, under a minute total. `build_series.py` re-runs Paper 1's Algorithm 2
with exp3/exp4's exact constants to recover the 243 Laplacians that Paper 1 discards
after extracting `λ₂`. Nothing in `paper1_gmrf_laplacian/` is written to; it is read
only.

Two gates must pass before any result is believed, both asserted in `build_series.py`:
the recomputed `λ₂` matches Paper 1's cached `data/lam2_series.npy` to 1.2e-10, and
`evaluate.py` reproduces the published `S1 = +0.078` against `S2 = −0.011`.

## What is tracked

| Tracked | Why |
|---|---|
| the six `.py` files, `FINDINGS.md` | the study itself |
| `series.parquet` | the 243 × 28 quantity menu, so results are checkable without a solver re-run |
| `scores.csv`, `scores_zdetector.csv`, `scores_nulls.csv`, `scores_significance.csv` | the scored results quoted in the report |

Gitignored as redundant or regenerable: `series.csv` (same data as the parquet),
`figdata.txt` (scratch from `report_numbers.py --figdata`), and `figures/` (rebuilt
by `figures.py`).

## Conventions worth knowing before reading the numbers

- **Two windows are in play.** Structural claims use all **243** learned graphs;
  every AUC and the backtest use exp4's **201-day** evaluation window ending
  2020-05-01. Some figures differ between them, so each is labelled.
- **Direction is never fitted.** Each candidate's stress direction is fixed by the
  sign of its correlation with `λ₂`, not by what scores best, and every trading gate
  is matched to `λ₂`'s own invest rate of 79/201 days.
- **Raw-level AUC is confounded** by FAAMUNG's elevated autumn-2019 regime, which
  puts `λ₂` itself at 0.472, below chance. The reported detector is a strictly-prior
  60-day trailing z-score, following Paper 3's convention.
- **One event, 24 crash days.** No detection claim here is certifiable, `λ₂`'s
  included. Treat every AUC comparison as indicative.
