# Engineering notes — Paper 1 solver

What was hard in building the GMRF graph-Laplacian solver, and how each problem was
resolved. Distilled from the full working log (`dev_log.md`, untracked) with one
unsupported claim corrected; see the note at the end of §1.

Every number here is reproducible from tracked artifacts. Commands assume
`paper1_gmrf_laplacian/` as the working directory.

---

## 1. ADMM was dropped at design stage

The plan was ADMM, splitting `L = Z` and alternating an `L`-update, a projection of
`Z` onto the Laplacian cone, and a dual update. It was abandoned **before any ADMM
code was written**, for two structural reasons:

1. **The `L`-update has no closed form.** With `−log gdet(L)` in the objective it is a
   structured eigenvalue problem, so it needs its own iterative inner solver. ADMM
   becomes a two-level loop.
2. **The `Z`-update is not a true projection.** The Laplacian cone requires symmetry,
   zero row sums, non-positive off-diagonals and PSD *simultaneously*, and these
   constraints interact. The usual "project then fix" order — PSD via eigen-flip, zero
   the violating off-diagonals, restore row sums — is not a projection, because zeroing
   off-diagonals after the PSD step changes the eigenvalues and the result is no longer
   PSD. Alternating to convergence works, at the cost of a second inner loop.

> **Correction.** The working log also asserted that "early experiments showed ADMM's
> dual variable U grows without bound near the boundary of the Laplacian cone". No such
> experiment exists in this repository: there is no ADMM implementation, no timing log
> and no iteration trace, so there are **no ADMM-versus-L-BFGS-B comparison numbers**.
> The two reasons above are design-stage arguments, not measurements, and that is all
> they are claimed to be.

## 2. The edge-weight parameterisation

For `p` nodes the complete graph has `m = p(p−1)/2` edges. With `K ∈ ℝ^{m×p}` the signed
incidence matrix (`K[l,i] = +1`, `K[l,j] = −1` for edge `l = (i,j)`) and weights `w ≥ 0`:

```
L(w) = Kᵀ diag(w) K
```

is a valid graph Laplacian for *any* `w ≥ 0`:

- `L1 = 0` because `K1 = 0` (every row has one `+1` and one `−1`);
- `L_ij = −w_ij ≤ 0` for `i ≠ j` by the incidence structure;
- `L ⪰ 0` because `L = Σ_l w_l k_l k_lᵀ` is a non-negative sum of rank-1 PSD terms.

All four constraints vanish. What remains is a box constraint on `m` variables, handled
by `scipy.optimize.minimize(method="L-BFGS-B")`.

The objective linearises cleanly. Since
`tr(L(w)S) = Σ_l w_l (S_ii + S_jj − 2S_ij)` and `β Σ|L_ij| = β Σ_l w_l` for `w ≥ 0`:

```
f(w) = q_effᵀ w − log gdet(L(w)),    q_eff,l = (S_ii + S_jj − 2S_ij) + β
```

strictly convex in `w` over connected graphs.

**Cost of the choice.** `w_ij = 0 ∀j` drives `log gdet → −∞`, so no node can reach zero
degree. The SGL baseline therefore cannot produce true isolated nodes, and the paper's
Fig 2 failure mode appears as a draw-threshold artifact instead (deviation D3 in
`CLAUDE.md` §9).

## 3. Four numerical traps

| Trap | Symptom | Fix |
|---|---|---|
| `w₀ = 0` | `L = 0` has no positive eigenvalues, `log gdet = −∞`, pseudoinverse undefined, gradient crashes on the first call | start at `w₀ = 1/p`, giving eigenvalues `{0, 2/p, …}` |
| Pseudo-determinant tolerance | `L` is singular by construction; float noise can make the null eigenvalue *slightly negative*, so `log(negative)` → NaN | treat `λ > 1e-10` as positive; if none are, return a large penalty and gradient so L-BFGS-B backs out of the degenerate region |
| L-BFGS-B default tolerances | the eigendecomposition injects ~1e-14 noise per gradient call, so the solver either declared convergence early or chased machine-precision gains for thousands of iterations | `ftol = tol²`, `gtol = tol`, and `maxls = 50` (default 20) because the log-gdet valley makes line-search steps undershoot |
| Boundary handling | L-BFGS-B returns `w ≈ −1e-15` at the bound, producing a *positive* off-diagonal and an invalid Laplacian | clip with `np.maximum(res.x, 0.0)` before forming `L` |

## 4. The temporal-penalty sign bug

The hardest defect in the project, because it looked like a modelling result.

**Symptom.** Algorithm 2's Fiedler value `λ₂(L_t)` was confined to `[0.00, 0.78]` with
**65 of 243 windows exactly zero** (disconnected estimates, all of September 2019). The
paper's threshold `τ = 1.0` was therefore never crossed, so strategy S2 held the market
every single day and was identical to buy-and-hold: Fig 4 was vacuous. Every unit test
still passed, because the solver was returning perfectly valid Laplacians.

**Cause.** The temporal-consistency penalty `δ‖L_t − L_{t−1}‖²_F` had been approximated
by augmenting the similarity matrix, `S_aug = S_t + (2δ/n_t)·L_prev`. The Frobenius
cross-term is `−2δ·tr(L·L_prev)`, so the augmentation must **subtract**. With the `+`
sign the linear cost `q_l = S_ii + S_jj − 2S_ij` was *raised* precisely on the edges
present in the previous graph: the "consistency" penalty repelled each window from its
predecessor and shrank every weight. At `δ = 100`, `n = 30` it was adding `6.7·L_prev`
to `S`.

A second bug compounded it: the docstring claimed each window warm-started from
`L_prev`, but `learn_laplacian` had no initial-point argument, so every window restarted
from `w₀ = 1/p`.

**Fix.** Implement the penalty exactly in edge-weight space, where it is a plain
quadratic:

```
γ(2‖Δw‖² + ‖K_sqᵀ Δw‖²),    gradient γ(4Δw + 2 K_sq K_sqᵀ Δw),    γ = δ/n_t
```

plus a real warm start (`w0 = w_prev`). An earlier claim that the quadratic term "breaks
the pseudoinverse gradient" was also wrong: it is orthogonal to the log-gdet gradient.

**Verification.** Finite-difference gradient check, an exact-Frobenius equality check
against the explicit matrix form, and a `γ → ∞` pin test.

**Outcome.**

| | before | after |
|---|---|---|
| `λ₂` range over 243 windows | `[0.00, 0.78]` | **`[0.6557, 1.5011]`**, median 1.1036 |
| windows with `λ₂` exactly 0 | **65** | **0** |
| `τ = 1.0` gate | never binding (S2 ≡ S1) | live |

Check the after-state directly:

```bash
python -c "import numpy as np; l=np.load('data/lam2_series.npy'); \
print(len(l), l.min(), l.max(), (l==0).sum())"
# 243 0.6557... 1.5011... 0
```

## 5. What the fix did and did not buy

The mechanism replicates; the paper's headline does not. Two different evaluation
windows are quoted in the history, so both are labelled here:

| Window | S1 (buy & hold) | S2 (`λ₂ < 1.0`) | Days invested |
|---|---|---|---|
| Full cached series, to 2020-06-30 | +0.214 | +0.038 | 103 / 242 |
| **Paper's window, to 2020-05-01** (what `exp4` reports) | **+0.0779** | **−0.0111** | **79 / 201** |

Both are cumulative log-returns, gross of costs. On the paper's window S2 sits out
autumn 2019 and the crash exactly as the paper's does — flat at ≈ −0.01 while S1 falls
to a −0.2123 trough — but our `λ₂` stays near 1.3 through April where the paper's
collapses below 1.0, so S2 re-enters late and misses the rebound. Crash-avoidance
replicates; the final P&L ranking does not. No threshold was tuned to close the gap;
the in-sample `τ = median` row is printed as reference only and is excluded from the
figure.

Reproduce the paper-window row:

```bash
python exp4_trading.py     # prints both rows and the tau table
```
