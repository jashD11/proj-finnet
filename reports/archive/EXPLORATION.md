# Exploration: Laplacian vs. modularity spectra

> **Archived 8 September 2026.** Superseded by
> `reports/2026-08-19_discrepancy_explained.tex`, which builds the same
> argument up from the papers' own definitions. Kept for the plain-language
> framing. Note that its λ₂(L) + μ₁(B) = d spine is now retracted: that
> identity needs a regular graph, see §4 of
> `reports/2026-09-08_modularity_equivalence.tex`.

**12 August 2026.** Plain-language account of the exploration that produced
`progress_report.tex`. Same content, no LaTeX and no equations — this is the
version to read first. Every number here is reproduced by the scripts in
`probe/`; none is retyped from a figure.

---

## 1. Where this came from

The three replications in this repository were treated as a literature review.
The question now is what to build on top of them. The prompt was the prof's:

> Paper 1's crash indicator is the Fiedler value λ₂(L). Paper 3's is modularity
> Q. Look at the **modularity matrix** as well — its eigenvalues and singular
> values may carry more. Can we get a study out of comparing Laplacian against
> modularity matrices?

## 2. What was looked at

**The repository itself**, end to end — the Cardoso & Palomar Laplacian-learning
pipeline, the TICC regime pipeline, the Silva modularity pipeline, and the
26-item methodological audit already run against Silva.

**Outside literature.** Spectral market-stress indicators in finance, and the
mathematical theory of the modularity matrix. The theory lives almost entirely
in physics and linear algebra — MacMahon & Garlaschelli on the correct
random-matrix null for correlation modularity, Fasino & Tudisco on the
modularity spectrum and anti-communities, Nadakuditi & Newman on when community
detection is possible at all. In finance, modularity appears essentially only as
the scalar Q. Repeated searches for modularity-matrix *spectra* used as
financial indicators returned nothing.

**An experiment.** Rather than speculate about what the modularity matrix would
show, it was computed. The Silva pipeline already builds one network per day for
6,315 days; the probe takes the full spectrum of three operators on each of those
networks and scores every resulting statistic through the replication's *own*
crisis detector. As a control, the probe reproduces the replication's stored Q
and τ scores to machine precision, so the new numbers sit on exactly the same
ruler as the old ones.

*Scoring convention.* AUC against the pre-registered "sharp" crisis windows,
base rate 2.68%. Each detector is a 250-day trailing z-score using strictly
prior data only. 0.5 is a coin flip, 1.0 is perfect.

## 3. The main result: the matrix beats the scalar

Paper 3 runs Louvain, gets a partition, and collapses the network into one
number Q(t). But the matrix that number comes from — **B = A − dd'/2m**, observed
edges minus expected edges — has 360 eigenvalues. Q is roughly one summary of one
eigenvector of it. The other 359 numbers are thrown away.

That turns out to be expensive.

| Signal | Operator | AUC |
|---|---|---|
| spectral entropy of `C` | correlation | 0.822 |
| `λ₁(C)/N` — absorption ratio | correlation | 0.812 |
| `τ(t)` — correlation threshold | — | 0.807 |
| **`μ_N(B)`** — most negative modularity eigenvalue | **modularity** | **0.760** |
| number of zero Laplacian eigenvalues (components) | Laplacian | 0.756 |
| `Σ_{μ>0} μ` — positive modularity mass | **modularity** | **0.750** |
| `λ_max(L)` | Laplacian | 0.731 |
| `#{μ > 0}` — count of positive modularity eigenvalues | **modularity** | **0.722** |
| `μ₁−μ₂` — modularity spectral gap | modularity | 0.636 |
| `λ₂(L)` on the largest connected component | Laplacian | 0.652 |
| `λ₂(L)` — Fiedler value | Laplacian | 0.617 |
| `μ₁(B)` — leading modularity eigenvalue | modularity | 0.607 |
| **`Q(t)`** — Louvain modularity | **partition scalar** | **0.538** |
| max Laplacian spectral gap | Laplacian | 0.526 |

Q(t) at 0.538 is barely distinguishable from a coin flip. Eigenvalues of the
matrix it was derived from reach 0.760, on the same networks and the same
scoring. **The information is in the matrix; the Louvain step destroys most of
it.**

Two details worth keeping:

- The strongest modularity signal is the *most negative* eigenvalue. Positive
  eigenvalues of B detect communities; negative ones detect **anti-communities**
  — groups trading against each other. Nobody in finance appears to have looked
  at this.
- `μ₁(B)` tracks Q(t) at r = 0.761 in levels while being fully deterministic.
  Louvain is stochastic and gives a different answer on every run. That alone
  makes it a useful drop-in.

## 4. The framing: dual in theory, orthogonal in practice

This is the finding worth building a study on.

There is an exact identity: on a connected graph where every node has the same
degree *d*,

```
λ₂(L) + μ₁(B) = d
```

The Fiedler value and the leading modularity eigenvalue are the *same quantity*
read from opposite ends. If that held here, comparing Paper 1's indicator with
Paper 3's would be meaningless — they would be one indicator.

Silva's fixed-density construction pins the *average* degree at exactly 35.90
every single day, so the identity ought to nearly hold. It does not:

| | value |
|---|---|
| residual `d̄ − (λ₂ + μ₁)`, all days | 9.489 ± 4.627 |
| residual, connected days only (37.1% of days) | 9.210 ± 3.595 |
| residual, largest connected component | 11.109 ± 5.314 |
| `corr(λ₂, μ₁)`, all days | **−0.187** |
| `corr(λ₂, μ₁)`, connected days | −0.119 |
| `corr(λ₂, μ₁)`, largest component | −0.169 |

**Dual in theory, orthogonal in practice.** The comparative study then has a
clean, non-arbitrary question at its centre: *what is that residual made of?*
Three candidates — degree heterogeneity, network fragmentation, and the market
mode — and establishing which one carries the crisis information is a real
result, not a leaderboard.

## 5. Two uncomfortable findings

Both should be known before committing to any of this.

**There may be nothing to detect.** At a 30-day window with 360 stocks,
random-matrix theory puts the noise band at [6.07, 19.93]: correlation
eigenvalues below 19.93 are indistinguishable from pure noise. The average
number of eigenvalues that escape that band — excluding the market-wide mode —
is **0.70**, with a maximum of 3 and often zero. Less than one resolvable group
mode per day. Meanwhile Louvain confidently reports a median of nine communities
daily. A large share of what this literature calls community structure is not
statistically resolvable at the window length the literature itself uses. This
is the rigorous version of the noise-control problem the earlier audit already
identified.

**The strong signals are all one signal, and it may not generalise.**
Everything scoring above 0.75 — τ, absorption ratio, spectral entropy — is a
single underlying signal wearing three hats; they correlate above 0.99 pairwise.
It is the market mode: in a crash, everything moves together. Worse, on a strict
chronological split (train pre-2002, test post-2002) the modularity signals stop
beating that market mode:

| feature set | in-sample | out-of-sample |
|---|---|---|
| market mode only | 0.812 | **0.730** |
| + `μ₁(B)` | 0.822 | 0.629 |
| + `μ_N(B)` | 0.810 | 0.719 |
| + `Q(t)` | 0.831 | 0.699 |
| + `λ₂(L)` on LCC | 0.810 | 0.700 |

In-sample the additions help; out-of-sample every one of them hurts. That is the
textbook overfitting signature. Only 27 test crisis events, so it is not
conclusive — but it is the first thing a referee will ask about, and it needs a
real answer before any of this is written up as a detector.

## 6. Three directions

**D1 — the operator comparison (recommended lead).** Decompose the 9.49
residual into its degree-heterogeneity, fragmentation and market-mode parts.
This gives a principled Laplacian-vs-modularity comparison anchored on a
theoretical identity, rather than a table of competing AUCs. It is the version
of the prof's question that has a defensible answer at the end.

**D2 — detectability (recommended, same project).** Map the region of
(window length, universe size, network density) over which community detection
on financial correlations is statistically possible at all, and build a
deterministic, Louvain-free replacement for the community count. This repairs
the weakest joint in all three papers and shares all of D1's machinery.

**D3 — anti-communities (gated).** `μ_N(B)` at 0.760 is the most promising
single new number here, and it is unexplored in finance. But it correlates 0.68
with the market mode, so part of that score is borrowed. Gate this on one quick
check: does it survive projecting the market mode out? If yes, it is a paper on
its own. If no, it folds back into D1 as a component of the residual.

**Recommendation: D1 + D2 as one project**, with D3 as the extension if the
check passes.

## 7. Caveats

**The Laplacian numbers are on Silva's graphs, not Paper 1's.** λ₂ scoring 0.617
here is *not* evidence against Cardoso & Palomar or against Kang et al. Those
papers build entirely different objects — a learned Laplacian and a filtration
respectively. What the number says is that on *thresholded correlation networks
of this kind*, the Fiedler value is a weak detector. Extending the comparison to
the learned Laplacian is itself part of D1.

**Two bibliography entries are unverified.** The *European Journal of Finance*
Laplacian-energy paper and the *International Review of Economics & Finance*
Laplacian-energy-like paper are cited by title and DOI because the publishers
block automated access to their author lists. They are flagged in red in
`progress_report.tex` and must be filled in from the published versions before
the document circulates beyond supervision.

**Everything here is preliminary.** These are one-pass probes designed to tell
which direction is worth months of work, not results ready for a paper.

## 8. Where things are

| | |
|---|---|
| `progress_report.tex` | the full write-up, 14 pages, Overleaf-ready, with five questions for the prof in §8 |
| `probe/spectral_probe.py` | spectra of `L`, `B`, `C` for all 6,315 days; scores each statistic (~6 min) |
| `probe/duality.py` | tests `λ₂ + μ₁ = d`; recomputes `λ₂` on the largest component (~6 min) |
| `probe/incremental.py` | does the modularity spectrum add anything to the market mode? (seconds) |
| `probe/*.json` | the scored results quoted above |
| `probe/spectral_probe.parquet` | per-day spectra, 6,315 × 15 |
