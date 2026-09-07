# reports/

Shareable write-ups for supervision. Everything here is generated from the
repository's own artifacts; no number is retyped from a figure.

**Naming convention.** Standalone reports are `YYYY-MM-DD_short_name.tex`, dated by
the day they were sent. `progress_report.tex` is the one exception: it is a running
log with no single date, and new entries go at its bottom. Superseded material moves
to `archive/` with a note saying what replaced it.

**House format.** Every `.tex` here is self-contained: no `.bib`, no
`\includegraphics`, no external data. Upload the single file to Overleaf and hit
Recompile; comment with *Review → Add comment*. Locally:

```bash
brew install tectonic          # if not already present
tectonic -X compile reports/<file>.tex
```

Prose in these documents uses no em-dashes, by request.

---

## `progress_report.tex` — running working log (last entry 7 Sep 2026)

A dated log rather than a paper: one `\logday{...}` entry per stretch of work,
newest at the bottom. Covers the three replications in timeline order, the
adversarial re-audit, the two matrices behind the two indicators, the spectral
probe, the 31 August supervision meeting, and the modularity-equivalence study.
5 pages.

Entries are a historical record and are not rewritten when later work supersedes
them. Where a conclusion is retracted, the newer entry says so explicitly; the
12 August D1/D2 framing is retracted in the 31 August entry.

## `2026-09-08_modularity_equivalence.tex` — the current report (8 Sep 2026)

Answers the question set at the 31 August meeting: find a quantity derived from the
modularity matrix `B` that carries the same information as Paper 1's Fiedler value
`λ₂(L)`, tested on Paper 1's own FAAMUNG data. 10 pages, four TikZ/pgfplots figures.

**The answer is negative, in two parts, and the negative answer is the deliverable.**

1. **After degree normalisation `B` and `L` are the same operator.**
   `B_n = I − L_sym − uuᵀ` where `u` is `L_sym`'s own null vector, so their
   eigenvalues pair one-to-one as `μ = 1 − λ` and their eigenvectors are shared.
   Verified across the full spectrum on all 243 windows to 1.8e-15, with eigenvector
   agreement 1.000000. No normalised-`B` indicator can be new; this is a proof, not
   a measurement.
2. **Unnormalised, the natural shortcut is unsound.** `d̄ − μ₁(B)` scored the highest
   agreement with `λ₂` of anything tested, but the identity behind it needs a
   *regular* graph. Substituting the mean degree leaves an error 89% as large as
   `λ₂`'s own standard deviation.

Three findings came out of chasing that down, and they matter more than the original
question: correlation is the wrong screen for novelty (it admitted the spurious
candidate and misses provable tautology); `λ₂` itself sits inside the noise band on
Paper 1's own setup (COVID at the 76th percentile of a sliding-event null,
`p = 0.243`); and `λ₂` decomposes into a structure part and a degree part, with
degree acting as a modifier rather than a signal. That last point localises where
anything new can live, since normalisation is exactly what destroys the degree
information.

Code and the number-checking script are in
[`modularity_indicator/`](../modularity_indicator/):

```bash
../.venv/bin/python modularity_indicator/report_numbers.py
```

## `2026-08-19_discrepancy_explained.tex` — the three-paper exploration (19 Aug 2026)

Written in reply to the note of 13 August asking, of the 12 August log entries,
*"what is the key discrepancy you have found in your simulations and what potential
directions you are envisioning?"* Those entries assert their conclusions without
building up to them; this document walks the whole investigation from Silva's own
definitions to each number, defining every object before it is used. 17 pages, eight
TikZ/pgfplots figures, nine tables.

- Opens with an introduction giving the whole argument in outline, then earns every
  claim in it over §1 to §9.
- Several figures are drawn from `probe/spectral_probe.parquet` directly (the
  λ₂-vs-μ₁ scatter and the escaping-mode histogram are real data, not schematics).
- The outside-literature survey is flagged for what it is: assembled largely with
  Claude, abstracts and main results rather than full readings. A set of pointers,
  not a literature review.
- §9 sets out D1, D2 and D3 at equal length with no ranking, deliberately.

**Read §9 with the later retraction in mind.** D1 and D2 are built on the identity
`λ₂(L) + μ₁(B) = d`, which holds only on a regular graph. The 2026-09-08 report
supersedes that framing: normalised, the two matrices are provably the same
operator, and the unnormalised version of the identity is not an identity at all.
The document's *measurements* stand; its proposed spine does not.

Re-derive every number in it:

```bash
/opt/anaconda3/bin/python reports/probe/check_numbers.py
```

`check_numbers.py` recomputes all 91 quoted values from the parquet, the probe JSONs
and the replication's phase results, *and* confirms each printed string occurs in the
source, so the prose cannot drift from the data in either direction. It exits
non-zero on any mismatch.

### Known gap in the bibliography

Two entries, the *European Journal of Finance* Laplacian-energy paper and the
*International Review of Economics & Finance* Laplacian-energy-like paper, are cited
by title and DOI because the publishers block automated access to their author lists.
They are marked in red and **must not be cited as-is**; fill the author lists from
the published versions before the document goes anywhere beyond supervision.

## `archive/` — superseded material

`EXPLORATION.md` and `EXPLORATION_EXPLAINED.md`, both 12 August 2026, the
plain-language and ground-up accounts of the same exploration that
`2026-08-19_discrepancy_explained.tex` later wrote up properly. Each carries a note
at the top saying what replaced it and which of its claims are retracted.

## `probe/` — preliminary spectral evidence

The numbers behind the 19 August report. These reuse the Paper-3 pipeline's own
`utils/crisis_dates.py`, `net/construct.py` and scoring conventions unchanged, so
they are directly comparable to the replication's published AUCs.

| File | What it does |
|---|---|
| `spectral_probe.py` | Full spectrum of `L`, `B` and the correlation matrix `C` for each of the 6,315 daily networks; scores each spectral statistic as a crisis detector. ~6 min. |
| `duality.py` | Tests the regular-graph identity `λ₂(L) + μ₁(B) = d`; recomputes the Fiedler value on the largest connected component to remove the fragmentation confound. ~6 min. |
| `incremental.py` | Asks whether modularity-spectrum signals add anything to the market mode, in-sample and on a strict pre/post-2002 chronological split. Seconds. |
| `check_numbers.py` | Re-derives every number quoted in the 19 August report. |
| `spectral_probe.parquet` | Per-day spectra, 6,315 × 15. Written by the first two scripts. |
| `*_auc.json` | The scored results. |

```bash
/opt/anaconda3/bin/python reports/probe/spectral_probe.py   # writes the parquet
/opt/anaconda3/bin/python reports/probe/duality.py          # extends the parquet
/opt/anaconda3/bin/python reports/probe/incremental.py      # reads it
```

Requires the Paper-3 derived artifacts (`data/edges.npy`, `data/returns.npy`,
`results/series/*.parquet`), which are gitignored as rebuildable; regenerate them
with `python run_phases.py 1 3` inside `paper3_market_modularity/` first.

**Headline.** Scored identically to the replication's own detector, against the same
pre-registered crisis windows (2.68% base rate):

| Signal | Operator | AUC |
|---|---|---|
| spectral entropy of `C` | correlation | 0.822 |
| `λ₁(C)/N` (absorption ratio) | correlation | 0.812 |
| `τ(t)` | n/a | 0.807 |
| `μ_N(B)` (anti-community edge) | **modularity** | 0.760 |
| `λ₂(L)` on the largest component | Laplacian | 0.652 |
| `λ₂(L)` (Fiedler value) | Laplacian | 0.617 |
| `μ₁(B)` (leading modularity eigenvalue) | **modularity** | 0.607 |
| `Q(t)` (Louvain modularity) | partition scalar | 0.538 |

The modularity **matrix** carries substantially more crisis information than the
modularity **scalar** derived from it; and at Δt = 30, N = 360 the mean number of
correlation eigenvalues escaping the Marchenko-Pastur bulk (excluding the market
mode) is **0.70**, so there is essentially no statistically resolvable community
structure to detect at the window length this literature uses.

`μ_N(B)` winning here and again on FAAMUNG in the 2026-09-08 study, on two unrelated
network constructions, is the one live lead in this direction. It is not significant
on either dataset alone.
