# reports/

Shareable write-ups for supervision. Everything here is generated from the
repository's own artifacts; no number is retyped from a figure.

## `EXPLORATION.md` — plain-language account (12 Aug 2026)

The same exploration as the report below, without LaTeX or equations: what was
looked at, the finding that the modularity **matrix** beats the modularity
**scalar**, the `λ₂ + μ₁ = d` framing, the two uncomfortable findings, and the
three proposed directions. **Read this one first.**

## `discrepancy_explained.tex` — the supervisor-facing explanation (19 Aug 2026)

Written in reply to the note of 13 August asking, of the 12 August log entries,
*"what is the key discrepancy you have found in your simulations and what
potential directions you are envisioning?"* Those entries assert their
conclusions without building up to them; this document walks the whole
investigation from Silva's own definitions to each number, defining every object
before it is used.

- **Opens with an introduction** that gives the whole argument in outline — the
  three replicated papers in one table, a second table pairing each observed
  discrepancy with what would explain it, how each was checked, and the summary —
  then earns every claim in it over §1–§9.
- **Eight TikZ/pgfplots figures and nine tables**, several drawn from
  `probe/spectral_probe.parquet` directly (the λ₂-vs-μ₁ scatter and the
  escaping-mode histogram are real data, not schematics).
- **The outside-literature survey is flagged for what it is** — assembled
  largely with Claude, abstracts and main results rather than full readings, in
  an aside in §1 and again in the appendix reference list. It is a set of
  pointers, not a literature review.
- **§9 sets out D1, D2 and D3 in one symmetric frame, at equal length, with no
  ranking and no combined recommendation** — unlike `EXPLORATION.md` §6, which
  does rank them. That is deliberate.
- **Self-contained.** No `.bib`, no `\includegraphics`, no external data files.
  Upload the single `.tex` to Overleaf and hit Recompile; comment with
  *Review → Add comment*. Verified with Tectonic 0.17.0 → 17 pages, no overfull
  boxes and no undefined references.

Three things it corrects, found while re-verifying against the artifacts:

1. The market-mode correlations quoted as `0.68` and `0.025` hold on the
   **signed, z-scored detector series**; in raw levels the same pairs are
   `−0.819` and `+0.294`. Neither `EXPLORATION.md` nor `progress_report.tex`
   says which basis it means, so the new document states it (§8.3).
2. `progress_report.tex` labels the AUC-0.756 row "isolated-node count". That
   series is `lap_n_zero`, the **number of connected components**; the
   isolated-node count is a different series scoring `0.7597`.
3. The noise-control figures come from two different runs — the Phase-3 level
   comparison (`Q = 0.2246`) and the Phase-4 run that carries its own matched
   null (`Q = 0.2253`). Only the second supports the excess calculation, and
   the document now says so.

### Compiling and checking

```bash
tectonic -X compile reports/discrepancy_explained.tex

# re-derive every number in the .tex from the stored artifacts:
/opt/anaconda3/bin/python reports/probe/check_numbers.py
```

`check_numbers.py` recomputes all 91 quoted values from the parquet, the probe
JSONs and the replication's phase results, *and* confirms each printed string
occurs in the source — so the prose cannot drift from the data in either
direction. It exits non-zero on any mismatch.

## `progress_report.tex` — running working log (last entry 12 Aug 2026)

A dated log rather than a paper: one `\logday{...}` entry per stretch of work,
newest at the bottom. Covers the three replications in timeline order, the
adversarial re-audit, the two matrices behind the two indicators, the spectral
probe, and the proposed directions. Add new entries at the end.

- **Self-contained.** No external `.bib`, no `\includegraphics`. Compiles
  standalone with Tectonic 0.17.0.
- **Built for commenting.** Upload the `.tex` to Overleaf and use
  *Review → Add comment*; no macros needed.
- The five open questions are collected in the final entry.

### Compiling

```bash
# Overleaf: upload progress_report.tex, hit Recompile. Nothing else needed.

# Locally (Tectonic pulls the packages it needs on first run):
brew install tectonic          # if not already present
tectonic -X compile reports/progress_report.tex
```

Only underfull-hbox (loose line) warnings remain; there are no overfull boxes,
undefined references or undefined citations.

### Known gap in the bibliography

Two entries — the *European Journal of Finance* Laplacian-energy paper and the
*International Review of Economics & Finance* Laplacian-energy-like paper — are
cited by title and DOI because the publishers block automated access to their
author lists. They are marked in red in the bibliography and **must not be
cited as-is**; fill the author lists from the published versions before the
document goes anywhere beyond supervision.

## `probe/` — preliminary spectral evidence

The numbers in Part V of the report. These reuse the Paper-3 pipeline's own
`utils/crisis_dates.py`, `net/construct.py` and scoring conventions unchanged,
so they are directly comparable to the replication's published AUCs.

| File | What it does |
|---|---|
| `spectral_probe.py` | Full spectrum of `L`, `B` and the correlation matrix `C` for each of the 6,315 daily networks; scores each spectral statistic as a crisis detector. ~6 min. |
| `duality.py` | Tests the regular-graph identity `λ₂(L) + μ₁(B) = d`; recomputes the Fiedler value on the largest connected component to remove the fragmentation confound. ~6 min. |
| `incremental.py` | Asks whether modularity-spectrum signals add anything to the market mode, in-sample and on a strict pre/post-2002 chronological split. Seconds. |
| `spectral_probe.parquet` | Per-day spectra, 6,315 × 15. Written by the first two scripts. |
| `*_auc.json` | The scored results. |

```bash
/opt/anaconda3/bin/python reports/probe/spectral_probe.py   # writes the parquet
/opt/anaconda3/bin/python reports/probe/duality.py          # extends the parquet
/opt/anaconda3/bin/python reports/probe/incremental.py      # reads it
```

Requires the Paper-3 derived artifacts (`data/edges.npy`, `data/returns.npy`,
`results/series/*.parquet`), which are gitignored as rebuildable — regenerate
them with `python run_phases.py 1 3` inside `paper3_market_modularity/` first.

**Headline.** Scored identically to the replication's own detector, against the
same pre-registered crisis windows (2.68% base rate):

| Signal | Operator | AUC |
|---|---|---|
| spectral entropy of `C` | correlation | 0.822 |
| `λ₁(C)/N` (absorption ratio) | correlation | 0.812 |
| `τ(t)` | — | 0.807 |
| `μ_N(B)` (anti-community edge) | **modularity** | 0.760 |
| `λ₂(L)` on the largest component | Laplacian | 0.652 |
| `λ₂(L)` (Fiedler value) | Laplacian | 0.617 |
| `μ₁(B)` (leading modularity eigenvalue) | **modularity** | 0.607 |
| `Q(t)` (Louvain modularity) | partition scalar | 0.538 |

The modularity **matrix** carries substantially more crisis information than the
modularity **scalar** derived from it; and at Δt = 30, N = 360 the mean number
of correlation eigenvalues escaping the Marchenko–Pastur bulk (excluding the
market mode) is **0.70**, so there is essentially no statistically resolvable
community structure to detect at the window length this literature uses.
