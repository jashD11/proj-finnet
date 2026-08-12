# reports/

Shareable write-ups for supervision. Everything here is generated from the
repository's own artifacts; no number is retyped from a figure.

## `EXPLORATION.md` — plain-language account (12 Aug 2026)

The same exploration as the report below, without LaTeX or equations: what was
looked at, the finding that the modularity **matrix** beats the modularity
**scalar**, the `λ₂ + μ₁ = d` framing, the two uncomfortable findings, and the
three proposed directions. **Read this one first.**

## `progress_report.tex` — Progress Report & Problem Statement (12 Aug 2026)

Covers, in timeline order, all three replications (Cardoso & Palomar; TICC;
Silva et al.), what reproduced and what did not, the findings that survived the
adversarial re-audit, a literature review on spectral market-stress indicators,
a problem statement on **Laplacian vs. modularity spectra**, three proposed
research directions, and preliminary numerical evidence.

- **Self-contained.** No external `.bib`, no `\includegraphics`. Compiles
  standalone; verified with Tectonic 0.17.0 → 14 pages, no undefined references
  or citations.
- **Built for commenting.** Upload the `.tex` to Overleaf and use
  *Review → Add comment* (needs no macros). If margin notes in the source are
  preferred, `\pc{...}` is a yellow supervisor note, `\jd{...}` a blue note of
  mine, `\open{...}` an orange unresolved question. Set `\commentsfalse` in the
  preamble to hide all of them for printing.
- Section 8 (*Decisions I would like your view on*) collects the five open
  questions in one place.

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
