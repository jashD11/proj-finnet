# modularity_graph_learning/

Work on Prof. Anguluri's October 2026 proposal, *Modularity Matrix Approach for Graph
Learning?* It asks what modularity's null-model view adds to Laplacian graph learning
(Paper 1, Cardoso and Palomar 2020), in three parts: I, the exact relation between the
Laplacian and the modularity matrix under the paper's degree constraint; II, the market
factor as an implicit null model; III, modularity as a prior replacing `diag(L) = 1`.

**Start with [`EXPLAINED.md`](EXPLAINED.md)**: a plain-language walk-through of the proposal's
Part I, every issue found, how each was fixed, and what remains.

Paper 1's solver is reused, not copied: `common.py` puts `paper1_gmrf_laplacian/` on
`sys.path`, as `modularity_indicator/` does.

| File | What it does |
|---|---|
| `common.py` | Data loading (the 753-day S&P panel), `B(W)`, Laplacians, eigen helpers, unit-degree scaling |
| `report_numbers.py` | Prints every number in `reports/2026-10-18_proposal_issues.tex` (A: Part III boundedness. B: Part I(c) on the FAAMUNG graphs. C: Fig 2 before and after) and `reports/2026-10-27_part1_lemma.tex` (D: Part I on the repaired Fig 2 graph. E: counterexamples. F: regular / normalised / unequal degrees. G: degree-penalty sweep) |
| `fig2_init_check.py` | Algorithm 1 from the uniform vs the Problem 1 start, under 1 vs default BLAS threads, to show the uniform start is round-off dependent (~8 min per run) |
| `degree_sweep.py` | Algorithm 1 at weaker degree penalties `ρ`, to show Part I's match breaking as degrees spread (a few minutes per run) |
| `tests/test_lemma.py` | Part I (a), (b), (c), the `K₄` and 6-cycle counterexamples, the c-regular and unequal-degree cases, the Part III identity and the normalised case, on random graphs (25 tests) |
| `cache_*.npy` | Committed so `report_numbers.py` runs in seconds; delete to rebuild |

```bash
cd modularity_graph_learning
python -m pytest tests/ -q
python fig2_init_check.py uniform 1        # and: uniform default, problem1 1
python degree_sweep.py 30 10 3 1 0.3
python report_numbers.py                   # --part1 for D to G only; --figdata, --figdata-part1 for pgfplots
```

## Findings so far

- **Part I is finished** (`reports/2026-10-27_part1_lemma.tex`). Under `diag(L) = 1`,
  modularity adds nothing: on the repaired Fig 2 graph, k-means on `B` returns the
  sectors exactly. Loosening the degree penalty breaks the combinatorial match; the
  normalised one stays exact.
- **Part I(c)** needs unit degrees; the paper's time-varying graphs (Problem 11) have none
  (mean degree 2.13, degree CV 0.305 on FAAMUNG).
- **Part III** as written is unbounded below: bounded for `η < 0.186`, unbounded for
  `η > 0.64` on the 753-day data. Fixing the total weight `1ᵀd = p` is the suggested repair.
  Identity: `−tr(VᵀBV) = tr(VᵀLV) − Σᵢ dᵢ‖vᵢ − c_d‖²`.
- **Paper 1 Fig 2** did not recover the sectors until Algorithm 1 was started from the
  Problem 1 estimate; from the uniform start the components are set by round-off.

Status of each part and next steps: `reports/2026-10-27_part1_lemma.tex` (Part I) and
`reports/2026-10-18_proposal_issues.tex` (Parts II and III).
