# The October 2026 proposal, explained from scratch

A plain-language record of Prof. Anguluri's proposal *Modularity Matrix Approach for Graph
Learning?* (received 16 October 2026), the issues found when checking it against this
repository, how they were fixed, and what has been done so far. Written 18 October 2026.

Every number here is printed by `report_numbers.py` in this folder. The formal version for
the professor is `reports/2026-10-18_proposal_issues.tex`.

---

## 1. Background

**What Paper 1 does.** Cardoso and Palomar (arXiv:2005.09958) take daily stock returns and
*learn a network* from them. Each stock is a dot; a line between two stocks means they move
together in a way the other stocks do not explain; a thicker line means a stronger link. A
stock's total line strength is its **degree**.

**The paper's Fig 2.** About 100 stocks from three sectors (Industrials, Consumer Staples,
Energy). The paper's Algorithm 1 is asked to split the network into exactly 3 separate pieces.
If it works, the pieces are the sectors. It also forces every stock's degree to be 1, and
argues that without this rule a competing method (SGL) leaves some stocks with no lines at all.

**The proposal.** Network science has a different tool, **modularity**, which judges a
grouping against a random network with the same degrees. The proposal asks what that tool adds
to Laplacian graph learning, in three parts:

- **Part I:** when every degree is 1, the paper's method and modularity are the same thing.
- **Part II:** the paper's handling of "the whole market moving together" is an implicit
  choice of null model.
- **Part III:** a new method using modularity in place of the "every degree equals 1" rule.

**Current focus: Part I only** (decision of 18 October 2026).

---

## 2. Part I of the proposal, step by step

### 2.1 The two matrices

Let `W` be the table of line strengths (`W_ij` = strength between stocks i and j), and `d` the
degrees.

- **Laplacian** `L = D − W` (the paper's object): degrees on the diagonal, minus the line
  strengths elsewhere.
- **Modularity matrix** `B = W − d dᵀ/2m` (network science's object). The term `d dᵀ/2m` is the
  line strength you would *expect* between two stocks if the network were randomly rewired
  keeping every degree. So `B` is "what we see minus what chance gives".

Both are read through their **eigenvalues**, which act as scores for possible ways of splitting
the network:

- For `L`, a *small* eigenvalue means a cheap split (few lines cut). Zero means a perfectly
  clean split, so the number of zeros equals the number of separate pieces.
- For `B`, a *large* eigenvalue means a good split (more links inside groups than chance).

They look at the same question from opposite ends. Part I asks when they agree exactly.

### 2.2 The key observation

The paper's Fig 2 problem forces **every degree to equal 1**. Then:

1. `D` is the identity, so `L = I − W`.
2. The total degree `2m` equals the number of stocks `p`.
3. The chance term `d dᵀ/2m` becomes `1/p` for every pair: a perfectly flat baseline.
4. Therefore **`B = W − (1/p)11ᵀ = I − L − (1/p)11ᵀ`**.

So `L` and `B` share the same split directions, and every eigenvalue `λ` of `L` becomes
`1 − λ` for `B`. The one exception is the "everyone together" direction (all-ones vector),
which scores 0 in both. Small for `L` becomes large for `B`: "cheap cut" and "good modularity
split" become the same thing.

*Example.* Four stocks in two separate pairs, each line strength 1 (every degree 1).
`L` has eigenvalues 0, 0, 2, 2 (two zeros, two pieces). `B` has 0 (everyone together), then
1 − 0 = 1, then 1 − 2 = −1 twice.

### 2.3 The three claims

**(a) Counting pieces.** `L` has exactly k pieces if and only if `B` has eigenvalue 1 repeated
exactly k − 1 times. (One of `L`'s k zeros is the everyone-together direction, which `B` sends
to 0, not 1.) The paper's "exactly k pieces" requirement is secretly a modularity requirement.

**(b) Algorithm 1 is secretly maximising modularity.** Algorithm 1 enforces k pieces by
penalising the sum of `L`'s k smallest eigenvalues. Under the degree rule,

> sum of L's k smallest eigenvalues = (k − 1) − sum of B's k − 1 largest eigenvalues,

**provided `λ_k(L) ≤ 1`**. The condition is needed because `B` always carries a 0 eigenvalue
(everyone together); if some `1 − λ` drops below 0, that 0 sneaks into "B's top k − 1" and the
bookkeeping breaks. It holds automatically once the algorithm succeeds (the k smallest are 0).

*Counterexample.* Four stocks all connected with strength ⅓ (degree 1 each). `L` has 0 and
then 4/3 three times; 4/3 > 1, so the condition fails. For k = 2 the left side is 4/3, but
`B`'s top eigenvalue is the 0, so the right side is 1 − 0 = 1. They differ.

**(c) The crash indicator.** The paper's trading signal is `λ₂(L)`. Under the degree rule,
`B`'s largest eigenvalue is `max{0, 1 − λ₂}`, so a modularity indicator would carry the same
information.

**The proposal's conclusion.** Under the paper's degree rule, modularity adds nothing new (a
negative but useful result). The match also holds for networks with all degrees equal, and for
degree-normalised versions of both matrices, and fails in general.

**The tasks set:** state and prove (a) to (c) with the role of `λ_k ≤ 1` and a failing example;
reproduce the paper's Fig 2 on S&P 500 sector data; compare its pieces with k-means on `B`'s
top k − 1 eigenvectors.

---

## 3. Issues found, and how each was fixed

### 3.1 Our Fig 2 had never really worked (fixed)

The repo called Fig 2 "reproduced" because Algorithm 1 returned 3 pieces. Nobody had checked
the contents: groups of 77, 12 and 8 stocks, each a random mix of sectors. Agreement with the
true sectors, measured by the **adjusted Rand index (ARI)** (1 = perfect, 0 = chance), was
**0.012**. Grouping stocks straight from their correlations scores 0.900.

This blocked Part I: its numerical check must run on a network whose pieces mean something.

**Cause A, wrong data window.** We used 4 years (1,005 days); the paper uses 3 (Jan 2016 to Jan
2019, 753 days). *Fix:* trimmed the existing file to the paper's window (same 97 stocks, no
re-download) and set the download script's end date to match. FAAMUNG data untouched.

**Cause B, a bad starting point (the main culprit).** Algorithm 1 works in rounds, each choosing
promising split directions from the current network. It started from a network where every
stock is linked to every other with equal strength. There, all splits tie, so the computer's
choice in round one is decided by rounding error. Proof: changing only the number of CPU
threads changed the answer.

| Start | Data | CPU threads | Pieces | ARI vs sectors |
|---|---|---|---|---|
| old (all equal) | old 4-year | default | 77 / 12 / 8 | 0.012 |
| old (all equal) | 753 days | 1 | 96 + one isolated stock | −0.009 |
| old (all equal) | 753 days | default | 93 / 2 / 2 | −0.020 |
| **new (Problem 1)** | 753 days | 1 | **52 / 28 / 17** | **1.000** |
| **new (Problem 1)** | 753 days | default | **52 / 28 / 17** | **1.000** |

*Fix:* Algorithm 1 now starts from the paper's basic learned network (Problem 1, β = 0), which
already contains the sector structure. The paper's Algorithm 1 takes an initial estimate, so
this is faithful. Result: the three sectors exactly, the two thread settings agree to within
7.7e-5, no isolated stocks. (`paper1_gmrf_laplacian/solver/algorithm1.py`, deviation D6.)

### 3.2 Our "SGL" baseline was not SGL (relabelled)

`sgl_benchmark.py` is Problem 1 with a sparsity penalty, without SGL's k-piece machinery. So
Fig 2 never tested the paper's claim that SGL leaves stocks isolated. *Fix:* labelled honestly
everywhere (code, figure, notebook, notes); a real SGL is deferred to Part III. (D5.)

### 3.3 A false claim that our solver cannot isolate a stock (corrected)

The docs said isolation was mathematically impossible. One of the runs above produced an
isolated stock: the solver can zero all of a stock's lines in one step, after which nothing
pulls them back. Corrected in `CLAUDE.md`, `ENGINEERING_NOTES.md`, `PIPELINE.md` and the
notebook.

### 3.4 Fig 1 is weaker than claimed (corrected)

The paper's preferred preparation (correlation, market not removed) wins, but by one line in
200: 199 of its 200 strongest lines stay within a sector, against 198 for the runner-up. Docs
now say "replicates narrowly". Our panel (c) is the paper's (d). (D7.)

### 3.5 Proposal Part I(c) applies to the wrong graphs (reported to the prof)

The degree rule exists only in the paper's Fig 2 problem. The trading experiment (Problem 11,
where `λ₂` comes from) has no degree rule; checked against the paper's formula. Our replicated
trading graphs have **average degree 2.13** (range 1.51 to 2.69) and degrees varying by about
**30%** (coefficient of variation 0.305). The September report showed `L` and `B` do not share
split directions there. Also, even on degree-1 graphs, `max{0, 1 − λ₂}` ignores every value
above 1, which is **58%** of our days (140 of 243).

What survives: *if* degrees were 1, the paper's rule "stay invested while λ₂ < 1" would be
exactly modularity's standard test (`λ_max(B) > 0`) for whether a network splits at all.
Suggestion to the prof: state (c) for degree-1 networks only, keep this as a remark.

### 3.6 Proposal Part III has no answer as written (reported to the prof)

Part III minimises `tr(LS) − log gdet(L) − η·tr(VᵀBV)`. If every line is made c times
stronger, the modularity reward grows in proportion to c while the resisting cost grows only
like log c. Once η is large enough, stronger lines always lower the score, so the solver would
strengthen them forever (the objective is unbounded below).

On our data: **bounded for η < 0.186** (proved: `Z_min/(k − 1)` with `Z_min = 0.373`);
**unbounded for η > 0.64** (explicit example: all weight on the three most related disjoint
pairs, RSG/WM, DAL/UAL, LMT/NOC; still true when the graph is kept connected). The paper does
not report its η; our replication needed 300.

Suggested repair: require only that the **total** degree equals the number of stocks
(`1ᵀd = p`, a fixed budget of line strength). Individual degrees may differ, the score cannot
run away, and the problem stays convex.

Identity found along the way: `−tr(VᵀBV) = tr(VᵀLV) − Σᵢ dᵢ‖vᵢ − c_d‖²`. The modularity term is
the paper's existing penalty minus a reward for giving degree to stocks far from their group's
centre. An isolated stock gets none of the reward, so the term discourages isolation.

### 3.7 A bug in `run_tests.sh` (fixed)

It ran `"../$PY"` inside each folder, so it broke when given an interpreter path, and also on
its own `python3` fallback. It now resolves the interpreter once, up front.

---

## 4. What has been carried out for Part I

1. **Maths checked by hand.** (a), (b), (c) are correct as algebra; the counterexample for (b)
   is the all-connected degree-1 network.
2. **Automated tests** (`tests/test_lemma.py`, 20 tests, all passing, run by `run_tests.sh`):
   `B = I − L − (1/p)11ᵀ` and the 1 − λ spectrum on random degree-1 networks; (a) for 1 to 4
   pieces; (b) when `λ_k ≤ 1`; the counterexample; (c) with λ₂ both below and above 1; and the
   normalised identity `B_n = I − L_sym − uuᵀ` for any network, which reduces to Part I when every
   degree is 1 (so Part I is a special case of the 8 September report's result).
3. **Fig 2 made meaningful** (Section 3.1): the prerequisite for the numerical task.
4. **Early numerical check, on the old Fig 2 network** (before repair): degrees 0.997 to 1.007;
   `‖B − (I − L − 11ᵀ/p)‖ = 0.044`; `B` had eigenvalue ≈1.003 twice (= k − 1); k-means on `B`'s top
   2 eigenvectors reproduced the 3 pieces exactly. The maths showed up in real output, but those
   pieces were not sectors, so this must be redone on the repaired network.
5. **Scope of (c) corrected** (Section 3.5) and reported in the note to the prof.

## 5. What Part I still needs

1. **Rerun the numerical check on the repaired Fig 2 network** (`part1_checks.py`): match of `B`
   to `I − L − 11ᵀ/p`, eigenvalue 1 twice, k-means on `B`'s top 2 eigenvectors versus the three
   sector pieces.
2. **Show the match breaking as degrees spread**: loosen the degree penalty step by step and
   measure the mismatch, making "fails for general degrees" visible.
3. **Write the Part I note** (`reports/YYYY-MM-DD_part1_lemma.tex`): lemma and proofs, the
   counterexample, the link to the 8 September result, the corrected scope of (c), the
   numerical section, and one figure of `L`'s eigenvalues against 1 minus `B`'s.

These wait on the professor's reply on (c), which may change how the note is framed.
