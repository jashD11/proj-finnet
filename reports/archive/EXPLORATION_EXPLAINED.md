# Exploration, explained from scratch

> **Archived 8 September 2026.** Superseded by
> `reports/2026-08-19_discrepancy_explained.tex`, the written-up version of
> the same material. Kept as the ground-up explanation of each object. Note
> that its λ₂(L) + μ₁(B) = d spine is now retracted: that identity needs a
> regular graph, see §4 of
> `reports/2026-09-08_modularity_equivalence.tex`.

**12 August 2026.** Ground-up companion to `EXPLORATION.md` and
`progress_report.tex`. Every object and every number is defined before it is
used. Nothing here is new evidence — it is the same probe results, explained for
someone meeting them for the first time.

---

# Part A — What the modularity matrix actually is

## A1. Start with the network

Each day you have 360 stocks. You compute the correlation between every pair over the last 30 days, then keep the strongest 10% of pairs as "edges" and throw the rest away. That gives you a graph: 360 dots, and a line between two dots if those two stocks moved together strongly enough that day.

You store that as the **adjacency matrix A**: a 360×360 grid of 0s and 1s. `A[i][j] = 1` if there's an edge between stock i and stock j, else 0.

Two quantities fall out immediately:
- **d_i**, the *degree* of stock i — how many edges it has, i.e. the row sum of A.
- **m**, the total number of edges. Here: 10% of the 64,620 possible pairs = **6,462 edges**. So `2m = 12,924` and the average degree is `12,924 / 360 = 35.90`. Every stock connects to about 36 others, every day, by construction.

## A2. The question modularity is trying to answer

Suppose stocks i and j have an edge. Is that *interesting*? Not necessarily. If i is a hub connected to 200 other stocks, it's connected to nearly everyone — an edge to j tells you almost nothing. But if i and j each have only 5 edges and one of them is to each other, that's meaningful.

So you need a baseline: **how many edges would we expect between i and j purely by accident**, given how many edges each of them has?

The standard answer (the "configuration model") is:

```
expected edges between i and j  =  d_i · d_j / 2m
```

The intuition: stock i has d_i edge-endpoints dangling. Each one picks a partner at random from the 2m endpoints in the whole network. d_j of those belong to j. So the chance a given endpoint of i lands on j is d_j/2m, and i has d_i tries.

Sanity check with your real numbers. Two typical stocks, each with degree 36:
```
36 × 36 / 12,924 = 0.100
```
A 10% chance — exactly the density of the network. Correct. Now two hubs, each with degree 100:
```
100 × 100 / 12,924 = 0.774
```
Those two are expected to be connected almost certainly, just from being popular. An edge between them is not news.

## A3. The matrix

The **modularity matrix** is: *what you observed, minus what you'd expect by accident.*

```
B  =  A  −  dd′/2m
```

Entry by entry: `B[i][j] = A[i][j] − d_i·d_j/(2m)`. Same size as A, 360×360. Its entries are surprise scores:

| observed | expected | B entry | reading |
|---|---|---|---|
| edge (1) | 0.10 | **+0.90** | strong surprise — these two are together far more than chance |
| edge (1) | 0.77 | **+0.23** | two hubs connected; barely news |
| no edge (0) | 0.10 | **−0.10** | mild — most pairs look like this |
| no edge (0) | 0.77 | **−0.77** | strong negative surprise — two hubs *avoiding* each other |

Notice the last row. B can go meaningfully negative, and that means something real: a pair that ought to be connected and isn't. The Laplacian has no equivalent of this. **A is 0/1; L is built from A and degrees; neither can express "should be there and isn't."** That asymmetry is the origin of direction D3.

Two structural facts, both used later:
- **Every row of B sums to zero.** The observed edges of node i sum to d_i, and the expected ones sum to `d_i·(Σ_j d_j)/2m = d_i`. They cancel exactly. So B always has a zero eigenvalue with the all-ones vector.
- **B is indefinite** — it always has both positive and negative eigenvalues. (Its diagonal is all negative, so its trace is negative, so it must have negative eigenvalues; and the row-sum-zero property forces positive ones too.) This is the crucial contrast with the Laplacian, which is positive semi-definite: L's eigenvalues are all ≥ 0.

## A4. Where Q comes from

Newman's modularity **Q** is: pick a way to split the stocks into groups, then add up B over all pairs that landed in the *same* group, and divide by 2m.

```
Q = (1/2m) · Σ_{i,j in same group} B[i][j]
```

That's it. Q is a **sum of selected entries of B**. High Q = you found groups whose members are together far more than chance.

Louvain is a search algorithm that tries to find the grouping maximising that sum. It's greedy and randomised, so it gives a different answer each run.

So the chain is:

```
360×360 matrix B  →  Louvain picks one grouping  →  one number Q
    129,600 entries          ~9 groups                1 number
```

Paper 3 reports the last box. **The middle step is stochastic and the last step is a massive compression.** Everything in the exploration follows from asking what's in the first box.

## A5. What "the spectrum" means and why you'd look at it

An **eigenvector** of a matrix is a direction the matrix doesn't rotate — it only stretches or shrinks it. The **eigenvalue** is the stretch factor. A symmetric 360×360 matrix has 360 of these pairs, and together they fully describe the matrix. The list of eigenvalues is the **spectrum**.

For B, order them `μ₁ ≥ μ₂ ≥ … ≥ μ_N`:

- **μ₁, the largest.** Its eigenvector is the single strongest "these stocks over here vs those over there" split in the surprise matrix. Newman's original method just reads off the signs of this eigenvector to split the network in two. So **μ₁ is doing the same job as Q — measuring how strong the best community split is — but computed directly and deterministically, no search, no random seed.**
- **How many μ are positive.** Theory (Fasino & Tudisco, Bolla et al.) says the count of positive eigenvalues upper-bounds the number of separable communities. So `#{μ > 0}` is a **community count that requires no algorithm** — a direct replacement for Louvain's unstable k(t).
- **μ_N, the most negative.** Its eigenvector picks out the strongest *anti*-community structure: groups that avoid each other more than chance. In markets that would be a genuine split — say, defensives vs cyclicals moving apart, or a flight to safety.
- **Σ of the positive μ's** — total "community mass" in the network.

None of these need Louvain. All are deterministic. **None of them appear anywhere in the finance literature.** That is the gap.

---

# Part B — What the probe did, and how to read a score

## B1. The experiment

Paper 3's pipeline already built 6,315 daily networks. The probe walks through them and, for each day, computes the **full spectrum of three matrices**:

| matrix | what it is | what its spectrum says |
|---|---|---|
| **L** = D − A | Laplacian | how connected / close to falling apart |
| **B** = A − dd′/2m | modularity | community and anti-community structure |
| **C** | the raw correlation matrix, *before* thresholding | how much everything moves together |

Three 360×360 eigendecompositions × 6,315 days. About 6 minutes. Result: `spectral_probe.parquet`, one row per day, 15 spectral statistics per row.

## B2. Turning a statistic into a crisis detector

You can't score a raw statistic directly. λ₂ drifts over 25 years; "λ₂ = 3.1" doesn't mean anything on its own. What you want is *unusual relative to recent normal*. So each series is converted to a **trailing 250-day z-score**:

```
z(today) = (value today − average of the last 250 days) / (std dev of the last 250 days)
```

z = 0 means today is completely typical; z = 3 means today is three standard deviations away from the last year's normal. Crucially, **the window uses only days strictly before today** — no peeking at the future. That's what makes it an honest detector rather than a description.

## B3. What AUC means

The crisis windows were **pre-registered in Phase 0**, before any of this — dates of known crashes (Black Monday, LTCM, dot-com, 2008, etc.), fixed in advance so no detector can be tuned to them after the fact. 2.68% of days are "sharp" crisis days — about 169 days out of 6,315.

That tiny base rate is why accuracy is useless as a measure: a detector that says "no crisis" every single day is 97.3% accurate and completely worthless.

**ROC-AUC** avoids that. Rank all 6,315 days by the detector's z-score. Then:

> **AUC = the probability that a randomly picked crisis day ranks above a randomly picked calm day.**

- **0.50** = coin flip. The detector carries no information.
- **1.00** = perfect. Every crisis day outranks every calm day.
- **0.75** = pick one crisis day and one calm day at random; 3 times in 4 the crisis day scores higher.

One generosity to note: for each statistic the probe tried both directions ("crisis = spike" and "crisis = drop") and **kept whichever scored better**. That flatters every number slightly. It's flagged in the report as such.

The whole point of this apparatus is that **every statistic — old and new — goes through the identical detector**. As proof, the probe re-derives Paper 3's own stored Q and τ scores to 10 decimal places (0.5380684370 and 0.8071316467). The new numbers sit on exactly the same ruler as the published ones.

---

# Part C — The results, explained

## C1. Result 1: the matrix beats the scalar

| Signal | What it measures | AUC |
|---|---|---|
| spectral entropy of C | how spread out the correlation structure is | 0.822 |
| λ₁(C)/N — absorption ratio | share of variance in the single biggest common factor | 0.812 |
| τ(t) | the threshold needed to keep 10% of pairs | 0.807 |
| **μ_N(B)** | **strongest anti-community** | **0.760** |
| # zero eigenvalues of L | number of disconnected pieces | 0.756 |
| **Σ positive μ(B)** | **total community mass** | **0.750** |
| λ_max(L) | the most connected node's local density | 0.731 |
| **#{μ > 0}** | **community count, algorithm-free** | **0.722** |
| λ₂(L) on largest component | Fiedler value, fragmentation removed | 0.652 |
| μ₁ − μ₂ | modularity spectral gap | 0.636 |
| λ₂(L) whole graph | Fiedler value — **Paper 1's indicator** | 0.617 |
| μ₁(B) | leading modularity eigenvalue | 0.607 |
| **Q(t)** | **Louvain modularity — Paper 3's indicator** | **0.538** |
| max Laplacian gap | Kang et al.'s indicator | 0.526 |

**How to read Q = 0.538.** Pick a random crash day and a random calm day. Q ranks the crash day higher 53.8% of the time. A coin does it 50% of the time. Paper 3 builds a 25-year narrative on this quantity, and it is very nearly uninformative about crises.

**And it came from a matrix that scores 0.760.** μ_N(B), Σμ⁺, and #{μ>0} are all statistics of the *same matrix B* on the *same networks* under the *same scoring*. So the information about crises was sitting inside B the whole time; the Louvain-then-collapse-to-one-number step is where it got destroyed.

Three supporting details:

- **The strongest modularity signal is the negative end.** μ_N — anti-communities, groups pulling apart — beats every positive-side statistic. That is the exact thing thresholding is structurally incapable of seeing, and nobody in finance has looked at it.
- **μ₁(B) correlates 0.761 with Q in levels.** So it is largely measuring the same thing Louvain is trying to measure — but deterministically. Recall from the audit that Louvain's seed-to-seed jitter on the same day is **2.09× the median day-to-day move**: run it twice on identical data and the disagreement is twice the size of the signal you're studying. μ₁ has none of that. Even as a straight drop-in replacement, that's worth something.
- **Paper 1's λ₂ scores 0.617 here — but that is *not* a refutation of Paper 1.** Paper 1 learns a regularised Laplacian on 7 stocks; this is a thresholded 360-node network. Different object entirely. What the number says is narrower: *on thresholded correlation networks of this kind*, the Fiedler value is weak.

## C2. Result 2: the duality breaks

**The claim being tested.** If every node in a graph has the *same* degree d, there's an exact algebraic identity:

```
λ₂(L)  +  μ₁(B)  =  d
```

Both operators are built on top of the same adjacency matrix, and on a regular graph they both reduce to reading the adjacency spectrum — one from the top, one from the bottom. **They are the same number, backwards.**

Why this matters for your project: if it held on your networks, then "compare Paper 1's indicator with Paper 3's" is a null question. There would be one indicator with two names, and the study would be measuring nothing.

**Why it should nearly hold here.** Silva's fixed-density construction forces the average degree to exactly 35.90 every single day. Not approximately — exactly, by construction. These graphs are as close to regular as any real financial network you'll build.

**What was measured.**

| quantity | value | reading |
|---|---|---|
| average degree d̄ | 35.900 | exact, every day |
| residual `d̄ − (λ₂ + μ₁)` | **9.49 ± 4.63** | should be 0 |
| residual, connected days only | 9.21 ± 3.60 | barely improves |
| residual, largest component only | 11.11 ± 5.31 | gets *worse* |
| **corr(λ₂, μ₁)** | **−0.187** | should be −1 |

Read the residual first. λ₂ + μ₁ should equal 35.90 and instead comes out around 26.4. **The identity misses by 26% of the mean degree.** And the "± 4.63" matters as much as the 9.49: the gap isn't a fixed offset you could subtract away, it *wobbles by a third of its own size* day to day. It's a live quantity.

Now the correlation, which is the cleaner statement. If `λ₂ + μ₁ = d` and d is fixed, then whenever λ₂ goes up, μ₁ must come down by exactly the same amount — they'd be **perfectly anti-correlated, r = −1.0**. Two variables with no relationship at all give r = 0.

Observed: **r = −0.187.** That is far closer to "unrelated" than to "the same number." The two indicators the field has been using are, on real market networks, **essentially independent measurements.**

**Why this is the good news.** It rescues the research question. Paper 1's indicator and Paper 3's indicator are *provably identical* in the idealised case and *empirically unrelated* in practice. So the question stops being the weak "which indicator wins?" and becomes the sharp one:

> **What is that 9.49 made of, and does it carry market-stress information?**

Three candidates: **degree heterogeneity** (real networks aren't regular — some stocks have 5 edges, some have 150, and the identity assumed they were all the same), **fragmentation** (the graph breaks into disconnected pieces on 63% of days, which pins λ₂ to zero and makes it meaningless), and the **market mode** (see below). Establishing which one carries the information is a genuine structural result, not a leaderboard. That's D1.

## C3. Result 3 — the two concerning findings

### Concern 1: there may be nothing there to detect

**The problem.** Each day, you estimate a 360×360 correlation matrix — that's **64,620 pairwise correlations** — from **30 days of data on 360 stocks**, which is 10,800 numbers total. You are asking for six times more output than you have input.

An immediate consequence: a correlation matrix built from 30 observations has **rank at most 29**. Of its 360 eigenvalues, at least 331 are exactly zero, forced by arithmetic and not by anything about markets.

**What random matrix theory adds.** Feed pure noise — 360 completely independent random series, no structure whatsoever — into this same pipeline, and you still get a correlation matrix with non-trivial eigenvalues, purely from estimation error. Marchenko–Pastur tells you *how large*. With N/Δt = 360/30 = 12, the noise band is:

```
[6.07 , 19.93]
```

**Any eigenvalue below 19.93 is indistinguishable from what pure noise produces.** It is not evidence of structure.

**What was found.** For each day, count the correlation eigenvalues that escape above 19.93 — excluding the largest one, because that's the market mode (see below), which is the whole market moving together, not a community.

> **Mean count: 0.70. Maximum: 3. Frequently zero.**

On a typical day, there is **less than one statistically resolvable group** in the data beyond the market itself.

**Why that's alarming.** Louvain, run on the same data, confidently reports a **median of nine communities every day**, and Paper 3 builds a 25-year story about their dynamics. Most of those nine are not detectable structure — they are the algorithm partitioning noise. Louvain *always* returns a partition; it has no way to say "there's nothing here."

This is the rigorous version of what the audit already found by simulation: run the entire pipeline on i.i.d. Gaussian noise and Louvain scores **Q = 0.2246**, against the real market's **Q = 0.2215**. *Structureless data is marginally more modular than the actual stock market.* The simulation showed it happens; MP explains why it must.

And this doesn't just indict Paper 3. Δt = 30 with a few hundred stocks is standard across this entire literature.

### Concern 2: the strong signals are all one signal, and it doesn't generalise

**What the "market mode" is.** On any given day the single biggest pattern in stock returns is that everything moves together — the whole market up or the whole market down. In the correlation matrix C this shows up as one dominant eigenvalue, λ₁(C), vastly larger than the rest. In a crash, correlations across the board go to ~1, so λ₁ balloons. That's the whole mechanism behind "diversification fails exactly when you need it."

This is old, well-established finance: the **absorption ratio** (Kritzman et al. 2011) is exactly λ₁(C)/N — the share of total variance in the leading factor. (Since a correlation matrix has trace = N, dividing by N makes it a clean percentage.) The RMT precursors go back to Laloux et al. and Plerou et al., 1999.

**The problem.** Look at the top of the leaderboard again: spectral entropy 0.822, absorption ratio 0.812, τ(t) 0.807. Those three are **pairwise correlated above 0.99**. A correlation of 0.99 means they are the *same series* with different labels. Three of them are one signal wearing three hats — and the isolated-node count (0.756) correlates 0.92 with τ, so really it's four.

So the "best" rows in that table are not four discoveries. **They are one discovery, and it was made in 1999.**

**The harder problem: out-of-sample.** The real question is whether the modularity spectrum adds anything *on top of* the market mode. To test it honestly: fit a detector on the early period (pre-2002) and evaluate it on data it has never seen (post-2002, 27 crisis days).

| model | in-sample | out-of-sample |
|---|---|---|
| market mode alone | 0.812 | **0.730** |
| + μ₁(B) | 0.822 | 0.629 |
| + μ_N(B) | 0.810 | 0.719 |
| + Q(t) | 0.831 | 0.699 |
| + λ₂(L) on LCC | 0.810 | 0.700 |

**Every single addition improves in-sample and hurts out-of-sample.** That is the textbook fingerprint of overfitting: the extra variable isn't learning a market mechanism, it's memorising the idiosyncrasies of 1987–2002 and then misfiring on 2002–2011.

Note the ranking flips completely, too. Adding Q looks like the second-best idea in-sample (0.831) and is one of the worst out-of-sample (0.699).

**The mitigating detail, and why it isn't a get-out.** 27 crisis events in the test set is a very small number — AUC on 27 events is noisy enough that none of this is statistically conclusive in *either* direction. But "our evidence is too weak to settle it" is not a defence; it's the same statement. Right now there is **no evidence the modularity spectrum adds anything to a 1999 baseline out of sample.**

**And this reaches into D3.** μ_N(B) scored 0.760 — the best new number in the whole probe. But μ_N correlates **0.68 with the market mode**. So a substantial chunk of that 0.760 is borrowed from a signal that already existed. Whether anything is left after you project the market mode out is exactly the gate that decides whether D3 is a paper or a footnote.

*(One genuine bright spot: μ₁(B) correlates only **0.025** with the market mode — essentially orthogonal — while correlating 0.761 with Q. So it really is a new axis measuring what Louvain measures. It just didn't convert that independence into out-of-sample gain, which is the whole tension.)*

## C4. So why do this at all?

Because both concerns are, read the other way, the contribution:

- "0.70 detectable modes per day while the method reports nine" isn't only a problem with your data — it's a **general result about the parameter settings this whole literature uses**, and nobody has stated it. That's D2, and it's the kind of finding that gets cited.
- "the incumbent indicators are duals in theory and independent in practice, and here's the decomposition of the gap" is a **structural claim**, checkable and surprising, and it doesn't depend on winning a horse race. That's D1.
- The out-of-sample failure tells you the *venue*: a physics/RMT audience wants the spectral and detectability result and will accept it; a finance audience wants you to beat the absorption ratio out of sample, and on current evidence you don't. That's Q4 in the report, and it's the decision that shapes everything downstream.

The thing that would actually be fatal is discovering all this in month six. It's month zero.
