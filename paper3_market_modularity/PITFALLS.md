# Methodological audit — Silva et al. (2015)

*Modular Dynamics of Financial Market Networks*, arXiv:1501.05040v3 · `docs/1501.05040v3.pdf`

26 pitfalls identified in a close read of the paper, recorded before implementation began.

**Labels.** **[EASY]** = a mechanical check or correction; the answer arrives within an hour of coding. **[THINK]** = a design question with no obvious right answer, where the fix changes what the paper is measuring.

**Status column.** `pending` → not yet reached · `addressed` → the phase ran and the result is recorded here · `UNTESTED` → deliberately out of scope, recorded so it is not silently dropped.

---

## Summary table

| # | Pitfall | Label | Phase | Status |
|---|---|---|---|---|
| A1 | Survivorship bias (348 of 3,799) | THINK | 0 | **addressed** |
| A2 | Adjusted vs. raw closes unstated | EASY | 0–1 | **addressed** (Phase 1 confirms) |
| A3 | Missing-data / trading-halt policy unstated | EASY | 0–1 | **addressed** |
| A4 | N/T ≈ 11.6 — degenerate correlation matrix | THINK | 2 | pending |
| A5 | Market mode never removed | THINK | — | **UNTESTED** |
| A6 | Δt = 30 never justified or robustness-tested | EASY | — | **UNTESTED** |
| A7 | Fixed density deletes the primary crisis signal | THINK | 2, 8a | pending |
| A8 | Negative correlations structurally invisible | THINK | 2 | pending |
| A9 | Binarization discards edge weights | THINK | 2 | pending |
| A10 | Window timestamp convention unstated | EASY* | 2 | **largely solved in Phase 0 — see below** |
| B1 | Louvain degeneracy, no seed, no stability check | EASY | 3, 8f | pending |
| B2 | Independent daily runs mix jitter with real change | THINK | 8f | pending |
| B3 | Resolution limit | THINK | 3 | pending |
| B4 | Number of communities *k* never reported | EASY | 3 | pending |
| B5 | Isolated nodes each become their own community | THINK | 3, 6 | pending |
| B6 | Modularity noise floor never drawn | EASY | 4 | pending |
| C1 | Does modularity rise or fall in a crisis? | THINK | 8c | pending |
| C2 | Crisis periods ill-defined | THINK | 0 | **addressed** |
| D1 | Compression claim unverified | EASY | 6 | pending |
| D2 | Coin-flip generation makes edge counts random | EASY | 4 | pending |
| D3 | One realization per day, no error bars | EASY | 4–5 | pending |
| D4 | Rich-club reported raw, not normalized | EASY | 5 | pending |
| E1 | **Window overlap inflates every ρ — the big one** | EASY/THINK | 6 | pending |
| E2 | ⟨χ²⟩ unnormalized, not comparable across measures | EASY | 6 | pending |
| E3 | ⟨χ²⟩ is not a chi-squared statistic | EASY | 6 | pending |
| E4 | No significance testing at any point | EASY | 6 | pending |
| E5 | The eight measures are not eight independent facts | EASY | 6 | pending |
| E6 | Disconnection handling unstated | EASY | 5 | pending |
| E7 | PCA1 asserted to be time, not shown | EASY | 7 | pending |
| E8 | Fig. 7 shows separation, described as agreement | THINK | 7 | pending |
| E9 | The 2002 split was chosen from the data | THINK | 6 | pending |
| E10 | "Crisis detection evaluation" never performed | THINK | 8b | pending |
| E11 | No mechanism proposed for the 2002 break | THINK | — | **UNTESTED** |

\* A10 is labelled EASY but critical: the ambiguity is 30 days, the same size as the effects being claimed.

---

## A. Data and network construction

### A1. Survivorship bias. [THINK]
348 of 3,799 stocks survive the "complete 1986–2011 history" filter. Lehman, Bear Stearns, Enron, WorldCom, every dot-com — all absent. A paper about crisis structure has systematically excluded the firms that failed in the crises. Unmentioned. You cannot fix this within their design; you can only quantify it by building a point-in-time universe and comparing.

**Phase 0 result.** Our funnel, against the paper's 3,799 → 348:

| Stage | Count |
|---|---|
| NYSE common-stock symbols requested | 1,575 |
| Yahoo returned no data at all | −590 |
| Not trading at both ends of the window | −607 |
| Coverage below 100 % of the calendar | −0 |
| Non-positive adjusted price (`VHI`) | −1 |
| **Surviving universe N** | **377** |

**1,198 of 1,575 requested symbols were removed before a single network existed** — a 76 % attrition rate against the paper's 91 %.

Our replication carries a **second layer** of the same bias that the paper does not: the NASDAQ Trader directory lists currently-listed symbols only, so firms delisted before the retrieval date never enter the candidate pool. That is why our pool is ~4× smaller than the paper's while our final N (377) is *larger* than theirs (348) — we start from a list that has already been filtered by survival once. Both layers are documented; neither is corrected.

### A2. Adjusted vs. raw closing prices, unstated. [EASY]
If raw closes were used, every stock split is a fake −69% log return that poisons 30 consecutive networks. Use adjusted close; assert no |Y| > 0.5 before proceeding.

*Our handling:* `yfinance(auto_adjust=True)`. Phase 1's `|Y| > 0.5` detector is the proof, not the intention.

**Phase 0 result — the modern data source has the opposite defect.** `yfinance` back-adjusts dividends by *subtraction*, so a stock whose cumulative historical distributions exceed its old price is returned with a **negative** adjusted close. `VHI` (Valhi Inc.) came back at a minimum of **−639.30**, negative on 3,964 of 6,345 days. `log()` is undefined there and the entire series is unusable. It is dropped and named in `UNIVERSE.md`. Raw closes would have hidden this behind plausible-looking positive numbers; a positivity assertion is now a permanent gate in `build_panel`.

### A3. Missing-data and trading-halt policy, unstated. [EASY]
Halts are frequent precisely during crises. Forward-filling produces zero returns, which deflates that stock's correlations. Document your policy, and test that results don't flip under the alternative.

*Our handling:* both policies implemented in `data/fetch_data.build_panel` (`drop_days` default, `ffill` alternative), and a unit test pins that they disagree by exactly one trading day per gap.

**Phase 0 result: the question is moot for our panel.** Under the strict reading of "complete history" (`MIN_COVERAGE = 1.0`), all 378 full-span survivors trade on *every* day of the 6,345-day calendar — the coverage ladder is flat at 378 from 100 % down to 95 %. **Zero days are dropped, so `drop_days` and `ffill` produce an identical panel here.** The policy still matters for any variant universe with a looser span or coverage rule, which is why both remain implemented and tested.

### A4. N/T ≈ 11.6 — the correlation matrix is degenerate. [THINK]
30 observations, 348 stocks. Per-entry noise standard deviation ≈ 0.19. Pure independent noise thresholded at top-10% yields ρ ≈ 0.24 and a network that looks exactly as full as the real one. The paper applies no random-matrix cleaning and never raises the issue. Fixing it (eigenvalue clipping, shrinkage) changes the object being studied.

*Our handling:* not corrected — correcting it would replicate a different paper. Quantified instead: Phase 2 runs the same pipeline on i.i.d. Gaussian returns of identical shape and reports the resulting τ and network statistics as a noise reference.

### A5. The market mode is never removed. [THINK]
Under ρ_ij ≈ β_i β_j σ²_M / (σ_i σ_j), the top 10% of correlations largely selects high-β_i β_j pairs. So the network is substantially a map of market beta, not of economic relatedness. Sector structure is only the second-order effect. Standard practice in this literature is to regress out the market factor first. Cheap to test as a variant, but it produces a genuinely different network.

**UNTESTED — out of scope.** Requires a full Phase 2–6 rerun on residual returns.

### A6. Δt = 30 is asserted, never justified or robustness-tested. [EASY]
They tested robustness to *f* (Fig. S1) but not to the more dangerous parameter. Rerun at 20/60/120.

**UNTESTED — out of scope.** Requires three additional full Phase 2–6 runs.

### A7. Fixed density deletes the primary crisis signal. [THINK]
"Correlations spike toward 1 in a crash" is the most replicated fact in this literature, and it's removed at step three. The choice is defensible — it is what makes Section III's comparison legitimate — but it means the paper's crisis signal is a weak residual. Partial fix is trivial (save τ(t)); deciding what the right construction is, is not.

*Our handling:* τ(t) stored for every day in Phase 2 and made a first-class result in Phase 8a, where it competes against modularity as a crisis indicator.

### A8. Negative correlations are structurally invisible. [THINK]
τ always lands around 0.4–0.6, so no anticorrelated pair ever becomes an edge. Two stocks that reliably move oppositely — a strong, tradeable relationship — are recorded identically to two unrelated stocks.

*Our handling:* not corrected. Quantified: Phase 2 reports the count and time series of pairs with ρ < −τ, i.e. the relationships the construction is blind to.

### A9. Binarization discards edge weights. [THINK]
ρ = 0.99 and ρ = 0.51 become the same edge. Justified by convenience (cliques, Louvain, rich-club are simpler on binary graphs), but it is real information loss.

*Our handling:* not corrected; the mean and spread of retained edge weights are reported per day so the magnitude of the discard is visible.

### A10. Window timestamp convention unstated. [EASY, but critical]
Is a network plotted at t₁, t₂, or the midpoint? The ambiguity is 30 days — the same size as the effects being claimed. Pick one, state it, and check your Black Monday alignment against Fig. 3.

**Largely solved in Phase 0.** Fig. 8 samples the sequence "every 1000 time steps" and prints five dates — the only hard link the paper gives between window index and calendar. On our panel calendar they land at 1-based day indices 1355, 2355, 3355, 4355, 5355 (`results/tables/date_calibration.csv`):

| Paper window *n* | Fig. 8 date | our day | offset |
|---|---|---|---|
| 1000 | 1991-05-10 | 1355 | 355 |
| 2000 | 1995-04-25 | 2355 | 355 |
| 3000 | 1999-04-12 | 3355 | 355 |
| 4000 | 2003-04-03 | 4355 | 355 |
| 5000 | 2007-03-26 | 5355 | 355 |

**The offset is exactly 355 at all five anchors, with zero drift across 16 years.** That is stronger than expected: the paper's trading calendar and the modern Yahoo one agree day-for-day from 1991 to 2007. Since the paper is 337 days short overall (6,008 vs our 6,345), **essentially all of that shortfall must fall before May 1991** — either their panel effectively begins around 1987 despite the stated January 1986 start, or their 1986–91 data is heavily gapped. Fig. 8 cannot separate those, but it does rule out the gap being spread across the sample. Note the implication: **the paper's stated date range and its own `C_p` = 6,008 are not mutually consistent on any modern NYSE calendar.**

**Correction to the pre-registered choice.** `PLAN.md` adopted **t₂** a priori, reasoning that a network is only knowable at window close. The anchors do not support that:

| Convention | predicted offset | residual vs. observed 355 |
|---|---|---|
| t₁ (window start) | 337 | +18 days |
| **midpoint** | **352** | **+3 days** |
| t₂ (window end) | 367 | −12 days |

**Midpoint fits four times better than either alternative.** The convention is therefore held provisional rather than settled: Phase 2 checks Black Monday alignment against Fig. 3 as an independent tiebreaker, and whichever wins, both the choice and its residual go in the report. A 3-day residual is within the noise of "≈337 missing days somewhere before 1991"; 12–18 days is not.

*(The working hypothesis previously recorded here — that our own missing-day filter would reproduce the 337-day gap, concentrated in 1986–87 — was **falsified**. Our panel drops zero days. The shortfall is a property of the paper's data, not of the completeness rule.)*

---

## B. Community detection

### B1. Louvain degeneracy, no seed, no stability check. [EASY]
Modularity's landscape has exponentially many near-optimal partitions with very different structure. Louvain also depends on node ordering. Fix: 10 seeds per day, report the spread in Q and partition agreement.

### B2. Independent daily runs mix algorithmic jitter with real change. [THINK]
No temporal coupling between consecutive days, so you cannot separate "the market changed" from "Louvain landed elsewhere." Multilayer/temporal community detection exists for exactly this — and the paper cites Mucha, Porter, and Fenn without using their method.

*Our handling:* not corrected (we replicate the independent-daily design), but Phase 8f measures the damage directly: seed-to-seed NMI versus day-to-day NMI. If they are comparable, the daily modularity series is substantially algorithmic noise, and that finding leads the report.

### B3. Resolution limit. [THINK]
Modularity maximization cannot resolve communities smaller than ~√(2m) ≈ 110 edges here. Genuine small sectors get absorbed. This mechanically bounds *k*, which determines the size of Π, which determines whether the compression claim holds.

### B4. The number of communities *k* is never reported. [EASY]
Eq. 2's behaviour and Section IV's central claim both depend on it entirely. We store and report k[t] for every day.

### B5. Isolated nodes each become their own community. [THINK]
The paper states components are always assigned distinct communities. During Black Monday the graph shatters into a hub plus hundreds of isolates, so *k* explodes and Π becomes huge and nearly empty — on exactly the days the paper cares most about. This may invert the compression argument where the story lives. Tested jointly with D1 in Phase 6.

### B6. The modularity noise floor is never drawn. [EASY]
A structureless graph of this size and density scores Q ≈ 0.1 — visible as the green line in Fig. 6(a1). Normal-times values of 0.20–0.40 are only modestly above it, and the crisis value sits at it. We draw the configuration-model floor explicitly on every modularity figure from Phase 4 onward.

---

## C. The internal contradiction

### C1. Does modularity rise or fall in a crisis? [THINK]
Abstract and Fig. 3 say structure is destroyed (Q falls). The Fig. 4 text says "we observe an increase of modularity around the time span of the crisis." Never reconciled. Best reconstruction: crash day is a trough, surrounding bear-market period is elevated. Resolvable empirically with event-window analysis — arguably the most interesting thing to settle. Phase 8c.

### C2. Crisis periods are ill-defined. [THINK] — **addressed in Phase 0**
Dates come from newspaper articles, and the paper admits "the duration of a crisis is not well-defined in the literature." With 17 fuzzy shaded bands over 25 years, a substantial fraction of the timeline is "crisis," so visual alignment proves little. Any real detection test needs precise, pre-registered event windows.

*Result:* `utils/crisis_dates.py` fixes all 17 windows **before any network exists**, split into 9 `sharp` events (dated, under ~4 weeks) and 8 `diffuse` regimes. Measured coverage over 6,303 business days, 1987–2011:

| Band set | Share of timeline |
|---|---|
| any crisis band | **32.1 %** |
| sharp events only | 2.8 % |
| diffuse regimes only | 29.8 % |

So under the paper's own event list, **roughly one trading day in three is inside a crisis band**. Any detector must be scored against that base rate, and the 2.8 % sharp subset is the only genuinely discriminating test. This is the quantified version of C2.

---

## D. The null model

### D1. The compression claim is unverified. [EASY]
Configuration model stores 348 numbers/day. Community model stores k + k(k+1)/2. At k = 26 they are equal; above that the "cheaper" claim inverts. Given B5, this likely fails during crises. Phase 6 reports the fraction of days on which the community summary is genuinely cheaper, overall and within crisis windows.

### D2. Coin-flip generation makes edge counts random, not exact. [EASY]
Π_αβ is matched only in expectation. Fine for large blocks, sloppy for small ones — and after a crisis shatter, most blocks are small. We keep the paper's coin-flip generation and report realized-vs-target deviation per block rather than silently fixing it.

### D3. One realization per day, no error bars anywhere. [EASY]
Both Louvain and the generative step are stochastic. Replicate spread is free. Clique number especially — a maximum over a random graph is a notoriously high-variance quantity. We run 10 replicates per day per model.

### D4. Rich-club is reported raw. [EASY]
φ(k) rises with k even in random graphs, which is why it is conventionally normalized against a null. Not done here. We store both raw and configuration-normalized φ.

---

## E. Statistics and evidence

### E1. Window overlap inflates every ρ in the paper. This is the big one. [EASY to check, THINK to interpret]
Consecutive networks share 29/30 of their data. Each series has ~200 effectively independent points, not 5,978. Two heavily smoothed series sharing any slow trend correlate highly almost regardless of whether one describes the other. This is the primary evidence for the primary claim, and it goes unadjusted and unmentioned. Recompute on non-overlapping windows — ten lines of code, and if 0.99 falls to 0.7 the headline needs rewriting.

*This is the single highest-value result in the replication.* Phase 6 reports overlapping and non-overlapping ρ side by side for every measure, with block-bootstrap confidence intervals.

### E2. ⟨χ²⟩ is unnormalized. [EASY]
Betweenness reports 419 and 2,522; transitivity reports 0.0188. Different units, different scales, not comparable — yet the tables invite exactly that comparison. Divide by the variance of the real series.

### E3. ⟨χ²⟩ is not a chi-squared statistic. [EASY]
No expected-value denominator, no degrees of freedom, no p-value. It is mean squared error with misleading notation. We report it under its real name.

### E4. No significance testing at any point. [EASY]
No confidence intervals, no bootstrap, no null distribution for ρ itself. The argument is "0.99 > 0.07, therefore." Phase 6 adds block-bootstrap CIs with block length ≥ Δt.

### E5. The eight measures are not eight independent facts. [EASY]
Path length, betweenness, transitivity, and matching index are mutually entangled and all coupled to modularity. Modularity is a guaranteed pass; transitivity/clique/rich-club are guaranteed failures. Realistically ~3 informative measures dressed as 8. Compute the inter-measure correlation matrix and you have quantified it.

### E6. Disconnection handling unstated for path length and betweenness. [EASY]
d_ij = ∞ for disconnected pairs, and Black Monday — the showcase event — is exactly when the graph shatters. Largest-component-only vs. harmonic-mean efficiency give visibly different crisis curves. Both stored in Phase 5.

### E7. PCA1 is asserted to be time, not shown. [EASY]
No loadings, no explained-variance ratios. "Ignore the dominant component" backed by one sentence. Correlating PCA1 scores against the date is a two-line check they never ran.

### E8. Fig. 7 shows separation and is described as agreement. [THINK]
The three clouds occupy visibly distinct territories. It is the softest evidence in the paper and it is positioned as the summary of everything. Phase 7 replaces eyeballing with centroid distances and a classifier's separation accuracy.

### E9. The 2002 split was chosen from the data, then used to report results. [THINK]
Not automatically wrong for exploratory work, but the post-2002 numbers are not independent evidence that 2002 is special — the period was defined to make it so.

### E10. "Crisis detection evaluation" (Fig. 1, step e) is never performed. [THINK]
There is a labelled box in the methodology flowchart for it, and the paper contains no detection rule, no threshold, no hit rate, no false-alarm rate. The entire crisis analysis is eyeballing wiggles against shaded bands. This is the largest gap and the most interesting thing to build. Phase 8b builds it, scored against the Phase 0 pre-registered windows.

### E11. No mechanism proposed for the 2002 break. [THINK]
Honest admission, but it means the central claim really only holds for the first 15 of 25 years. Untested candidates: decimalization (2001), ETF/index-fund growth, algorithmic trading.

**UNTESTED — out of scope.** Testing any of these needs data the replication does not gather.
