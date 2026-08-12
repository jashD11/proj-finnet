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
| A4 | N/T ≈ 11.6 — degenerate correlation matrix | THINK | 2 | **addressed** |
| A5 | Market mode never removed | THINK | — | **UNTESTED** |
| A6 | Δt = 30 never justified or robustness-tested | EASY | — | **UNTESTED** |
| A7 | Fixed density deletes the primary crisis signal | THINK | 2, 8a | **addressed** (8a pending) |
| A8 | Negative correlations structurally invisible | THINK | 2 | **addressed** |
| A9 | Binarization discards edge weights | THINK | 2 | **addressed** |
| A10 | Window timestamp convention unstated | EASY* | 2 | **RESOLVED — t₂** |
| B1 | Louvain degeneracy, no seed, no stability check | EASY | 3, 8f | **addressed (P3)** |
| B2 | Independent daily runs mix jitter with real change | THINK | 8f | pending |
| B3 | Resolution limit | THINK | 3 | **addressed (P3)** |
| B4 | Number of communities *k* never reported | EASY | 3 | **addressed (P3)** |
| B5 | Isolated nodes each become their own community | THINK | 3, 6 | **confirmed (P3)**, compression test P6 |
| B6 | Modularity noise floor never drawn | EASY | 4 | **addressed (P4)** |
| B7 | *(new, P3)* t_Δ = 100 samples the plateau, not the decay | — | 3 | **new finding** |
| C1 | Does modularity rise or fall in a crisis? | THINK | 8c | **settled (P8)** |
| C2 | Crisis periods ill-defined | THINK | 0 | **addressed** |
| D1 | Compression claim unverified | EASY | 6 | **inverts in crises (P6)** |
| D2 | Coin-flip generation makes edge counts random | EASY | 4 | **addressed (P4)** |
| D3 | One realization per day, no error bars | EASY | 4–5 | **addressed (P4)**, P5 pending |
| D4 | Rich-club reported raw, not normalized | EASY | 5 | **addressed (P5)** |
| E1 | **Window overlap inflates every ρ — the big one** | EASY/THINK | 6 | pending |
| E2 | ⟨χ²⟩ unnormalized, not comparable across measures | EASY | 6 | **addressed** |
| E3 | ⟨χ²⟩ is not a chi-squared statistic | EASY | 6 | pending |
| E4 | No significance testing at any point | EASY | 6 | **addressed** |
| E5 | The eight measures are not eight independent facts | EASY | 6 | **addressed** |
| E6 | Disconnection handling unstated | EASY | 5 | **addressed** |
| E7 | PCA1 asserted to be time, not shown | EASY | 7 | **addressed** |
| E8 | Fig. 7 shows separation, described as agreement | THINK | 7 | **addressed** |
| E9 | The 2002 split was chosen from the data | THINK | 6 | pending |
| E10 | "Crisis detection evaluation" never performed | THINK | 8b | **addressed** |
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

*Our handling:* not corrected — correcting it would replicate a different paper. Quantified instead: Phase 2 runs the same pipeline on i.i.d. Gaussian returns of identical shape.

**Phase 2 result — the audit's prediction is confirmed almost exactly.** Pure i.i.d. Gaussian noise, 360 series × 30 observations, put through the identical construction, gives **τ = 0.241 ± 0.001** and a network that is just as full and just as connected as the real one (0 isolated nodes, mean retained edge weight 0.324). The audit predicted ρ ≈ 0.24 from the 1/√29 ≈ 0.19 per-entry noise alone.

**Phase 3 result — the same reference applied to modularity, and this one is worse.** Running Louvain on those same pure-noise networks gives **Q = 0.2246 ± 0.0067** (median k = 7). The real market networks give **Q = 0.2215 ± 0.0413** (median k = 9). *The average level of the paper's central quantity is exactly what structureless data produces under its own construction* — noise is, if anything, marginally more modular. What separates the real series is not its level but its **variance**: 6× that of the noise reference, with a range of 0.09–0.42 against noise's 0.21–0.24.

The consequence for reading the paper: no statement of the form "the market network has modularity ≈ 0.2, therefore it is modularly organised" is supported. Only statements about *changes* in Q — the crisis drops, the 2002 break — survive this reference. The paper makes both kinds of statement and does not distinguish them.

This comparison is bounded by A5 being out of scope: real returns carry a dominant market factor that i.i.d. noise does not, and that factor pushes measured modularity *down*. A factor-matched null would be the stronger test and is not run here, so this is evidence that the *level* of Q is uninformative — not evidence that the real market is indistinguishable from noise.

Our real τ ranges 0.263–0.861 with a mean of **0.444**. So on a *typical* day the threshold sits only ~0.20 above the level pure noise would produce unaided, and the quietest days in the sample (τ ≈ 0.26) are barely distinguishable from noise at all. The crisis days are unambiguous — τ = 0.861 on Black Monday is far outside anything noise generates — but the paper's normal-times networks contain a great deal of estimation error, and it is never acknowledged.

### A5. The market mode is never removed. [THINK]
Under ρ_ij ≈ β_i β_j σ²_M / (σ_i σ_j), the top 10% of correlations largely selects high-β_i β_j pairs. So the network is substantially a map of market beta, not of economic relatedness. Sector structure is only the second-order effect. Standard practice in this literature is to regress out the market factor first. Cheap to test as a variant, but it produces a genuinely different network.

**UNTESTED — out of scope.** Requires a full Phase 2–6 rerun on residual returns.

### A6. Δt = 30 is asserted, never justified or robustness-tested. [EASY]
They tested robustness to *f* (Fig. S1) but not to the more dangerous parameter. Rerun at 20/60/120.

**UNTESTED — out of scope.** Requires three additional full Phase 2–6 runs.

### A7. Fixed density deletes the primary crisis signal. [THINK]
"Correlations spike toward 1 in a crash" is the most replicated fact in this literature, and it's removed at step three. The choice is defensible — it is what makes Section III's comparison legitimate — but it means the paper's crisis signal is a weak residual. Partial fix is trivial (save τ(t)); deciding what the right construction is, is not.

*Our handling:* τ(t) stored for every day in Phase 2 and made a first-class result in Phase 8a, where it competes against modularity as a crisis indicator.

**Phase 2 result — the discarded signal is very strong.** τ(t) peaks at **0.861 on 1987-10-19, Black Monday, which is its global maximum over 25 years**, against a mean of 0.444. Autumn 2008 reaches 0.798 (99th percentile). Mean pairwise correlation behaves the same way, peaking at 0.683. Both are recovered at zero cost and are available before any community is detected. Phase 8a will test whether modularity — which requires Louvain, a null model and eight measures — actually beats them.

### A8. Negative correlations are structurally invisible. [THINK]
τ always lands around 0.4–0.6, so no anticorrelated pair ever becomes an edge. Two stocks that reliably move oppositely — a strong, tradeable relationship — are recorded identically to two unrelated stocks.

*Our handling:* not corrected. Quantified: Phase 2 reports the count and time series of pairs with ρ < −τ, i.e. the relationships the construction is blind to.

**Phase 2 result, and it turned up something the audit did not anticipate.** Across the sample there are **4,246,798 pair-days** with ρ < −τ — strong anticorrelations recorded identically to no relationship at all — peaking at 4,596 pairs in a single day (~7 % of all pairs, comparable to the 6,462 edges actually kept).

More interesting is the *shape* of the series: invisible pairs are common and volatile from 1987 through about 2002, then **collapse to near zero from roughly 2003 onward and never recover** (see `figures/fig2_networks.png`, panel C). Whatever changed in the cross-sectional structure of NYSE returns, it changed close to the paper's own unexplained 2002 break point. This is a candidate mechanism for **E11**, which the paper leaves as "the topic of future work" — noted here rather than pursued, since E11 is out of scope.

### A9. Binarization discards edge weights. [THINK]
ρ = 0.99 and ρ = 0.51 become the same edge. Justified by convenience (cliques, Louvain, rich-club are simpler on binary graphs), but it is real information loss.

*Our handling:* not corrected; the mean and spread of retained edge weights are reported per day so the magnitude of the discard is visible.

**Phase 2 result.** Retained edge weights span **0.263 to 0.996** across the sample. Within a single day the spread is routinely as wide as 0.3 to 0.99. A pair correlated at 0.99 and a pair correlated at 0.27 are stored as the same 1, so the binarization is discarding roughly the entire dynamic range of the quantity the network is built from.

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

**Resolved: t₂, confirmed twice independently.**

The anchors fix, for each convention, where the paper's day 1 must sit on our calendar; `C_p` = 6,008 then fixes where its last day must sit. A convention whose implied last day runs past the end of our calendar is impossible, because the paper's data ends in February 2011 and ours ends 2011-02-28:

| Convention | implied first day | implied last day | verdict |
|---|---|---|---|
| t₁ (window start) | 1987-05-29 | past 2011-02-28 | impossible |
| midpoint | 1987-05-07 | past 2011-02-28 | impossible |
| **t₂ (window end)** | **1987-04-15** | **2011-02-09** | **consistent** |

**t₂ is the only convention that fits, and it fits exactly** — 6,008 days ending in February 2011, as stated.

Phase 2 then confirms it from the data rather than from arithmetic. Our isolated-node count peaks at 152 (from a baseline near 0) in a single window; under t₂ that window is stamped **1987-10-19, Black Monday itself, to the day.** Under midpoint it would be stamped 21 days before the crash and under t₁ 45 days before — i.e. both alternatives would show the network shattering *before* the event that shattered it.

**The corollary is a substantive finding about the paper: its data does not begin in January 1986 as stated, but around April 1987.** The stated date range and the reported `C_p` = 6,008 cannot both be true. This also explains why Fig. 4's x-axis starts at 1988 and Fig. 3's panel opens in August 1987.

*(Two superseded readings are kept here as record. First: the hypothesis that our own missing-day filter would reproduce the 337-day gap was **falsified** — our panel drops zero days. Second: an initial comparison concluded "midpoint fits four times better", by wrongly assuming the paper's panel ends on our last day rather than 6,008 days after its own start. Fixing that assumption reverses the conclusion.)*

---

## B. Community detection

### B1. Louvain degeneracy, no seed, no stability check. [EASY]
Modularity's landscape has exponentially many near-optimal partitions with very different structure. Louvain also depends on node ordering. Fix: 10 seeds per day, report the spread in Q and partition agreement.

*Our handling — addressed in Phase 3, and the damage is real.* Ten seeds per day, all retained; seed 0 is the canonical `C(t)` so the pipeline stays as faithful as the paper's single run. On the median day the ten seeds disagree by **sd 0.0038, range 0.0118** in Q. The median day-to-day move in Q is **0.0056**. So the seed-to-seed spread on a single day is **2.09× the typical daily move, and exceeds it on 77.9 % of days.** Any reading of the daily modularity series finer than ~0.012 in Q is reading Louvain, not the market. Partition-level agreement (NMI) is Phase 8f.

### B2. Independent daily runs mix algorithmic jitter with real change. [THINK]
No temporal coupling between consecutive days, so you cannot separate "the market changed" from "Louvain landed elsewhere." Multilayer/temporal community detection exists for exactly this — and the paper cites Mucha, Porter, and Fenn without using their method.

*Our handling:* not corrected (we replicate the independent-daily design), but Phase 8f measures the damage directly: seed-to-seed NMI versus day-to-day NMI. If they are comparable, the daily modularity series is substantially algorithmic noise, and that finding leads the report.

### B3. Resolution limit. [THINK]
Modularity maximization cannot resolve communities smaller than ~√(2m) ≈ 110 edges here. Genuine small sectors get absorbed. This mechanically bounds *k*, which determines the size of Π, which determines whether the compression claim holds.

*Our handling — measured in Phase 3.* With 6,462 edges the bound is √(2m) ≈ 114 edges, and the realized median *k* is **9** communities over 360 stocks — i.e. groups averaging 40 stocks, far coarser than the ~20 GICS sub-industries a 360-stock NYSE panel spans. The resolution limit is therefore binding, not hypothetical: whatever the market's sector structure is, Louvain at this density cannot report more than about a dozen groups. Consequence for the compression claim carried into Phase 6.

### B4. The number of communities *k* is never reported. [EASY]
Eq. 2's behaviour and Section IV's central claim both depend on it entirely. We store and report k[t] for every day.

*Our handling — addressed in Phase 3.* `k[t]` is stored for all 6,315 days and for all ten seeds. **Median 9, range 4–156**, maximum on 1987-10-19. Median 42 inside the sharp-tier crisis bands against 9 outside — the number of communities is itself a crisis indicator, and the paper never plots it.

### B5. Isolated nodes each become their own community. [THINK]
The paper states components are always assigned distinct communities. During Black Monday the graph shatters into a hub plus hundreds of isolates, so *k* explodes and Π becomes huge and nearly empty — on exactly the days the paper cares most about. This may invert the compression argument where the story lives. Tested jointly with D1 in Phase 6.

*Our handling — the prediction is confirmed in Phase 3.* On 1987-10-19 the partition contains **156 communities, 152 of them singletons**, against a median of 9 on an ordinary day. Π therefore grows from a 9×9 summary to a 156×156 one, i.e. from 45 numbers to 12,246, on the single day the paper's argument most needs the compression to hold. Storage had to be made ragged for exactly this reason (`net/communities.MixingStore`). Whether this inverts the compression claim is the Phase 6 test.

### B7. The lag t_Δ = 100 is past the point where a partition means anything. *(not in the original audit — found in Phase 3)*
The paper reports a "lagged" modularity using a partition 100 trading days stale, and never says why 100. Measuring `Q(g_t, C(t−L))` across L = 0…500 gives a decay curve with a **half-life of ≈ 14 trading days**: Q falls from 0.226 at L = 0 to 0.081 at L = 20, then flattens onto a plateau of ≈ 0.03 from L = 50 onward, against a random-partition floor of −0.004.

So t_Δ = 100 does not sample the decay — it samples the plateau, where only **15.4 %** of Q survives and a further doubling of the lag costs almost nothing. The paper's lagged curve is therefore close to a flat "stale partitions are worthless" line rather than a measurement of how fast structure turns over, and the interesting quantity — that ~14-day half-life, which is what "modular dynamics" should mean — is not reported at any point.

A corollary settles an expectation this replication got wrong. The plan asserted `Q_fixed ≤ Q_lagged ≤ Q_dyn` on every day and called a violation a bug. `Q_stale ≤ Q_dyn` holds on **all 6,215 days**, as it must. But `Q_fixed ≤ Q_lagged` fails on 1,451 days (23.3 %), and the failures are not random: **46 % of days before 1999, 0 % after.** There is no theorem behind that ordering — both quantities sit near the random-partition floor, where their ranking is not meaningful, and the fixed partition `C(0)` additionally encodes a stable split that a 100-day-old Louvain fit does not. The acceptance test was corrected to assert what is actually required; the violation is reported here rather than asserted away.

### B6. The modularity noise floor is never drawn. [EASY]
A structureless graph of this size and density scores Q ≈ 0.1 — visible as the green line in Fig. 6(a1). Normal-times values of 0.20–0.40 are only modestly above it, and the crisis value sits at it. We draw the configuration-model floor explicitly on every modularity figure from Phase 4 onward.

*Our handling — Phase 4 reproduces the floor and then shows it is the wrong floor.* The configuration-model null gives **Q = 0.1065 ± 0.0145**, matching the ≈0.10 the paper draws but never states. The real market's 0.2215 sits **+0.115** above it, and that gap is the paper's evidence of modular organisation.

**The control the paper never runs: put i.i.d. Gaussian returns through the entire pipeline and compare them against *their* degree-sequence null.** Result: Q = 0.2253 over a floor of 0.1332, an excess of **+0.0921** — **80 % of the market's excess is reproduced by data containing no communities whatsoever.**

The mechanism is that a correlation matrix is positive semi-definite, so high ρ_ij and ρ_ik force ρ_jk up; thresholding one therefore yields a graph that is transitive by construction (noise transitivity 0.219), and transitive graphs are modular. A degree-sequence null cannot control for this because rewiring destroys the geometry along with the structure. Any modularity excess over a configuration model, in any paper using thresholded correlation networks, needs this control before it can be read as evidence about the system. Bounded by A5 remaining out of scope.

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

*Our handling — measured in Phase 4, and it is a smaller problem than expected.* Over all 6,315 days the generated edge count averages **6,462 ± 70** against a real 6,462: bias **+0.002 %**, worst single day **1.42 %**, and within 3 % on **100 %** of days. The coin-flip variance is real but is roughly the sqrt(m) ≈ 80 you would predict, so it never approaches the size of the effects the paper reports. Kept unfixed, as the paper wrote it.

### D3. One realization per day, no error bars anywhere. [EASY]
Both Louvain and the generative step are stochastic. Replicate spread is free. Clique number especially — a maximum over a random graph is a notoriously high-variance quantity. We run 10 replicates per day per model.

*Our handling — done in Phase 4 for both generators* (126,300 graphs; every seed a pure function of `(FIT_SEED, day, replicate)`, so the run is bit-reproducible at any worker count). Carried into Phase 5 for the eight measures.

### D4. Rich-club is reported raw. [EASY]
φ(k) rises with k even in random graphs, which is why it is conventionally normalized against a null. Not done here. We store both raw and configuration-normalized φ.

*Related finding — Phase 4.* "Preserve the degree sequence" has two standard implementations and they are not interchangeable here. Classical stub matching (igraph's `Degree_Sequence`) emits self-loops and multi-edges; collapsing them to a simple graph **destroys 13.0 % of the edges on average and 22.2 % on the worst day**, so the null ends up materially sparser than the network it is being compared against — which confounds every density-sensitive measure (transitivity, path length, clique number) with an artifact of the generator. Degree-preserving edge swaps keep the degree sequence *and* the edge count exactly, on 100 % of days, and are used instead. igraph's third option, `vl`, cannot be used at all: it requires a connected realization, and the crisis networks contain hundreds of degree-0 nodes. Recorded in `net/nullmodels.CONFIG_METHOD`.

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


---

## Results, phases 5–9

Added after the run. Numbers read from `results/phase*.json`; nothing retyped.

### E1 — the overlap correction. **The audit's central hypothesis is not supported.**
The prediction was that recomputing every ρ on non-overlapping windows would collapse the paper's evidence — "if 0.99 falls to 0.7 the headline needs rewriting."

It does not fall. Across all 32 cells the mean ρ moves **0.783 → 0.777**, a change of +0.006. Only 12 of 32 cells drop at all, and the worst drop is 0.164 (rich club, community model, post-2002).

The mechanism in the audit is real — a unit test in `tests/test_paper3.py` shows two *independent* random walks passed through 30-day windows correlating at |ρ| > 0.3 on overlapping points. But overlap inflates correlations that are **spurious**, and these are not: the community model genuinely tracks the real series. Overlap cannot add much to a correlation already at 0.98. Effective sample size drops from 6,315 to 211, so the *confidence intervals* should widen — but the point estimates stand.

This is the single largest thing this replication expected to find and did not.

### E2 — normalized MSE. The rescaling changes the reading completely.
Dividing each χ² by the variance of the real series (1.0 = no better than predicting the mean):

| measure | community NMSE | configuration NMSE |
|---|---|---|
| modularity | **0.034** | 8.508 |
| path length | 0.371 | **0.298** |
| assortativity | 1.138 | 8.897 |
| transitivity | 9.668 | **3.218** |
| betweenness | 3.168 | **1.703** |
| clique number | 5.360 | **3.014** |
| rich club | 14.340 | **1.457** |
| matching index | 4.430 | **0.561** |

**The community model beats the degree sequence on exactly one measure: modularity — the one quantity Π directly encodes.** On six of the remaining seven the degree sequence is better, and on five of them the community model scores worse than 1.0, i.e. worse than predicting the real series' own mean. The paper's raw χ² table hides this because its units differ by five orders of magnitude.

### E4 — block bootstrap. 95 % CIs on all 48 cells, block length = Δt = 30.

### E5 — the eight measures are about three.
Mean |ρ| between measures **0.607** (max 0.955); **3 of 8** principal components carry 90 % of the variance. Eight agreements are not eight pieces of evidence.

### E6 — the disconnection convention matters more than expected.
The network is fragmented on **3,973 of 6,315 days**, worst on 1987-10-19 when the largest component holds only 57.8% of stocks. The two conventions for average path length correlate just **+0.391** — they are substantially different series, and the paper does not say which it used.

### E7/E8 — PCA.
PC1 explains 52.7%, PC1+PC2 80.1%. corr(PC1, date) = **+0.439** — a moderate relationship, not the "PCA1 is essentially time" the paper asserts.

The separation the paper describes as agreement: a logistic classifier tells the three families apart **98.6%** of the time against 33.3% chance, and real from community model **97.8%** against 50 %. A null model genuinely indistinguishable from the real networks would sit at chance.

### E10 — the detector, built and scored.
Against the Phase-0 pre-registered windows (base rate 2.7%):

| signal | ROC-AUC |
|---|---|
| mean correlation | 0.807 |
| **τ(t)** | **0.807** |
| isolated nodes | 0.760 |
| dynamical modularity | 0.538 |

**τ(t) — the quantity fixed-density thresholding throws away — detects crises far better than modularity, which is barely above chance at 0.538.** The paper builds its entire crisis narrative on the weakest of the four signals its own pipeline produces.

### C1 — rise or fall: neither.
Averaging over all 17 pre-registered onsets, in units of each event's pre-window sd: modularity at onset **-0.015** (nothing happens on the day), over the following 60 days **-0.495**. The abstract's "crises destroy structure" is directionally right but slow; the Fig. 4 text's "increase of modularity around the crisis" is not supported.

### D1/B5 — the compression claim inverts exactly where the story lives.
Community summary `k + k(k+1)/2` vs the degree sequence's N = 360: cheaper on **69.3%** of all days but only **34.3%** of sharp-crisis days. Break-even is k = 25.4; observed k reaches 156. On the days the paper is about, describing the market by its communities costs *more* than listing every stock's degree.

### B1/B2 — Louvain stability.
NMI between seeds on the same day **0.543**; between consecutive days at fixed seed **0.463**. Two runs on identical data agree only +0.079 more than runs on different days.

### Not reproduced — 4 of 32 cells
| measure | model | era | ours | paper | Δ |
|---|---|---|---|---|---|
| assortativity | conf | pre | +0.573 | -0.470 | +1.043 |
| modularity | conf | pre | +0.555 | +0.070 | +0.485 |
| assortativity | comm | pre | +0.283 | +0.760 | -0.477 |
| betweenness | comm | pre | +0.619 | +0.930 | -0.311 |

Concentrated in the configuration model, which in our hands tracks the real networks far more closely than in the paper's — mean ρ **0.861** vs **0.721** across all 16 configuration cells, and higher on every measure. Two explanations were tested on a 500-day pre-2002 sample and **both ruled out**: rewiring at 5× and 50× the edge count gives identical results (ρ 0.510 vs 0.512, edge overlap 26.4 % both), and classical stub matching gives ρ +0.443 / +0.418 — closer to ours than to the paper's +0.07 / −0.47.

This matters because the configuration model is the paper's **yardstick**. The community model looks impressive largely in proportion to how badly the degree null performs, and we cannot reproduce the degree null performing that badly.
