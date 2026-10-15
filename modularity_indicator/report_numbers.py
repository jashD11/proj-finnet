"""
Re-derive every number quoted in reports/2026-09-08_modularity_equivalence.tex.

Same discipline as reports/probe/check_numbers.py: the report transcribes this
script's output and nothing else. Read-only; writes only figure-data blocks for
pasting into the .tex (pgfplots coordinates), never into paper1_gmrf_laplacian.

Run:  ../.venv/bin/python report_numbers.py
      ../.venv/bin/python report_numbers.py --figdata   (also emit coordinates)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from numpy.linalg import eigh, lstsq
from sklearn.metrics import roc_auc_score

from evaluate import (HERE, P1, TAU, COVID_START, COVID_END, load, backtest)
from robustness import trailing_z
from build_series import (WINDOW, K, ETA, BETA, DELTA, DEGREE_CONTROL,
                          adjacency_from_laplacian)

sys.path.insert(0, P1)
from solver.algorithm2 import algorithm2  # noqa: E402

EPS = 1e-12
REPRESENTATIVE = "2020-03-16"      # the COVID-crash snapshot exp3 plots


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def spectra(A):
    """Return the four spectra and eigenbases for one adjacency matrix."""
    n = A.shape[0]
    d = A.sum(axis=1)
    two_m = d.sum()
    L = np.diag(d) - A
    B = A - np.outer(d, d) / two_m
    Dm = np.diag(1.0 / np.sqrt(np.maximum(d, EPS)))
    Lsym = np.eye(n) - Dm @ A @ Dm
    Bn = Dm @ B @ Dm
    sym = lambda M: (M + M.T) / 2.0
    wl, Vl = eigh(sym(L))
    wb, Vb = eigh(sym(B))
    wls, Vls = eigh(sym(Lsym))
    wbn, Vbn = eigh(sym(Bn))
    return dict(d=d, two_m=two_m, wl=wl, Vl=Vl, wb=wb, Vb=Vb,
                wls=wls, Vls=Vls, wbn=wbn, Vbn=Vbn)


def main():
    figdata = "--figdata" in sys.argv

    # ── source data ──────────────────────────────────────────────────────────
    rets = pd.read_csv(os.path.join(P1, "data", "faamung.csv"),
                       index_col=0, parse_dates=True)
    graphs, day_idx = algorithm2(rets.values, window=WINDOW, k=K, eta=ETA,
                                 beta=BETA, delta=DELTA,
                                 degree_control=DEGREE_CONTROL, verbose=False)
    gdates = rets.index[day_idx]
    # parquet, not csv: series.csv is a local convenience copy and is gitignored,
    # so a fresh clone only has the parquet (which evaluate.load() also reads).
    S = pd.read_parquet(os.path.join(HERE, "series.parquet")).reset_index()
    ev = load()                                    # 201-day evaluation window
    sig = pd.read_csv(os.path.join(HERE, "scores_significance.csv")).set_index("quantity")
    zdet = pd.read_csv(os.path.join(HERE, "scores_zdetector.csv")).set_index("quantity")
    nulls = pd.read_csv(os.path.join(HERE, "scores_nulls.csv")).set_index("quantity")

    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])

    rule("SETUP  (section 1, section 6)")
    print(f"  FAAMUNG            : {rets.shape[1]} stocks, {rets.shape[0]} trading days, "
          f"{rets.index[0].date()} -> {rets.index[-1].date()}")
    print(f"  rolling windows    : {len(graphs)}  (window={WINDOW}, k={K}, "
          f"eta={ETA}, beta={BETA}, delta={DELTA}, degree_control={DEGREE_CONTROL})")
    print(f"  evaluation window  : {len(ev)} signal days, "
          f"{pd.DatetimeIndex(ev.index)[0].date()} -> "
          f"{pd.DatetimeIndex(ev['invest_date'])[-1].date()}")
    crash_ev = np.asarray((pd.DatetimeIndex(ev.index) >= COVID_START) &
                          (pd.DatetimeIndex(ev.index) <= COVID_END))
    lam2_ev = ev["lam2"].values
    n_invest = int((lam2_ev < TAU).sum())
    s1 = ev["s1_ret"].values.sum()
    s2 = backtest(lam2_ev, n_invest, ev["s1_ret"].values,
                  pd.DatetimeIndex(ev["invest_date"]))["final"]
    print(f"  COVID crash days   : {int(crash_ev.sum())} "
          f"({COVID_START.date()} -> {COVID_END.date()})")
    print(f"  lambda2 gate       : tau={TAU}, invests {n_invest}/{len(ev)} days "
          f"({100*n_invest/len(ev):.1f}%)")
    print(f"  S1 buy & hold      : {s1:+.4f}          S2 (lambda2 gate): {s2:+.4f}")

    # ── SECTION 2: the normalisation collapse ────────────────────────────────
    rule("SECTION 2  normalised B and normalised L are the same operator")
    e_spec, e_ident, ov_n, ov_u, fied = [], [], [], [], []
    for L in graphs:
        s = spectra(adjacency_from_laplacian(L))
        pred = np.sort(np.concatenate([[0.0], 1.0 - s["wls"][1:]]))
        e_spec.append(np.abs(np.sort(s["wbn"]) - pred).max())
        e_ident.append(abs(s["wbn"][-1] - (1.0 - s["wls"][1])))
        ov_n.append(np.abs(s["Vls"].T @ s["Vbn"]).max(axis=0).mean())
        ov_u.append(np.abs(s["Vl"].T @ s["Vb"]).max(axis=0).mean())
        fied.append(abs(s["Vl"][:, 1] @ s["Vb"][:, -1]))
    print(f"  FULL spectrum   max |spec(B_n) - ({{0}} u {{1-lam_i(L_sym)}})|  "
          f"= {max(e_spec):.2e}   <- exact")
    print(f"  top eigenvalue  max |mu1(B_n) - (1 - lam2(L_sym))|             "
          f"= {max(e_ident):.2e}")
    print(f"  eigenvectors    B_n vs L_sym   mean best-overlap {np.mean(ov_n):.6f}  "
          f"min {np.min(ov_n):.6f}   <- shared basis")
    print(f"  eigenvectors    B   vs L       mean best-overlap {np.mean(ov_u):.4f}  "
          f"min {np.min(ov_u):.4f}   <- NOT shared")
    print(f"  |<Fiedler vec of L, top eigvec of B>|  mean {np.mean(fied):.3f}  "
          f"range [{np.min(fied):.3f}, {np.max(fied):.3f}]")
    print(f"  connectedness    k={K}, so L_sym has a 1-dim null space on all "
          f"{len(graphs)} windows (identity requires this)")

    # ── SECTION 3: the candidate menu ────────────────────────────────────────
    rule("SECTION 3  provenance of the candidate menu")
    NORMALISED = ["mu1_norm", "muN_norm", "one_minus_mu1n"]
    LAP_REF = ["lam2", "lam_max", "lam2_sym", "dbar", "two_m", "deg_cv"]
    plain = [q for q in S.columns
             if q not in ("date", "day_index") and q not in NORMALISED + LAP_REF]
    print(f"  from normalised B_n : {len(NORMALISED):2d}  {NORMALISED}")
    print(f"  from plain B        : {len(plain):2d}  {plain}")
    print(f"  Laplacian reference : {len(LAP_REF):2d}  {LAP_REF}")
    print(f"  => {len(NORMALISED)+len(plain)} B-derived candidates, "
          f"{len(NORMALISED)+len(plain)+len(LAP_REF)} quantities scored in total")
    lam_max_sym = []
    for L in graphs:
        lam_max_sym.append(spectra(adjacency_from_laplacian(L))["wls"][-1])
    lam_max_sym = np.array(lam_max_sym)
    print(f"\n  the three normalised candidates are AFFINE images of Laplacian "
          f"quantities:")
    print(f"    corr( mu1_norm      , lam2_sym        ) = "
          f"{c(S.mu1_norm, S.lam2_sym):+.10f}")
    print(f"    corr( muN_norm      , lam_max(L_sym)  ) = "
          f"{c(S.muN_norm, lam_max_sym):+.10f}")
    print(f"    corr( one_minus_mu1n, lam2_sym        ) = "
          f"{c(S.one_minus_mu1n, S.lam2_sym):+.10f}")
    print(f"    max |one_minus_mu1n - lam2_sym|         = "
          f"{np.abs(S.one_minus_mu1n - S.lam2_sym).max():.2e}")
    print(f"\n  collapse WITHIN the plain-B block (correlation with mu1):")
    for q in ["gap12", "q_bound", "spread", "absmax", "mu_ratio", "muN",
              "trace_pos", "energy", "frob2", "spec_entropy"]:
        print(f"    corr( {q:<12}, mu1 ) = {c(S[q], S.mu1):+.3f}")
    n_absmax = int((np.abs(S.muN) > S.mu1).sum())
    print(f"\n  |muN| > mu1 on {n_absmax}/{len(S)} windows "
          f"({'absmax IS |muN| relabelled' if n_absmax == len(S) else 'MIXED - do not claim absmax = |muN|'})")

    # ── SECTION 4: the retracted theorem ─────────────────────────────────────
    rule("SECTION 4  the d-bar substitution is not an identity")
    e_dbar = []
    for L in graphs:
        s = spectra(adjacency_from_laplacian(L))
        pred = np.sort(np.concatenate([[0.0], s["d"].mean() - s["wl"][1:]]))
        e_dbar.append(np.abs(np.sort(s["wb"]) - pred).max())
    resid = (S.dbar - S.mu1 - S.lam2).values          # = dual_resid
    crash_all = np.asarray((S.date >= COVID_START) & (S.date <= COVID_END))
    print(f"  FULL spectrum   max |spec(B) - ({{0}} u {{dbar-lam_i(L)}})|  "
          f"= {max(e_dbar):.2f}   <- NOT exact")
    print(f"  (compare with lam2's own sd = {S.lam2.std():.3f}: the error is "
          f"{max(e_dbar)/S.lam2.std():.1f}x it)")
    print(f"\n  degree regularity : deg_cv mean {S.deg_cv.mean():.3f}, "
          f"range [{S.deg_cv.min():.3f}, {S.deg_cv.max():.3f}]  (0 = regular)")
    print(f"                      dbar mean {S.dbar.mean():.3f} sd {S.dbar.std():.3f}")
    print(f"\n  residual r = (dbar - mu1(B)) - lam2  over all {len(S)} windows:")
    print(f"    mean {resid.mean():+.3f}   sd {resid.std():.3f}   "
          f"range [{resid.min():+.3f}, {resid.max():+.3f}]")
    print(f"    lam2 mean {S.lam2.mean():.3f}  sd {S.lam2.std():.3f}")
    print(f"    residual sd as % of lam2 sd : {100*resid.std()/S.lam2.std():.0f}%")
    print(f"    corr(resid, lam2)   = {c(resid, S.lam2):+.3f}   "
          f"<- error moves WITH the signal")
    print(f"    corr(resid, deg_cv) = {c(resid, S.deg_cv):+.3f}   "
          f"<- not a simple irregularity effect")
    print(f"    residual in-crash {resid[crash_all].mean():.3f}  vs "
          f"out {resid[~crash_all].mean():.3f}   <- error is event-dependent")
    proxy = S.dbar_minus_mu1.values
    v_l, v_r = np.var(S.lam2), np.var(resid)
    cov = np.cov(S.lam2, resid)[0, 1]
    tot = np.var(proxy)
    print(f"\n  variance decomposition of (dbar - mu1) = lam2 + r, total sd {np.sqrt(tot):.3f}:")
    print(f"    from lam2 {100*v_l/tot:.0f}% | from residual {100*v_r/tot:.0f}% "
          f"| from covariance {100*2*cov/tot:.0f}%")
    print(f"\n  does the dbar term do any work?  (partial correlation, controlling for dbar)")

    def partial(x, y, z):
        x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
        rx = x - np.polyval(np.polyfit(z, x, 1), z)
        ry = y - np.polyval(np.polyfit(z, y, 1), z)
        return c(rx, ry)
    print(f"    corr(lam2, mu1)            raw {c(S.lam2, S.mu1):+.3f}   "
          f"partial|dbar {partial(S.lam2, S.mu1, S.dbar):+.3f}")
    print(f"    corr(lam2, dbar - mu1)     raw {c(S.lam2, proxy):+.3f}   "
          f"partial|dbar {partial(S.lam2, proxy, S.dbar):+.3f}")
    print(f"\n  RECONCILIATION of the two correlation figures in FINDINGS.md:")
    print(f"    corr(lam2, dbar-mu1) over all {len(S)} windows            = "
          f"{c(S.lam2, proxy):.3f}")
    print(f"    corr(lam2, dbar-mu1) over the {len(ev)}-day eval window   = "
          f"{c(ev.lam2, ev.dbar_minus_mu1):.3f}   <- the figure quoted in FINDINGS.md")

    # ── SECTION 5: correlation is a bad novelty test ─────────────────────────
    rule("SECTION 5  correlation fails as a novelty test")
    ent_B, nuc_B, WLS = [], [], []
    for L in graphs:
        s = spectra(adjacency_from_laplacian(L))
        p = np.abs(s["wbn"]); p = p / p.sum()
        ent_B.append(-np.sum(p[p > 0] * np.log(p[p > 0])))
        nuc_B.append(np.abs(s["wbn"]).sum())
        WLS.append(s["wls"])
    ent_B, nuc_B, WLS = np.array(ent_B), np.array(nuc_B), np.array(WLS)
    MU = np.concatenate([np.zeros((len(WLS), 1)), 1.0 - WLS[:, 1:]], axis=1)
    P = np.abs(MU); P = P / P.sum(1, keepdims=True)
    ent_hat = -np.sum(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0.0), axis=1)
    nuc_hat = np.abs(MU).sum(1)
    print("  Example A - correlation DOES catch the tautology (affine map):")
    print(f"    corr( mu1(B_n), lam2(L_sym) ) = {c(S.mu1_norm, S.lam2_sym):+.10f}")
    print("  Example B - correlation MISSES the tautology (non-affine map):")
    print(f"    corr( spectral entropy(B_n), lam2 ) = {c(ent_B, S.lam2):+.3f}  "
          f"(looks independent)")
    print(f"    ... but rebuilt from the L_sym spectrum alone: max abs error "
          f"= {np.abs(ent_B - ent_hat).max():.2e}")
    print(f"    corr( nuclear norm(B_n)    , lam2 ) = {c(nuc_B, S.lam2):+.3f}")
    print(f"    ... rebuilt from the L_sym spectrum alone: max abs error "
          f"= {np.abs(nuc_B - nuc_hat).max():.2e}")
    print("  Converse trap (section 4): a SPURIOUS quantity scored the highest "
          "correlation of all.")

    # ── SECTION 6: lambda2 inside the noise band ─────────────────────────────
    rule("SECTION 6  lambda2 under Paper 1's own conditions")
    print(f"  lambda2 raw-level COVID AUC      = {zdet.loc['lam2','auc_raw']:.3f}  "
          f"<- BELOW chance (autumn-2019 regime confound)")
    print(f"  lambda2 z-scored COVID AUC       = {zdet.loc['lam2','auc_z']:.3f}")
    r = sig.loc["lam2"]
    print(f"  bootstrap 95% CI on that AUC     = [{r.lo:.3f}, {r.hi:.3f}]   "
          f"-> EXCLUDES 0.5, so lambda2 does beat a random DAY")
    print(f"  sliding-event null (24-day win)  : COVID at the "
          f"{100*r.slide_pctile:.0f}th percentile, p = {r.slide_p:.3f}   "
          f"-> FAILS to beat a random 24-day STRETCH")
    print(f"    null median {r.null_med:.3f}, 95th pct {r.null_p95:.3f}, "
          f"max {r.null_max:.3f}")
    print(f"  => the two nulls answer different questions; state both.")
    n_beat = int((nulls["s2"] > s1).sum())
    rng = np.random.default_rng(42)
    s1r = ev["s1_ret"].values
    rand = np.array([s1r[rng.choice(len(s1r), n_invest, replace=False)].sum()
                     for _ in range(20000)])
    p_rand = float((rand > s1).mean())
    best = nulls["s2"].idxmax()
    print(f"\n  backtest luck  : {n_beat}/{len(nulls)} candidates beat S1 = {s1:+.4f}")
    print(f"    uniform random gate at the same invest rate beats S1 "
          f"{100*p_rand:.1f}% of the time")
    print(f"    -> ~{p_rand*len(nulls):.1f} of {len(nulls)} candidates expected to "
          f"beat S1 by luck alone; {n_beat} did")
    print(f"    best-PERFORMING candidate      : {best} "
          f"S2={nulls.loc[best,'s2']:+.4f}, circular-shift p={nulls.loc[best,'p_value']:.3f}")
    print(f"    smallest p-value ANY candidate : {nulls['p_value'].min():.3f} "
          f"({nulls['p_value'].idxmin()})  -> nothing reaches significance")
    print(f"  multiplicity   : {len(zdet)} quantities scored against ONE event, "
          f"no correction applied anywhere in this study.")

    # ── SECTION 7: decomposing lambda2 ───────────────────────────────────────
    rule("SECTION 7  lambda2 = structure x scale")
    print("  NOTE two windows are in play: the 243 learned graphs (structural claims)")
    print("       and the 201-day exp4 evaluation window (all AUCs). Both printed.\n")
    print(f"  {'quantity':<32} {'all 243':>9} {'eval 201':>9}")
    for lbl, ca, cb in [
            ("corr( lam2(L), lam2(L_sym) )  structure", c(S.lam2, S.lam2_sym),
             c(ev.lam2, ev.lam2_sym)),
            ("corr( lam2(L), dbar        )  scale    ", c(S.lam2, S.dbar),
             c(ev.lam2, ev.dbar)),
            ("corr( lam2(L), 2m          )           ", c(S.lam2, S.two_m),
             c(ev.lam2, ev.two_m))]:
        print(f"  {lbl:<32} {ca:>+9.3f} {cb:>+9.3f}")
    for nm, D in [("all 243 ", S), ("eval 201", ev)]:
        X = np.c_[np.ones(len(D)), D.lam2_sym, D.dbar]
        b = lstsq(X, D.lam2.values, rcond=None)[0]
        r2 = 1 - ((D.lam2 - X @ b) ** 2).sum() / ((D.lam2 - D.lam2.mean()) ** 2).sum()
        print(f"  R^2 of lam2(L) ~ lam2(L_sym) + dbar  [{nm}] = {r2:.3f}")

    lo_b, hi_b, pos_b = [], [], []
    for L in graphs:
        s = spectra(adjacency_from_laplacian(L))
        lo_i = s["d"].min() * s["wls"][1]
        hi_i = s["d"].max() * s["wls"][1]
        lo_b.append(lo_i); hi_b.append(hi_i)
        pos_b.append((s["wl"][1] - lo_i) / max(hi_i - lo_i, EPS))
    lo_b, hi_b, pos_b = np.array(lo_b), np.array(hi_b), np.array(pos_b)
    held = int(((S.lam2.values >= lo_b - 1e-9) & (S.lam2.values <= hi_b + 1e-9)).sum())
    print(f"\n  bound  d_min*lam2(L_sym) <= lam2(L) <= d_max*lam2(L_sym)")
    print(f"    holds on {held}/{len(S)} windows"
          f"{'  <- exact containment, so the split is structural not fitted' if held == len(S) else '  <- VIOLATED, do not state as a bound'}")
    print(f"    lam2 sits at {pos_b.mean():.2f} of the way across the band "
          f"(sd {pos_b.std():.2f}); band width mean {np.mean(hi_b-lo_b):.3f}")

    print(f"\n  z-scored COVID AUC of each part (project convention: sign fixed by "
          f"correlation with lam2)")
    parts = [("lam2      full, unnormalised", "lam2"),
             ("lam2_sym  structure only    ", "lam2_sym"),
             ("dbar      scale only        ", "dbar"),
             ("two_m     total weight      ", "two_m"),
             ("deg_cv    degree spread     ", "deg_cv"),
             ("muN       plain B           ", "muN")]
    for label, col in parts:
        print(f"    {label} : {zdet.loc[col,'auc_z']:.3f}   "
              f"(raw {zdet.loc[col,'auc_raw']:+.3f}, sign {int(zdet.loc[col,'sign']):+d})")

    # ── SECTION 8: the surviving lead ────────────────────────────────────────
    rule("SECTION 8  muN(B) as the one live lead")
    print(f"  corr( muN, mu1  ) levels      = {c(S.muN, S.mu1):+.3f}   "
          f"<- least redundant with the mu1 family")
    print(f"  corr( muN, lam2 ) levels      = {c(S.muN, S.lam2):+.3f}")
    zl, zm = trailing_z(ev.lam2.values), trailing_z(ev.muN.values)
    ok = np.isfinite(zl) & np.isfinite(zm)
    print(f"  corr( z[muN], z[lam2] )       = {c(zm[ok], zl[ok]):+.3f}   "
          f"(mu1_norm's is {c(trailing_z(ev.mu1_norm.values)[ok], zl[ok]):+.3f})")
    rm = sig.loc["muN"]
    print(f"  muN z-AUC {rm.auc:.3f}  CI [{rm.lo:.3f}, {rm.hi:.3f}]   "
          f"dAUC vs lam2 {rm.d_auc:+.3f} [{rm.d_lo:+.3f}, {rm.d_hi:+.3f}]  "
          f"-> NOT significant")
    print(f"  muN sliding-event p = {rm.slide_p:.3f}  -> also inside the noise band")
    probe = os.path.join(os.path.dirname(HERE), "reports", "probe",
                         "spectral_probe_auc.json")
    if os.path.exists(probe):
        with open(probe) as f:
            pj = json.load(f)
        det = pj.get("detectors", {})
        print(f"\n  independent corroboration, reports/probe/ "
              f"(6315 days, 360 stocks, 17 pre-registered crises):")
        labels = {"mod_lambda_min": "muN(B)      most negative modularity eigenvalue",
                  "mod_trace_pos": "sum mu>0    total group mass",
                  "mod_n_pos":     "#{mu>0}     group count, no algorithm",
                  "mod_lambda1":   "mu1(B)      leading modularity eigenvalue",
                  "mod_gap12":     "mu1-mu2     modularity spectral gap",
                  "lap_lambda2":   "lam2(L)     Fiedler value, PAPER 1's indicator",
                  "q_dyn":         "Q(t)        Louvain modularity, PAPER 3's indicator"}
        for k, lab in sorted(labels.items(),
                             key=lambda kv: -det.get(kv[0], {}).get("auc_sharp", 0)):
            if k in det:
                print(f"    {lab:<50} AUC {det[k]['auc_sharp']:.3f}")

    # ── figure data ──────────────────────────────────────────────────────────
    if figdata:
        out = os.path.join(HERE, "figdata.txt")
        with open(out, "w") as f:
            i = int(np.argmin(np.abs(gdates - pd.Timestamp(REPRESENTATIVE))))
            s = spectra(adjacency_from_laplacian(graphs[i]))
            f.write(f"% FIG 2 - spectrum pairing, window {gdates[i].date()}\n")
            f.write("% lam_i(L_sym):  " +
                    " ".join(f"{v:.4f}" for v in s["wls"]) + "\n")
            f.write("% mu_i(B_n)   :  " +
                    " ".join(f"{v:.4f}" for v in np.sort(s["wbn"])[::-1]) + "\n")
            f.write("% pairs (lam, mu=1-lam), i>=2:\n")
            for j in range(1, len(s["wls"])):
                f.write(f"({s['wls'][j]:.4f},{1-s['wls'][j]:.4f}) ")
            f.write("\n\n% FIG 4 - residual vs lam2 sd band (every 2nd window)\n")
            for j in range(0, len(S), 2):
                f.write(f"({j},{resid[j]:.4f}) ")
            f.write(f"\n% lam2 sd = {S.lam2.std():.4f}, resid sd = {resid.std():.4f}\n")
            f.write("\n% FIG 6 - sliding-event null, lam2 z-detector AUCs\n")
            zz = trailing_z(ev.lam2.values)
            okz = np.isfinite(zz)
            v, n_ok = zz[okz], int(okz.sum())
            w = int(crash_ev.sum())
            nullv = []
            for st in range(n_ok - w + 1):
                lab = np.zeros(n_ok, dtype=int); lab[st:st + w] = 1
                nullv.append(roc_auc_score(lab, v))
            hist, edges = np.histogram(nullv, bins=20, range=(0, 1))
            for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
                f.write(f"({(e0+e1)/2:.3f},{h}) ")
            f.write(f"\n% COVID AUC = {zdet.loc['lam2','auc_z']:.3f}, "
                    f"pctile {100*r.slide_pctile:.0f}\n")
            f.write("\n% FIG 7 - lam2 / structure / scale, z-scored, every 2nd window\n")
            for nm, col in [("lam2", S.lam2), ("lam2_sym", S.lam2_sym), ("dbar", S.dbar)]:
                zc = (col - col.mean()) / col.std()
                f.write(f"% {nm}\n")
                for j in range(0, len(S), 2):
                    f.write(f"({j},{zc.values[j]:.3f}) ")
                f.write("\n")
        print(f"\nWrote {out}")

    print("\nDone. Every number in "
          "reports/2026-09-08_modularity_equivalence.tex comes from above.")


if __name__ == "__main__":
    main()
