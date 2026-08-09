"""
Every figure in the replication, as a reusable function.

Phase 9's notebook calls these; it never recomputes anything. Each function
takes already-computed arrays or a results dict and returns a matplotlib Figure,
saving to figures/ when given a path.

House rules, applied to every figure here:
  - One y-axis per panel. The paper's own Fig. 4 overlays modularity and path
    length on twin axes; we stack panels instead, because a dual-axis chart lets
    the reader infer any correlation the axis scaling implies.
  - Categorical colour is assigned by entity and never by rank: inferred
    networks are always blue, the community null always orange, the
    configuration null always aqua, in every figure in the project.
  - Crisis bands are neutral grey, never a categorical hue - they are context,
    not a series.
  - Recessive grid and spines; series are the only saturated ink.
  - Committed to a light surface: these are static PNGs embedded in a notebook
    and compared against the paper's own figures.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ── Palette ────────────────────────────────────────────────────────────────
# Slots 1-3 of the validated categorical palette, used unchanged. This trio is
# documented as clearing the all-pairs floors in both modes (CVD dE 9.2,
# normal-vision dE 24.0 on a light surface), which matters because these series
# appear together as scatter points in Figs. 6/7, not only as lines.
INFERRED = "#2a78d6"      # slot 1, blue   - the real networks
COMMUNITY = "#eb6834"     # slot 2, orange - community (blockmodel) null
CONFIG = "#1baf7a"        # slot 3, aqua   - configuration (degree) null
ACCENT = "#4a3aa7"        # slot 7, violet - a secondary quantity (tau, rho-bar)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#dcdbd6"
CRISIS_SHADE = "#9a9a94"

FAMILY_COLOR = {"inferred": INFERRED, "community": COMMUNITY, "configuration": CONFIG}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "figure.dpi": 130,
})

DPI = 130


def _style(ax, title=None, ylabel=None, xlabel=None):
    if title:
        ax.set_title(title, loc="left", color=INK)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def shade_crises(ax, tiers=("sharp", "diffuse"), label=True):
    """Draw the pre-registered crisis bands as neutral context."""
    from utils.crisis_dates import crisis_frame
    df = crisis_frame()
    df = df[df["tier"].isin(tiers)]
    for k, (_, r) in enumerate(df.iterrows()):
        ax.axvspan(r["start"], r["end"], color=CRISIS_SHADE, alpha=0.18, lw=0,
                   label="crisis period (pre-registered)" if (label and k == 0) else None)


def _stamp_label(stamp: str) -> str:
    """X-axis label naming the window timestamp convention in force (audit A10)."""
    return {
        "t1": "window (stamped at its first day, t₁)",
        "midpoint": "window (stamped at its midpoint)",
        "t2": "window (stamped at its last day, t₂)",
    }.get(stamp, "window")


def _finish(fig, save):
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=DPI, bbox_inches="tight")
        print(f"[figure] {save}")
    return fig


# ── Phase 1 ────────────────────────────────────────────────────────────────

def plot_return_diagnostics(returns, per_ticker_std, dates, rho_bar,
                            std_band=(0.01, 0.04), stamp="t2", save=None):
    """Three-panel Phase 1 validation figure."""
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0))

    # A: the return distribution against a Gaussian of the same variance.
    ax = axes[0]
    flat = returns.ravel()
    sd = flat.std()
    bins = np.linspace(-0.25, 0.25, 301)
    # Outline, not a filled block: on a log density axis a filled histogram
    # floods the panel and hides the tails, which are the entire point here.
    ax.hist(flat, bins=bins, color=INFERRED, histtype="step", lw=1.2,
            density=True, label="observed log returns")
    centres = 0.5 * (bins[1:] + bins[:-1])
    gauss = np.exp(-0.5 * (centres / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    ax.plot(centres, gauss, color=ACCENT, lw=1.6, ls="--",
            label=f"Gaussian, same σ ({sd:.4f})")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, None)
    ax.legend(loc="upper right")
    ax.annotate("tails sit orders of magnitude\nabove the Gaussian",
                xy=(-0.19, 0.35), xytext=(-0.24, 0.02),
                textcoords="data", fontsize=8.5, color=INK_SECONDARY,
                arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=0.8))
    _style(ax, "A · Daily log returns are fat-tailed, not Gaussian",
           "density (log scale)", "log return")

    # B: per-ticker volatility against the acceptance band.
    ax = axes[1]
    ax.hist(per_ticker_std, bins=60, color=INFERRED, alpha=0.85, lw=0)
    ax.axvspan(*std_band, color=CRISIS_SHADE, alpha=0.18, lw=0)
    med = float(np.median(per_ticker_std))
    ax.axvline(med, color=ACCENT, lw=1.6)
    ax.annotate(f"median {med:.4f}", xy=(med, ax.get_ylim()[1] * 0.85),
                xytext=(6, 0), textcoords="offset points",
                color=ACCENT, fontsize=8.5, va="top")
    ax.annotate("expected band\n0.01–0.04", xy=(std_band[1], ax.get_ylim()[1] * 0.45),
                xytext=(8, 0), textcoords="offset points",
                color=INK_SECONDARY, fontsize=8.5, va="center")
    _style(ax, "B · Per-ticker daily volatility", "tickers", "standard deviation of daily log return")

    # C: the signal the paper's construction is about to throw away.
    ax = axes[2]
    shade_crises(ax)
    ax.plot(dates, rho_bar, color=ACCENT, lw=0.8)
    ax.legend(loc="upper left")
    _style(ax, "C · Mean pairwise correlation per 30-day window — the signal fixed density discards",
           "mean off-diagonal ρ", _stamp_label(stamp))
    return _finish(fig, save)


# ── Phase 2 ────────────────────────────────────────────────────────────────

def plot_network_construction(dates, tau, n_isolated, n_below_neg_tau,
                              rho_bar=None, stamp="t2", save=None):
    """Four-panel Phase 2 figure: what fixed-density thresholding does."""
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 11.0), sharex=True)

    ax = axes[0]
    shade_crises(ax)
    ax.plot(dates, tau, color=ACCENT, lw=0.8, label="τ(t)")
    ax.legend(loc="upper left")
    _style(ax, "A · The threshold τ(t) that holds edge density at 10 %",
           "τ (correlation cutoff)")

    ax = axes[1]
    shade_crises(ax, label=False)
    ax.plot(dates, n_isolated, color=INFERRED, lw=0.8)
    _style(ax, "B · Isolated nodes (degree 0) — the Fig. 3B crisis signature",
           "stocks with no edges")

    ax = axes[2]
    shade_crises(ax, label=False)
    ax.plot(dates, n_below_neg_tau, color=COMMUNITY, lw=0.8)
    _style(ax, "C · Pairs with ρ < −τ — strong anticorrelations the construction cannot see",
           "invisible pairs")

    ax = axes[3]
    shade_crises(ax, label=False)
    if rho_bar is not None:
        ax.plot(dates, rho_bar, color=CONFIG, lw=0.8)
    _style(ax, "D · Mean pairwise correlation, for comparison with τ(t)",
           "mean off-diagonal ρ", _stamp_label(stamp))
    return _finish(fig, save)


# ── Phase 3 ────────────────────────────────────────────────────────────────

# The three modularities are not three entities — they are one quantity at
# three levels of partition staleness, so they take an ordinal ramp of the
# single "inferred network" hue rather than three categorical slots. Fresh is
# darkest. Steps 650/450/250 of the blue ramp; 250 is the documented ordinal
# floor on a light surface (2.06:1), so the palest line still reads.
STALENESS = ("#104281", "#2a78d6", "#86b6ef")   # dynamical / lagged / fixed


def plot_modularity_overview(dates, q_dyn, k, seed_lo=None, seed_hi=None,
                             stamp="t2", save=None):
    """Fig. 4 companion: the modularity series with its Louvain noise band.

    The band is ±1 sd across the ten Louvain seeds on each day. Drawing it is
    the whole point: the paper reports a single Louvain run per day, and any
    daily feature narrower than this band is an artifact of the algorithm.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)

    ax = axes[0]
    shade_crises(ax)
    if seed_lo is not None:
        ax.fill_between(dates, seed_lo, seed_hi, color=INFERRED, alpha=0.35, lw=0,
                        label="±1 sd across 10 Louvain seeds")
    ax.plot(dates, q_dyn, color=STALENESS[0], lw=0.7, label="dynamical Q(t), seed 0")
    ax.legend(loc="upper left", ncol=3)
    _style(ax, "A · Dynamical modularity — with the Louvain seed spread it is usually read without",
           "modularity Q")

    ax = axes[1]
    shade_crises(ax, label=False)
    ax.plot(dates, k, color=ACCENT, lw=0.7)
    _style(ax, "B · Number of communities k(t) — singletons make it spike when the network shatters",
           "communities detected", _stamp_label(stamp))
    return _finish(fig, save)


def plot_three_modularities(dates, q_dyn, q_lagged, q_fixed, q_diff,
                            lag=100, stamp="t2", save=None):
    """Fig. 5: the same graphs scored with partitions of three different ages."""
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True)

    ax = axes[0]
    shade_crises(ax)
    for series, colour, label in (
            (q_fixed, STALENESS[2], "fixed — C(0), first day's partition"),
            (q_lagged, STALENESS[1], f"lagged — C(t − {lag})"),
            (q_dyn, STALENESS[0], "dynamical — C(t), re-detected daily")):
        ax.plot(dates, series, color=colour, lw=0.7, label=label)
    ax.legend(loc="upper left", ncol=2)
    _style(ax, "A · One modularity formula, three partition ages", "modularity Q")

    ax = axes[1]
    shade_crises(ax, label=False)
    ax.plot(dates, q_diff, color=ACCENT, lw=0.7)
    ax.axhline(0, color=INK_SECONDARY, lw=0.8, ls=":")
    _style(ax, f"B · Q_dyn − Q_lagged — modularity that exists only because communities are allowed to move",
           "Q gained by re-detecting", _stamp_label(stamp))
    return _finish(fig, save)


def plot_partition_decay(lags, q_by_lag, floor, q_real, q_noise, lag_marker=100,
                         save=None):
    """What a modularity number is worth: how fast it decays, and its floor.

    The noise reference is drawn in neutral grey rather than a categorical hue,
    following the same rule as the crisis bands: it is context to read the real
    series against, not a fourth entity competing with it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

    ax = axes[0]
    lags = np.asarray(lags, dtype=float)
    ax.axhline(floor, color=CRISIS_SHADE, lw=1.2, ls="--",
               label=f"random partition, same sizes ({floor:+.3f})")
    ax.plot(lags, q_by_lag, color=INFERRED, lw=1.6, marker="o", ms=4,
            label="Q(gₜ, C(t − L))")
    ax.axvline(lag_marker, color=INK, lw=1.0, ls=":")
    ax.annotate(f"the paper's t_Δ = {lag_marker}", xy=(lag_marker, q_by_lag[0] * 0.85),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, lags.max() * 1.15)   # lag 0 exists; negative ticks do not
    ax.legend(loc="lower left")
    _style(ax, "A · Community structure has a half-life of days, not months",
           "modularity Q", "partition age L (trading days, symlog)")

    ax = axes[1]
    bins = np.linspace(min(q_noise.min(), q_real.min()) - 0.01,
                       max(q_noise.max(), q_real.max()) + 0.01, 70)
    ax.hist(q_noise, bins=bins, color=CRISIS_SHADE, alpha=0.55, lw=0,
            density=True, label="i.i.d. Gaussian returns (A4 reference)")
    ax.hist(q_real, bins=bins, color=INFERRED, histtype="step", lw=1.4,
            density=True, label="real market networks")
    ax.legend(loc="upper right")
    _style(ax, "B · The level of Q is what noise gives; only its spread is market",
           "density", "dynamical modularity Q")
    return _finish(fig, save)


def plot_event_zoom(dates, series_map, start, end, title, ylabel_map=None,
                    onset=None, save=None):
    """Zoom several series onto one event window, one panel per series."""
    import pandas as pd
    dates = pd.DatetimeIndex(dates)
    sel = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    n = len(series_map)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 2.3 * n + 0.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, values) in zip(axes, series_map.items()):
        colour = FAMILY_COLOR.get(name, ACCENT)
        ax.plot(dates[sel], np.asarray(values)[sel], color=colour, lw=1.3)
        if onset is not None:
            ax.axvline(pd.Timestamp(onset), color=INK, lw=1.0, ls=":")
        _style(ax, name, (ylabel_map or {}).get(name))
    if onset is not None:
        axes[0].annotate(f"onset {onset}", xy=(pd.Timestamp(onset), 1.0),
                         xycoords=("data", "axes fraction"),
                         xytext=(4, -10), textcoords="offset points",
                         fontsize=8.5, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    return _finish(fig, save)
