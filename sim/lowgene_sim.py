"""
lowgene_sim.py -- simulation study for the MoMa meta-analysis (ASAP-030013).

Question: how can low-abundance genes (TYR, MAOB, MAOA, ALDH1A1) be measured and
related to neuronal vulnerability in human substantia nigra (SN) dopaminergic (DA)
neurons and locus coeruleus (LC) noradrenergic (NE) neurons, given single-nucleus,
spatial and bulk RNA-seq data?

Everything here is SYNTHETIC. No CRN data are read. The parameters are calibrated
to structural facts of the Lee SN dataset observed in Workbench (counts of nuclei,
sequencing depth per nucleus; no expression values):
  - marker-defined DA nuclei: 217 in total over 25 donors (median ~5 per donor,
    max 30; only 5 donors with >= 20)
  - DA nuclei are deep: ~11,000 UMIs per nucleus (per-donor median 12,000)
  - the pipeline's curated object keeps 3,000 highly-variable genes; none of the
    panel genes (TH, TYR, MAOA, MAOB, ALDH1A1, ...) is in it
  - the 'unfiltered' object has 1.3 M barcodes with median 59 UMIs (77% below 500)

Sections (each produces one figure in sim/figures/):
  S1  per-nucleus detection vs depth; 'expressed at all?' test against a noise floor
  S2  why HVG selection drops these genes (cohort-wide Pearson-residual HVG with batches)
  S3  the QC / ambient trap: unfiltered barcodes, ambient MAOB, ambient-aware estimate
  S4  donor-level power, PD vs control, for SN and LC capture scenarios
  S5  subtype contrast within controls (vulnerable vs resistant DA subtypes) vs pooled nuclei
  S6  noise amplification traps: binarising a sparse gene; nucleus-level pseudoreplication
  S7  GeoMx TH-masked ROI designs (SN as in Edwards/Halliday; a hypothetical LC design)
  S8  bulk tissue: composition confound and the TYR/TH ratio

Run:  python sim/lowgene_sim.py [--fast]
Only numpy / scipy / pandas / matplotlib are needed (ASCII-only source on purpose).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# global parameters
# ----------------------------------------------------------------------------

RNG = np.random.default_rng(20261030)

# Relative abundance = fraction of a nucleus' UMIs coming from the gene, in the
# target neuron (DA or NE). These are PRIORS, to be replaced by measured values
# from notebook 01 (fraction detected and UMI per nucleus in DA nuclei).
GENE_P = {
    "TH":       5e-3,   # ~50 UMIs per 10k-UMI DA nucleus
    "ALDH1A1":  1e-3,   # ~10
    "MAOA":     3e-4,   # ~3
    "MAOB":     1e-4,   # ~1 in DA/NE neurons (astrocytes ~15x higher)
    "TYR_opt":  1e-5,   # ~0.1   optimistic
    "TYR_mid":  2e-6,   # ~0.02  plausible
    "TYR_pes":  5e-7,   # ~0.005 pessimistic
}
TYR_LEVELS = ["TYR_opt", "TYR_mid", "TYR_pes"]
PANEL = ["ALDH1A1", "MAOA", "MAOB", "TYR_opt", "TYR_mid", "TYR_pes"]

# Okabe-Ito (colour-blind safe), fixed order
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F"}
GENE_COLORS = {"ALDH1A1": C["green"], "MAOA": C["blue"], "MAOB": C["sky"],
               "TYR_opt": C["orange"], "TYR_mid": C["red"], "TYR_pes": C["purple"], "TH": C["grey"]}

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
RESULTS: list[dict] = []


def record(section, **kw):
    kw["section"] = section
    RESULTS.append(kw)


def lognormal(median, sigma, size, rng=RNG):
    return rng.lognormal(np.log(median), sigma, size)


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.relpath(path, HERE))


# ----------------------------------------------------------------------------
# small GLM toolkit (Poisson IRLS with offset, quasi-Poisson Wald test)
# ----------------------------------------------------------------------------

def poisson_irls(X, y, offset, max_iter=50, tol=1e-8):
    """Poisson GLM with log link. Returns (beta, cov_unscaled, pearson_chi2, df_resid)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(max(y.sum(), 0.5) / np.exp(offset).sum())
    for _ in range(max_iter):
        eta = X @ beta + offset
        mu = np.exp(np.clip(eta, -30, 30))
        W = mu
        z = eta - offset + (y - mu) / np.maximum(mu, 1e-12)
        XtW = X.T * W
        try:
            H = XtW @ X
            beta_new = np.linalg.solve(H + 1e-10 * np.eye(p), XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta + offset
    mu = np.exp(np.clip(eta, -30, 30))
    H = (X.T * mu) @ X
    cov = np.linalg.pinv(H)
    chi2 = np.sum((y - mu) ** 2 / np.maximum(mu, 1e-12))
    return beta, cov, chi2, max(n - p, 1)


def qp_test(X, y, offset, k):
    """Quasi-Poisson Wald test for coefficient k (t distribution, df = n - p).
    Dispersion is floored at 1 (never 'under-dispersed')."""
    if y.sum() == 0:
        return 1.0, 0.0
    beta, cov, chi2, df = poisson_irls(X, y, offset)
    scale = max(chi2 / df, 1.0)
    se = np.sqrt(max(cov[k, k], 1e-12) * scale)
    t = beta[k] / se
    return 2 * stats.t.sf(abs(t), df), beta[k]


def two_group_test(y, exposure, group):
    """y: counts per unit (donor/ROI); exposure: UMIs per unit; group 0/1."""
    X = np.column_stack([np.ones(len(y)), group])
    return qp_test(X, y, np.log(exposure), 1)


def bg_poisson_test(y, exposure, bg, x):
    """Counts y ~ Poisson(bg + exp(b0 + b1 x) * exposure) with KNOWN additive background bg
    (negative-probe expectation). Likelihood-ratio test of b1 = 0, scaled by a quasi-Poisson
    dispersion estimated from Pearson residuals (units = donors)."""
    from scipy.optimize import minimize
    y = np.asarray(y, float); e = np.asarray(exposure, float); bg = np.asarray(bg, float); x = np.asarray(x, float)
    if y.sum() == 0:
        return 1.0, 0.0

    def nll(b, with_slope):
        mu = bg + np.exp(np.clip(b[0] + (b[1] * x if with_slope else 0.0), -40, 40)) * e
        return np.sum(mu - y * np.log(mu))
    b0 = np.log(max((y - bg).sum(), 0.5) / e.sum())
    f0 = minimize(lambda b: nll(b, False), [b0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=2000))
    f1 = minimize(lambda b: nll(b, True), [f0.x[0], 0.0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    lrt = max(2 * (f0.fun - f1.fun), 0.0)
    mu1 = bg + np.exp(f1.x[0] + f1.x[1] * x) * e
    chi2 = np.sum((y - mu1) ** 2 / np.maximum(mu1, 1e-9))
    scale = max(chi2 / max(len(y) - 2, 1), 1.0)
    return stats.chi2.sf(lrt / scale, 1), f1.x[1]


# ----------------------------------------------------------------------------
# S1  detection per nucleus vs depth; expressed-at-all test
# ----------------------------------------------------------------------------

def s1_detection():
    print("S1 detection vs depth")
    p_grid = np.logspace(-7, -2, 120)
    depth_scen = {
        "sn, shallow 'Unknown' barcode (median 300 UMI)": 300,
        "sn, typical nucleus (median 3,000 UMI)": 3000,
        "sn, DA/NE neuron nucleus (median 10,000 UMI)": 10000,
        "sc, whole large neuron (median 40,000 UMI)": 40000,
    }
    cols = [C["grey"], C["sky"], C["blue"], C["green"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for (name, med), col in zip(depth_scen.items(), cols):
        d = lognormal(med, 0.6, 20000)
        pdet = np.array([np.mean(1 - np.exp(-p * d)) for p in p_grid])
        ax.plot(p_grid, pdet, color=col, lw=2, label=name)
        for g in PANEL:
            pd_g = np.mean(1 - np.exp(-GENE_P[g] * d))
            record("S1", scenario=name, gene=g, p_detect=round(float(pd_g), 4),
                   umi_per_nucleus=round(float(GENE_P[g] * med), 4))
    for g in PANEL:
        ax.axvline(GENE_P[g], color=GENE_COLORS[g], ls=":", lw=1)
        ax.text(GENE_P[g], 1.02, g, rotation=90, fontsize=7, color=GENE_COLORS[g],
                ha="center", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("true relative abundance (fraction of UMIs in the neuron)")
    ax.set_ylabel("P(>= 1 UMI in a nucleus)")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.set_title("A. per-nucleus detection is set by depth x abundance", fontsize=10, loc="left")

    # expressed-at-all test: DA nuclei pooled (depth 10k each) vs a noise floor
    ax = axes[1]
    floor = 1e-7       # spurious counts per UMI in non-expressing nuclei (ambient/mismapping)
    n_nuclei = np.array([50, 217, 765, 1000, 5000, 22000, 60000])
    depth = 10000
    for g in ["TYR_opt", "TYR_mid", "TYR_pes", "MAOB"]:
        lam = n_nuclei * depth * (GENE_P[g] + floor)
        lam0 = n_nuclei * depth * floor
        # power of a one-sided Poisson test at alpha 0.001 vs the floor
        crit = stats.poisson.ppf(0.999, lam0)
        power = stats.poisson.sf(crit, lam)
        ax.plot(n_nuclei, power, "-o", color=GENE_COLORS[g], lw=2, ms=5, label=g)
        for n, pw, l in zip(n_nuclei, power, lam):
            record("S1b", gene=g, n_nuclei=int(n), expected_umis=round(float(l), 2),
                   power_vs_floor=round(float(pw), 3))
    for n, lab, yy in [(217, "Lee SN DA\n(217)", 0.05), (765, "public LC NE\n(~765)", 0.30),
                       (22000, "Kamath FANS\n(~22k)", 0.05)]:
        ax.axvline(n, color=C["grey"], ls="--", lw=1)
        ax.text(n, yy, lab, fontsize=7, ha="center", color=C["grey"],
                bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xscale("log")
    ax.set_xlabel("pooled DA/NE nuclei (10,000 UMIs each)")
    ax.set_ylabel("power to show expression above noise floor")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_title("B. 'is it expressed at all?' needs pooled nuclei, not depth", fontsize=10, loc="left")
    fig.suptitle("S1  Sparse genes: detection per nucleus vs detection in a pooled population", fontsize=11)
    savefig(fig, "S1_detection_vs_depth.png")


# ----------------------------------------------------------------------------
# S2  HVG selection drops low / rare-cell-type genes
# ----------------------------------------------------------------------------

def pearson_residual_var(X, theta=100.0):
    """Analytic Pearson residual variance per gene (Lause et al. 2021), clip sqrt(n)."""
    n = X.shape[0]
    tot = X.sum()
    mu = np.outer(X.sum(1), X.sum(0)) / tot
    r = (X - mu) / np.sqrt(mu + mu ** 2 / theta + 1e-12)
    r = np.clip(r, -np.sqrt(n), np.sqrt(n))
    return r.var(0)


def s2_hvg(fast=False):
    print("S2 HVG selection")
    n_batches, n_cells, n_genes = (8, 1200, 5000) if fast else (12, 1500, 6000)
    rng = np.random.default_rng(2)
    # baseline gene abundance distribution (heavy-tailed: most genes sparse), normalised
    base = rng.lognormal(-5.5, 2.2, n_genes)
    base /= base.sum()
    types = ["oligo", "astro", "micro", "neuron", "da"]
    props_sn = np.array([0.55, 0.20, 0.08, 0.16, 0.01])
    props_ctx = np.array([0.45, 0.20, 0.08, 0.27, 0.00])
    # cell-type marker structure: half of all genes carry some specificity (fold 1.5x - 30x)
    prof = np.tile(base, (len(types), 1))
    n_mark = n_genes // 10
    for t in range(len(types)):
        idx = rng.choice(n_genes, n_mark, replace=False)
        prof[t, idx] *= np.exp(rng.uniform(np.log(1.5), np.log(30), n_mark))
    # panel genes appended: TH-like, ALDH1A1-like, MAOA-like, MAOB-like, TYR-like
    panel = {"TH": [0, 0, 0, 0, 5e-3], "ALDH1A1": [0, 1e-4, 0, 0, 1e-3],
             "MAOA": [0, 0, 0, 5e-5, 3e-4], "MAOB": [1e-5, 1.5e-3, 0, 1e-5, 1e-4],
             "TYR_opt": [0, 0, 0, 0, 1e-5], "TYR_mid": [0, 0, 0, 0, 2e-6]}
    names = [f"g{i}" for i in range(n_genes)] + list(panel)
    prof = np.hstack([prof, np.array([panel[k] for k in panel]).T])
    prof /= prof.sum(1, keepdims=True)
    depth_med = {"oligo": 2500, "astro": 3000, "micro": 1500, "neuron": 8000, "da": 10000}
    n_sel = int(round(len(names) * 3000 / 36601))   # same fraction as the pipeline: 3,000 of 36,601 genes
    sel_count = np.zeros(len(names))
    rank_sum = np.zeros(len(names))
    means = np.zeros(len(names))
    for b in range(n_batches):
        props = props_sn if b < n_batches // 3 else props_ctx    # a third of samples are SN
        ct = rng.choice(len(types), n_cells, p=props)
        d = np.array([lognormal(depth_med[types[t]], 0.5, 1, rng)[0] for t in ct])
        lam = prof[ct] * d[:, None]
        X = rng.poisson(lam).astype(float)
        v = pearson_residual_var(X)
        order = np.argsort(-v)
        rank = np.empty(len(names)); rank[order] = np.arange(len(names))
        sel_count[order[:n_sel]] += 1
        rank_sum += rank
        means += X.mean(0) / n_batches
    # scanpy batch_key behaviour: rank by number of batches selected, then by mean rank
    final_order = np.lexsort((rank_sum, -sel_count))
    final_sel = np.zeros(len(names), bool); final_sel[final_order[:n_sel]] = True
    df = pd.DataFrame({"gene": names, "mean_count": means, "n_batches_selected": sel_count,
                       "selected_final": final_sel})
    # summary: selection probability by mean-count bin (background genes only)
    bg = df[~df["gene"].isin(panel)]
    bins = np.logspace(-4, 2, 13)
    bg = bg.assign(bin=pd.cut(bg["mean_count"], bins))
    frac = bg.groupby("bin", observed=True)["selected_final"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    centers = [b.mid for b in frac.index]
    ax.plot(centers, frac.values, "-o", color=C["blue"], lw=2, ms=5, label="background genes (binned)")
    offsets = {"TH": (6, -14), "ALDH1A1": (6, 8), "MAOA": (-70, 10), "MAOB": (6, 10), "TYR_opt": (6, 8), "TYR_mid": (-70, 8)}
    for g in panel:
        row = df[df["gene"] == g].iloc[0]
        yv = 1.0 if row["selected_final"] else 0.0
        xv = max(row["mean_count"], 1e-4)
        ax.scatter([xv], [yv], s=60, color=GENE_COLORS.get(g, C["grey"]), zorder=5)
        ax.annotate(f"{g} ({int(row['n_batches_selected'])}/{n_batches})", (xv, yv),
                    textcoords="offset points", xytext=offsets[g], fontsize=7, color=GENE_COLORS.get(g, C["grey"]))
        record("S2", gene=g, mean_count=round(float(row["mean_count"]), 5),
               batches_selected=int(row["n_batches_selected"]), in_final_hvg=bool(row["selected_final"]))
    ax.set_xscale("log")
    ax.set_xlabel("mean count per nucleus across the cohort")
    ax.set_ylabel(f"P(gene kept among the {n_sel} HVGs of {len(names)})")
    ax.set_ylim(-0.08, 1.15)
    ax.set_title("S2  Cohort-wide HVG selection (per-batch Pearson residuals, ranked by number of batches):\n"
                 "survival depends on cohort-wide variance, not on the level in the neurons of interest (label: batches selected)",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    savefig(fig, "S2_hvg_selection_drops_panel.png")


# ----------------------------------------------------------------------------
# S3  QC / ambient / empty-droplet trap
# ----------------------------------------------------------------------------

def s3_qc_ambient(fast=False):
    print("S3 QC and ambient trap")
    rng = np.random.default_rng(3)
    genes = ["TH", "SLC6A3", "SLC18A2", "DBH", "TYR", "MAOA", "MAOB", "ALDH1A1", "AQP4"]
    gi = {g: i for i, g in enumerate(genes)}
    types = ["da", "neuron", "astro", "oligo", "micro"]
    props = np.array([0.012, 0.15, 0.20, 0.55, 0.088])
    size = np.array([10000, 8000, 3000, 2500, 1500])   # median depth per type
    # relative abundance per type (rows) x gene (cols); 'rest' absorbs the remainder
    P = np.zeros((5, len(genes)))
    P[0] = [5e-3, 2e-3, 2e-3, 0, 1e-5, 3e-4, 1e-4, 1e-3, 0]          # DA
    P[1] = [0, 0, 0, 0, 0, 5e-5, 1e-5, 0, 0]                           # other neuron
    P[2] = [0, 0, 0, 0, 0, 2e-5, 1.5e-3, 1e-4, 2e-3]                   # astro
    P[3] = [0, 0, 0, 0, 0, 1e-5, 5e-5, 0, 0]                           # oligo
    P[4] = [0, 0, 0, 0, 0, 1e-5, 2e-5, 0, 0]                           # micro
    # ambient pool = RNA-weighted average of cell types (what leaks into every droplet)
    w = props * size; w /= w.sum()
    amb = w @ P
    n_true = 4000 if fast else 8000
    n_empty = 3 * n_true
    reps = 20 if fast else 40
    rows = []
    for r in range(reps):
        ct = rng.choice(5, n_true, p=props)
        d = np.array([lognormal(size[t], 0.5, 1, rng)[0] for t in ct])
        rho = rng.beta(2, 18, n_true)                       # ambient fraction per nucleus (~10%)
        lam = (1 - rho)[:, None] * P[ct] * d[:, None] + rho[:, None] * amb[None, :] * d[:, None]
        X = rng.poisson(lam)
        d_e = lognormal(150, 0.7, n_empty, rng)
        Xe = rng.poisson(amb[None, :] * d_e[:, None])
        is_da_true = np.r_[ct == 0, np.zeros(n_empty, bool)]
        Xall = np.vstack([X, Xe]); dall = np.r_[d, d_e]
        th = Xall[:, gi["TH"]]; dat = np.maximum(Xall[:, gi["SLC6A3"]], Xall[:, gi["SLC18A2"]]); dbh = Xall[:, gi["DBH"]]
        qc = dall >= 500
        # marker thresholds: lenient (1 UMI) vs scaled to what a real DA nucleus carries (~50 TH UMIs)
        for thr_name, t_th, t_dat in [("TH>=1 & DAT/VMAT2>=1", 1, 1), ("TH>=3 & DAT/VMAT2>=2", 3, 2), ("TH>=10 & DAT/VMAT2>=3", 10, 3)]:
            c = (th >= t_th) & (dat >= t_dat) & (dbh == 0)
            for sname, m in [("unfiltered barcodes", c), ("QC-passed nuclei", c & qc)]:
                rows.append(dict(rep=r, set=sname, threshold=thr_name, kind="threshold",
                                 precision=is_da_true[m].mean() if m.sum() else np.nan,
                                 recall=(m & is_da_true).sum() / max(is_da_true.sum(), 1)))
        call = (th >= 1) & (dat >= 1) & (dbh == 0)
        for name, mask in [("unfiltered barcodes", call), ("QC-passed nuclei", call & qc)]:
            n_called = mask.sum(); prec = is_da_true[mask].mean() if n_called else np.nan
            maob_det = (Xall[mask, gi["MAOB"]] >= 1).mean()
            maob_amb_exp = np.mean(1 - np.exp(-0.1 * amb[gi["MAOB"]] * dall[mask]))
            tyr_cpm = Xall[mask, gi["TYR"]].sum() / dall[mask].sum() * 1e6
            # ambient-aware estimate: subtract expected ambient contribution (rho * amb * depth)
            tyr_cpm_corr = (Xall[mask, gi["TYR"]].sum() - 0.1 * amb[gi["TYR"]] * dall[mask].sum()) / dall[mask].sum() * 1e6
            maob_cpm = Xall[mask, gi["MAOB"]].sum() / dall[mask].sum() * 1e6
            maob_cpm_corr = (Xall[mask, gi["MAOB"]].sum() - 0.1 * amb[gi["MAOB"]] * dall[mask].sum()) / dall[mask].sum() * 1e6
            rows.append(dict(rep=r, set=name, kind="lenient", n_called=int(n_called), precision=prec,
                             maob_detected=maob_det, maob_ambient_expected=maob_amb_exp,
                             tyr_cpm=tyr_cpm, tyr_cpm_corr=tyr_cpm_corr,
                             maob_cpm=maob_cpm, maob_cpm_corr=maob_cpm_corr))
    df_all = pd.DataFrame(rows)
    thr = df_all[df_all["kind"] == "threshold"].groupby(["set", "threshold"])[["precision", "recall"]].mean()
    for (sname, tname), rr in thr.iterrows():
        record("S3a", set=sname, threshold=tname, precision=round(float(rr["precision"]), 3), recall=round(float(rr["recall"]), 3))
    df = df_all[df_all["kind"] == "lenient"].drop(columns=["kind", "threshold"])
    truth_tyr = P[0, gi["TYR"]] * 1e6
    truth_maob = P[0, gi["MAOB"]] * 1e6
    s = df.groupby("set").agg(["mean", "std"])
    for name in ["unfiltered barcodes", "QC-passed nuclei"]:
        record("S3", set=name, n_called=round(float(s.loc[name, ("n_called", "mean")]), 1),
               da_call_precision=round(float(s.loc[name, ("precision", "mean")]), 3),
               maob_detected_frac=round(float(s.loc[name, ("maob_detected", "mean")]), 3),
               maob_detected_expected_from_ambient=round(float(s.loc[name, ("maob_ambient_expected", "mean")]), 3),
               tyr_cpm=round(float(s.loc[name, ("tyr_cpm", "mean")]), 2),
               tyr_cpm_truth=round(truth_tyr, 2),
               maob_cpm=round(float(s.loc[name, ("maob_cpm", "mean")]), 1),
               maob_cpm_ambient_corrected=round(float(s.loc[name, ("maob_cpm_corr", "mean")]), 1),
               maob_cpm_truth=round(truth_maob, 1))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    sets = ["unfiltered barcodes", "QC-passed nuclei"]
    x = np.arange(2)
    ax = axes[0]
    thr_names = ["TH>=1 & DAT/VMAT2>=1", "TH>=3 & DAT/VMAT2>=2", "TH>=10 & DAT/VMAT2>=3"]
    wd = 0.26
    for j, tname in enumerate(thr_names):
        pr = [thr.loc[(k, tname), "precision"] for k in sets]
        rc = [thr.loc[(k, tname), "recall"] for k in sets]
        ax.bar(x + (j - 1) * wd, pr, wd, color=[C["red"], C["blue"], C["green"]][j], label=tname)
        for xi, (p_, r_) in enumerate(zip(pr, rc)):
            ax.text(xi + (j - 1) * wd, 0.03, f"recall {r_:.2f}", ha="center", va="bottom", fontsize=6, color="white", rotation=90)
    ax.set_xticks(x, sets, fontsize=8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("precision of marker-based DA call")
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, title="marker threshold", title_fontsize=6.5)
    ax.set_title("A. ambient TH/DAT makes 1-UMI calls impure;\nreal DA nuclei carry ~50 TH UMIs, so raise the bar", fontsize=9.5, loc="left")
    ax = axes[1]
    wd = 0.35
    ax.bar(x - wd / 2, [s.loc[k, ("maob_detected", "mean")] for k in sets], wd, color=C["sky"], label="MAOB detected (>= 1 UMI)")
    ax.bar(x + wd / 2, [s.loc[k, ("maob_ambient_expected", "mean")] for k in sets], wd, color=C["grey"], label="expected from ambient alone")
    ax.set_xticks(x, sets, fontsize=8); ax.set_ylabel("fraction of called DA nuclei")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("B. astrocytic MAOB leaks into DA droplets", fontsize=10, loc="left")
    ax = axes[2]
    ax.bar(x - wd / 2, [s.loc[k, ("maob_cpm", "mean")] for k in sets], wd, color=C["sky"], label="naive pseudobulk CPM")
    ax.bar(x + wd / 2, [s.loc[k, ("maob_cpm_corr", "mean")] for k in sets], wd, color=C["blue"], label="ambient-corrected")
    ax.axhline(truth_maob, color=C["red"], ls="--", lw=1.5, label="truth in DA neurons")
    ax.set_xticks(x, sets, fontsize=8); ax.set_ylabel("MAOB CPM in called DA nuclei")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("C. subtracting the ambient expectation recovers the truth", fontsize=10, loc="left")
    fig.suptitle("S3  'Raw' must mean all genes for QC-passed nuclei, not all barcodes; ambient has to be modelled", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "S3_qc_ambient_trap.png")


# ----------------------------------------------------------------------------
# S4  donor-level power: PD vs control
# ----------------------------------------------------------------------------

SCENARIOS = {
    # name: (n_ctrl, n_pd, median n nuclei/donor, sigma, depth median)
    "SN unsorted, Lee (8 ctrl / 15 PD, ~5 DA/donor)":        (8, 15, 5, 1.1, 10000),
    "SN unsorted, CRN pooled (25 / 40, ~5 DA/donor)":        (25, 40, 5, 1.1, 10000),
    "SN FANS-enriched, Kamath-like (8 / 10, ~1,200 DA/donor)": (8, 10, 1200, 0.6, 10000),
    "SN mega-pool (40 / 50, mixed 5 and 1,200 DA/donor)":    (40, 50, None, None, 10000),
    "LC unsorted pons (10 / 10, ~40 NE/donor)":              (10, 10, 40, 0.8, 10000),
    "LC enriched FANS/LCM (10 / 10, ~400 NE/donor)":         (10, 10, 400, 0.6, 10000),
}


def sim_cohort(rng, n_ctrl, n_pd, med_n, sigma, depth_med, p, fc, bio_cv=0.3, floor=1e-7):
    """Donor-level pseudobulk counts for one gene. Returns (counts, exposures, group)."""
    n_d = n_ctrl + n_pd
    group = np.r_[np.zeros(n_ctrl), np.ones(n_pd)]
    if med_n is None:   # mega-pool: 18 FANS donors (8 ctrl, 10 PD), rest unsorted
        n_nuc = np.r_[lognormal(5, 1.1, n_ctrl, rng), lognormal(5, 1.1, n_pd, rng)]
        n_nuc[:8] = lognormal(1200, 0.6, 8, rng); n_nuc[n_ctrl:n_ctrl + 10] = lognormal(1200, 0.6, 10, rng)
    else:
        n_nuc = lognormal(med_n, sigma, n_d, rng)
    n_nuc = np.maximum(np.round(n_nuc), 1).astype(int)
    p_donor = p * rng.lognormal(0, bio_cv, n_d) * np.where(group == 1, fc, 1.0)
    counts = np.zeros(n_d); expo = np.zeros(n_d)
    for i in range(n_d):
        d = lognormal(depth_med, 0.6, n_nuc[i], rng)
        expo[i] = d.sum()
        counts[i] = rng.poisson((p_donor[i] + floor) * expo[i])
    return counts, expo, group, n_nuc


def s4_power(fast=False):
    print("S4 donor-level power PD vs control")
    rng = np.random.default_rng(4)
    reps = 150 if fast else 400
    fcs = [0.5, 0.7]
    res = {}
    for fc in fcs:
        for sname, (nc, npd, medn, sig, dep) in SCENARIOS.items():
            for g in PANEL:
                hits = 0
                for r in range(reps):
                    y, e, grp, _ = sim_cohort(rng, nc, npd, medn, sig, dep, GENE_P[g], fc)
                    pval, _ = two_group_test(y, e, grp)
                    hits += pval < 0.05
                res[(fc, sname, g)] = hits / reps
                record("S4", fold_change=fc, scenario=sname, gene=g, power=round(hits / reps, 3))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, fc in zip(axes, fcs):
        M = np.array([[res[(fc, s, g)] for g in PANEL] for s in SCENARIOS])
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(PANEL)), PANEL, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(SCENARIOS)), list(SCENARIOS), fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.6 else "black")
        ax.set_title(f"fold change in PD survivors = {fc}", fontsize=10, loc="left")
    fig.colorbar(im, ax=axes, fraction=0.02, label="power (alpha 0.05, quasi-Poisson, donor = unit)")
    fig.suptitle("S4  Donor-level pseudobulk power, PD vs control (biological CV 30%, depth 10k UMIs/nucleus)", fontsize=11)
    savefig(fig, "S4_power_pd_vs_control.png")


# ----------------------------------------------------------------------------
# S5  subtype contrast within controls vs pooled nuclei
# ----------------------------------------------------------------------------

def s5_subtype(fast=False):
    print("S5 subtype contrast")
    rng = np.random.default_rng(5)
    reps = 150 if fast else 400
    totals = [200, 1000, 5000, 20000, 60000]
    n_donors = 10
    fc = 2.0     # gene 2x higher in the vulnerable (SOX6+/AGTR1+) than the resistant (CALB1+) subtype
    frac_vuln = 0.6
    out = {}
    for g in PANEL:
        pw = []
        for tot in totals:
            hits = 0
            for r in range(reps):
                n_per = np.maximum(np.round(lognormal(tot / n_donors, 0.5, n_donors, rng)), 2).astype(int)
                y = []; e = []; sub = []; don = []
                for dd in range(n_donors):
                    pdon = GENE_P[g] * rng.lognormal(0, 0.3)
                    nv = rng.binomial(n_per[dd], frac_vuln); nr = n_per[dd] - nv
                    for s_id, n_s, mult in [(1, nv, fc), (0, nr, 1.0)]:
                        if n_s == 0:
                            continue
                        d = lognormal(10000, 0.6, n_s, rng)
                        e.append(d.sum()); y.append(rng.poisson(pdon * mult * d.sum()))
                        sub.append(s_id); don.append(dd)
                y = np.array(y, float); e = np.array(e); sub = np.array(sub); don = np.array(don)
                # donor fixed effects + subtype
                X = np.column_stack([np.ones(len(y)), sub] + [(don == k).astype(float) for k in range(1, n_donors)])
                pval, _ = qp_test(X, y, np.log(e), 1)
                hits += pval < 0.05
            pw.append(hits / reps)
            record("S5", gene=g, pooled_nuclei=tot, power=round(hits / reps, 3))
        out[g] = pw
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for g in PANEL:
        ax.plot(totals, out[g], "-o", color=GENE_COLORS[g], lw=2, ms=5, label=g)
    ax.axhline(0.8, color=C["grey"], ls=":", lw=1)
    for n, lab in [(217, "Lee"), (22000, "Kamath")]:
        ax.axvline(n, color=C["grey"], ls="--", lw=1); ax.text(n, 0.02, lab, fontsize=7, ha="center", color=C["grey"])
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("pooled DA nuclei from control donors (10 donors, 10k UMIs each)")
    ax.set_ylabel("power (2-fold subtype difference, donor-blocked)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_title("S5  Vulnerable vs resistant DA subtype within control brains:\nTYR-level genes need 10^4 - 10^5 pooled nuclei", fontsize=10, loc="left")
    savefig(fig, "S5_subtype_power_vs_pooled_nuclei.png")


# ----------------------------------------------------------------------------
# S6  noise-amplification traps
# ----------------------------------------------------------------------------

def s6_traps(fast=False):
    print("S6 binarisation and pseudoreplication traps")
    rng = np.random.default_rng(6)
    reps = 150 if fast else 400
    n_cells = 5000
    p_tyr = GENE_P["TYR_opt"]      # 'TYR+' call on >= 1 UMI (10% of nuclei at 10k UMIs)
    p_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]   # abundance of the SECOND gene (no true association)
    tests = ["Wilcoxon on log1p(CPM)", "detection-rate chi2", "Poisson GLM + log(depth) offset", "depth-matched Wilcoxon"]
    fpr = {t: [] for t in tests}
    for p_g in p_grid:
        hits = dict.fromkeys(tests, 0)
        for r in range(reps):
            d = lognormal(10000, 0.8, n_cells, rng)
            tyr = rng.poisson(p_tyr * d) >= 1
            g = rng.poisson(p_g * d)
            if tyr.sum() < 5 or (~tyr).sum() < 5 or g.sum() == 0:
                continue
            cpm = np.log1p(g / d * 1e6)
            hits["Wilcoxon on log1p(CPM)"] += stats.mannwhitneyu(cpm[tyr], cpm[~tyr]).pvalue < 0.05
            tab = np.array([[(g[tyr] > 0).sum(), (g[tyr] == 0).sum()], [(g[~tyr] > 0).sum(), (g[~tyr] == 0).sum()]]) + 0.5
            hits["detection-rate chi2"] += stats.chi2_contingency(tab)[1] < 0.05
            X = np.column_stack([np.ones(n_cells), tyr.astype(float)])
            hits["Poisson GLM + log(depth) offset"] += qp_test(X, g.astype(float), np.log(d), 1)[0] < 0.05
            # depth matching: for each TYR+ cell pick the TYR- cell with the nearest depth
            neg = np.flatnonzero(~tyr); pos = np.flatnonzero(tyr)
            order = np.argsort(d[neg]); dn = d[neg][order]
            idx = np.clip(np.searchsorted(dn, d[pos]), 0, len(dn) - 1)
            matched = neg[order][idx]
            hits["depth-matched Wilcoxon"] += stats.mannwhitneyu(cpm[pos], cpm[matched]).pvalue < 0.05
        for t in tests:
            fpr[t].append(hits[t] / reps)
            record("S6a", test=t, second_gene_abundance=p_g, false_positive_rate=round(hits[t] / reps, 3))
    # pseudoreplication: nucleus-level test PD vs ctrl with donor effects, no true difference
    n_don = 8; n_nuc = 200
    fp_cell = 0; fp_donor = 0
    for r in range(reps):
        y = []; grp = []; don = []; expo = []
        for dd in range(2 * n_don):
            pdon = GENE_P["MAOA"] * rng.lognormal(0, 0.3)
            d = lognormal(10000, 0.6, n_nuc, rng)
            y.append(rng.poisson(pdon * d)); grp.append(np.full(n_nuc, dd >= n_don)); don.append(np.full(n_nuc, dd)); expo.append(d)
        y = np.concatenate(y); grp = np.concatenate(grp); don = np.concatenate(don); expo = np.concatenate(expo)
        cpm = np.log1p(y / expo * 1e6)
        fp_cell += stats.mannwhitneyu(cpm[grp], cpm[~grp]).pvalue < 0.05
        yd = np.array([y[don == k].sum() for k in range(2 * n_don)], float)
        ed = np.array([expo[don == k].sum() for k in range(2 * n_don)])
        gd = np.array([k >= n_don for k in range(2 * n_don)], float)
        fp_donor += two_group_test(yd, ed, gd)[0] < 0.05
    record("S6b", test="nucleus-level Wilcoxon (donor ignored)", false_positive_rate=round(fp_cell / reps, 3))
    record("S6b", test="donor-level pseudobulk quasi-Poisson", false_positive_rate=round(fp_donor / reps, 3))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    cols = [C["red"], C["orange"], C["blue"], C["green"]]
    for t, col in zip(tests, cols):
        ax.plot(p_grid, fpr[t], "-o", color=col, lw=2, ms=5, label=t)
    for g in ["TYR_opt", "MAOB", "MAOA", "ALDH1A1"]:
        ax.axvline(GENE_P[g], color=GENE_COLORS[g], ls=":", lw=1)
        ax.text(GENE_P[g], 1.02, g, fontsize=7, ha="center", color=GENE_COLORS[g])
    ax.axhline(0.05, color=C["grey"], ls="--", lw=1)
    ax.set_xscale("log"); ax.set_ylim(0, 1.1)
    ax.set_xlabel("abundance of the second gene (no true association with TYR)")
    ax.set_ylabel("false-positive rate (alpha 0.05)")
    ax.legend(fontsize=7, frameon=False, loc="center left")
    ax.set_title("A. 'TYR+ vs TYR- nuclei' (TYR+ = any UMI): deeper nuclei are called TYR+,\n"
                 "so every other sparse gene looks 'co-expressed' unless depth is modelled", fontsize=9.5, loc="left")
    ax = axes[1]
    ax.bar(["nuclei as\nreplicates", "donors as\nreplicates"], [fp_cell / reps, fp_donor / reps], color=[C["red"], C["blue"]], width=0.6)
    ax.axhline(0.05, color=C["grey"], ls="--", lw=1); ax.set_ylim(0, 1)
    ax.set_ylabel("false-positive rate")
    ax.set_title("B. pseudoreplication (8 vs 8 donors,\n200 nuclei each, 30% donor CV, no true effect)", fontsize=9.5, loc="left")
    fig.suptitle("S6  How count-level analyses amplify noise, and the fixes (depth offsets or matching; donor as the unit)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "S6_noise_amplification_traps.png")


# ----------------------------------------------------------------------------
# S7  GeoMx TH-masked ROI designs
# ----------------------------------------------------------------------------

# Expected counts per gene in a ~120-cell TH+ ROI, calibrated to the public Edwards/Chatterton TH-mask
# WTA object (median raw counts per ROI at a ~1.2e5-count library: TH 22, ALDH1A1 12, MAOA 10, MAOB 9;
# TYR did not pass the limit of quantification). TYR scenarios are scaled from MAOB by the snRNA-seq
# abundance ratios in GENE_P.
GEOMX_ROI_COUNTS = {"TH": 22.0, "ALDH1A1": 12.0, "MAOA": 10.0, "MAOB": 9.0,
                    "TYR_opt": 0.9, "TYR_mid": 0.18, "TYR_pes": 0.045}
GEOMX_REF_CELLS = 120.0     # ROI size (cells) that the counts above refer to

GEOMX_DESIGNS = {
    # name: (donors per stage [ctrl, ILBD, ePD, lPD], ROIs per donor, median cells/ROI ctrl, cell multipliers per stage)
    "SN TH+ ROIs, Edwards/Halliday-like (10/10/5/5 donors, 348 ROIs)": ([10, 10, 5, 5], 12, 120, [1.0, 0.9, 0.5, 0.3]),
    "LC TH+ ROIs, Halliday existing (9 ctrl/0 ILBD/8 ePD/2 lPD, 39 ROIs)": ([9, 0, 8, 2], 2, 50, [1.0, 0.8, 0.5, 0.3]),
    "LC TH+ ROIs, proposed (8/8/8/8 donors, ~8 ROIs/donor)":           ([8, 8, 8, 8], 8, 50, [1.0, 0.8, 0.5, 0.3]),
    "LC TH+ ROIs, small pilot (5/5/5/5 donors, ~6 ROIs/donor)":        ([5, 5, 5, 5], 6, 50, [1.0, 0.8, 0.5, 0.3]),
}


def s7_geomx(fast=False):
    print("S7 GeoMx designs")
    rng = np.random.default_rng(7)
    reps = 150 if fast else 400
    bg_mean = 6.0       # negative-probe-like background counts per gene per ROI (assumption; use the NegProbe geomean)
    res = {}
    for stage_fc in [0.8, 0.6]:      # per-stage multiplicative change (trend effect)
      for dname, (dons, rois, cells_med, mult) in GEOMX_DESIGNS.items():
        for g in PANEL:
            hits_trend = 0; hits_ilbd = 0
            for r in range(reps):
                y = []; expo = []; stage = []; don = []
                k = 0
                for s_idx, n_d in enumerate(dons):
                    for dd in range(n_d):
                        pdon = GEOMX_ROI_COUNTS[g] / GEOMX_REF_CELLS * rng.lognormal(0, 0.3) * stage_fc ** s_idx
                        for rr in range(rois):
                            cells = max(lognormal(cells_med * mult[s_idx], 0.5, 1, rng)[0], 3)
                            lam = pdon * cells + bg_mean
                            y.append(rng.poisson(lam)); expo.append(cells); stage.append(s_idx); don.append(k)
                        k += 1
                y = np.array(y, float); expo = np.array(expo); stage = np.array(stage, float); don = np.array(don)
                # donor-level aggregation (donor = unit); background enters the model additively,
                # it must NOT be treated as scaling with the number of cells in the ROI
                nd = int(don.max()) + 1
                yd = np.array([y[don == i].sum() for i in range(nd)])
                ed = np.array([expo[don == i].sum() for i in range(nd)])
                bd = np.array([bg_mean * (don == i).sum() for i in range(nd)])
                sd = np.array([stage[don == i][0] for i in range(nd)])
                hits_trend += bg_poisson_test(yd, ed, bd, sd)[0] < 0.05
                m = sd <= 1
                hits_ilbd += bg_poisson_test(yd[m], ed[m], bd[m], sd[m])[0] < 0.05
            res[(stage_fc, dname, g)] = (hits_trend / reps, hits_ilbd / reps)
            record("S7", per_stage_fc=stage_fc, design=dname, gene=g, power_stage_trend=round(hits_trend / reps, 3),
                   power_ilbd_vs_ctrl=round(hits_ilbd / reps, 3))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharey=True, sharex=True)
    for row, stage_fc in enumerate([0.8, 0.6]):
        for col, (title, idx) in enumerate([("stage trend over 4 stages", 0), ("ILBD vs control only", 1)]):
            ax = axes[row, col]
            M = np.array([[res[(stage_fc, d, g)][idx] for g in PANEL] for d in GEOMX_DESIGNS])
            im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(PANEL)), PANEL, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(GEOMX_DESIGNS)), list(GEOMX_DESIGNS), fontsize=8)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.6 else "black")
            ax.set_title(f"{title}, {round((1 - stage_fc) * 100)}% change per stage", fontsize=10, loc="left")
    fig.colorbar(im, ax=axes, fraction=0.02, label="power (donor = unit; background modelled additively)")
    fig.suptitle("S7  TH-masked GeoMx ROIs = deep pseudobulk of the right neurons, with disease stage\n"
                 f"(calibrated to the public TH-mask object: MAOA ~10, MAOB ~9, ALDH1A1 ~12 counts per ~120-cell ROI; "
                 f"background {bg_mean:.0f} counts/ROI/gene)", fontsize=11)
    savefig(fig, "S7_geomx_designs.png")


# ----------------------------------------------------------------------------
# S8  bulk tissue: composition confound and ratio estimator
# ----------------------------------------------------------------------------

def s8_bulk(fast=False):
    print("S8 bulk composition confound")
    rng = np.random.default_rng(8)
    reps = 200 if fast else 500
    reads = 40e6
    n_ctrl = n_pd = 20
    # DA neuron share of tissue RNA: ~9% in control SN, ~3% in PD
    share_ctrl, share_pd = 0.09, 0.03
    p_th = GENE_P["TH"]
    out = []
    for g in ["TYR_opt", "TYR_mid", "MAOA", "ALDH1A1"]:
        for true_fc in [1.0, 0.5]:
            fp_bulk = 0; hit_ratio = 0
            for r in range(reps):
                share = np.r_[share_ctrl * rng.lognormal(0, 0.4, n_ctrl), share_pd * rng.lognormal(0, 0.4, n_pd)]
                grp = np.r_[np.zeros(n_ctrl), np.ones(n_pd)]
                pcell = GENE_P[g] * rng.lognormal(0, 0.3, n_ctrl + n_pd) * np.where(grp == 1, true_fc, 1.0)
                y = rng.poisson(reads * share * pcell)              # gene reads from DA neurons only
                th = rng.poisson(reads * share * p_th)
                # naive bulk test: gene vs library size
                fp_bulk += two_group_test(y.astype(float), np.full(len(y), reads), grp)[0] < 0.05
                # ratio test: gene vs TH exposure (per-DA-neuron expression)
                hit_ratio += two_group_test(y.astype(float), np.maximum(th, 1).astype(float), grp)[0] < 0.05
            out.append((g, true_fc, fp_bulk / reps, hit_ratio / reps))
            record("S8", gene=g, true_per_cell_fc=true_fc, bulk_naive_significant=round(fp_bulk / reps, 3),
                   bulk_ratio_to_TH_significant=round(hit_ratio / reps, 3))
    df = pd.DataFrame(out, columns=["gene", "true_fc", "naive", "ratio"])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(df)); wd = 0.38
    ax.bar(x - wd / 2, df["naive"], wd, color=C["red"], label="naive bulk (gene vs library size)")
    ax.bar(x + wd / 2, df["ratio"], wd, color=C["blue"], label="gene vs TH (per-DA-neuron ratio)")
    ax.set_xticks(x, [f"{g}\nFC {f}" for g, f in zip(df["gene"], df["true_fc"])], fontsize=8)
    ax.set_xlabel("gene and TRUE per-DA-neuron fold change in PD (FC 1.0 = no change)")
    ax.axhline(0.05, color=C["grey"], ls="--", lw=1)
    ax.set_ylabel("fraction of cohorts called 'significant'")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("S8  Bulk SN (20 vs 20, 40M reads): DA-neuron loss alone makes every DA gene 'down in PD';\n"
                 "normalising to TH removes the composition effect but leaves TYR Poisson-limited", fontsize=10, loc="left")
    savefig(fig, "S8_bulk_composition_confound.png")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer replicates (smoke test)")
    ap.add_argument("--only", default="", help="comma-separated sections, e.g. S1,S4")
    args = ap.parse_args()
    only = set(s.strip().upper() for s in args.only.split(",") if s.strip())
    t0 = time.time()
    steps = [("S1", lambda: s1_detection()), ("S2", lambda: s2_hvg(args.fast)),
             ("S3", lambda: s3_qc_ambient(args.fast)), ("S4", lambda: s4_power(args.fast)),
             ("S5", lambda: s5_subtype(args.fast)), ("S6", lambda: s6_traps(args.fast)),
             ("S7", lambda: s7_geomx(args.fast)), ("S8", lambda: s8_bulk(args.fast))]
    for name, fn in steps:
        if only and name not in only:
            continue
        t1 = time.time(); fn(); print(f"  ({time.time() - t1:.0f}s)")
    df = pd.DataFrame(RESULTS)
    out = os.path.join(HERE, "sim_results.csv")
    df.to_csv(out, index=False)
    print(f"wrote {os.path.relpath(out, HERE)} ({len(df)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
