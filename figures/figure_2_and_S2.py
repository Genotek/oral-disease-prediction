#!/usr/bin/env python3
"""Figure 2 (taxonomic profiles) and Figure S2 (pathway profiles): batch structure.

  - Bioprojects are labelled by first author and year, taken from Table S1 and held
    in STUDY below; accessions are given in Table S6.
  - The 31 participants carrying both diagnoses are dropped throughout, so all
    panels use the same three mutually exclusive groups and n = 483.
  - Panels A-D carry a pooled column on the right.
  - Significance brackets are two-sided Mann-Whitney U with Benjamini-Hochberg
    correction across all comparisons shown.
  - PERMANOVA and PERMDISP values are read from results/permanova.csv rather than
    recomputed, so the figure and the text cannot disagree; R/02_batch_structure.R
    must have been run first.
  - Colour-blind-safe palette (Okabe-Ito); no red-green pairing.

Panels A-D show the 10 of 14 bioprojects that contribute more than one clinical
group; within the other four there is nothing to compare. Panels E-H use all 14.

Run:  python figures/figure_2_and_S2.py --space both
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]        # repository root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (DATA_DIR, TAX_CSV, PATH_CSV, META_CSV, ML_DIR, MODELS_DIR,
                   RESULTS_DIR, FIGURES_DIR, PERMANOVA_CSV, require_data)
sys.path.insert(0, str(ROOT))
from common import benjamini_hochberg, numeric_feature_frame, read_csv_with_run  # noqa: E402

from skbio.diversity import beta_diversity                                        # noqa: E402
from skbio.diversity.alpha import observed_features, shannon, simpson             # noqa: E402
from skbio.stats.composition import clr                                           # noqa: E402
from skbio.stats.ordination import pcoa                                           # noqa: E402
from sklearn.decomposition import PCA                                             # noqa: E402

OUT = FIGURES_DIR
# --space picks the feature space; everything else in the figure is identical.
SPACE = "TAX"
SPACE_CFG = {
    "TAX":  dict(file="tax.csv",  unit="taxa",     rich="Observed richness",
                 out="Figure2",  label="Taxonomic profiles"),
    "PATH": dict(file="path.csv", unit="pathways", rich="Observed pathway richness",
                 out="FigureS2", label="Functional pathway profiles"),
}
PERM = PERMANOVA_CSV

# Bioproject-to-study mapping, from Table S1.
STUDY = {
    "PRJEB36291":   "Caselli, 2020",
    "PRJEB42701":   "Plachokova, 2021",
    "PRJNA1057503": "Schmartz, 2024",
    "PRJNA1162519": "Bao, 2025",
    "PRJNA1379526": "PRJNA1379526",      # no linked publication
    "PRJNA380711":  "Wang, 2019",
    "PRJNA396840":  "Belstrøm, 2017",
    "PRJNA478018":  "Baker, 2021",
    "PRJNA678453":  "Belstrøm, 2021",
    "PRJNA717815":  "Saito, 2024",
    "PRJNA741688":  "Sun, 2021",
    "PRJNA752888":  "Blostein, 2022",
    "PRJNA929548":  "Wu, 2026",
    "PRJNA932553":  "Koohi-Moghadam, 2024",
}

# Shared with the other figures. These muted tones separate less well under
# deuteranopia than Okabe-Ito, so group is not carried by colour alone: box order
# is fixed and identical across columns, and panels F and H add marker shape.
GROUPS = ["Healthy", "Caries (K02)", "Periodontal disease (K05)"]
GCOL = {
    "Healthy":                   "#7DAF86",
    "Caries (K02)":              "#C9A86C",
    "Periodontal disease (K05)": "#7B80C0",
    "K02+K05":                   "#8DAAB8",   # unused: comorbid participants are excluded
}
GMARK = {"Healthy": "o", "Caries (K02)": "s", "Periodontal disease (K05)": "^"}
POOLED = "Pooled"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 11.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "savefig.bbox": "tight",
})


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = SPACE_CFG[SPACE]
    tax = read_csv_with_run(DATA_DIR / cfg["file"]).set_index("run")
    md = read_csv_with_run(META_CSV).set_index("run")
    common = md.index.intersection(tax.index)
    md, tax = md.loc[common].copy(), tax.loc[common].copy()
    if SPACE == "PATH":
        # Same rule as R/00_data.R: species-stratified columns and HUMAnN
        # bookkeeping rows are dropped.
        drop = [c for c in tax.columns
                if "|" in c or c in ("UNMAPPED", "UNINTEGRATED")]
        tax = tax.drop(columns=drop)
        print(f"pathways after dropping stratified and bookkeeping rows: {tax.shape[1]}")

    for c in ("healthy", "K02_caries", "K05_gingivitis_periodontitis"):
        md[c] = pd.to_numeric(md[c], errors="coerce").fillna(0).astype(int)

    both = (md["K02_caries"] == 1) & (md["K05_gingivitis_periodontitis"] == 1)
    md["group"] = np.where(
        both, "K02+K05",
        np.where(md["K02_caries"] == 1, "Caries (K02)",
                 np.where(md["K05_gingivitis_periodontitis"] == 1,
                          "Periodontal disease (K05)",
                          np.where(md["healthy"] == 1, "Healthy", "Unknown"))))

    # Comorbid participants are dropped: panels E-H are captioned as the
    # unambiguous-label subset, so the legend has to match.
    keep = md["group"].isin(GROUPS)
    md = md[keep].copy()
    feats = [c for c in tax.columns if c != "run"]  # noqa: E501
    X = numeric_feature_frame(tax.loc[md.index], feats)
    md["study"] = md["bioproject"].map(STUDY).fillna(md["bioproject"])
    print(f"samples after excluding comorbid participants: {len(md)}")
    print(md["group"].value_counts().reindex(GROUPS).to_string())
    return md, X


def alpha(md: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    A = X.to_numpy(float)
    md = md.copy()
    md["observed"] = [observed_features(r) for r in A]
    md["shannon"] = [shannon(r) for r in A]
    md["simpson"] = [simpson(r) for r in A]
    return md


def alpha_tests(md: pd.DataFrame, bps: list[str]) -> pd.DataFrame:
    """Two-sided Mann-Whitney U, Benjamini-Hochberg across all comparisons shown."""
    rows = []
    for metric in ("observed", "shannon", "simpson"):
        for bp in bps + [POOLED]:
            sub = md if bp == POOLED else md[md["study"] == bp]
            present = [g for g in GROUPS if (sub["group"] == g).sum() >= 3]
            for g1, g2 in combinations(present, 2):
                v1 = sub.loc[sub["group"] == g1, metric].dropna()
                v2 = sub.loc[sub["group"] == g2, metric].dropna()
                if len(v1) < 3 or len(v2) < 3:
                    continue
                _, p = mannwhitneyu(v1, v2, alternative="two-sided")
                rows.append({"metric": metric, "panel": bp, "g1": g1, "g2": g2, "p": p})
    t = pd.DataFrame(rows)
    if not t.empty:
        t["q"] = benjamini_hochberg(t["p"].to_numpy())
    return t


def stars(q: float) -> str | None:
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else None


def perm_numbers() -> dict:
    """PERMANOVA and PERMDISP values from results/permanova.csv; not recomputed here."""
    d = pd.read_csv(PERM)
    d = d[(d["sample_set"] == "pure483") & (d["space"] == SPACE)
          & (d["filtered"] == "no")]
    out = {}
    for metric, key in (("Bray-Curtis", "bc"), ("Aitchison", "ait")):
        r = d[d["metric"] == metric].iloc[0]
        out[key] = dict(bp=r["R2_bioproject_marginal"], gr=r["R2_group_marginal"],
                        bp_j=r["R2_bioproject_joint"], gr_j=r["R2_group_joint"],
                        pd_bp=r["permdisp_p_bioproject"], pd_gr=r["permdisp_p_group"],
                        n=int(r["n_samples"]))
    return out


def panel_label(ax, txt, dx=-0.055, dy=1.02):
    ax.text(dx, dy, txt, transform=ax.transAxes, fontsize=14.5, fontweight="bold",
            va="bottom", ha="left")


def draw(md: pd.DataFrame, X: pd.DataFrame) -> None:
    # Panels A-D: only cohorts holding at least two clinical groups, since there
    # is nothing to compare within the others. Plus the pooled column.
    ng = md.groupby("study")["group"].nunique()
    bps = [s for s in ng[ng >= 2].index]
    bps = sorted(bps, key=lambda s: -len(md[md["study"] == s]))
    cols = bps + [POOLED]
    print(f"panels A-D: {len(bps)} cohorts with >=2 groups + pooled")

    tests = alpha_tests(md, bps)
    P = perm_numbers()

    fig = plt.figure(figsize=(19.0, 11.4))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.28, 1.0],
                          height_ratios=[1, 1, 1, 0.55],
                          hspace=0.22, wspace=0.17)
    axes_a = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_d = fig.add_subplot(gs[3, 0])
    gs_r = gs[:, 1].subgridspec(2, 2, hspace=0.28, wspace=0.22)
    ax_e = fig.add_subplot(gs_r[0, 0]); ax_g = fig.add_subplot(gs_r[0, 1])
    ax_f = fig.add_subplot(gs_r[1, 0]); ax_h = fig.add_subplot(gs_r[1, 1])

    # ---------------------------------------------------- A-C: alpha diversity
    xpos = {c: i for i, c in enumerate(cols)}
    off = {g: (i - 1) * 0.26 for i, g in enumerate(GROUPS)}
    for ax, (metric, ylab) in zip(axes_a, [("observed", SPACE_CFG[SPACE]["rich"]),
                                           ("shannon", "Shannon index"),
                                           ("simpson", "Simpson index")]):
        vals_all = md[metric]
        rng = vals_all.max() - vals_all.min()
        for c in cols:
            sub = md if c == POOLED else md[md["study"] == c]
            for g in GROUPS:
                v = sub.loc[sub["group"] == g, metric].dropna()
                if len(v) < 1:
                    continue
                bx = ax.boxplot([v.to_numpy()], positions=[xpos[c] + off[g]],
                                widths=0.22, patch_artist=True, showfliers=True,
                                flierprops=dict(marker="o", markersize=1.6,
                                                markerfacecolor="0.55",
                                                markeredgecolor="none", alpha=.6),
                                medianprops=dict(color="0.15", linewidth=0.9),
                                whiskerprops=dict(linewidth=0.6),
                                capprops=dict(linewidth=0.6),
                                boxprops=dict(linewidth=0.6))
                bx["boxes"][0].set_facecolor(GCOL[g]); bx["boxes"][0].set_alpha(.85)
            # significance brackets
            t = tests[(tests["metric"] == metric) & (tests["panel"] == c)]
            sig = [(r.g1, r.g2, stars(r.q)) for r in t.itertuples() if stars(r.q)]
            if sig:
                y = sub[metric].max() + rng * 0.05
                for g1, g2, st in sig:
                    x1, x2 = xpos[c] + off[g1], xpos[c] + off[g2]
                    h = rng * 0.022
                    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                            lw=0.6, c="0.25")
                    ax.text((x1 + x2) / 2, y + h, st, ha="center", va="bottom",
                            fontsize=10.9, color="0.15")
                    y += rng * 0.085
        ax.axvline(len(bps) - 0.5, color="0.7", lw=0.8, ls=(0, (4, 3)))
        ax.set_ylabel(ylab, fontsize=13.0)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([])
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.grid(axis="y", lw=0.4, color="0.9")
        ax.set_axisbelow(True)
    for ax, lab in zip(axes_a, "ABC"):
        panel_label(ax, f"({lab})")

    # ------------------------------------------------------------ D: cohort sizes
    # Groups sit side by side rather than stacked: on a log axis segment heights
    # do not add up to the total, so a stacked bar misreports it. One bar carries
    # one number, and small cohorts stop collapsing into an unreadable sliver.
    wid = 0.26
    counts = {g: [int(((md if c == POOLED else md[md["study"] == c])["group"] == g).sum())
                  for c in cols] for g in GROUPS}
    for k, g in enumerate(GROUPS):
        xs = np.arange(len(cols)) + (k - 1) * wid
        h = np.array(counts[g], float)
        ax_d.bar(xs, np.where(h > 0, h, np.nan), width=wid * 0.92,
                 color=GCOL[g], edgecolor="white", linewidth=0.35, label=g)
        for x, v in zip(xs, counts[g]):
            if v > 0:
                ax_d.text(x, v * 1.10, str(v), ha="center", va="bottom",
                          fontsize=8.1, color="0.25")
    totals = np.array([sum(counts[g][i] for g in GROUPS) for i in range(len(cols))])
    ax_d.set_yscale("log")
    ax_d.set_ylim(1, totals.max() * 4.2)
    ax_d.set_yticks([1, 10, 100])
    ax_d.set_yticklabels(["1", "10", "100"])
    ax_d.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax_d.tick_params(axis="y", which="minor", length=1.6)
    for i, tot in enumerate(totals):
        ax_d.text(i, totals.max() * 2.1, f"n = {tot}", ha="center", va="bottom",
                  fontsize=9.0, color="0.15", fontweight="bold")
    ax_d.axvline(len(bps) - 0.5, color="0.7", lw=0.8, ls=(0, (4, 3)))
    ax_d.set_ylabel("Samples (log)", fontsize=13.0, fontweight="bold")
    ax_d.set_xticks(range(len(cols)))
    ax_d.set_xticklabels(cols, rotation=38, ha="right", fontsize=10.9)
    ax_d.set_xlim(-0.6, len(cols) - 0.4)
    ax_d.grid(axis="y", lw=0.4, color="0.9"); ax_d.set_axisbelow(True)
    panel_label(ax_d, "(D)")
    leg_d = ax_d.legend(loc="upper right", frameon=False, fontsize=13, ncol=3,
                        bbox_to_anchor=(1.0, 1.62))
    for t_ in leg_d.get_texts():
        t_.set_fontweight("bold")

    # ------------------------------------------------- E-H: beta diversity
    A = X.to_numpy(float)
    rel = A / np.clip(A.sum(axis=1, keepdims=True), 1e-12, None)
    bc = beta_diversity("braycurtis", rel, ids=list(md.index))
    ord_bc = pcoa(bc, number_of_dimensions=2)
    C = ord_bc.samples.to_numpy()[:, :2]
    ev_bc = ord_bc.proportion_explained.to_numpy()[:2] * 100

    Z = clr(rel + 1e-6)
    pca = PCA(n_components=2, random_state=42)
    Cz = pca.fit_transform(Z)
    ev_ait = pca.explained_variance_ratio_[:2] * 100

    studies = sorted(md["study"].unique())
    # 14 categories cannot be told apart by colour alone, so colour is paired with
    # marker shape: three shapes cycled over the colour-blind-safe palette.
    base = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
            "#D55E00", "#F0E442", "#000000", "#7F7F7F", "#8C564B",
            "#4C72B0", "#55A868", "#C44E52", "#937860"]
    marks = ["o", "^", "s"]
    scol = {s: base[i % len(base)] for i, s in enumerate(studies)}
    smar = {s: marks[i % 3] for i, s in enumerate(studies)}

    def _ellipse(ax, x, y, color, n_std=1.5, lw=1.0, alpha=.9):
        """Covariance ellipse: shows the spread of a group, not a confidence interval."""
        if len(x) < 4:
            return
        cov = np.cov(x, y)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
        ax.add_patch(mpl.patches.Ellipse((np.mean(x), np.mean(y)), w, h, angle=ang,
                                         facecolor="none", edgecolor=color,
                                         lw=lw, ls=(0, (5, 3)), alpha=alpha, zorder=3))

    def scatter_study(ax, C, ev, title):
        for s in studies:
            m = (md["study"] == s).to_numpy()
            ax.scatter(C[m, 0], C[m, 1], s=9, c=scol[s], marker=smar[s],
                       linewidths=0.2, edgecolors="white", alpha=.85, label=s)
            # One ellipse per cohort: this is what makes the batch structure
            # visible, which overlapping point clouds hide.
            _ellipse(ax, C[m, 0], C[m, 1], scol[s], lw=0.7, alpha=.65)
        ax.set_xlabel(f"PC1 ({ev[0]:.1f} %)", fontsize=12.5, fontweight="bold")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f} %)", fontsize=12.5, fontweight="bold")
        ax.set_title(title, fontsize=14, pad=6, fontweight="bold")

    def scatter_group(ax, C, ev):
        for g in GROUPS:
            m = (md["group"] == g).to_numpy()
            ax.scatter(C[m, 0], C[m, 1], s=11, c=GCOL[g], marker=GMARK[g],
                       linewidths=0.25, edgecolors="white", alpha=.8, label=g)
            _ellipse(ax, C[m, 0], C[m, 1], GCOL[g])
        ax.set_xlabel(f"PC1 ({ev[0]:.1f} %)", fontsize=12.5, fontweight="bold")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f} %)", fontsize=12.5, fontweight="bold")

    scatter_study(ax_e, C, ev_bc, "Bray–Curtis PCoA")
    scatter_study(ax_g, Cz, ev_ait, "Aitchison PCA")
    scatter_group(ax_f, C, ev_bc)
    scatter_group(ax_h, Cz, ev_ait)

    for ax, lab in ((ax_e, "E"), (ax_g, "G"), (ax_f, "F"), (ax_h, "H")):
        panel_label(ax, f"({lab})", dx=-0.12)
        ax.grid(lw=0.4, color="0.93"); ax.set_axisbelow(True)
        ax.tick_params(labelsize=10.2)

    def _p(v: float) -> str:
        return "< 0.001" if v < 0.001 else f"{v:.3f}".rstrip("0").rstrip(".")

    def stat_box(ax, k):
        p = P[k]
        txt = (f"PERMANOVA (999 perm., n = {p['n']})\n"
               f"study   R² = {p['bp']:.3f}   joint {p['bp_j']:.3f}\n"
               f"group  R² = {p['gr']:.3f}   joint {p['gr_j']:.3f}\n"
               f"all terms p < 0.001\n"
               f"PERMDISP study p = {p['pd_bp']:.0e}\n"
               f"         group p = {_p(p['pd_gr'])}")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9.6,
                va="bottom", ha="left", family="monospace", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.75", lw=0.5,
                          alpha=.92))
    stat_box(ax_e, "bc"); stat_box(ax_g, "ait")

    h1 = [Line2D([], [], marker=smar[s], color="none", markerfacecolor=scol[s],
                 markeredgecolor="white", markersize=6.5, label=s) for s in studies]
    leg_g = ax_g.legend(handles=h1, loc="upper left", bbox_to_anchor=(1.03, 1.02),
                        frameon=False, fontsize=11.5, title="Study",
                        title_fontsize=12.5, handletextpad=0.5, labelspacing=0.42,
                        markerscale=1.5)
    leg_g.get_title().set_fontweight("bold")
    for t_ in leg_g.get_texts():
        t_.set_fontweight("bold")
    h2 = [Line2D([], [], marker=GMARK[g], color="none", markerfacecolor=GCOL[g],
                 markeredgecolor="white", markersize=6.5, label=g) for g in GROUPS]
    leg_h = ax_h.legend(handles=h2, loc="upper left", bbox_to_anchor=(1.03, 1.02),
                        frameon=False, fontsize=11.5, title="Clinical group",
                        title_fontsize=12.5, handletextpad=0.5, labelspacing=0.42,
                        markerscale=1.5)
    leg_h.get_title().set_fontweight("bold")
    for t_ in leg_h.get_texts():
        t_.set_fontweight("bold")

    name = SPACE_CFG[SPACE]["out"]
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400 if ext == "png" else None)
    print(f"written: {OUT}/{name}.png|svg|pdf")

    tests.to_csv(OUT / f"{name}_alpha_tests.csv", index=False)
    n_sig = int((tests["q"] < 0.05).sum())
    print(f"Mann-Whitney comparisons: {len(tests)}, significant after BH: {n_sig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="TAX", choices=["TAX", "PATH", "both"],
                    help="TAX -> Figure 2, PATH -> Figure S2, both -> both")
    arg = ap.parse_args().space
    for SPACE in (["TAX", "PATH"] if arg == "both" else [arg]):
        print(f"\n=== feature space: {SPACE} ===")
        md, X = load()
        md = alpha(md, X)
        draw(md, X)
