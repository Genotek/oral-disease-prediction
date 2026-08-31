#!/usr/bin/env python3
"""Figure 3: differentially abundant taxa and pathways.

Layout: half violins with a median tick, a grid of squares by cohort, bars for
-log10(q) and for the coefficient, and a volcano panel.

Two modes share one layout:
  --mode pooled  -> Figure 3, main text. Pooled analysis, the screening step.
  --mode meta    -> Figure S3, supplementary. Within-cohort effects combined by
                    meta-analysis, the confirmatory step; I2 is printed to the
                    right of each coefficient bar.

Panels A and F show within-cohort significance:
  --consistency p  (default) nominal within-cohort p, uncorrected. The panel
                   describes consistency of direction, it does not test hypotheses.
  --consistency q  BH correction across the family shown in the panel.
Whichever is used has to be named in the caption.

Top-5 ranking:
  --rank q       by q value;
  --rank effect  by effect magnitude among significant features. Needed because
                 there are 86 significant caries pathways, and ranking by q
                 brings up features that are significant but negligible in size
                 (PENTOSE-P-PWY, beta = -0.066).

Everything is read from results/; nothing is recomputed here.

Run:  python figures/figure_3.py --mode both
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]        # repository root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (DATA_DIR, TAX_CSV, PATH_CSV, META_CSV, ML_DIR, MODELS_DIR,
                   RESULTS_DIR, FIGURES_DIR, PERMANOVA_CSV, require_data)
sys.path.insert(0, str(ROOT))
from common import benjamini_hochberg, read_csv_with_run    # noqa: E402

warnings.filterwarnings("ignore")
RES = RESULTS_DIR
OUT = FIGURES_DIR

TOP, SIG, PSEUDO = 5, 0.05, 1e-6
PALETTE = {"Healthy": "#7DAF86", "K05": "#7B80C0", "K02": "#C9A86C"}
C_H, C_NS = PALETTE["Healthy"], "#CECECE"
BG_TAX, BG_PATH = "#F4F4FA", "#FAF9F3"
FS = {"row_title": 14, "col_title": 12, "axis_label": 10.5, "tick": 8.5,
      "ytick": 10.5, "small": 7.5, "star": 7.0, "legend": 10}

STUDY = {
    "PRJEB36291": "Caselli, 2020", "PRJEB42701": "Plachokova, 2021",
    "PRJNA1057503": "Schmartz, 2024", "PRJNA1162519": "Bao, 2025",
    "PRJNA1379526": "PRJNA1379526", "PRJNA380711": "Wang, 2019",
    "PRJNA396840": "Belstrøm, 2017", "PRJNA478018": "Baker, 2021",
    "PRJNA678453": "Belstrøm, 2021", "PRJNA717815": "Saito, 2024",
    "PRJNA741688": "Sun, 2021", "PRJNA752888": "Blostein, 2022",
    "PRJNA929548": "Wu, 2026", "PRJNA932553": "Koohi-Moghadam, 2024",
}
ROW_TITLE = {"K05": "Periodontal disease status (K05)", "K02": "Caries (K02)"}

mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.linewidth": 0.6, "savefig.bbox": "tight"})


def fmt_taxon(x: str) -> str:
    p = str(x).replace("_", " ").split()
    return f"{p[0][0]}. {' '.join(p[1:])}" if len(p) >= 2 else str(x)


def fmt_label(raw: str, is_taxon: bool, maxlen: int = 24) -> str:
    lab = fmt_taxon(raw) if is_taxon else str(raw)
    return lab[:maxlen - 3] + "..." if len(lab) > maxlen else lab


def p2star(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if np.isnan(p) or p >= SIG:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*"


def half_violin(ax, vals, yc, side, color, w=0.40):
    """Half violin with a white median tick, as in the original notebook."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5 or (vals.max() - vals.min()) < 1e-8:
        return
    kde = gaussian_kde(vals, bw_method=0.4)
    xg = np.linspace(vals.min(), vals.max(), 200)
    den = kde(xg); den = den / den.max() * w
    lo, hi = (yc, yc + den) if side == "up" else (yc - den, yc)
    ax.fill_between(xg, lo, hi, color=color, alpha=0.30, lw=0, zorder=3)
    ax.plot(xg, yc + den if side == "up" else yc - den, color=color, lw=1.1,
            alpha=0.90, zorder=4)
    med = np.median(vals)
    ax.vlines(med, yc - w * .22, yc + w * .22, color="white", lw=2.0, zorder=5)
    ax.vlines(med, yc - w * .22, yc + w * .22, color=color, lw=0.9, alpha=.7, zorder=6)


# ------------------------------------------------------------------ loading
def load_results(task: str, space: str, mode: str) -> pd.DataFrame:
    if mode == "pooled":
        f = RES / "differential_abundance" / f"S3_{task}_{'taxa' if space=='TAX' else 'pathways'}.csv"
        d = pd.read_csv(f)
        key = d["feature"] if space == "TAX" else d["pathway_id"].str.strip()
        return pd.DataFrame({"key": key.to_numpy(), "coef": d["coef"].to_numpy(),
                             "qval": d["qval"].to_numpy(), "I2": np.nan})
    d = pd.read_csv(RES / "meta_analysis_pooled.csv")
    d = d[(d["task"] == task) & (d["space"] == space)].dropna(subset=["qval"])
    key = d["feature"] if space == "TAX" else d["feature"].str.split(":").str[0].str.strip()
    return pd.DataFrame({"key": key.to_numpy(), "coef": d["coef"].to_numpy(),
                         "qval": d["qval"].to_numpy(), "I2": d["I2"].to_numpy()})


def load_per_study(task: str, space: str) -> pd.DataFrame:
    d = pd.read_csv(RES / "per_study_effects.csv")
    d = d[(d["task"] == task) & (d["space"] == space)].copy()
    d["key"] = (d["feature"] if space == "TAX"
                else d["feature"].str.split(":").str[0].str.strip())
    return d


def abundances():
    md = read_csv_with_run(META_CSV).set_index("run")
    tax = read_csv_with_run(TAX_CSV).set_index("run")
    pth = read_csv_with_run(PATH_CSV).set_index("run")
    common = md.index.intersection(tax.index).intersection(pth.index)
    md, tax, pth = md.loc[common], tax.loc[common], pth.loc[common]
    pth = pth[[c for c in pth.columns
               if "|" not in c and c not in ("UNMAPPED", "UNINTEGRATED")]]
    for c in ("healthy", "K02_caries", "K05_gingivitis_periodontitis"):
        md[c] = pd.to_numeric(md[c], errors="coerce").fillna(0).astype(int)
    tax = tax.apply(pd.to_numeric, errors="coerce").fillna(0)
    pth = pth.apply(pd.to_numeric, errors="coerce").fillna(0)
    # relative abundance: pathways arrive as raw RPK, so both spaces are put on one scale
    tax = tax.div(tax.sum(axis=1).replace(0, 1), axis=0)
    pth = pth.div(pth.sum(axis=1).replace(0, 1), axis=0)
    pth.columns = [str(c).split(":")[0].strip() for c in pth.columns]
    return md, tax, pth


def get_ab(key, tbl, h_idx, d_idx):
    if key not in tbl.columns:
        return np.array([]), np.array([])
    h = tbl.loc[tbl.index.intersection(h_idx), key].to_numpy(float)
    d = tbl.loc[tbl.index.intersection(d_idx), key].to_numpy(float)
    return (np.log10(h[np.isfinite(h) & (h >= 0)] + PSEUDO),
            np.log10(d[np.isfinite(d) & (d >= 0)] + PSEUDO))


def pick_top(d: pd.DataFrame, rank: str) -> pd.DataFrame:
    if rank == "effect":
        s = d[d["qval"] < SIG]
        return s.reindex(s["coef"].abs().sort_values(ascending=False).index).head(TOP)
    return d.sort_values("qval").head(TOP)


# -------------------------------------------------------------------- one row
def draw_row(fig, slot, task, mode, rank, consistency, md, tax, pth,
             labels, show_col_titles=True):
    C_D = PALETTE[task]
    n = 2 * TOP
    # A sixth, empty column spaces the effect bars from the volcano: in meta mode
    # I2 is printed to the right of the bars and would otherwise overlap it.
    gap = 1.15 if mode == "meta" else 0.35
    inner = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=slot,
                                             width_ratios=[4.0, 3.0, 1.55, 1.55, gap, 6.4],
                                             wspace=0.10)
    ax_st = fig.add_subplot(inner[0])
    vg = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=inner[1],
                                          height_ratios=[TOP, TOP], hspace=0.0)
    ax_vt, ax_vp = fig.add_subplot(vg[0]), fig.add_subplot(vg[1])
    ax_q = fig.add_subplot(inner[2], sharey=ax_st)
    ax_c = fig.add_subplot(inner[3], sharey=ax_st)
    ax_v = fig.add_subplot(inner[5])

    res = {sp: load_results(task, sp, mode) for sp in ("TAX", "PATH")}
    ps = {sp: load_per_study(task, sp) for sp in ("TAX", "PATH")}
    top_tax, top_path = pick_top(res["TAX"], rank), pick_top(res["PATH"], rank)
    info = pd.concat([top_tax, top_path], ignore_index=True)
    feats = info["key"].tolist()
    y_of = {f: n - 1 - i for i, f in enumerate(feats)}

    for ax in (ax_st, ax_q, ax_c):
        ax.axhspan(TOP - .5, n - .5, color=BG_TAX, zorder=0)
        ax.axhspan(-.5, TOP - .5, color=BG_PATH, zorder=0)
    ax_vt.set_facecolor(BG_TAX); ax_vp.set_facecolor(BG_PATH)

    # ---- A: consistency across cohorts
    #
    # Multiplicity correction is applied to the family actually shown in the
    # panel: 10 selected features x 6 cohorts = 60 tests. Correcting within a
    # cohort across all ~350 features penalises tests the paper never reports and
    # leaves the panel nearly empty; nominal p is not corrected at all.
    # The features were selected on the pooled analysis of the same data, so these
    # p values are conditional on that selection either way: the panel describes
    # consistency of direction, and formal cross-cohort inference comes from the
    # meta-analysis.
    cohorts = sorted(set(ps["TAX"]["batch"]) | set(ps["PATH"]["batch"]))
    rows_ = []
    for fi, feat in enumerate(feats):
        e = ps["TAX" if fi < TOP else "PATH"]
        for b in cohorts:
            m = e[(e["key"] == feat) & (e["batch"] == b)]
            gv = lambda c: (float(m[c].iloc[0])
                            if len(m) and pd.notna(m[c].iloc[0]) else np.nan)
            rows_.append({"feat": feat, "batch": b, "pval": gv("pval"), "qval": gv("qval")})
    SH = pd.DataFrame(rows_)
    ok_ = SH["pval"].notna()
    SH["q_shown"] = np.nan
    SH.loc[ok_, "q_shown"] = benjamini_hochberg(SH.loc[ok_, "pval"].to_numpy())
    col = {"p": "pval", "q": "qval", "shown": "q_shown"}[consistency]
    print(f"  panel {labels[0]}/{labels[0]}: significant cells "
          f"{int((SH[col] < SIG).sum())} of {int(ok_.sum())} (mode {consistency})")
    for bi, b in enumerate(cohorts):
        for fi, feat in enumerate(feats):
            m = SH[(SH["feat"] == feat) & (SH["batch"] == b)]
            p = float(m[col].iloc[0]) if len(m) and pd.notna(m[col].iloc[0]) else np.nan
            cf = info.iloc[fi]["coef"]
            sig = not (np.isnan(p) or p >= SIG)
            bg = (C_D if cf > 0 else C_H) if sig else C_NS
            ax_st.scatter(bi, y_of[feat], s=250, c=bg, marker="s", linewidths=.4,
                          edgecolors="white", alpha=.92, zorder=2)
            st = p2star(p)
            if st:
                ax_st.text(bi, y_of[feat], st, ha="center", va="center",
                           fontsize=FS["star"], color="white", fontweight="bold", zorder=3)

    ys = sorted(y_of.items(), key=lambda kv: kv[1])
    ax_st.set_yticks([v for _, v in ys])
    ax_st.set_yticklabels([fmt_label(f, yp >= TOP) for f, yp in ys], fontsize=FS["ytick"])
    for lbl, (_, yp) in zip(ax_st.get_yticklabels(), ys):
        if yp >= TOP:
            lbl.set_fontstyle("italic")
    ax_st.set_xlim(-.6, len(cohorts) - .4); ax_st.set_ylim(-.6, n - .4)
    ax_st.set_xticks(range(len(cohorts)))
    ax_st.set_xticklabels([STUDY.get(b, b) for b in cohorts], rotation=-35,
                          ha="right", fontsize=FS["tick"])
    ax_st.xaxis.tick_top()
    ax_st.axhline(TOP - .5, color="#999", lw=1.1, ls="--", zorder=4)
    ax_st.tick_params(length=0, pad=3)
    for s in ax_st.spines.values():
        s.set_visible(False)
    ax_st.set_title(ROW_TITLE[task], fontsize=FS["row_title"], fontweight="bold",
                    loc="left", pad=34, color=C_D)

    # ---- B: half violins
    disease_col = "K05_gingivitis_periodontitis" if task == "K05" else "K02_caries"
    h_idx = md.index[md["healthy"] == 1]
    d_idx = md.index[md[disease_col] == 1]
    ax_vt.set_ylim(TOP - .5, n - .5); ax_vp.set_ylim(-.5, TOP - .5)
    for fi, feat in enumerate(feats):
        ax, tbl = (ax_vt, tax) if fi < TOP else (ax_vp, pth)
        hv, dv = get_ab(feat, tbl, h_idx, d_idx)
        half_violin(ax, hv, y_of[feat], "down", C_H)
        half_violin(ax, dv, y_of[feat], "up", C_D)
    for ax in (ax_vt, ax_vp):
        ax.tick_params(axis="x", labelsize=FS["tick"]); ax.tick_params(axis="y", length=0)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
    ax_vt.spines["bottom"].set_color("#AAAAAA"); ax_vt.spines["bottom"].set_linewidth(.6)
    ax_vt.spines["bottom"].set_linestyle("--")
    ax_vt.set_title("Abundance (log$_{10}$)", fontsize=FS["col_title"], pad=34, color="#333")
    ax_vp.set_xlabel(r"log$_{10}$(rel. abundance + $\varepsilon$)", fontsize=FS["axis_label"])
    ax_vt.text(.99, .96, "taxa", transform=ax_vt.transAxes, ha="right", va="top",
               fontsize=FS["small"], color="#888", style="italic")
    ax_vp.text(.99, .04, "pathways", transform=ax_vp.transAxes, ha="right", va="bottom",
               fontsize=FS["small"], color="#888", style="italic")

    # ---- C: significance
    for fi in range(n):
        r = info.iloc[fi]
        ax_q.barh(y_of[feats[fi]], -np.log10(r["qval"] + 1e-300), height=.65,
                  color=(C_D if r["coef"] > 0 else C_H), alpha=.85, zorder=2)
    ax_q.axvline(-np.log10(SIG), color="#777", lw=.8, ls="--")
    ax_q.axhline(TOP - .5, color="#999", lw=1.1, ls="--", zorder=4)
    ax_q.tick_params(axis="x", labelsize=FS["tick"]); ax_q.tick_params(axis="y", length=0)
    plt.setp(ax_q.get_yticklabels(), visible=False)
    for s in ("top", "right", "left"):
        ax_q.spines[s].set_visible(False)
    ax_q.set_title("Significance", fontsize=FS["col_title"], pad=34, color="#333")
    ax_q.set_xlabel(r"$-\log_{10}(q)$", fontsize=FS["axis_label"])

    # ---- D: effect size
    for fi in range(n):
        r = info.iloc[fi]
        ax_c.barh(y_of[feats[fi]], r["coef"], height=.65,
                  color=(C_D if r["coef"] > 0 else C_H), alpha=.85, zorder=2)
    ax_c.axvline(0, color="#555", lw=.8)
    ax_c.axhline(TOP - .5, color="#999", lw=1.1, ls="--", zorder=4)
    ax_c.tick_params(axis="x", labelsize=FS["tick"]); ax_c.tick_params(axis="y", length=0)
    plt.setp(ax_c.get_yticklabels(), visible=False)
    for s in ("top", "right", "left"):
        ax_c.spines[s].set_visible(False)
    ax_c.set_title("Effect size", fontsize=FS["col_title"], pad=34, color="#333")
    ax_c.set_xlabel(r"$\beta$", fontsize=FS["axis_label"])
    if mode == "meta":
        for fi in range(n):
            v = info.iloc[fi]["I2"]
            if pd.notna(v):
                ax_c.text(1.04, y_of[feats[fi]], f"I²={v:.0f}%",
                          transform=ax_c.get_yaxis_transform(), va="center",
                          ha="left", fontsize=FS["small"], color="#666")

    # ---- E: volcano
    lab_pool = []
    for sp, mk, sz in (("TAX", "o", 15), ("PATH", "D", 12)):
        d = res[sp]
        q = np.clip(d["qval"].to_numpy(float), 1e-300, 1)
        y, x = -np.log10(q), d["coef"].to_numpy(float)
        s_ = q < SIG
        ax_v.scatter(x[~s_], y[~s_], s=sz * .55, c=C_NS, marker=mk, lw=0, alpha=.55, zorder=1)
        ax_v.scatter(x[s_ & (x > 0)], y[s_ & (x > 0)], s=sz, c=C_D, marker=mk,
                     lw=.2, edgecolors="white", alpha=.9, zorder=2)
        ax_v.scatter(x[s_ & (x <= 0)], y[s_ & (x <= 0)], s=sz, c=C_H, marker=mk,
                     lw=.2, edgecolors="white", alpha=.9, zorder=2)
        lab_pool.append(d[s_].assign(sp=sp))
    ax_v.axhline(-np.log10(SIG), color="#777", lw=.8, ls="--")
    ax_v.axvline(0, color="#999", lw=.6)
    ax_v.set_xlabel(r"$\beta$", fontsize=FS["axis_label"])
    ax_v.set_ylabel(r"$-\log_{10}(q)$", fontsize=FS["axis_label"])
    ax_v.tick_params(labelsize=FS["tick"])
    for s in ("top", "right"):
        ax_v.spines[s].set_visible(False)
    ax_v.set_title("Volcano", fontsize=FS["col_title"], pad=34, color="#333")

    # Four labels in each quadrant: taxa and pathways, enriched in disease and in
    # health. Selected by effect magnitude among the significant features.
    N_LAB = 4
    lab = pd.concat(lab_pool).copy()
    lab["_y"] = -np.log10(lab["qval"].clip(lower=1e-300))
    # Ranked by distance from the origin rather than by effect alone: the corners
    # of the cloud are what should be labelled. Ranking by |beta| misses points
    # with a moderate effect but very high significance, and the reverse.
    mx = max(lab["coef"].abs().max(), 1e-9)
    my = max(lab["_y"].max(), 1e-9)
    lab["_d"] = np.hypot(lab["coef"].abs() / mx, lab["_y"] / my)
    picks = []
    for sp in ("TAX", "PATH"):
        d = lab[lab["sp"] == sp]
        picks.append(d[d["coef"] > 0].nlargest(N_LAB, "_d"))
        picks.append(d[d["coef"] <= 0].nlargest(N_LAB, "_d"))
    picked = pd.concat(picks).drop_duplicates(subset="key")
    # Labels are repelled separately for taxa and pathways: pathways get a larger
    # horizontal offset, so the two sets sit in different columns and need half as
    # much vertical spreading.
    span_x = max(lab["coef"].max() - lab["coef"].min(), 1e-9)
    span_y = max(picked["_y"].max() - picked["_y"].min(), 1e-9)
    for side in (1, -1):
        for sp, xmul in (("TAX", 0.05), ("PATH", 0.20)):
            sub = picked[(np.sign(picked["coef"]).replace(0, -1) == side)
                         & (picked["sp"] == sp)].sort_values("_y")
            if sub.empty:
                continue
            prev, shift = -np.inf, {}
            for k, yv in zip(sub["key"], sub["_y"]):
                dy = max(0.0, span_y * 0.075 - (yv - prev))
                prev = yv + dy
                shift[k] = dy
            for _, r in sub.iterrows():
                t = fmt_taxon(r["key"]) if sp == "TAX" else r["key"]
                col = C_D if side > 0 else C_H
                ax_v.annotate(
                    t,
                    xy=(r["coef"], r["_y"]),
                    xytext=(r["coef"] + side * span_x * xmul, r["_y"] + shift[r["key"]]),
                    textcoords="data",
                    fontsize=6.4, ha="left" if side > 0 else "right", va="center",
                    color=col, style="italic" if sp == "TAX" else "normal", zorder=6,
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=col, alpha=.65,
                                    shrinkA=0, shrinkB=2.2))
    # margin so labels are not clipped
    xr = ax_v.get_xlim(); pad = (xr[1] - xr[0]) * 0.40
    ax_v.set_xlim(xr[0] - pad, xr[1] + pad)

    for ax, l in zip((ax_st, ax_vt, ax_q, ax_c, ax_v), labels):
        ax.text(-0.02, 1.16 if ax is ax_st else 1.10, f"({l})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom", ha="right")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pooled", "meta"], default="pooled")
    ap.add_argument("--rank", choices=["q", "effect"], default="q")
    ap.add_argument("--consistency", choices=["shown", "p", "q"], default="shown",
                    help="correction family for panels A/F")
    a = ap.parse_args()

    md, tax, pth = abundances()
    fig = plt.figure(figsize=(25, 13.5), facecolor="white")
    gs = fig.add_gridspec(2, 1, hspace=0.30)
    i1 = draw_row(fig, gs[0], "K05", a.mode, a.rank, a.consistency, md, tax, pth, list("ABCDE"))
    i2 = draw_row(fig, gs[1], "K02", a.mode, a.rank, a.consistency, md, tax, pth, list("FGHIJ"))

    handles = [Patch(fc=PALETTE["K05"], ec="none", label="Enriched in periodontal disease"),
               Patch(fc=PALETTE["K02"], ec="none", label="Enriched in caries"),
               Patch(fc=C_H, ec="none", label="Enriched in health"),
               Patch(fc=C_NS, ec="none", label="Not significant"),
               Line2D([], [], marker="o", color="none", markerfacecolor="0.45",
                      markersize=6, label="Taxon"),
               Line2D([], [], marker="D", color="none", markerfacecolor="0.45",
                      markersize=5.5, label="Pathway")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=FS["legend"], bbox_to_anchor=(0.5, -0.005))

    name = "Figure3" if a.mode == "pooled" else "Figure_S_META"
    if a.rank == "effect":
        name += "_by_effect"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=400 if ext == "png" else None)
    print(f"written: {OUT}/{name}.png|svg|pdf   "
          f"(mode={a.mode}, rank={a.rank}, panel A by {a.consistency})")

    t = pd.concat([i1.assign(task="K05"), i2.assign(task="K02")])
    t.to_csv(OUT / f"{name}_top_features.csv", index=False)
    print(t[["task", "key", "coef", "qval"]].to_string(
        index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
