#!/usr/bin/env python3
"""Figures 4 and 5: machine-learning panels for K05 and K02.

Panel A  ROC under class-stratified 5-fold cross-validation. Transformation and
         classifier are fixed per disease so the three curves differ only in
         feature space; otherwise the TAX / TAX+PATH / PATH comparison would be
         confounded with model choice. Study-grouped estimates are quoted in the
         text and come from ml/grid_and_schemes.py.

Panel B  Nested leave-one-study-out. The configuration is chosen by an inner
         leave-one-study-out loop over the training cohorts, so the held-out
         cohort takes no part in its own selection; selection runs separately
         within each feature space, giving the three columns. Fold sizes are read
         from the data rather than written by hand.

Panel C  SHAP for the configuration panel B selected, restricted to configurations
         in which one model input is one taxon. The full ranking is written to
         shap_feature_ranking_<task>.csv and becomes Table S18.

Run:  python figures/figure_4_and_5.py --task both
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared with every other figure in the paper; TrueType keeps the text editable
# in the vector output, which most journals require.
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.labelweight": "bold",
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (DATA_DIR, TAX_CSV, PATH_CSV, META_CSV, ML_DIR, MODELS_DIR,
                   RESULTS_DIR, FIGURES_DIR, PERMANOVA_CSV, require_data)
RES  = ML_DIR
OUT  = FIGURES_DIR
MODELS = MODELS_DIR
MODELS.mkdir(parents=True, exist_ok=True)

# --- palette ------------------------------------------------------------------
# violet -> gold, defined in palettes.py. The light end stops short of white so it
# stays visible on the page, and the red-green axis is barely used, so dichromacy
# leaves the scale readable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from palettes import get as _get_cmap            # noqa: E402
HEAT_CMAP = SHAP_CMAP = _get_cmap("violet-gold")
# ROC lines take three points of the same scale; the lightest end is skipped
# because a near-white line disappears against the page.
ROC_COLORS = {"TAX": HEAT_CMAP(0.03), "TAX+PATH": HEAT_CMAP(0.46),
              "PATH": HEAT_CMAP(0.80)}
# Line style duplicates colour so the figure survives greyscale printing.
ROC_STYLE  = {"TAX": "-", "TAX+PATH": "--", "PATH": "-."}
DIS_LABEL  = {"K05": "Periodontal disease status (K05)", "K02": "Caries (K02)"}

SPACES = ["TAX", "TAX+PATH", "PATH"]
SHAP_TOP_N = 24

# Panel A: only the feature space varies; transformation and classifier are fixed.
PANEL_A_CFG = {"K05": ("CLR", "none", "LR_Ridge"),
               "K02": ("rank.std", "none", "SVM")}

# Cohort labels: first author and year, as in Figures 2 and 3. Accessions are
# given in Table S6.
COHORT_LABEL = {
    "PRJNA1057503": "Schmartz, 2024", "PRJNA396840": "Belstrøm, 2017",
    "PRJNA678453": "Belstrøm, 2021",  "PRJNA717815": "Saito, 2024",
    "PRJNA741688": "Sun, 2021",       "PRJNA932553": "Koohi-M., 2024",
    "PRJNA380711": "Wang, 2019",      "PRJNA478018": "Baker, 2021",
    "PRJNA752888": "Blostein, 2022",  "PRJNA929548": "Wu, 2026",
}


def cohort(b: str) -> str:
    return COHORT_LABEL.get(str(b), str(b))


# ---------------------------------------------------------------- panel A
def panel_a(ax, task: str) -> None:
    """ROC under class-stratified CV; one configuration, three feature spaces."""
    preds = pd.read_csv(RES / "per_sample_predictions.csv")
    split = pd.read_csv(RES / "per_split_results.csv")
    split = split[split["status"] == "ok"]
    trf, dr, mn = PANEL_A_CFG[task]
    grid = np.linspace(0, 1, 200)

    for fs in SPACES:
        cfg = f"{fs}|{trf}|{dr}|{mn}"
        sub = preds[(preds["task"] == task) & (preds["scheme"] == "stratified_cv")
                    & (preds["config"] == cfg)]
        if sub.empty:
            continue
        met = split[(split["task"] == task) & (split["scheme"] == "stratified_cv")
                    & (split["feature_set"] == fs) & (split["transform"] == trf)
                    & (split["dimred"] == dr) & (split["model_name"] == mn)]
        tprs = []
        for _, g in sub.groupby("fold"):
            if g["y_true"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(g["y_true"], g["prob"])
            t = np.interp(grid, fpr, tpr); t[0], t[-1] = 0.0, 1.0
            tprs.append(t)
        if not tprs:
            continue
        tprs = np.asarray(tprs)
        m, s = tprs.mean(0), tprs.std(0, ddof=1)
        c = ROC_COLORS[fs]
        ax.plot(grid, m, lw=2.6, color=c, ls=ROC_STYLE[fs],
                label=f"{fs}   AUC = {met.AUC.mean():.3f} ± {met.AUC.std(ddof=1):.3f}")
        ax.fill_between(grid, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                        color=c, alpha=0.15, lw=0)

    ax.plot([0, 1], [0, 1], ":", color="#999999", lw=1.4, label="Random   AUC = 0.500")
    ax.set_xlabel("False positive rate", fontsize=16.7, fontweight="bold")
    ax.set_ylabel("True positive rate", fontsize=16.7, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    leg = ax.legend(loc="lower right", fontsize=14.4, framealpha=0.96,
                    borderpad=0.7, labelspacing=0.55)
    for t_ in leg.get_texts():
        t_.set_fontweight("bold")
    ax.grid(alpha=0.13, color="#CCCCCC")
    ax.tick_params(labelsize=13.8)
    for sp_ in ax.spines.values():
        sp_.set_edgecolor("#DDDDDD")
    ax.set_title(f"Class-stratified CV · {trf} | {mn}", fontsize=16.7, pad=9, fontweight="bold")


# ---------------------------------------------------------------- panel B
def panel_b(ax, cax, task: str) -> pd.DataFrame:
    """Nested LOSO: rows are held-out cohorts, columns are feature spaces."""
    nb = pd.read_csv(RES / "nested_by_space.csv")
    nb = nb[(nb["task"] == task) & (nb["status"] == "ok")]
    folds = sorted(nb["fold"].astype(str).unique(),
                   key=lambda f: -int(nb.loc[nb["fold"].astype(str) == f, "n_test"].iloc[0]))
    M = np.full((len(folds), len(SPACES)), np.nan)
    txt = [["" for _ in SPACES] for _ in folds]
    for i, f in enumerate(folds):
        for j, fs in enumerate(SPACES):
            r = nb[(nb["fold"].astype(str) == f) & (nb["restriction"] == fs)]
            if r.empty:
                continue
            r = r.iloc[0]
            M[i, j] = r["AUC"]
            txt[i][j] = f"AUC={r['AUC']:.3f}\nMCC={r['MCC']:.3f}"

    ax.imshow(M, cmap=HEAT_CMAP, vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(len(folds)):
        for j in range(len(SPACES)):
            if np.isnan(M[i, j]):
                continue
            # Text sits on the fill, so its colour follows cell lightness.
            col = "#FFFFFF" if M[i, j] < 0.74 else "#1A1A1A"
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=14.4,
                    color=col, fontweight="bold", linespacing=1.45)

    ylab = []
    for f in folds:
        r = nb[nb["fold"].astype(str) == f].iloc[0]
        ylab.append(f"{cohort(f)} ({int(r['n_test'])})\n"
                    f"{int(r['n_pos_test'])} cases   {int(r['n_neg_test'])} controls")
    ax.set_yticks(range(len(folds))); ax.set_yticklabels(ylab, fontsize=13.8, linespacing=1.5)
    ax.set_xticks(range(len(SPACES)))
    ax.set_xticklabels(SPACES, fontsize=19.5, fontweight="bold")
    ax.tick_params(length=0)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)

    # bottom row: mean +/- SD across folds
    rows = []
    for j, fs in enumerate(SPACES):
        v = nb[nb["restriction"] == fs]
        rows.append(f"AUC={v.AUC.mean():.3f} ± {v.AUC.std(ddof=1):.3f}\n"
                    f"MCC={v.MCC.mean():.3f} ± {v.MCC.std(ddof=1):.3f}")
    # Summary row below the heatmap. get_xaxis_transform puts x in data
    # coordinates and y in axes coordinates, hence the negative y.
    for j, s in enumerate(rows):
        ax.text(j, -0.075, s, ha="center", va="top", fontsize=14.4, linespacing=1.45,
                fontweight="bold", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(-0.62, -0.075, "Mean ± SD", ha="right", va="top", fontsize=15.5,
            fontweight="bold", transform=ax.get_xaxis_transform(), clip_on=False)

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0.4, 1.0), cmap=HEAT_CMAP),
                      cax=cax, orientation="horizontal")
    cb.set_label("AUC", fontsize=14.9, labelpad=4, fontweight="bold")
    cb.ax.tick_params(labelsize=13.2)
    return nb


# ---------------------------------------------------------------- panel C
def fit_final_model(task: str, cfg_str: str):
    """Fit the final model on all eligible samples and save it for reuse."""
    import joblib
    sys.path.insert(0, str(ROOT))
    import ml_pipeline_article_configs as base

    fs, trf, dr, mn = cfg_str.split("|")
    D = base.load_data(str(TAX_CSV),
                       str(PATH_CSV),
                       str(META_CSV))
    meta, tax_cols, path_cols = D["meta"], D["tax_cols"], D["path_cols"]
    Xtax_all, Xpath_all = D["X_tax"].to_numpy(float), D["X_path"].to_numpy(float)
    y, _, bp, keep = base.task_vectors(meta, task)
    Xtax, Xpath = Xtax_all[keep], Xpath_all[keep]

    cfg = base.Config(task=task, feature_set=fs, transform=trf, dimred=dr,
                      batch_strategy="none", model_name=mn)
    idx = np.arange(len(y))
    trb, teb, names, info = base.select_feature_blocks(
        cfg, Xtax, Xpath, idx, idx, bp, tax_cols, path_cols)
    Xt, _, _ = base.transform_blocks(trb, teb, cfg)
    probs, model, mmeta = base.tune_and_predict(Xt, y, Xt, mn, 1)

    joblib.dump({"config": cfg_str, "task": task, "model": model,
                 "feature_names": list(names), "best_params": mmeta.get("best_params"),
                 "n_samples": int(len(y)), "n_features": int(Xt.shape[1])},
                MODELS / f"final_{task}.joblib")
    print(f"  model saved -> models/final_{task}.joblib  "
          f"({Xt.shape[0]}x{Xt.shape[1]}, {mmeta.get('best_params')})")
    return model, Xt, list(names), y


def pick_interpretable(task: str):
    """Configuration for panel C: best inner-loop score among those where one
    model input corresponds to one feature.

    ILR maps D components onto D-1 coordinates, each a linear combination of log
    ratios, and PCA mixes all features; a SHAP plot over either cannot be labelled
    with species names. Selection uses mean inner-loop AUC, so the held-out
    cohorts play no part in it.
    """
    inn = pd.read_csv(RES / "inner_results.csv")
    inn = inn[(inn["status"] == "ok") & (inn["task"] == task)].copy()
    parts = inn["config"].str.split("|", expand=True)
    inn["trf"], inn["dr"] = parts[1], parts[2]
    grp = inn.groupby("config")["AUC"].mean()
    ok = inn[(inn["trf"].isin(["CLR", "rank.std"])) & (inn["dr"] == "none")]
    g2 = ok.groupby("config")["AUC"].mean().sort_values(ascending=False)
    return g2.index[0], float(g2.iloc[0]), grp.idxmax(), float(grp.max())


def wrap_label(s: str, width: int = 21) -> str:
    """Wrap long names such as "TM7 phylum sp oral taxon 351" onto two lines;
    on one line they overrun the neighbouring panel."""
    if len(s) <= width:
        return s
    words, lines, cur = s.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines[:2]) if len(lines) <= 2 else lines[0] + "\n" + " ".join(lines[1:])


def fmt_feature(name: str) -> str:
    """Abbreviate names: genus to an initial, GGB/SGB identifiers kept in full."""
    import re
    s = str(name)
    for pref in ("TAX__", "PATH__"):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    if len(s) > 3 and s[1:3] == "__":
        s = s[3:]
    if ": " in s:
        return s.split(": ")[0].strip()
    s = s.replace("_", " ").strip()
    if bool(re.match(r"^(PWY|ARGININE|GLYCO|POLY|P\d|[A-Z0-9]{2,}-)", s)):
        return s
    p = s.split()
    if len(p) < 2 or re.match(r"^(GGB|TM\d|\d)", p[0]):
        return s
    if re.match(r"^(SGB|GGB|\d)", p[1]):
        return f"{p[0]} sp."
    return f"{p[0][0]}. {' '.join(p[1:])}"


def is_pathway(name: str) -> bool:
    return str(name).startswith("PATH__")


def export_shap_ranking(task: str, names, vals, X) -> None:
    """Write the full SHAP ranking for every feature; this becomes Table S18.

    Direction is the sign of the correlation between a feature value and its SHAP
    value. Mean signed SHAP is not usable here: for a linear model it is
    identically zero, because phi = coef * (x - E[x]).
    """
    imp = np.abs(vals).mean(0).ravel()
    corr = np.zeros(len(imp))
    for j in range(vals.shape[1]):
        xj, pj = X[:, j], vals[:, j]
        if np.std(xj) > 0 and np.std(pj) > 0:
            corr[j] = np.corrcoef(xj, pj)[0, 1]
    order = np.argsort(imp)[::-1]
    fig_panel = {"K05": "4C", "K02": "5C"}[task]
    pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "feature": [str(names[i]).replace("TAX__", "").replace("PATH__", "")
                    for i in order],
        "figure_label": [fmt_feature(names[i]) for i in order],
        "feature_type": ["pathway" if is_pathway(names[i]) else "taxon" for i in order],
        "mean_abs_shap": imp[order],
        "corr_value_vs_shap": corr[order],
        "direction": np.where(corr[order] > 0,
                              "higher abundance increases predicted probability of disease",
                              "higher abundance decreases predicted probability of disease"),
        "displayed_in_figure": [f"Figure {fig_panel}" if r < SHAP_TOP_N else "no"
                                for r in range(len(order))],
    }).to_csv(RESULTS_DIR / f"shap_feature_ranking_{task}.csv", index=False)
    print(f"  SHAP ranking -> results/shap_feature_ranking_{task}.csv "
          f"({len(order)} features)")


def panel_c(ax, cax, task: str, cfg_str: str) -> None:
    import shap
    model, X, names, y = fit_final_model(task, cfg_str)
    mn = cfg_str.split("|")[3]

    if mn == "LR_Ridge":
        vals = shap.LinearExplainer(model, X).shap_values(X)
    elif mn in ("RF", "XGBoost"):
        raw = shap.TreeExplainer(model).shap_values(X)
        vals = raw[1] if isinstance(raw, list) else (
            raw[:, :, 1] if np.asarray(raw).ndim == 3 else np.asarray(raw))
    else:                                   # SVM: no exact explainer, sample instead
        rng = np.random.default_rng(42)
        bg = shap.kmeans(X, 25)
        sel = rng.choice(len(X), size=min(150, len(X)), replace=False)
        vals = shap.KernelExplainer(lambda z: model.predict_proba(z)[:, 1], bg
                                    ).shap_values(X[sel], nsamples=200, silent=True)
        X = X[sel]
    vals = np.asarray(vals)
    if vals.shape[1] != len(names):
        raise RuntimeError(
            f"{task}: {len(names)} names against {vals.shape[1]} coordinates; "
            "the transformation does not preserve the feature mapping")

    export_shap_ranking(task, names, vals, X)

    order = np.argsort(np.abs(vals).mean(0).ravel())[::-1][:SHAP_TOP_N][::-1]
    rng = np.random.default_rng(0)
    for row, fi in enumerate(order):
        v, d = vals[:, fi], X[:, fi]
        lo, hi = np.nanpercentile(d, [5, 95])
        col = np.clip((d - lo) / (hi - lo) if hi > lo else np.zeros_like(d), 0, 1)
        jitter = rng.uniform(-0.16, 0.16, size=len(v))
        ax.scatter(v, np.full(len(v), row) + jitter, c=col, cmap=SHAP_CMAP,
                   s=7.0, lw=0, alpha=0.8, vmin=0, vmax=1, rasterized=True)
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([wrap_label(fmt_feature(names[i])) for i in order],
                       fontsize=13.8, linespacing=1.15)
    for t_, i in zip(ax.get_yticklabels(), order):
        if not is_pathway(names[i]):
            t_.set_fontstyle("italic")
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=16.7, fontweight="bold")
    ax.tick_params(axis="x", labelsize=13.8); ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", alpha=0.08); ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_edgecolor("#DDDDDD")

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=SHAP_CMAP), cax=cax)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value", fontsize=14.9, fontweight="bold")
    cb.ax.tick_params(labelsize=13.2)


# ---------------------------------------------------------------- assembly
def build(task: str) -> None:
    fig = plt.figure(figsize=(19.5, 12.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0],
                           height_ratios=[1.0, 1.20], wspace=0.30, hspace=0.34,
                           left=0.075, right=0.905, top=0.955, bottom=0.195)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[:, 1])
    cbB = fig.add_axes([0.098, 0.082, 0.150, 0.013])
    cbC = fig.add_axes([0.918, 0.36, 0.009, 0.28])

    panel_a(axA, task)
    nb = panel_b(axB, cbB, task)

    cfg_str, inner_auc, best_any, auc_any = pick_interpretable(task)
    panel_c(axC, cbC, task, cfg_str)
    axC.set_title(f"Final model: {cfg_str.replace('|', ' | ')}", fontsize=16.7,
                  pad=11, fontweight="bold")

    for ax_, lab in ((axA, "A"), (axB, "B"), (axC, "C")):
        ax_.text(-0.115, 1.015, f"({lab})", transform=ax_.transAxes,
                 fontsize=21.8, fontweight="bold", va="bottom")

    name = "Figure4" if task == "K05" else "Figure5"
    for ext, kw in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    (OUT / f"{name}_config.json").write_text(json.dumps(
        {"task": task, "panel_A": "|".join(PANEL_A_CFG[task]),
         "panel_C_final_model": cfg_str,
         "nested_selections": nb[nb["restriction"] == "all"][
             ["fold", "selected_config", "AUC"]].to_dict("records")}, indent=2))
    print(f"  -> {name}.[png|svg|pdf]   final model: {cfg_str}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="K05", choices=["K05", "K02", "both"])
    a = ap.parse_args()
    for t in (["K05", "K02"] if a.task == "both" else [a.task]):
        print(f"\n=== {t} ===")
        build(t)
