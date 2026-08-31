#!/usr/bin/env python3
"""
The comparisons quoted in the manuscript.

Reads per_split_results.csv and nested_loso.csv and builds four of them:

  1. Study-grouped against class-stratified CV: how much performance came from
     recognising the cohort.
  2. Unweighted against weighted averaging over LOSO folds.
  3. Nested selection against the grid maximum: the size of the optimism.
  4. The distribution over the whole grid, so the reported value can be placed
     within it rather than presented on its own.

Fits nothing. Run from the repository root.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (DATA_DIR, TAX_CSV, PATH_CSV, META_CSV, ML_DIR, MODELS_DIR,
                   RESULTS_DIR, FIGURES_DIR, PERMANOVA_CSV, require_data)
RES = ML_DIR
OUT = RES

df = pd.read_csv(RES / "per_split_results.csv")
ok = df[df["status"] == "ok"].copy()
ok["config"] = ok[["feature_set", "transform", "dimred", "model_name"]].agg("|".join, axis=1)

def agg(sub: pd.DataFrame) -> pd.Series:
    """Unweighted mean and mean weighted by test-fold size."""
    w = sub["n_test"].to_numpy(float)
    return pd.Series({
        "n_folds": len(sub),
        "AUC_unweighted": sub["AUC"].mean(),
        "AUC_weighted": float(np.average(sub["AUC"], weights=w)),
        "MCC_unweighted": sub["MCC"].mean(),
        "MCC_weighted": float(np.average(sub["MCC"], weights=w)),
    })

per_cfg = (ok.groupby(["task", "scheme", "feature_set", "config"])
             .apply(agg, include_groups=False).reset_index())
per_cfg.to_csv(OUT / "per_config_summary.csv", index=False)

lines = []
def say(s=""): lines.append(s); print(s)

# ---- 1. best configuration in each feature space, by validation scheme
say("=" * 78)
say("1. STUDY-GROUPED CV AGAINST CLASS-STRATIFIED")
say("   The gap shows how much performance came from recognising the cohort.")
say("=" * 78)
say(f"{'task/space':16}{'strat CV':>10}{'grouped CV':>12}{'gap':>9}{'LOSO':>9}")
gap_rows = []
for (t, fs), g in per_cfg.groupby(["task", "feature_set"]):
    def best(scheme, metric="AUC_unweighted"):
        s = g[g["scheme"] == scheme]
        return (float(s[metric].max()), s.loc[s[metric].idxmax(), "config"]) if len(s) else (np.nan, "")
    a_s, c_s = best("stratified_cv"); a_g, c_g = best("grouped_cv"); a_l, c_l = best("loso")
    say(f"{t+'/'+fs:16}{a_s:>10.3f}{a_g:>12.3f}{a_g-a_s:>+9.3f}{a_l:>9.3f}")
    gap_rows.append({"task": t, "feature_set": fs, "AUC_stratified": a_s,
                     "AUC_grouped": a_g, "gap": a_g - a_s, "AUC_loso": a_l,
                     "best_config_stratified": c_s, "best_config_grouped": c_g})
pd.DataFrame(gap_rows).to_csv(OUT / "comparison_cv_schemes.csv", index=False)

# ---- 2. weighted against unweighted averaging over LOSO folds
say(); say("=" * 78)
say("2. UNWEIGHTED AGAINST WEIGHTED MEAN OVER LOSO FOLDS")
say("=" * 78)
say(f"{'task/space':16}{'AUC unw':>10}{'AUC wgt':>10}{'MCC unw':>10}{'MCC wgt':>10}")
w_rows = []
for (t, fs), g in per_cfg[per_cfg["scheme"] == "loso"].groupby(["task", "feature_set"]):
    b = g.loc[g["AUC_unweighted"].idxmax()]
    say(f"{t+'/'+fs:16}{b['AUC_unweighted']:>10.3f}{b['AUC_weighted']:>10.3f}"
        f"{b['MCC_unweighted']:>10.3f}{b['MCC_weighted']:>10.3f}")
    w_rows.append({"task": t, "feature_set": fs, "config": b["config"],
                   "AUC_unweighted": b["AUC_unweighted"], "AUC_weighted": b["AUC_weighted"],
                   "MCC_unweighted": b["MCC_unweighted"], "MCC_weighted": b["MCC_weighted"]})
pd.DataFrame(w_rows).to_csv(OUT / "comparison_weighted.csv", index=False)

# ---- 3. nested selection against the grid maximum
nested_path = RES / "nested_loso.csv"
if nested_path.exists():
    nl = pd.read_csv(nested_path)
    say(); say("=" * 78)
    say("3. NESTED SELECTION AGAINST THE GRID MAXIMUM")
    say("   The grid maximum is biased upward: selected on the same folds.")
    say("=" * 78)
    say(f"{'task':8}{'nested AUC':>15}{'grid max':>12}{'median':>10}{'optimism':>10}")
    n_rows = []
    for t, g in nl.groupby("task"):
        na, nm = g["AUC"].mean(), g["MCC"].mean()
        grid = per_cfg[(per_cfg["task"] == t) & (per_cfg["scheme"] == "loso")]["AUC_unweighted"]
        say(f"{t:8}{na:>15.3f}{grid.max():>12.3f}{grid.median():>10.3f}"
            f"{grid.max()-na:>+10.3f}")
        n_rows.append({"task": t, "AUC_nested": na, "MCC_nested": nm,
                       "AUC_grid_max": grid.max(), "AUC_grid_median": grid.median(),
                       "optimism": grid.max() - na,
                       "configs_selected": "; ".join(sorted(set(g["selected_config"])))})
        for _, r in g.iterrows():
            say(f"    fold {r['fold']:<16} selected {r['selected_config']:<32} AUC={r['AUC']:.3f}")
    pd.DataFrame(n_rows).to_csv(OUT / "comparison_nested.csv", index=False)

# ---- 4. distribution over the grid
say(); say("=" * 78)
say("4. DISTRIBUTION OVER THE GRID")
say("=" * 78)
say(f"{'task/scheme':24}{'n':>5}{'min':>8}{'25%':>8}{'median':>9}{'75%':>8}{'max':>8}")
d_rows = []
for (t, sc), g in per_cfg.groupby(["task", "scheme"]):
    v = g["AUC_unweighted"]
    say(f"{t+'/'+sc:24}{len(v):>5}{v.min():>8.3f}{v.quantile(.25):>8.3f}"
        f"{v.median():>9.3f}{v.quantile(.75):>8.3f}{v.max():>8.3f}")
    d_rows.append({"task": t, "scheme": sc, "n_configs": len(v), "min": v.min(),
                   "q25": v.quantile(.25), "median": v.median(),
                   "q75": v.quantile(.75), "max": v.max()})
pd.DataFrame(d_rows).to_csv(OUT / "grid_distribution.csv", index=False)

(OUT / "comparison_report.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nWritten to {OUT}")
