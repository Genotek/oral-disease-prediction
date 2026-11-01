#!/usr/bin/env python3
"""Sensitivity of the final models to the largest cohort (leave-one-study-out).

The largest bioproject, PRJNA1057503, contributes 41 % of the periodontal and 36 %
of the caries analysis set. This script asks how much the cross-cohort estimates
depend on it. The models are not re-selected: the two final configurations are fixed

  K05: TAX | CLR      | none | LR_Ridge
  K02: TAX | rank.std | none | RF

and hyperparameters are tuned by the same inner GridSearchCV as in the main analysis
(ml_pipeline_article_configs), so the baseline scenario reproduces the per-fold LOSO
values of the main analysis (sheet "Check", if ml/per_split_results.csv is present).

The outer scheme is always leave-one-study-out over the bioprojects that contain both
classes. Only the training set changes:

  baseline             all other cohorts, unweighted (as in the main analysis)
  weighted             all other cohorts, sample weight = 1 / size of its cohort in the
                       training set, so that every training cohort carries equal total weight
  downsample           each training cohort capped at n_cap samples (the median cohort
                       size; class ratio kept within the cohort), N_REPEATS random repeats
  exclude_<BioProject> the cohort is removed from training; evaluated on the remaining
                       folds (undefined for its own fold, where it is already absent)

Feature filtering and transformation are fitted on the training set of each scenario
only, as in the main analysis.

Output (<output>/ml/cohort_sensitivity/): per_fold.csv (raw results, written
incrementally and resumable) and cohort_sensitivity.xlsx (summary sheets).

Run:    python ml/cohort_sensitivity.py --n-jobs 8
Check:  python ml/cohort_sensitivity.py --repeats 2 --tasks K05
Then:   python ml/cohort_sensitivity_subgroups.py
        python ml/make_table_S18.py
"""
from __future__ import annotations

import argparse, json, platform, sys, time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]        # repository root
sys.path.insert(0, str(ROOT))
from paths import TAX_CSV, PATH_CSV, META_CSV, ML_DIR, require_data   # noqa: E402
from common import compute_metrics                                      # noqa: E402
import ml_pipeline_article_configs as base                              # noqa: E402
from config import RANDOM_STATE                                         # noqa: E402

OUT = ML_DIR / "cohort_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT / "per_fold.csv"

MIN_PER_CLASS = 3        # minimum cases and controls in a held-out cohort (same as grid_and_schemes.py)
FINAL = {"K05": dict(feature_set="TAX", transform="CLR", dimred="none", model_name="LR_Ridge"),
         "K02": dict(feature_set="TAX", transform="rank.std", dimred="none", model_name="RF")}
DOMINANT = "PRJNA1057503"
EXCLUDE = [DOMINANT, "PRJNA741688"]      # PRJNA741688: periodontitis cases with Crohn's disease
N_REPEATS = 30


def splits_loso(y, bp):
    """Leave-one-study-out folds over cohorts with enough cases and controls."""
    out = []
    for b in sorted(set(bp)):
        te = np.where(bp == b)[0]
        tr = np.where(bp != b)[0]
        if int(y[te].sum()) >= MIN_PER_CLASS and int((y[te] == 0).sum()) >= MIN_PER_CLASS:
            out.append((tr, te, b))
    return out


# ------------------------------------------------------------------ fitting
def fit_predict(cfg, Xtax, Xpath, y, bp, tax_cols, path_cols, tr_idx, te_idx,
                weights: Optional[np.ndarray] = None) -> Optional[Dict]:
    """Same steps as the main analysis, with optional sample weights."""
    y_tr, y_te = y[tr_idx], y[te_idx]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None
    trb, teb, _, info = base.select_feature_blocks(
        cfg, Xtax, Xpath, tr_idx, te_idx, bp, tax_cols, path_cols)
    Xtr, Xte, _ = base.transform_blocks(trb, teb, cfg)
    Xtr, Xte, _ = base.apply_dimred(Xtr, Xte, cfg)

    model, grid = base.get_model_and_grid(cfg.model_name)
    fit_kw = {} if weights is None else {"sample_weight": weights}
    inner = StratifiedKFold(n_splits=base.INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(model, grid, cv=inner, scoring="roc_auc", refit=True, n_jobs=1)
    gs.fit(Xtr, y_tr, **fit_kw)
    probs = gs.best_estimator_.predict_proba(Xte)[:, 1]
    m = compute_metrics(y_te, probs)
    if m is None:
        return None
    return {"AUC": m["AUC"], "MCC": m["MCC"], "BACC": m.get("BACC", np.nan),
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "n_features": int(info["n_tax_features"] + info["n_path_features"]),
            "best_params": json.dumps(gs.best_params_, default=str)}


def cohort_weights(bp_tr: np.ndarray) -> np.ndarray:
    """1 / cohort size, rescaled to mean 1 (every cohort carries the same total weight)."""
    vals, counts = np.unique(bp_tr, return_counts=True)
    size = dict(zip(vals, counts))
    w = np.array([1.0 / size[b] for b in bp_tr])
    return w * (len(w) / w.sum())


def downsample(train_idx, y, bp, cap: int, seed: int) -> np.ndarray:
    """At most cap samples per cohort; the class ratio is kept within each cohort."""
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for b in np.unique(bp[train_idx]):
        ids = train_idx[bp[train_idx] == b]
        if len(ids) <= cap:
            keep.extend(ids.tolist())
            continue
        sel: List[int] = []
        for cls in (0, 1):
            c = ids[y[ids] == cls]
            if len(c) == 0:
                continue
            k = max(1, int(round(cap * len(c) / len(ids))))
            sel.extend(rng.choice(c, size=min(k, len(c)), replace=False).tolist())
        keep.extend(sel)
    return np.array(sorted(keep))


# -------------------------------------------------------------- checkpoints
def done() -> set:
    if not RAW.exists():
        return set()
    d = pd.read_csv(RAW, usecols=["task", "scenario", "fold", "repeat"])
    return {tuple(map(str, r)) for r in d.itertuples(index=False, name=None)}


def append(rows: List[Dict]) -> None:
    rows = [r for r in rows if r]
    if rows:
        pd.DataFrame(rows).to_csv(RAW, mode="a", header=not RAW.exists(), index=False)


# ------------------------------------------------------------------ running
def run_job(task, cfg, scen, fold, rep, Xtax, Xpath, y, bp, tax_cols, path_cols,
            tr, te, cap):
    base_row = {"task": task, "scenario": scen, "fold": fold, "repeat": rep}
    w = None
    if scen == "weighted":
        w = cohort_weights(bp[tr])
    elif scen == "downsample":
        tr = downsample(tr, y, bp, cap, seed=int(rep))
    elif scen.startswith("exclude_"):
        tr = tr[bp[tr] != scen.split("_", 1)[1]]
    try:
        r = fit_predict(cfg, Xtax, Xpath, y, bp, tax_cols, path_cols, tr, te, w)
    except Exception as exc:                                  # noqa: BLE001
        return {**base_row, "status": f"error:{type(exc).__name__}:{exc}"}
    if r is None:
        return {**base_row, "status": "undefined"}
    return {**base_row, "status": "ok", **r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--tasks", nargs="*", default=["K05", "K02"])
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    args = ap.parse_args()
    require_data()
    t0 = time.time()

    D = base.load_data(str(TAX_CSV), str(PATH_CSV), str(META_CSV))
    meta, tax_cols, path_cols = D["meta"], D["tax_cols"], D["path_cols"]
    Xtax_all, Xpath_all = D["X_tax"].to_numpy(float), D["X_path"].to_numpy(float)

    caps: Dict[str, int] = {}
    for task in args.tasks:
        y, _, bp, keep = base.task_vectors(meta, task)
        Xtax, Xpath = Xtax_all[keep], Xpath_all[keep]
        cfg = base.Config(task=task, batch_strategy="none", **FINAL[task])
        folds = splits_loso(y, bp)
        sizes = pd.Series(bp).value_counts()
        caps[task] = int(np.median(sizes.values))
        print(f"\n[{task}] n={len(y)} cohorts={len(sizes)} LOSO folds={len(folds)} "
              f"n_cap={caps[task]} | {FINAL[task]}", flush=True)
        print("  cohort sizes:", dict(sizes), flush=True)

        seen = done()
        jobs = []
        for tr, te, held in folds:
            scens = [("baseline", "0"), ("weighted", "0")]
            scens += [(f"exclude_{x}", "0") for x in EXCLUDE if x != held and x in set(bp)]
            scens += [("downsample", str(r)) for r in range(args.repeats)]
            for scen, rep in scens:
                if (task, scen, str(held), rep) not in seen:
                    jobs.append((scen, held, rep, tr, te))
        print(f"  jobs to run: {len(jobs)}", flush=True)
        for i in range(0, len(jobs), 40):                # checkpoint every 40 jobs
            chunk = jobs[i:i + 40]
            res = Parallel(n_jobs=args.n_jobs)(
                delayed(run_job)(task, cfg, s, h, r, Xtax, Xpath, y, bp, tax_cols,
                                 path_cols, tr, te, caps[task])
                for (s, h, r, tr, te) in chunk)
            append(res)
            print(f"  {min(i + 40, len(jobs))}/{len(jobs)}  {time.time() - t0:.0f} s", flush=True)

    build_excel(caps, time.time() - t0)
    print(f"\nDone in {time.time() - t0:.0f} s -> {OUT}")


# ------------------------------------------------------------------ summary
def build_excel(caps: Dict[str, int], elapsed: float) -> None:
    raw = pd.read_csv(RAW)
    bad = raw[raw.status != "ok"]
    ok = raw[raw.status == "ok"].copy()

    # Every comparison is made on the same set of folds.
    rows, per_fold_rows, ds_rows = [], [], []
    for task, g in ok.groupby("task"):
        base_f = g[g.scenario == "baseline"].set_index("fold")
        for scen in ["baseline", "weighted"] + sorted(
                s for s in g.scenario.unique() if s.startswith("exclude_")) + ["downsample"]:
            s = g[g.scenario == scen]
            if s.empty:
                continue
            if scen == "downsample":
                per = s.groupby(["repeat"])[["AUC", "MCC"]].mean()   # mean over folds within a repeat
                folds = sorted(s.fold.unique())
                fm = s.groupby("fold")[["AUC", "MCC"]].agg(["mean", "std"])
                for f in folds:
                    ds_rows.append({"task": task, "fold": f,
                                    "AUC mean over repeats": fm.loc[f, ("AUC", "mean")],
                                    "AUC SD over repeats": fm.loc[f, ("AUC", "std")],
                                    "MCC mean over repeats": fm.loc[f, ("MCC", "mean")],
                                    "baseline AUC": base_f.loc[f, "AUC"]})
                d_auc = per["AUC"].values - base_f.loc[folds, "AUC"].mean()
                rows.append({"task": task, "scenario": scen, "n folds": len(folds),
                             "repeats": per.shape[0],
                             "AUC (mean across folds)": per["AUC"].mean(),
                             "SD": per["AUC"].std(),
                             "SD means": "across repeats",
                             "MCC (mean across folds)": per["MCC"].mean(),
                             "baseline AUC, same folds": base_f.loc[folds, "AUC"].mean(),
                             "difference vs baseline": d_auc.mean(),
                             "difference 2.5%": np.percentile(d_auc, 2.5),
                             "difference 97.5%": np.percentile(d_auc, 97.5)})
                continue
            s1 = s.set_index("fold")
            folds = sorted(s1.index)
            diff = s1.loc[folds, "AUC"] - base_f.loc[folds, "AUC"]
            rows.append({"task": task, "scenario": scen, "n folds": len(folds), "repeats": 1,
                         "AUC (mean across folds)": s1.loc[folds, "AUC"].mean(),
                         "SD": s1.loc[folds, "AUC"].std(),
                         "SD means": "across folds",
                         "MCC (mean across folds)": s1.loc[folds, "MCC"].mean(),
                         "baseline AUC, same folds": base_f.loc[folds, "AUC"].mean(),
                         "difference vs baseline": diff.mean(),
                         "difference 2.5%": np.nan, "difference 97.5%": np.nan,
                         "SE of paired difference": (diff.std() / np.sqrt(len(diff))
                                                     if len(diff) > 1 else np.nan)})
            for f in folds:
                per_fold_rows.append({"task": task, "scenario": scen, "held-out bioproject": f,
                                      "n train": int(s1.loc[f, "n_train"]),
                                      "n test": int(s1.loc[f, "n_test"]),
                                      "AUC": s1.loc[f, "AUC"], "MCC": s1.loc[f, "MCC"],
                                      "AUC baseline": base_f.loc[f, "AUC"],
                                      "difference": s1.loc[f, "AUC"] - base_f.loc[f, "AUC"]})
    summary = pd.DataFrame(rows)
    per_fold = pd.DataFrame(per_fold_rows)
    ds = pd.DataFrame(ds_rows)

    # Check: the baseline against the per-fold LOSO values of the same configuration
    # from the main analysis (written by ml/grid_and_schemes.py).
    chk = []
    ref_p = ML_DIR / "per_split_results.csv"
    if ref_p.exists():
        ref = pd.read_csv(ref_p, keep_default_na=False)
        ref = ref[(ref.scheme == "loso") & (ref.status == "ok")]
        for task, g in ok[ok.scenario == "baseline"].groupby("task"):
            c = FINAL[task]
            r = ref[(ref.task == task) & (ref.feature_set == c["feature_set"]) &
                    (ref["transform"] == c["transform"]) & (ref.dimred == c["dimred"]) &
                    (ref.model_name == c["model_name"])].set_index("fold")
            for _, x in g.iterrows():
                f = str(x.fold)
                chk.append({"task": task, "fold": f, "AUC here": x.AUC,
                            "AUC in per_split_results.csv": r.AUC.get(f, np.nan),
                            "abs difference": abs(x.AUC - r.AUC.get(f, np.nan))})
    check = pd.DataFrame(chk)

    settings = pd.DataFrame([
        ("Final models", "K05: TAX | CLR | none | LR_Ridge; K02: TAX | rank.std | none | RF (Figures 4C, 5C)"),
        ("Outer scheme", "leave-one-study-out over the bioprojects with both classes (as in Figures 4B, 5B)"),
        ("Hyperparameters", "inner 5-fold GridSearchCV on AUC, same grids as the main analysis"),
        ("baseline", "all other bioprojects, unweighted"),
        ("weighted", "sample weight = 1 / (training-set size of the sample's bioproject), rescaled to mean 1"),
        ("downsample", f"each training bioproject capped at the median bioproject size "
                       f"(K05: {caps.get('K05')}, K02: {caps.get('K02')}), class ratio kept within bioproject; "
                       f"{N_REPEATS} random repeats"),
        ("exclude_<BioProject>", "bioproject removed from training; compared with baseline on the same folds"),
        ("Averaging", "unweighted mean across folds, as in the main analysis"),
        ("Failed/undefined jobs", int(len(bad))),
        ("Elapsed, s", round(elapsed)),
        ("Python / sklearn", f"{platform.python_version()} / " + __import__('sklearn').__version__),
        ("random_state", RANDOM_STATE)], columns=["Item", "Value"])

    with pd.ExcelWriter(OUT / "cohort_sensitivity.xlsx", engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        per_fold.to_excel(xw, sheet_name="Per fold", index=False)
        ds.to_excel(xw, sheet_name="Downsample per fold", index=False)
        ok[ok.scenario == "downsample"].to_excel(xw, sheet_name="Downsample raw", index=False)
        check.to_excel(xw, sheet_name="Check", index=False)
        settings.to_excel(xw, sheet_name="Settings", index=False)
        if len(bad):
            bad.to_excel(xw, sheet_name="Failed", index=False)


if __name__ == "__main__":
    main()
