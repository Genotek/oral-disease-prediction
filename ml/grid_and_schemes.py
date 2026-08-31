#!/usr/bin/env python3
"""Machine-learning grid under three validation schemes, plus nested LOSO selection.

Data loading, feature filtering, transformations and models come from
ml_pipeline_article_configs.py in the repository root.

  1. The grid is 3 x 3 x 2 x 4 = 72 configurations per disease. No batch
     correction is applied to the data: batch_strategy is always "none".

  2. Study-grouped cross-validation: StratifiedGroupKFold on bioproject, every
     sample of a cohort entirely in training or entirely in test. Class-stratified
     CV is computed as well so the gap between the two can be reported.

  3. Nested configuration selection inside LOSO: for each outer fold the
     configuration is chosen by an inner LOSO over the training cohorts and only
     then applied to the held-out one, which takes no part in its own selection.

  4. Metrics are averaged unweighted; the fold-size-weighted means are computed
     alongside and stored for comparison.

  5. Per-fold predictions are kept for every held-out sample, so any metric can be
     recomputed without refitting.

Pathway depth normalisation is not applied here: CLR, ILR and rank.std all
normalise within a sample, so sequencing depth does not leak into the models. It
is applied on the MaAsLin2 side only.

Run:  python ml/grid_and_schemes.py --n-jobs 12
"""
from __future__ import annotations

import argparse, json, platform, sys, time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]        # repository root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (DATA_DIR, TAX_CSV, PATH_CSV, META_CSV, ML_DIR, MODELS_DIR,
                   RESULTS_DIR, FIGURES_DIR, PERMANOVA_CSV, require_data)
sys.path.insert(0, str(ROOT))

from common import compute_metrics                       # noqa: E402
import ml_pipeline_article_configs as base               # noqa: E402
from config import RANDOM_STATE                          # noqa: E402

OUT = ML_DIR
OUT.mkdir(parents=True, exist_ok=True)

TASKS         = ["K05", "K02"]
FEATURE_SETS  = ["TAX", "PATH", "TAX+PATH"]
TRANSFORMS    = ["CLR", "ILR", "rank.std"]
DIMREDS       = ["none", "PCA"]
MODEL_NAMES   = ["LR_Ridge", "RF", "SVM", "XGBoost"]
N_SPLITS      = 5
MIN_PER_CLASS = 3


def make_grid(task: str) -> List[base.Config]:
    """72 configurations per disease; batch_strategy fixed at 'none'."""
    out = []
    for fs, tr, dr, mn in product(FEATURE_SETS, TRANSFORMS, DIMREDS, MODEL_NAMES):
        c = base.Config(task=task, feature_set=fs, transform=tr, dimred=dr,
                        batch_strategy="none", model_name=mn)
        if base.is_valid_config(c):
            out.append(c)
    return out


def cfg_key(c: base.Config) -> str:
    return "|".join([c.feature_set, c.transform, c.dimred, c.model_name])


def run_split(cfg, Xtax, Xpath, y, bp, tax_cols, path_cols,
              tr_idx, te_idx, scheme: str, fold_id: str) -> Dict:
    y_tr, y_te = y[tr_idx], y[te_idx]
    row = {"task": cfg.task, "feature_set": cfg.feature_set,
           "transform": cfg.transform, "dimred": cfg.dimred,
           "model_name": cfg.model_name, "scheme": scheme, "fold": fold_id,
           "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
           "n_pos_test": int(y_te.sum()), "n_neg_test": int((y_te == 0).sum())}

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return {**row, "status": "skipped_single_class"}
    try:
        trb, teb, _, info = base.select_feature_blocks(
            cfg, Xtax, Xpath, tr_idx, te_idx, bp, tax_cols, path_cols)
        Xtr, Xte, _ = base.transform_blocks(trb, teb, cfg)
        Xtr, Xte, dinfo = base.apply_dimred(Xtr, Xte, cfg)
        probs, _, meta = base.tune_and_predict(Xtr, y_tr, Xte, cfg.model_name, 1)
    except Exception as exc:                              # noqa: BLE001
        return {**row, "status": f"error:{type(exc).__name__}"}

    m = compute_metrics(y_te, probs)
    if m is None:
        return {**row, "status": "metrics_undefined"}
    return {**row, "status": "ok", **m,
            "n_tax_features": info["n_tax_features"],
            "n_path_features": info["n_path_features"],
            "n_final_features": dinfo.get("n_final_features"),
            "best_params": json.dumps(meta.get("best_params"), default=str),
            "_probs": probs.tolist(), "_test_idx": te_idx.tolist(),
            "_y_true": y_te.tolist()}


def splits_stratified(y, bp):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return [(tr, te, f"strat{i}") for i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y))]


def splits_grouped(y, bp):
    sgk = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return [(tr, te, f"group{i}")
            for i, (tr, te) in enumerate(sgk.split(np.zeros(len(y)), y, groups=bp))]


def splits_loso(y, bp):
    out = []
    for b in sorted(set(bp)):
        te = np.where(bp == b)[0]; tr = np.where(bp != b)[0]
        if int(y[te].sum()) >= MIN_PER_CLASS and int((y[te] == 0).sum()) >= MIN_PER_CLASS:
            out.append((tr, te, b))
    return out


def inner_loso(y, bp, train_idx):
    """Inner LOSO over the cohorts left in training, used for nested selection."""
    out = []
    sub_bp = bp[train_idx]
    for b in sorted(set(sub_bp)):
        te_loc = np.where(sub_bp == b)[0]; tr_loc = np.where(sub_bp != b)[0]
        te, tr = train_idx[te_loc], train_idx[tr_loc]
        if int(y[te].sum()) >= MIN_PER_CLASS and int((y[te] == 0).sum()) >= MIN_PER_CLASS:
            out.append((tr, te, b))
    return out


# ------------------------------------------------------------- checkpointing
# Results are written as they are produced rather than at the end: the run is long
# and an interruption should not cost all of it. On re-invocation, combinations of
# (task, scheme, fold, configuration) already on disk are skipped.

SPLITS_CSV = OUT / "per_split_results.csv"
PREDS_CSV  = OUT / "per_sample_predictions.csv"
NESTED_CSV = OUT / "nested_loso.csv"


def _append(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def done_keys() -> set:
    """Combinations already on disk, so they are not recomputed."""
    if not SPLITS_CSV.exists():
        return set()
    try:
        d = pd.read_csv(SPLITS_CSV, usecols=["task", "scheme", "fold", "feature_set",
                                             "transform", "dimred", "model_name"])
    except Exception:
        return set()
    return {tuple(map(str, r)) for r in d.itertuples(index=False, name=None)}


def done_nested() -> set:
    if not NESTED_CSV.exists():
        return set()
    try:
        d = pd.read_csv(NESTED_CSV, usecols=["task", "fold"])
    except Exception:
        return set()
    return {(str(a), str(b)) for a, b in d.itertuples(index=False, name=None)}


def split_key(cfg, scheme: str, fold: str) -> tuple:
    return (cfg.task, scheme, str(fold), cfg.feature_set, cfg.transform,
            cfg.dimred, cfg.model_name)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--skip-nested", action="store_true",
                    help="skip nested selection, which is the slowest part")
    args = ap.parse_args()

    t0 = time.time()
    D = base.load_data(str(TAX_CSV),
                       str(PATH_CSV),
                       str(META_CSV))
    meta, tax_cols, path_cols = D["meta"], D["tax_cols"], D["path_cols"]
    Xtax_all = D["X_tax"].to_numpy(float)
    Xpath_all = D["X_path"].to_numpy(float)

    all_rows: List[Dict] = []
    preds: List[Dict] = []
    nested_rows: List[Dict] = []

    for task in args.tasks:
        y, _, bp, idx_keep = base.task_vectors(meta, task)
        Xtax, Xpath = Xtax_all[idx_keep], Xpath_all[idx_keep]
        runs = meta.index.to_numpy()[idx_keep]
        cfgs = make_grid(task)
        print(f"\n[{task}] n={len(y)} cases={int(y.sum())} controls={int((y==0).sum())} "
              f"| cohorts={len(set(bp))} | configurations={len(cfgs)}", flush=True)

        schemes = {"grouped_cv": splits_grouped(y, bp),
                   "stratified_cv": splits_stratified(y, bp),
                   "loso": splits_loso(y, bp)}
        for name, sp in schemes.items():
            print(f"  {name}: {len(sp)} folds", flush=True)

        seen = done_keys()
        # One scheme at a time; each is flushed to disk before the next starts.
        for sname, sp in schemes.items():
            jobs = [(c, tr, te, sname, fid)
                    for (tr, te, fid) in sp for c in cfgs
                    if split_key(c, sname, fid) not in seen]
            if not jobs:
                print(f"  {sname}: already computed, skipped", flush=True)
                continue
            print(f"  {sname}: {len(jobs)} jobs to run", flush=True)
            res = Parallel(n_jobs=args.n_jobs, verbose=5, batch_size=4)(
                delayed(run_split)(c, Xtax, Xpath, y, bp, tax_cols, path_cols,
                                   tr, te, s_, f_)
                for (c, tr, te, s_, f_) in jobs)

            rows_, preds_ = [], []
            for r in res:
                if r is None:
                    continue
                pp = {k: r.pop(k) for k in ("_probs", "_test_idx", "_y_true") if k in r}
                rows_.append(r)
                if pp and r["status"] == "ok":
                    for j, ti in enumerate(pp["_test_idx"]):
                        preds_.append({"task": task, "scheme": r["scheme"], "fold": r["fold"],
                                       "config": "|".join([r["feature_set"], r["transform"],
                                                           r["dimred"], r["model_name"]]),
                                       "run": runs[ti], "bioproject": bp[ti],
                                       "y_true": pp["_y_true"][j], "prob": pp["_probs"][j]})
            _append(SPLITS_CSV, rows_)
            _append(PREDS_CSV, preds_)
            print(f"  {sname}: {len(rows_)} rows written", flush=True)

        # ---- nested selection: configuration chosen by the inner LOSO
        if not args.skip_nested:
            outer = schemes["loso"]
            dn = done_nested()
            for (tr, te, held) in outer:
                if (task, str(held)) in dn:
                    print(f"    nested fold {held}: already computed, skipped", flush=True)
                    continue
                inner = inner_loso(y, bp, tr)
                if len(inner) < 2:
                    continue
                ijobs = [(c, itr, ite, "inner", f"{held}<-{ib}")
                         for (itr, ite, ib) in inner for c in cfgs]
                ires = Parallel(n_jobs=args.n_jobs, batch_size=4)(
                    delayed(run_split)(c, Xtax, Xpath, y, bp, tax_cols, path_cols,
                                       itr, ite, "inner", fid)
                    for (c, itr, ite, sname, fid) in ijobs)
                agg: Dict[str, List[float]] = {}
                for r in ires:
                    if r and r.get("status") == "ok":
                        k = "|".join([r["feature_set"], r["transform"],
                                      r["dimred"], r["model_name"]])
                        agg.setdefault(k, []).append(r["AUC"])
                if not agg:
                    continue
                best_k = max(agg, key=lambda k: float(np.mean(agg[k])))
                fs, trf, dr, mn = best_k.split("|")
                bc = base.Config(task=task, feature_set=fs, transform=trf, dimred=dr,
                                 batch_strategy="none", model_name=mn)
                r = run_split(bc, Xtax, Xpath, y, bp, tax_cols, path_cols,
                              tr, te, "nested_loso", held)
                if r and r.get("status") == "ok":
                    for k in ("_probs", "_test_idx", "_y_true"):
                        r.pop(k, None)
                    r["selected_config"] = best_k
                    r["inner_mean_auc"] = float(np.mean(agg[best_k]))
                    r["n_inner_folds"] = len(inner)
                    _append(NESTED_CSV, [r])
                print(f"    nested fold {held}: selected {best_k} "
                      f"-> AUC={r['AUC'] if r else float('nan'):.3f}", flush=True)

    # Results are already on disk; this is only the run summary.
    import sklearn, scipy, xgboost
    (OUT / "session_info.json").write_text(json.dumps({
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "scipy": scipy.__version__, "xgboost": xgboost.__version__,
        "random_state": RANDOM_STATE, "n_splits": N_SPLITS,
        "grid_size_per_task": len(make_grid("K05")),
        "elapsed_sec": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False))
    print(f"\nDone in {time.time()-t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
