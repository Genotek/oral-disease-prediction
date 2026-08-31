#!/usr/bin/env python3
"""
Nested configuration selection inside LOSO, separately within each feature space.

Panel B of Figures 4 and 5 needs three columns (TAX, TAX+PATH, PATH), and in each
the configuration must be chosen without the held-out cohort taking part. The
inner loop is therefore run once and stored in full, after which selection is
performed four times: over the whole grid and within each feature space.

Output:
  inner_results.csv          every inner-loop run (task, held-out cohort, inner
                             cohort, configuration, AUC/MCC)
  nested_by_space.csv        outer estimates for the four selections
  nested_by_space_preds.csv  per-fold predictions behind those estimates

Run:  python ml/nested_selection.py --n-jobs 12
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                       # repository root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))                            # paths.py, common.py, config.py
from paths import TAX_CSV, PATH_CSV, META_CSV, require_data   # noqa: E402

import ml_pipeline_article_configs as base                # noqa: E402
from grid_and_schemes import (make_grid, run_split, splits_loso,  # noqa: E402
                      inner_loso, _append, OUT)

INNER_CSV  = OUT / "inner_results.csv"
NBS_CSV    = OUT / "nested_by_space.csv"
NBS_PREDS  = OUT / "nested_by_space_preds.csv"
SPACES     = ["TAX", "TAX+PATH", "PATH"]


def done_inner() -> set:
    if not INNER_CSV.exists():
        return set()
    d = pd.read_csv(INNER_CSV)
    return set(zip(d.task, d.held.astype(str), d.inner_fold.astype(str), d.config))


def done_outer() -> set:
    if not NBS_CSV.exists():
        return set()
    d = pd.read_csv(NBS_CSV)
    return set(zip(d.task, d.restriction, d.fold.astype(str)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--tasks", nargs="+", default=["K05", "K02"])
    args = ap.parse_args()
    require_data()

    t0 = time.time()
    D = base.load_data(str(TAX_CSV),
                       str(PATH_CSV),
                       str(META_CSV))
    meta, tax_cols, path_cols = D["meta"], D["tax_cols"], D["path_cols"]
    Xtax_all = D["X_tax"].to_numpy(float)
    Xpath_all = D["X_path"].to_numpy(float)

    for task in args.tasks:
        y, _, bp, idx_keep = base.task_vectors(meta, task)
        Xtax, Xpath = Xtax_all[idx_keep], Xpath_all[idx_keep]
        runs = meta.index.to_numpy()[idx_keep]
        cfgs = make_grid(task)
        outer = splits_loso(y, bp)
        print(f"\n[{task}] outer folds {len(outer)}, configurations {len(cfgs)}", flush=True)

        di, do = done_inner(), done_outer()

        for (tr, te, held) in outer:
            inner = inner_loso(y, bp, tr)
            if len(inner) < 2:
                print(f"  {held}: fewer than 2 inner folds, skipped", flush=True)
                continue

            jobs = [(c, itr, ite, str(ib))
                    for (itr, ite, ib) in inner for c in cfgs
                    if (task, str(held), str(ib),
                        "|".join([c.feature_set, c.transform, c.dimred,
                                  c.model_name])) not in di]
            if jobs:
                print(f"  {held}: inner loop, {len(jobs)} jobs", flush=True)
                res = Parallel(n_jobs=args.n_jobs, verbose=5, batch_size=4)(
                    delayed(run_split)(c, Xtax, Xpath, y, bp, tax_cols, path_cols,
                                       itr, ite, "inner", f"{held}<-{ib}")
                    for (c, itr, ite, ib) in jobs)
                rows = []
                for r, (c, _, _, ib) in zip(res, jobs):
                    if not r:
                        continue
                    for k in ("_probs", "_test_idx", "_y_true"):
                        r.pop(k, None)
                    rows.append({"task": task, "held": str(held), "inner_fold": ib,
                                 "config": "|".join([c.feature_set, c.transform,
                                                     c.dimred, c.model_name]),
                                 "feature_set": c.feature_set,
                                 "status": r.get("status"),
                                 "AUC": r.get("AUC"), "MCC": r.get("MCC")})
                _append(INNER_CSV, rows)
                print(f"  {held}: {len(rows)} inner rows written", flush=True)

            # ---- selection: over the whole grid and within each feature space
            inn = pd.read_csv(INNER_CSV)
            inn = inn[(inn.task == task) & (inn.held.astype(str) == str(held))
                      & (inn.status == "ok")]
            if inn.empty:
                continue

            for restriction in ["all"] + SPACES:
                if (task, restriction, str(held)) in do:
                    continue
                sub = inn if restriction == "all" else inn[inn.feature_set == restriction]
                if sub.empty:
                    continue
                agg = sub.groupby("config")["AUC"].agg(["mean", "count"])
                best = agg["mean"].idxmax()
                fs, trf, dr, mn = best.split("|")
                bc = base.Config(task=task, feature_set=fs, transform=trf, dimred=dr,
                                 batch_strategy="none", model_name=mn)
                r = run_split(bc, Xtax, Xpath, y, bp, tax_cols, path_cols,
                              tr, te, "nested_by_space", str(held))
                if not r or r.get("status") != "ok":
                    print(f"    {held} / {restriction}: outer fold not evaluated", flush=True)
                    continue
                pp = {k: r.pop(k) for k in ("_probs", "_test_idx", "_y_true") if k in r}
                r["restriction"] = restriction
                r["selected_config"] = best
                r["inner_mean_auc"] = float(agg.loc[best, "mean"])
                r["n_inner_folds"] = int(agg.loc[best, "count"])
                _append(NBS_CSV, [r])
                if pp:
                    _append(NBS_PREDS, [
                        {"task": task, "restriction": restriction, "fold": str(held),
                         "config": best, "run": runs[ti], "bioproject": bp[ti],
                         "y_true": pp["_y_true"][j], "prob": pp["_probs"][j]}
                        for j, ti in enumerate(pp["_test_idx"])])
                print(f"    {held} / {restriction:9s}: {best:34s} "
                      f"inner={r['inner_mean_auc']:.3f} outer={r['AUC']:.3f}", flush=True)

    print(f"\nDone in {time.time() - t0:.0f} s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
