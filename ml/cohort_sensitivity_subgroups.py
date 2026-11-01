#!/usr/bin/env python3
"""Subgroup check for the held-out PRJNA1057503 fold (leave-one-study-out).

Why is the AUC lowest when the largest cohort is held out? No model is refitted: the
saved per-sample predictions of the final configurations (per_sample_predictions.csv,
written by ml/grid_and_schemes.py) are combined with the labels in metadata.csv.

Controls are always the same; cases are split into
  - all cases;
  - cases with both diagnoses (K02 and K05) / with one;
  - cases with a systemic condition recorded in the free-text health_status field
    (diabetes, hypertension, obesity, colitis, Crohn's disease) / without one.
Limitations: the annotation is free text and incomplete, and controls carry no
systemic conditions. The subgroups were defined after prediction. Confidence
intervals: bootstrap over cases and controls (2000 resamples, seed 42).

Run:  python ml/cohort_sensitivity_subgroups.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paths import META_CSV, ML_DIR, require_data          # noqa: E402

COHORT = "PRJNA1057503"
CFG = {"K05": "TAX|CLR|none|LR_Ridge", "K02": "TAX|rank.std|none|RF"}
KEYS = ["diabet", "hypertens", "obes", "colitis", "crohn"]
OUT = ML_DIR / "cohort_sensitivity"
rng = np.random.default_rng(42)


def auc_ci(ctrl, cas, n_boot=2000):
    d = pd.concat([ctrl, cas])
    a = roc_auc_score(d.y_true, d.prob)
    bs = []
    for _ in range(n_boot):
        c = ctrl.sample(len(ctrl), replace=True, random_state=int(rng.integers(1e9)))
        k = cas.sample(len(cas), replace=True, random_state=int(rng.integers(1e9)))
        bs.append(roc_auc_score(pd.concat([c, k]).y_true, pd.concat([c, k]).prob))
    return round(a, 3), round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)


def main() -> None:
    require_data()
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(META_CSV, keep_default_na=False).set_index("run")
    hs = m.health_status.str.lower()
    m["systemic"] = hs.apply(lambda s: any(k in s for k in KEYS))
    pred_csv = ML_DIR / "per_sample_predictions.csv"
    if not pred_csv.exists():
        raise SystemExit(f"{pred_csv} not found; run ml/grid_and_schemes.py first.")
    p = pd.read_csv(pred_csv, keep_default_na=False)

    rows = []
    for t, c in CFG.items():
        q = p[(p.task == t) & (p.scheme == "loso") & (p.config == c) & (p.bioproject == COHORT)].copy()
        q["both"] = ((q.run.map(m.K02_caries).astype(int) == 1) &
                     (q.run.map(m.K05_gingivitis_periodontitis).astype(int) == 1))
        q["systemic"] = q.run.map(m.systemic)
        ctrl, cas = q[q.y_true == 0], q[q.y_true == 1]
        groups = [("all cases", cas),
                  ("cases with both diagnoses", cas[cas.both]),
                  ("cases with one diagnosis", cas[~cas.both]),
                  ("cases with systemic condition", cas[cas.systemic]),
                  ("cases without systemic condition", cas[~cas.systemic]),
                  ("cases without systemic condition, one diagnosis", cas[~cas.systemic & ~cas.both])]
        for name, g in groups:
            a, lo, hi = auc_ci(ctrl, g)
            rows.append({"task": t, "config": c, "group": name, "n_cases": len(g),
                         "n_controls": len(ctrl), "AUC": a, "CI95_low": lo, "CI95_high": hi})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(OUT / "subgroups_PRJNA1057503.csv", index=False)


if __name__ == "__main__":
    main()
