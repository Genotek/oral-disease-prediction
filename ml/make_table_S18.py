#!/usr/bin/env python3
"""Builds Table S18 (sensitivity to the largest cohort) from the outputs of
ml/cohort_sensitivity.py and ml/cohort_sensitivity_subgroups.py.

Run:  python ml/make_table_S18.py
"""
import sys
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ML_DIR   # noqa: E402

R = ML_DIR / "cohort_sensitivity"
OUT = R / "Table_S18_sensitivity_to_largest_cohort.xlsx"
DIS = {"K05": "Periodontal disease status", "K02": "Caries"}
COND = {"baseline": "All other cohorts, unweighted (main analysis)",
        "weighted": "Equal weight per cohort",
        "downsample": "Each training cohort capped at 20 samples (mean of 30 repeats)",
        "exclude_PRJNA1057503": "PRJNA1057503 removed from training",
        "exclude_PRJNA741688": "PRJNA741688 removed from training"}
ORDER = list(COND)
SUB = {"all cases": "All cases", "cases with systemic condition": "Cases with a systemic condition",
       "cases without systemic condition": "Cases without a systemic condition",
       "cases with both diagnoses": "Cases with both diagnoses",
       "cases with one diagnosis": "Cases with one diagnosis"}

x = pd.read_excel(R / "cohort_sensitivity.xlsx", sheet_name=None)
summ, pf, ds = x["Summary"], x["Per fold"], x["Downsample per fold"]
com = pd.read_csv(R / "subgroups_PRJNA1057503.csv")

wb = Workbook(); ws = wb.active; ws.title = "Table S18"
bold = Font(bold=True)


def block(title, header, rows):
    ws.append([title]); ws.cell(ws.max_row, 1).font = bold
    ws.append(header)
    for c in ws[ws.max_row]:
        c.font = bold; c.alignment = Alignment(wrap_text=True, vertical="top")
    for r in rows:
        ws.append(r)
    ws.append([])


# A. summary
rows = []
for d in ["K05", "K02"]:
    for s in ORDER:
        r = summ[(summ.task == d) & (summ.scenario == s)]
        if r.empty: continue
        r = r.iloc[0]
        spread = (f"{r['difference 2.5%']:.3f} to {r['difference 97.5%']:.3f} (repeats, 2.5-97.5%)"
                  if s == "downsample" else f"SE {r['SE of paired difference']:.3f}")
        rows.append([DIS[d], COND[s], int(r["n folds"]), round(r["AUC (mean across folds)"], 3),
                     round(r["SD"], 3), round(r["MCC (mean across folds)"], 3),
                     round(r["baseline AUC, same folds"], 3), round(r["difference vs baseline"], 3), spread])
block("A. Leave-one-study-out performance of the final models under different training conditions",
      ["Disease", "Training condition", "Folds", "Mean AUC across folds", "SD (across folds; across repeats for capped)",
       "Mean MCC", "Mean AUC of the main analysis on the same folds", "Difference", "Spread of difference"], rows)

# B. per fold
rows = []
for d in ["K05", "K02"]:
    for s in ORDER:
        if s == "downsample":
            g = ds[ds.task == d]
            for _, r in g.iterrows():
                base_row = pf[(pf.task == d) & (pf.scenario == "baseline") & (pf["held-out bioproject"] == r["fold"])].iloc[0]
                rows.append([DIS[d], COND[s], r["fold"], int(base_row["n test"]), "", round(r["AUC mean over repeats"], 3),
                             round(r["MCC mean over repeats"], 3), round(r["baseline AUC"], 3),
                             round(r["AUC mean over repeats"] - r["baseline AUC"], 3),
                             f"SD over repeats {r['AUC SD over repeats']:.3f}"])
            continue
        g = pf[(pf.task == d) & (pf.scenario == s)]
        for _, r in g.iterrows():
            rows.append([DIS[d], COND[s], r["held-out bioproject"], int(r["n test"]), int(r["n train"]),
                         round(r["AUC"], 3), round(r["MCC"], 3), round(r["AUC baseline"], 3),
                         round(r["difference"], 3), ""])
block("B. Per held-out bioproject",
      ["Disease", "Training condition", "Held-out bioproject", "Test n", "Training n", "AUC", "MCC",
       "AUC in the main analysis", "Difference", "Note"], rows)

# C. PRJNA1057503 subgroups
rows = []
for d in ["K05", "K02"]:
    for k in ["all cases", "cases with systemic condition", "cases without systemic condition",
              "cases with both diagnoses", "cases with one diagnosis"]:
        r = com[(com.task == d) & (com.group == k)].iloc[0]
        rows.append([DIS[d], SUB[k], int(r.n_cases), int(r.n_controls), r.AUC, r.CI95_low, r.CI95_high])
block("C. Held-out PRJNA1057503: AUC of controls versus subgroups of cases (final models, leave-one-study-out)",
      ["Disease", "Cases included", "Cases n", "Controls n", "AUC", "95% CI lower", "95% CI upper"], rows)

for t in ["Notes",
          "Final models: periodontal disease status, taxonomic profiles, CLR, ridge logistic regression; caries, taxonomic profiles, rank standardisation, random forest (Figures 4C, 5C). Configurations were not re-selected.",
          "Evaluation is leave-one-study-out over the six bioprojects containing both classes. Hyperparameters were tuned by inner 5-fold cross-validation, and feature filtering and transformation were fitted on the training data only, as in the main analysis.",
          "Equal weight per cohort: sample weight proportional to 1 / (size of its bioproject in the training set).",
          "Capped: each training bioproject reduced to at most 20 samples (the median bioproject size), class ratio preserved within bioproject; 30 random repeats.",
          "Removal of a bioproject from training changes the set of evaluated folds (the removed bioproject is not tested); the difference is calculated against the main analysis on the same folds. This analysis reduces the training set (by 41% for periodontal disease status and 36% for caries when PRJNA1057503 is removed).",
          "Part C: systemic conditions (diabetes, hypertension, obesity, colitis, Crohn's disease) were taken from the free-text health_status field and occur in cases only; annotation is incomplete. Confidence intervals by bootstrap over cases and controls (2000 resamples). The subgroups were defined after prediction."]:
    ws.append([t])
ws.cell(ws.max_row - 6, 1).font = bold
for col, w in zip("ABCDEFGHI", [26, 52, 20, 14, 18, 12, 14, 18, 30]):
    ws.column_dimensions[col].width = w
wb.save(OUT)
print("saved", OUT)
