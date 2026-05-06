#!/usr/bin/env python3
"""
ml_pipeline_article_configs.py
==============================

Article-aligned ML pipeline for the salivary WGS meta-analysis.

What is intentionally kept:
  1) Classification setting:
       train: target disease vs healthy only
       test : target disease vs healthy only
     All non-target disease-only samples are excluded from both train and test.
     Mixed samples that contain the target disease are treated as positive for that task.

  2) Feature spaces described in the manuscript:
       TAX, PATH, TAX+PATH

  3) Transformations described in the manuscript:
       CLR, ILR, rank.std

  4) Dimensionality control described in the manuscript:
       none, PCA retaining 95% variance

  5) Batch correction variants described in the manuscript:
       none
       pycombat_global_leak
     The pycombat_global_leak option deliberately performs batch correction on the
     train+test matrix inside each fold/LOSO split. This is a leakage-prone check,
     kept only to test whether global batch correction can improve performance.

  6) Classifiers described in the manuscript:
       LR_Ridge, RF, SVM, XGBoost

"""

from __future__ import annotations

import argparse
import itertools
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    BATCH_STRATEGIES,
    DEFAULT_METADATA_PATH,
    DEFAULT_OUT_DIR,
    DEFAULT_PATH_PATH,
    DEFAULT_TAX_PATH,
    DIMREDS,
    FEATURE_SETS,
    INNER_FOLDS,
    META_COLS,
    MODEL_NAMES,
    OUTER_FOLDS,
    PATHWAY_SCALE,
    RANDOM_STATE,
    TASKS,
    TRANSFORMS,
    UNWANTED_PATHWAY_COLS,
)
from common import (
    HAVE_SKBIO,
    apply_clr,
    apply_ilr,
    compute_metrics,
    json_dumps_safe,
    numeric_feature_frame,
    prevalence_bioproject_filter,
    rank_std_fit,
    rank_std_transform,
    read_csv_with_run,
)

warnings.filterwarnings("ignore")
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning",
)

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAVE_XGBOOST = False

@dataclass(frozen=True)
class Config:
    task: str
    feature_set: str
    transform: str
    dimred: str
    batch_strategy: str
    model_name: str

    @property
    def name(self) -> str:
        return " | ".join([
            self.task,
            self.feature_set,
            self.transform,
            self.dimred,
            self.batch_strategy,
            self.model_name,
            "vs_healthy_only",
        ])

    @property
    def slug(self) -> str:
        return "__".join([
            self.task,
            self.feature_set.replace("+", "plus"),
            self.transform.replace(".", "_"),
            self.dimred,
            self.batch_strategy,
            self.model_name,
            "vs_healthy_only",
        ])


# =============================================================================
# Data loading
# =============================================================================


def load_data(tax_path: str, path_path: str, metadata_path: str) -> Dict[str, object]:
    print("[DATA] Loading input tables")
    meta = read_csv_with_run(metadata_path)
    tax = read_csv_with_run(tax_path)
    path = read_csv_with_run(path_path)

    required_meta = {"run", "bioproject", "healthy", "K02_caries", "K05_gingivitis_periodontitis"}
    missing = sorted(required_meta - set(meta.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    for col in ["healthy", "K02_caries", "K05_gingivitis_periodontitis"]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0).astype(int)

    meta["is_healthy"] = (meta["healthy"] == 1).astype(int)
    meta["is_k02"] = (meta["K02_caries"] == 1).astype(int)
    meta["is_k05"] = (meta["K05_gingivitis_periodontitis"] == 1).astype(int)
    meta["bioproject"] = meta["bioproject"].astype(str)

    tax_cols = [c for c in tax.columns if c not in META_COLS]
    path_cols = [
        c for c in path.columns
        if c not in META_COLS
        and c not in UNWANTED_PATHWAY_COLS
        and "|" not in c
    ]

    leaked = [
        c for c in tax_cols + path_cols
        if c in {"healthy", "K02_caries", "K05_gingivitis_periodontitis", "K02", "K05"}
    ]
    if leaked:
        raise ValueError(f"Label leakage in feature columns: {leaked}")

    common_runs = sorted(set(meta["run"]) & set(tax["run"]) & set(path["run"]))
    if not common_runs:
        raise ValueError("No common runs across metadata, tax and path tables")

    def _align(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        use_cols = ["run"] + list(cols) if cols is not None else list(df.columns)
        out = (
            df.loc[df["run"].isin(common_runs), use_cols]
            .drop_duplicates("run")
            .sort_values("run")
            .set_index("run")
        )
        return out

    meta_a = _align(meta)
    tax_a = _align(tax, tax_cols)
    path_a = _align(path, path_cols)

    # Match Fig. 4 notebook: HUMAnN pathway CPM -> relative scale.
    X_tax = numeric_feature_frame(tax_a, tax_cols)
    X_path = numeric_feature_frame(path_a, path_cols) / PATHWAY_SCALE

    print(f"[DATA] Samples: {len(meta_a)}")
    print(
        "[DATA] Labels: "
        f"healthy={meta_a['is_healthy'].sum()}, "
        f"K05={meta_a['is_k05'].sum()}, "
        f"K02={meta_a['is_k02'].sum()}"
    )
    print(f"[DATA] TAX features: {X_tax.shape[1]}")
    print(f"[DATA] PATH features: {X_path.shape[1]}")
    print(f"[DATA] Bioprojects: {sorted(meta_a['bioproject'].unique())}")

    return {
        "meta": meta_a,
        "X_tax": X_tax,
        "X_path": X_path,
        "tax_cols": list(X_tax.columns),
        "path_cols": list(X_path.columns),
    }


# =============================================================================
# Configuration grid
# =============================================================================

def is_valid_config(cfg: Config) -> bool:
    if cfg.transform == "ILR" and not HAVE_SKBIO:
        return False
    if cfg.transform == "CLR" and not HAVE_SKBIO:
        return False
    if cfg.model_name == "XGBoost" and not HAVE_XGBOOST:
        return False
    return True


def build_config_grid(final_only: bool = False) -> List[Config]:
    if final_only:
        # Exactly the six configurations used for Fig. 4/5 plotting in the notebook.
        base = [
            ("K05", "TAX", "CLR", "none", "none", "LR_Ridge"),
            ("K05", "TAX+PATH", "CLR", "none", "none", "LR_Ridge"),
            ("K05", "PATH", "CLR", "none", "none", "LR_Ridge"),
            ("K02", "TAX", "rank.std", "none", "none", "XGBoost"),
            ("K02", "TAX+PATH", "CLR", "none", "none", "RF"),
            ("K02", "PATH", "CLR", "none", "none", "RF"),
        ]
        return [Config(*x) for x in base if is_valid_config(Config(*x))]

    configs = []
    for task, fs, tr, dr, bs, mn in itertools.product(
        TASKS, FEATURE_SETS, TRANSFORMS, DIMREDS, BATCH_STRATEGIES, MODEL_NAMES
    ):
        cfg = Config(task, fs, tr, dr, bs, mn)
        if is_valid_config(cfg):
            configs.append(cfg)
    return configs


# =============================================================================
# Split construction: target disease vs healthy only
# =============================================================================

def task_vectors(meta: pd.DataFrame, task: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    disease_col = "is_k05" if task == "K05" else "is_k02"
    y_full = meta[disease_col].to_numpy(dtype=int)
    healthy = meta["is_healthy"].to_numpy(dtype=bool)
    bp = meta["bioproject"].to_numpy(dtype=str)

    # The only comparison mode retained in this script.
    # Positive = target disease. Negative = healthy.
    keep = (y_full.astype(bool)) | healthy
    y = y_full[keep]
    bp_keep = bp[keep]
    idx_keep = np.where(keep)[0]
    return y, healthy[keep], bp_keep, idx_keep


def safe_stratification_labels(y: np.ndarray, bp: np.ndarray, n_splits: int) -> np.ndarray:
    """Prefer bioproject+label stratification, but collapse rare strata if needed."""
    labels = np.array([f"{b}__{int(yi)}" for b, yi in zip(bp, y)], dtype=object)
    counts = pd.Series(labels).value_counts()
    if counts.min() >= n_splits:
        return labels

    # Collapse rare bioproject-label strata to disease label only.
    collapsed = np.array([
        lab if counts[lab] >= n_splits else f"rare__{int(yi)}"
        for lab, yi in zip(labels, y)
    ], dtype=object)
    counts2 = pd.Series(collapsed).value_counts()
    if counts2.min() >= n_splits:
        return collapsed

    # Last fallback: stratify only by target class.
    return y.astype(str)


# =============================================================================
# Feature filtering and transformations
# =============================================================================


def select_feature_blocks(
    cfg: Config,
    X_tax_all: np.ndarray,
    X_path_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    bp_subset: np.ndarray,
    tax_cols: Sequence[str],
    path_cols: Sequence[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], Dict[str, object]]:
    bp_train = bp_subset[train_idx]

    keep_tax = prevalence_bioproject_filter(X_tax_all[train_idx], bp_train)
    keep_path = prevalence_bioproject_filter(X_path_all[train_idx], bp_train)

    if cfg.feature_set in {"TAX", "TAX+PATH"} and keep_tax.sum() == 0:
        raise ValueError("No TAX features after prevalence filter")
    if cfg.feature_set in {"PATH", "TAX+PATH"} and keep_path.sum() == 0:
        raise ValueError("No PATH features after prevalence filter")

    tr_blocks: List[np.ndarray] = []
    te_blocks: List[np.ndarray] = []
    names: List[str] = []

    if cfg.feature_set in {"TAX", "TAX+PATH"}:
        tr_blocks.append(X_tax_all[train_idx][:, keep_tax])
        te_blocks.append(X_tax_all[test_idx][:, keep_tax])
        names.extend([f"TAX__{c}" for c, k in zip(tax_cols, keep_tax) if k])

    if cfg.feature_set in {"PATH", "TAX+PATH"}:
        tr_blocks.append(X_path_all[train_idx][:, keep_path])
        te_blocks.append(X_path_all[test_idx][:, keep_path])
        names.extend([f"PATH__{c}" for c, k in zip(path_cols, keep_path) if k])

    info = {
        "n_tax_features": int(keep_tax.sum()),
        "n_path_features": int(keep_path.sum()),
        "keep_tax": keep_tax,
        "keep_path": keep_path,
    }
    return tr_blocks, te_blocks, names, info


def transform_blocks(
    tr_blocks: List[np.ndarray],
    te_blocks: List[np.ndarray],
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Transform selected feature blocks. CLR/ILR are blockwise for TAX+PATH."""
    block_names: List[str] = []

    if cfg.transform == "rank.std":
        X_tr_raw = np.concatenate(tr_blocks, axis=1)
        X_te_raw = np.concatenate(te_blocks, axis=1)
        mu, std = rank_std_fit(X_tr_raw)
        return rank_std_transform(X_tr_raw, mu, std), rank_std_transform(X_te_raw, mu, std), []

    tr_out: List[np.ndarray] = []
    te_out: List[np.ndarray] = []

    for bi, (Xtr, Xte) in enumerate(zip(tr_blocks, te_blocks)):
        # If both blocks are present, block 0 is TAX and block 1 is PATH.
        # If only one block is present, PATH needs relative normalization for CLR.
        is_path_block = (
            cfg.feature_set == "PATH"
            or (cfg.feature_set == "TAX+PATH" and bi == 1)
        )

        if cfg.transform == "CLR":
            tr = apply_clr(Xtr, force_relative=is_path_block)
            te = apply_clr(Xte, force_relative=is_path_block)
        elif cfg.transform == "ILR":
            tr = apply_ilr(Xtr)
            te = apply_ilr(Xte)
        else:
            raise ValueError(f"Unknown transform: {cfg.transform}")

        tr_out.append(tr)
        te_out.append(te)

        if cfg.transform == "ILR":
            prefix = "PATH_ilr" if is_path_block else "TAX_ilr"
            block_names.extend([f"{prefix}_{j}" for j in range(tr.shape[1])])

    return np.concatenate(tr_out, axis=1), np.concatenate(te_out, axis=1), block_names


# =============================================================================
# Batch correction
# =============================================================================

def manual_global_combat_like(X: np.ndarray, batches: np.ndarray) -> np.ndarray:
    """
    Fallback if pycombat/neuroCombat is unavailable.
    This is not empirical-Bayes ComBat. It standardizes each batch to the global
    mean/std and is provided only so the pipeline fails less often.
    """
    X = np.asarray(X, dtype=float)
    out = X.copy()
    grand_mean = X.mean(axis=0)
    grand_std = X.std(axis=0) + 1e-8
    for b in np.unique(batches):
        m = batches == b
        if m.sum() < 2:
            continue
        bm = X[m].mean(axis=0)
        bs = X[m].std(axis=0) + 1e-8
        out[m] = (X[m] - bm) / bs * grand_std + grand_mean
    return out


def try_pycombat_global(X: np.ndarray, batches: np.ndarray) -> np.ndarray:
    """Try several common ComBat APIs, then fall back to manual batch standardization."""
    X = np.asarray(X, dtype=float)
    batches = np.asarray(batches).astype(str)
    errors: List[str] = []

    # API variant 1: pycombat.Combat, samples x features.
    try:
        from pycombat import Combat  # type: ignore
        combat = Combat()
        for kwargs in (
            {"Y": X, "b": batches},
            {"data": X, "batch": batches},
        ):
            try:
                out = combat.fit_transform(**kwargs)
                out = np.asarray(out, dtype=float)
                if out.shape == X.shape and np.isfinite(out).all():
                    return out
            except Exception as e:
                errors.append(f"pycombat.Combat kwargs={list(kwargs)}: {e}")
    except Exception as e:
        errors.append(f"import pycombat.Combat: {e}")

    # API variant 2: neuroCombat, features x samples.
    try:
        from neuroCombat import neuroCombat  # type: ignore
        covars = pd.DataFrame({"batch": batches})
        out_dict = neuroCombat(dat=X.T, covars=covars, batch_col="batch")
        out = np.asarray(out_dict["data"].T, dtype=float)
        if out.shape == X.shape and np.isfinite(out).all():
            return out
    except Exception as e:
        errors.append(f"neuroCombat: {e}")

    print("[WARN] pycombat/neuroCombat failed; using manual global batch standardization fallback")
    print("       " + " | ".join(errors[:3]))
    return manual_global_combat_like(X, batches)


def apply_batch_strategy(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    bp_tr: np.ndarray,
    bp_te: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.batch_strategy == "none":
        return X_tr, X_te

    if cfg.batch_strategy == "pycombat_global_leak":
        X_all = np.vstack([X_tr, X_te])
        bp_all = np.concatenate([bp_tr, bp_te])
        X_corr = try_pycombat_global(X_all, bp_all)
        return X_corr[: len(X_tr)], X_corr[len(X_tr):]

    raise ValueError(f"Unknown batch_strategy: {cfg.batch_strategy}")


# =============================================================================
# Dimensionality reduction
# =============================================================================

def apply_dimred(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)

    if cfg.dimred == "none":
        return Xtr_s, Xte_s, {"n_final_features": int(Xtr_s.shape[1])}

    if cfg.dimred == "PCA":
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        Xtr_p = pca.fit_transform(Xtr_s)
        Xte_p = pca.transform(Xte_s)
        return Xtr_p, Xte_p, {
            "n_final_features": int(Xtr_p.shape[1]),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        }

    raise ValueError(f"Unknown dimred: {cfg.dimred}")


# =============================================================================
# Models
# =============================================================================

def get_model_and_grid(name: str):
    if name == "LR_Ridge":
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
        return model, grid

    if name == "RF":
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
        grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 5, 10],
            "max_features": ["sqrt", 0.3],
        }
        return model, grid

    if name == "SVM":
        model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
        grid = {
            "C": [0.01, 0.1, 1, 10],
            "gamma": ["scale", "auto", 0.01, 0.1],
        }
        return model, grid

    if name == "XGBoost":
        if not HAVE_XGBOOST:
            raise RuntimeError("XGBoost is not installed")
        model = XGBClassifier(
            eval_metric="logloss",
            verbosity=0,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        grid = {
            "n_estimators": [100, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
        }
        return model, grid

    raise ValueError(f"Unknown model: {name}")


def tune_and_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    model_name: str,
    n_jobs_inner: int,
) -> Tuple[np.ndarray, object, Dict[str, object]]:
    model, grid = get_model_and_grid(model_name)

    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_te), 0.5), model, {"best_params": None}

    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    if n_pos >= INNER_FOLDS and n_neg >= INNER_FOLDS:
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=inner_cv,
            scoring="roc_auc",
            refit=True,
            n_jobs=n_jobs_inner,
        )
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        probs = best.predict_proba(X_te)[:, 1]
        return probs, best, {"best_params": gs.best_params_}

    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    return probs, model, {"best_params": None}


# =============================================================================
# One split execution
# =============================================================================

def run_one_split(
    cfg: Config,
    X_tax_subset: np.ndarray,
    X_path_subset: np.ndarray,
    y_subset: np.ndarray,
    bp_subset: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    tax_cols: Sequence[str],
    path_cols: Sequence[str],
    n_jobs_inner: int,
) -> Tuple[Optional[Dict[str, float]], Optional[object], Dict[str, object], Optional[str]]:
    try:
        y_tr = y_subset[train_idx]
        y_te = y_subset[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            return None, None, {}, "single_class_train_or_test"

        tr_blocks, te_blocks, feature_names, info = select_feature_blocks(
            cfg,
            X_tax_subset,
            X_path_subset,
            train_idx,
            test_idx,
            bp_subset,
            tax_cols,
            path_cols,
        )

        Xtr_t, Xte_t, transformed_names = transform_blocks(tr_blocks, te_blocks, cfg)
        if transformed_names:
            feature_names = transformed_names

        Xtr_b, Xte_b = apply_batch_strategy(
            Xtr_t,
            Xte_t,
            bp_subset[train_idx],
            bp_subset[test_idx],
            cfg,
        )

        Xtr_d, Xte_d, dim_info = apply_dimred(Xtr_b, Xte_b, cfg)
        probs, model, model_info = tune_and_predict(Xtr_d, y_tr, Xte_d, cfg.model_name, n_jobs_inner)
        metrics = compute_metrics(y_te, probs)
        if metrics is None:
            return None, None, {}, "metrics_single_class_test"

        info.update(dim_info)
        info.update(model_info)
        info["n_raw_features_after_filter"] = int(Xtr_t.shape[1])
        info["n_train"] = int(len(train_idx))
        info["n_pos_train"] = int(y_tr.sum())
        info["n_neg_train"] = int((y_tr == 0).sum())
        info["n_test"] = int(len(test_idx))
        info["n_pos_test"] = int(y_te.sum())
        info["n_neg_test"] = int((y_te == 0).sum())
        info["feature_names"] = feature_names if cfg.dimred == "none" else []
        return metrics, model, info, None

    except Exception as e:
        return None, None, {}, str(e)


# =============================================================================
# Cross-validation
# =============================================================================

def run_cv_for_config(
    cfg: Config,
    data: Dict[str, object],
    out_dir: Path,
    n_jobs_inner: int,
) -> List[Dict[str, object]]:
    meta: pd.DataFrame = data["meta"]  # type: ignore
    X_tax: pd.DataFrame = data["X_tax"]  # type: ignore
    X_path: pd.DataFrame = data["X_path"]  # type: ignore
    tax_cols: List[str] = data["tax_cols"]  # type: ignore
    path_cols: List[str] = data["path_cols"]  # type: ignore

    y, _, bp, idx_keep = task_vectors(meta, cfg.task)
    X_tax_sub = X_tax.to_numpy(dtype=float)[idx_keep]
    X_path_sub = X_path.to_numpy(dtype=float)[idx_keep]

    labels = safe_stratification_labels(y, bp, OUTER_FOLDS)
    skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows: List[Dict[str, object]] = []
    oof = np.full(len(y), np.nan)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tax_sub, labels)):
        metrics, _, info, err = run_one_split(
            cfg,
            X_tax_sub,
            X_path_sub,
            y,
            bp,
            train_idx,
            test_idx,
            tax_cols,
            path_cols,
            n_jobs_inner,
        )

        base = {
            "task": cfg.task,
            "comparison_mode": "vs_healthy_only",
            "stage": "CV",
            "config_name": cfg.name,
            "config_slug": cfg.slug,
            "feature_set": cfg.feature_set,
            "transform": cfg.transform,
            "dimred": cfg.dimred,
            "batch_strategy": cfg.batch_strategy,
            "batch_method": "pycombat" if cfg.batch_strategy == "pycombat_global_leak" else "none",
            "model_name": cfg.model_name,
            "fold": fold,
        }

        if err:
            rows.append({**base, "status": "skip_or_error", "error": err})
            continue

        assert metrics is not None
        # Re-run only probabilities for OOF is avoided; not needed for S2 grid.
        rows.append({
            **base,
            "status": "ok",
            "error": "",
            "n_train": info.get("n_train"),
            "n_pos_train": info.get("n_pos_train"),
            "n_neg_train": info.get("n_neg_train"),
            "n_test": info.get("n_test"),
            "n_pos_test": info.get("n_pos_test"),
            "n_neg_test": info.get("n_neg_test"),
            "n_tax_features": info.get("n_tax_features"),
            "n_path_features": info.get("n_path_features"),
            "n_raw_features_after_filter": info.get("n_raw_features_after_filter"),
            "n_final_features": info.get("n_final_features"),
            "best_params": json_dumps_safe(info.get("best_params")),
            **metrics,
        })

    return rows


def summarize_cv(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = [
        "task", "comparison_mode", "feature_set", "transform", "dimred",
        "batch_strategy", "batch_method", "model_name", "config_name", "config_slug",
    ]
    out = (
        ok.groupby(group_cols)
        .agg(
            n_folds=("fold", "nunique"),
            mean_AUC=("AUC", "mean"),
            sd_AUC=("AUC", "std"),
            se_AUC=("AUC", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            mean_MCC=("MCC", "mean"),
            sd_MCC=("MCC", "std"),
            se_MCC=("MCC", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            mean_BACC=("BACC", "mean"),
            mean_n_train=("n_train", "mean"),
            mean_n_test=("n_test", "mean"),
            mean_n_final_features=("n_final_features", "mean"),
        )
        .reset_index()
    )
    return out.sort_values(["task", "mean_MCC", "mean_AUC"], ascending=[True, False, False])


# =============================================================================
# LOSO
# =============================================================================

def run_loso_for_config(
    cfg: Config,
    data: Dict[str, object],
    out_dir: Path,
    n_jobs_inner: int,
    save_models: bool,
) -> List[Dict[str, object]]:
    meta: pd.DataFrame = data["meta"]  # type: ignore
    X_tax: pd.DataFrame = data["X_tax"]  # type: ignore
    X_path: pd.DataFrame = data["X_path"]  # type: ignore
    tax_cols: List[str] = data["tax_cols"]  # type: ignore
    path_cols: List[str] = data["path_cols"]  # type: ignore

    y, _, bp, idx_keep = task_vectors(meta, cfg.task)
    X_tax_sub = X_tax.to_numpy(dtype=float)[idx_keep]
    X_path_sub = X_path.to_numpy(dtype=float)[idx_keep]

    rows: List[Dict[str, object]] = []
    model_dir = out_dir / "models" / "loso" / cfg.task / cfg.slug
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    for test_bp in sorted(np.unique(bp)):
        test_idx = np.where(bp == test_bp)[0]
        train_idx = np.where(bp != test_bp)[0]

        # Skip non-informative held-out projects; they remain in training for other folds.
        if len(test_idx) < 3 or len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
            continue

        metrics, model, info, err = run_one_split(
            cfg,
            X_tax_sub,
            X_path_sub,
            y,
            bp,
            train_idx,
            test_idx,
            tax_cols,
            path_cols,
            n_jobs_inner,
        )

        base = {
            "task": cfg.task,
            "comparison_mode": "vs_healthy_only",
            "stage": "LOSO",
            "config_name": cfg.name,
            "config_slug": cfg.slug,
            "feature_set": cfg.feature_set,
            "transform": cfg.transform,
            "dimred": cfg.dimred,
            "batch_strategy": cfg.batch_strategy,
            "batch_method": "pycombat" if cfg.batch_strategy == "pycombat_global_leak" else "none",
            "model_name": cfg.model_name,
            "test_bioproject": test_bp,
        }

        if err:
            rows.append({**base, "status": "skip_or_error", "error": err})
            continue

        assert metrics is not None
        model_path = ""
        if save_models and model is not None:
            model_path = str(model_dir / f"{test_bp}.joblib")
            joblib.dump(model, model_path)

        rows.append({
            **base,
            "status": "ok",
            "error": "",
            "n_train": info.get("n_train"),
            "n_pos_train": info.get("n_pos_train"),
            "n_neg_train": info.get("n_neg_train"),
            "n_test": info.get("n_test"),
            "n_pos_test": info.get("n_pos_test"),
            "n_neg_test": info.get("n_neg_test"),
            "n_tax_features": info.get("n_tax_features"),
            "n_path_features": info.get("n_path_features"),
            "n_raw_features_after_filter": info.get("n_raw_features_after_filter"),
            "n_final_features": info.get("n_final_features"),
            "best_params": json_dumps_safe(info.get("best_params")),
            "model_path": model_path,
            **metrics,
        })

    return rows


def summarize_loso(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    group_cols = [
        "task", "comparison_mode", "feature_set", "transform", "dimred",
        "batch_strategy", "batch_method", "model_name", "config_name", "config_slug",
    ]
    out = (
        ok.groupby(group_cols)
        .agg(
            n_folds=("test_bioproject", "nunique"),
            total_n_test=("n_test", "sum"),
            mean_AUC=("AUC", "mean"),
            sd_AUC=("AUC", "std"),
            se_AUC=("AUC", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            weighted_AUC=("AUC", lambda x: np.average(x, weights=ok.loc[x.index, "n_test"])),
            mean_MCC=("MCC", "mean"),
            sd_MCC=("MCC", "std"),
            se_MCC=("MCC", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            weighted_MCC=("MCC", lambda x: np.average(x, weights=ok.loc[x.index, "n_test"])),
            mean_BACC=("BACC", "mean"),
            weighted_BACC=("BACC", lambda x: np.average(x, weights=ok.loc[x.index, "n_test"])),
            mean_n_train=("n_train", "mean"),
            mean_n_test=("n_test", "mean"),
            mean_n_final_features=("n_final_features", "mean"),
        )
        .reset_index()
    )
    return out.sort_values(["task", "weighted_MCC", "weighted_AUC"], ascending=[True, False, False])


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tax", default=DEFAULT_TAX_PATH)
    parser.add_argument("--path", default=DEFAULT_PATH_PATH)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs over configurations")
    parser.add_argument("--n-jobs-inner", type=int, default=-1, help="GridSearchCV parallel jobs")
    parser.add_argument("--stage", choices=["cv", "loso", "both"], default="both")
    parser.add_argument("--final-only", action="store_true", help="Run only the six final plotting configs from the notebook")
    parser.add_argument("--save-models", action="store_true", help="Save LOSO models as joblib files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.tax, args.path, args.metadata)
    configs = build_config_grid(final_only=args.final_only)

    if not configs:
        raise RuntimeError("No valid configurations. Check optional dependencies: scikit-bio, xgboost")

    pd.DataFrame([c.__dict__ | {"config_name": c.name, "config_slug": c.slug} for c in configs]).to_csv(
        out_dir / "article_config_grid.csv", index=False
    )

    print("\n[GRID] Article-aligned configurations")
    print(f"  configs: {len(configs)}")
    print(f"  feature_sets: {FEATURE_SETS}")
    print(f"  transforms: {[t for t in TRANSFORMS if t != 'ILR' or HAVE_SKBIO]}")
    print(f"  dimreds: {DIMREDS}")
    print(f"  batch: {BATCH_STRATEGIES}")
    print(f"  models: {[m for m in MODEL_NAMES if m != 'XGBoost' or HAVE_XGBOOST]}")
    print("  comparison_mode: vs_healthy_only")

    t0 = time.time()

    if args.stage in {"cv", "both"}:
        print("\n[RUN] Cross-validation")
        cv_rows_nested = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(run_cv_for_config)(cfg, data, out_dir, args.n_jobs_inner)
            for cfg in configs
        )
        cv_rows = [row for rows in cv_rows_nested for row in rows]
        cv_df = pd.DataFrame(cv_rows)
        cv_df.to_csv(out_dir / "cv_article_configs.csv", index=False)
        cv_summary = summarize_cv(cv_df)
        cv_summary.to_csv(out_dir / "cv_article_configs_summary.csv", index=False)
        print(f"[SAVE] {out_dir / 'cv_article_configs.csv'}")
        print(f"[SAVE] {out_dir / 'cv_article_configs_summary.csv'}")
        if not cv_summary.empty:
            print("\n[TOP CV]")
            print(cv_summary.head(20).to_string(index=False))

    if args.stage in {"loso", "both"}:
        print("\n[RUN] LOSO")
        loso_rows_nested = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(run_loso_for_config)(cfg, data, out_dir, args.n_jobs_inner, args.save_models)
            for cfg in configs
        )
        loso_rows = [row for rows in loso_rows_nested for row in rows]
        loso_df = pd.DataFrame(loso_rows)
        loso_df.to_csv(out_dir / "loso_article_configs.csv", index=False)
        loso_summary = summarize_loso(loso_df)
        loso_summary.to_csv(out_dir / "loso_article_configs_summary.csv", index=False)
        print(f"[SAVE] {out_dir / 'loso_article_configs.csv'}")
        print(f"[SAVE] {out_dir / 'loso_article_configs_summary.csv'}")
        if not loso_summary.empty:
            print("\n[TOP LOSO]")
            print(loso_summary.head(20).to_string(index=False))

    print(f"\n[DONE] elapsed={(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
