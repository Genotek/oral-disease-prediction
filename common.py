"""Shared helper functions for the salivary WGS meta-analysis.

Only put functions here when they are reused across scripts/notebooks.
Figure-specific plotting code should stay in the corresponding notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

from config import CLR_PSEUDO, MIN_BIOPROJECTS, MIN_PREVALENCE

try:
    from skbio.stats.composition import clr as skbio_clr
    from skbio.stats.composition import ilr as skbio_ilr
    HAVE_SKBIO = True
except Exception:
    skbio_clr = None
    skbio_ilr = None
    HAVE_SKBIO = False

try:
    from skbio.stats.distance import permanova
    HAVE_SKBIO_DISTANCE = True
except Exception:
    permanova = None
    HAVE_SKBIO_DISTANCE = False


def read_csv_with_run(path: str | Path) -> pd.DataFrame:
    """Read a CSV table and normalize the required 'run' column."""
    df = pd.read_csv(path, low_memory=False)
    if "run" not in df.columns:
        raise ValueError(f"No 'run' column in {path}")
    df["run"] = df["run"].astype(str).str.strip()
    return df


def numeric_feature_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Return selected columns as numeric features with missing values set to 0."""
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def assign_group(row: pd.Series) -> str:
    """Map metadata disease flags to a compact clinical group label."""
    has_k02 = row.get("K02_caries", 0) == 1
    has_k05 = row.get("K05_gingivitis_periodontitis", 0) == 1
    has_healthy = row.get("healthy", 0) == 1

    if has_k02 and has_k05:
        return "K02+K05"
    if has_k02:
        return "Caries (K02)"
    if has_k05:
        return "Periodontitis (K05)"
    if has_healthy:
        return "Healthy"
    return "Unknown"


def pval_to_stars(p: float) -> Optional[str]:
    """Convert a p-value or q-value to a compact significance label."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, preserving the original order."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])

    order = np.argsort(pvals)
    ranked = pvals[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    qvals = np.empty(n)
    qvals[order] = q_ranked
    return qvals


def permanova_r2(dm, grouping: Sequence[object], permutations: int = 999) -> Dict[str, float]:
    """Run PERMANOVA and estimate R2 from the pseudo-F statistic."""
    if not HAVE_SKBIO_DISTANCE:
        raise RuntimeError("scikit-bio is required for PERMANOVA")

    grouping = pd.Series(grouping, index=dm.ids)
    res = permanova(dm, grouping, permutations=permutations)
    n = len(dm.ids)
    k = grouping.nunique()
    pseudo_f = float(res["test statistic"])
    r2 = pseudo_f * (k - 1) / (pseudo_f * (k - 1) + (n - k))

    return {
        "pseudo_F": round(pseudo_f, 3),
        "p_value": float(res["p-value"]),
        "R2": round(float(r2), 4),
    }


def confidence_ellipse(x, y, ax, n_std: float = 1.5, **kwargs):
    """Draw a covariance ellipse on an existing matplotlib axis."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        return None

    cov = np.cov(x, y)
    denom = np.sqrt(max(cov[0, 0] * cov[1, 1], 1e-20))
    pearson = np.clip(cov[0, 1] / denom, -1, 1)

    ellipse = Ellipse(
        (0, 0),
        width=np.sqrt(max(1 + pearson, 1e-9)) * 2,
        height=np.sqrt(max(1 - pearson, 1e-9)) * 2,
        **kwargs,
    )

    transform = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(
            np.sqrt(max(cov[0, 0], 1e-9)) * n_std,
            np.sqrt(max(cov[1, 1], 1e-9)) * n_std,
        )
        .translate(np.mean(x), np.mean(y))
    )

    ellipse.set_transform(transform + ax.transData)
    ax.add_patch(ellipse)
    return ellipse


def prevalence_bioproject_filter(
    X: np.ndarray,
    bp: np.ndarray,
    min_prevalence: float = MIN_PREVALENCE,
    min_bioprojects: int = MIN_BIOPROJECTS,
) -> np.ndarray:
    """Keep features present in enough samples and enough bioprojects."""
    X = np.asarray(X)
    bp = np.asarray(bp).astype(str)
    present = X > 0
    prev = present.mean(axis=0)
    n_bp = np.array([
        len(np.unique(bp[present[:, j]])) if present[:, j].any() else 0
        for j in range(X.shape[1])
    ])
    return (prev >= min_prevalence) & (n_bp >= min_bioprojects)


def to_relative(X: np.ndarray) -> np.ndarray:
    """Convert rows to relative abundances while preserving all-zero rows."""
    X = np.asarray(X, dtype=float)
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return X / s


def apply_clr(
    X: np.ndarray,
    force_relative: bool = False,
    pseudocount: float = CLR_PSEUDO,
) -> np.ndarray:
    """Apply CLR transform. Optionally close rows to relative abundances first."""
    if not HAVE_SKBIO:
        raise RuntimeError("scikit-bio is required for CLR")
    X_in = to_relative(X) if force_relative else np.asarray(X, dtype=float)
    return skbio_clr(X_in + pseudocount)


def apply_ilr(X: np.ndarray, pseudocount: float = CLR_PSEUDO) -> np.ndarray:
    """Apply ILR transform after row closure and pseudocount addition."""
    if not HAVE_SKBIO:
        raise RuntimeError("scikit-bio is required for ILR")
    X_in = to_relative(np.asarray(X, dtype=float)) + pseudocount
    X_in = to_relative(X_in)
    if X_in.shape[1] < 2:
        raise ValueError("ILR requires at least 2 features")
    return skbio_ilr(X_in)


def rank_std_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit rank.std parameters: within-sample ranks, then feature-wise scaling."""
    r = np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, X)
    mu = r.mean(axis=0)
    std = r.std(axis=0)
    std[std == 0] = 1.0
    return mu, std


def rank_std_transform(X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply rank.std using parameters fitted on the training data."""
    r = np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, X)
    return (r - mu) / std


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Optional[Dict[str, float]]:
    """Compute binary classification metrics used in the article tables."""
    if len(np.unique(y_true)) < 2:
        return None
    pred = (probs >= 0.5).astype(int)
    return {
        "AUC": round(float(roc_auc_score(y_true, probs)), 6),
        "MCC": round(float(matthews_corrcoef(y_true, pred)), 6),
        "BACC": round(float(balanced_accuracy_score(y_true, pred)), 6),
    }


def write_rows(path: Path, rows: List[Dict[str, object]], mode: str = "a") -> None:
    """Append rows to a CSV file."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists() or mode == "w"
    df.to_csv(path, mode=mode, header=header, index=False)


def json_dumps_safe(obj: object) -> str:
    """Serialize an object for CSV output without breaking the pipeline."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)
