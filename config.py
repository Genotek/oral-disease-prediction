from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def env_path(name: str, default: str | Path) -> Path:
    """Read a path from an environment variable or use a portable default."""
    return Path(os.getenv(name, str(default))).expanduser()


# =============================================================================
# Main project paths
# =============================================================================

DATA_DIR = env_path("SALIVARY_WGS_DATA_DIR", PROJECT_ROOT / "data")
OUTPUT_DIR = env_path("SALIVARY_WGS_OUTPUT_DIR", PROJECT_ROOT / "outputs")

DEFAULT_TAX_PATH = DATA_DIR / "tax.csv"
DEFAULT_PATH_PATH = DATA_DIR / "path.csv"
DEFAULT_METADATA_PATH = DATA_DIR / "metadata.csv"
DEFAULT_OUT_DIR = OUTPUT_DIR

DIST_DIR = DATA_DIR / "distances"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURE_OUTPUT_DIR = FIGURES_DIR
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"


# =============================================================================
# Technical QC paths
# =============================================================================

QC_CLEAN_DATA_DIR = env_path(
    "SALIVARY_WGS_QC_CLEAN_DATA_DIR",
    DATA_DIR / "qc" / "clean_data",
)
QC_MULTIQC_BASE = env_path(
    "SALIVARY_WGS_QC_MULTIQC_BASE",
    DATA_DIR / "qc" / "multiqc_final",
)
QC_RESULTS_DIR = RESULTS_DIR / "technical_qc"
QC_FIGURES_DIR = FIGURES_DIR / "technical_qc"


# =============================================================================
# Reproducibility and model evaluation
# =============================================================================

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3


# =============================================================================
# Feature filtering and compositional transforms
# =============================================================================

CLR_PSEUDO = 1e-6
MIN_PREVALENCE = 0.10
MIN_BIOPROJECTS = 3
PATHWAY_SCALE = 1e6


# =============================================================================
# Article-aligned configuration grid
# =============================================================================

TASKS = ["K05", "K02"]
FEATURE_SETS = ["TAX", "PATH", "TAX+PATH"]
TRANSFORMS = ["CLR", "ILR", "rank.std"]
DIMREDS = ["none", "PCA"]
BATCH_STRATEGIES = ["none", "pycombat_global_leak"]
MODEL_NAMES = ["LR_Ridge", "RF", "SVM", "XGBoost"]


# =============================================================================
# Metadata and feature-column handling
# =============================================================================

META_COLS = {
    "run",
    "bioproject",
    "source",
    "layout",
    "year",
    "country",
    "sequencing_platform",
    "instrument",
    "instrument_model",
    "library_strategy",
    "library_source",
    "health_status",
    "healthy",
    "K02_caries",
    "K05_gingivitis_periodontitis",
    "K02",
    "K05",
    "is_healthy",
    "is_k05",
    "is_k02",
    "disease_label",
}

UNWANTED_PATHWAY_COLS = {"UNMAPPED", "UNINTEGRATED"}

GROUP_ORDER = [
    "Healthy",
    "Caries (K02)",
    "Periodontitis (K05)",
    "K02+K05",
]
