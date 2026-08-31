"""Single place where the Python side resolves paths.

Data and output directories come from environment variables; without them,
directories inside the repository are used. R/00_data.R reads the same two
variables, so the R and Python halves always look in the same place.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("SALIVARY_WGS_DATA_DIR", REPO / "data")).expanduser()
OUT_DIR = Path(os.getenv("SALIVARY_WGS_OUTPUT_DIR", REPO / "outputs")).expanduser()

TAX_CSV = DATA_DIR / "tax.csv"
PATH_CSV = DATA_DIR / "path.csv"
META_CSV = DATA_DIR / "metadata.csv"

ML_DIR = OUT_DIR / "ml"
MODELS_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"
FIGURES_DIR = OUT_DIR / "figures"
PERMANOVA_CSV = RESULTS_DIR / "permanova.csv"

for _d in (ML_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require_data() -> None:
    """A named error instead of a traceback from inside pandas."""
    missing = [p.name for p in (TAX_CSV, PATH_CSV, META_CSV) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Input files not found in {DATA_DIR}: {', '.join(missing)}.\n"
            "Put tax.csv, path.csv and metadata.csv in <repo>/data/ "
            "or set SALIVARY_WGS_DATA_DIR."
        )
