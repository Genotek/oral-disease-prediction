# =============================================================================
# 00_data.R - shared data loading and feature filtering
#
# Sourced by every other script; not run on its own.
#
# Loading, alignment and filtering only; no models. Every choice that affects a
# reported number is a constant at the top of this file and is echoed to the log,
# so it can be carried into Methods verbatim.
# =============================================================================

# The local library must come first: it holds MMUPHin 1.20.0, where the function
# is lm_meta and runs on Maaslin2, the same engine used throughout the paper.
# MMUPHin 2.x renames it to maaslin_meta and runs it on maaslin3, a different
# model, so it is not used here.
# Pinning the version in a separate library keeps the run independent of whatever
# is installed system-wide. Path from SALIVARY_WGS_RLIB, default Rlib/.
# If that directory is absent, system libraries are used and the version check
# below reports the mismatch.
.RLIB <- Sys.getenv("SALIVARY_WGS_RLIB", unset = "Rlib")
if (dir.exists(.RLIB)) .libPaths(c(normalizePath(.RLIB), .libPaths()))

suppressPackageStartupMessages({
  library(MMUPHin)
  library(Maaslin2)
  library(metafor)
  library(vegan)
})

# --- Pinned package versions --------------------------------------------------
# Also listed in DESCRIPTION. A mismatch stops the run: the reported
# numbers are reproducible on this set of versions only.
REQUIRED_VERSIONS <- c(MMUPHin  = "1.20.0",   # 1.x only: has lm_meta, Maaslin2 engine
                       Maaslin2 = "1.26.0",
                       metafor  = "5.0.1",
                       vegan    = "2.7.5",
                       ggplot2  = "4.0.3",
                       ggrepel  = "0.9.8",
                       dplyr    = "1.2.1")

check_versions <- function(strict = TRUE) {
  have <- vapply(names(REQUIRED_VERSIONS),
                 function(p) as.character(packageVersion(p)), character(1))
  bad <- have != REQUIRED_VERSIONS
  msg <- paste(sprintf("  %-9s need %-8s have %s%s", names(REQUIRED_VERSIONS),
                       REQUIRED_VERSIONS, have, ifelse(bad, "   <-- MISMATCH", "")),
               collapse = "\n")
  cat("Package versions:\n", msg, "\n", sep = "")
  if (any(bad) && strict)
    stop("Package versions do not match the pinned set. ",
         "MMUPHin must be 1.x, which provides lm_meta on the Maaslin2 engine.")
  if (!"lm_meta" %in% ls("package:MMUPHin"))
    stop("The loaded MMUPHin has no lm_meta. ",
         "Check that SALIVARY_WGS_RLIB comes first in .libPaths().")
  invisible(TRUE)
}

write_session_info <- function(path) {
  writeLines(c(capture.output(sessionInfo()),
               "", "--- .libPaths() ---", .libPaths()), path)
}

set.seed(42)

# --- Paths --------------------------------------------------------------------
# paths.py reads the same two environment variables, so the R and Python
# halves always look in the same place. Without them, directories inside the
# repository are used.
PROJ     <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."), ".."),
                          mustWork = FALSE)
if (!file.exists(file.path(PROJ, "paths.py"))) PROJ <- getwd()
DATA_DIR <- Sys.getenv("SALIVARY_WGS_DATA_DIR", unset = file.path(PROJ, "data"))
OUT_DIR  <- Sys.getenv("SALIVARY_WGS_OUTPUT_DIR",
                       unset = file.path(PROJ, "outputs"))
if (!all(file.exists(file.path(DATA_DIR, c("tax.csv", "path.csv", "metadata.csv")))))
  stop("tax.csv, path.csv, metadata.csv not found in ", DATA_DIR,
       ". Put them in <repo>/data/ or set SALIVARY_WGS_DATA_DIR.")
RES_DIR  <- file.path(OUT_DIR, "results")
FIG_DIR  <- file.path(OUT_DIR, "figures")
dir.create(RES_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# --- Analysis parameters (these are what Methods reports) ---------------------
# The 10 % threshold sits below the one-third detection level at
# which Gibbons et al. (2018) found ComBat acceptable; since no correction is
# applied here, that constraint does not bind, and the threshold is reported as
# a limitation instead.
# Samples carrying both diagnoses (K02 and K05).
# TRUE  = drop them; this is the sensitivity analysis reported in Table S4
# FALSE = keep them as cases for the disease under test; this is the primary
#         analysis, 365 and 384 samples
# All 31 come from one bioproject, PRJNA1057503.
EXCLUDE_DUAL_POSITIVE <- FALSE

MIN_PREVALENCE  <- 0.10   # fraction of samples in which a feature is non-zero
MIN_BIOPROJECTS <- 3      # minimum number of bioprojects a feature appears in
# MMUPHin::lm_meta defaults, applied to the pooled runs as well so that the two
# levels of analysis are compared under identical preprocessing
NORMALIZATION   <- "TSS"  # required: pathways are raw RPK, and 66.8 % of the variance in log depth is explained by bioproject
TRANSFORM       <- "LOG"  # keeps the coefficient scale interpretable (AST does not)
ANALYSIS_METHOD <- "LM"
RMA_METHOD      <- "REML" # between-study variance estimator in the meta-analysis
SIG             <- 0.05   # Benjamini-Hochberg adjusted q threshold

`%||%` <- function(a, b) if (is.null(a)) b else a

# --- Loading ------------------------------------------------------------------
load_all <- function() {
  meta <- read.csv(file.path(DATA_DIR, "metadata.csv"), check.names = FALSE,
                   stringsAsFactors = FALSE)
  tax  <- read.csv(file.path(DATA_DIR, "tax.csv"),  check.names = FALSE,
                   stringsAsFactors = FALSE)
  path <- read.csv(file.path(DATA_DIR, "path.csv"), check.names = FALSE,
                   stringsAsFactors = FALSE)

  for (nm in c("meta", "tax", "path")) {
    d <- get(nm); d$run <- trimws(as.character(d$run)); assign(nm, d)
  }

  # One sample set: the intersection of all three tables, so every step of the
  # analysis works on the same samples.
  common <- Reduce(intersect, list(meta$run, tax$run, path$run))
  common <- sort(common)

  meta <- meta[match(common, meta$run), , drop = FALSE]
  tax  <- tax [match(common, tax$run ), , drop = FALSE]
  path <- path[match(common, path$run), , drop = FALSE]
  rownames(meta) <- rownames(tax) <- rownames(path) <- common

  for (cc in c("healthy", "K02_caries", "K05_gingivitis_periodontitis")) {
    meta[[cc]] <- as.integer(ifelse(is.na(suppressWarnings(as.numeric(meta[[cc]]))), 0,
                                    as.numeric(meta[[cc]])))
  }
  meta$bioproject <- as.character(meta$bioproject)

  # Bookkeeping and species-stratified pathway columns are dropped
  drop_path <- grepl("\\|", colnames(path)) |
               colnames(path) %in% c("run", "UNMAPPED", "UNINTEGRATED")
  X_path <- as.matrix(path[, !drop_path, drop = FALSE])
  X_tax  <- as.matrix(tax [, setdiff(colnames(tax), "run"), drop = FALSE])
  storage.mode(X_tax) <- "numeric"; storage.mode(X_path) <- "numeric"
  X_tax[is.na(X_tax)] <- 0; X_path[is.na(X_path)] <- 0

  list(meta = meta, tax = X_tax, path = X_path)
}

# --- Per-disease subset -------------------------------------------------------
# Cases for the target disease, or healthy controls.
# Samples carrying both diagnoses are dropped when EXCLUDE_DUAL_POSITIVE = TRUE,
# which is the sensitivity analysis; the primary analysis keeps them as cases.
task_subset <- function(D, task, exclude_dual = EXCLUDE_DUAL_POSITIVE) {
  stopifnot(task %in% c("K05", "K02"))
  case_col <- if (task == "K05") "K05_gingivitis_periodontitis" else "K02_caries"
  is_case    <- D$meta[[case_col]] == 1
  is_healthy <- D$meta$healthy == 1
  is_dual    <- D$meta$K02_caries == 1 & D$meta$K05_gingivitis_periodontitis == 1
  if (exclude_dual) is_case <- is_case & !is_dual
  keep <- is_case | is_healthy

  md <- D$meta[keep, , drop = FALSE]
  md$exposure  <- factor(ifelse(is_case[keep], "case", "control"),
                         levels = c("control", "case"))
  md$batch     <- factor(md$bioproject)
  list(meta = md,
       tax  = D$tax [keep, , drop = FALSE],
       path = D$path[keep, , drop = FALSE],
       task = task)
}

# --- Feature filtering --------------------------------------------------------
filter_features <- function(X, batch, min_prev = MIN_PREVALENCE,
                            min_bp = MIN_BIOPROJECTS) {
  present <- X > 0
  prev    <- colMeans(present)
  n_bp    <- apply(present, 2, function(p)
                     if (any(p)) length(unique(batch[p])) else 0L)
  keep <- prev >= min_prev & n_bp >= min_bp
  attr(keep, "prevalence") <- prev
  attr(keep, "n_bioprojects") <- n_bp
  keep
}

# Safe identifiers: pathway names contain ':', ',' and '(', which MaAsLin2
# mangles. Work by ID and restore the real names only in the output tables.
make_feature_ids <- function(feature_names, prefix) {
  ids <- sprintf("%s%04d", prefix, seq_along(feature_names))
  data.frame(feature_id = ids, feature = feature_names,
             stringsAsFactors = FALSE)
}

# --- Which bioprojects can enter the meta-analysis ----------------------------
# A case-control effect cannot be estimated inside a study that contributes only
# one of the two groups. This is a property of the data, not a choice.
two_class_batches <- function(md, min_per_class = 3) {
  tb <- table(md$batch, md$exposure)
  ok <- rownames(tb)[tb[, "case"] >= min_per_class & tb[, "control"] >= min_per_class]
  sort(ok)
}

log_msg <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), ..., "\n", sep = "")
