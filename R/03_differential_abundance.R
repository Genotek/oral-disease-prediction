# =============================================================================
# 03_differential_abundance.R - pooled MaAsLin2 for both diseases and both spaces
#
# Pathway abundances are depth-normalised and the feature filter is applied
# here, before the call.
#
# MaAsLin2 settings:
#   normalization  = "TSS"          (see the note below)
#   transform      = "LOG"
#   min_prevalence = 0.0            (filtering is done by us, before the call)
#   fixed_effects  = "target"       (1 = case, 0 = control)
#   random_effects = "bioproject"
#   pathways: UNMAPPED, UNINTEGRATED and species-stratified columns are dropped
#
# The random intercept matters: without it the pathway comparison returns 214
# significant features against 24 with it. With it, coefficients reproduce the
# published ones to r = 1.0000.
#
# The random-effects meta-analysis is a separate script, 04_meta_analysis.R.
# =============================================================================

source("R/00_data.R")   # paths, analysis parameters, version check

.libPaths(c(normalizePath("Rlib", mustWork = FALSE), .libPaths()))
suppressPackageStartupMessages({ library(Maaslin2); library(dplyr) })

# Samples carrying both diagnoses (K02 and K05):
#   FALSE = keep them as cases for the disease under test (primary analysis)
#   TRUE  = drop them (sensitivity analysis, Table S4)
EXCLUDE_DUAL <- as.logical(Sys.getenv("EXCLUDE_DUAL", "FALSE"))

# DATA_DIR and OUT_DIR come from 00_data.R and read the environment variables
OUT_DIR  <- file.path(RES_DIR, if (EXCLUDE_DUAL)
  "differential_abundance_sensitivity_no_dual" else "differential_abundance")
RAW_DIR  <- file.path(OUT_DIR, "maaslin2_raw")
dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)

tax  <- read.csv(file.path(DATA_DIR, "tax.csv"),  check.names = FALSE)
path <- read.csv(file.path(DATA_DIR, "path.csv"), check.names = FALSE)
meta <- read.csv(file.path(DATA_DIR, "metadata.csv"), check.names = FALSE)

rownames(tax) <- tax$run; tax$run <- NULL
tax[] <- lapply(tax, as.numeric); tax[is.na(tax)] <- 0

rownames(path) <- path$run; path$run <- NULL
path <- path[, !colnames(path) %in% c("UNMAPPED", "UNINTEGRATED"), drop = FALSE]
path <- path[, !grepl("\\|", colnames(path)), drop = FALSE]
path[] <- lapply(path, as.numeric); path[is.na(path)] <- 0

meta$run <- as.character(meta$run); rownames(meta) <- meta$run
for (cc in c("healthy", "K02_caries", "K05_gingivitis_periodontitis"))
  meta[[cc]] <- as.integer(meta[[cc]])

mixed_ids <- meta %>%
  filter(K02_caries == 1, K05_gingivitis_periodontitis == 1) %>% pull(run)
cat(if (EXCLUDE_DUAL) "MODE: comorbid samples EXCLUDED, n =" else
    "MODE: comorbid samples KEPT as cases, n =", length(mixed_ids), "\n\n")

# Feature filter, as in common.py of the original code:
#   prevalence_bioproject_filter(MIN_PREVALENCE = 0.10, MIN_BIOPROJECTS = 3)
# Without it the output holds roughly three times as many features.
# --- Normalisation ------------------------------------------------------------
# The original analysis ran with normalization = "NONE". That is defensible for
# taxa, whose MetaPhlAn profiles are already relative (row sums 0.43-1), but not
# for pathways: HUMAnN returns raw RPK, per-sample totals run from 1,841 to
# 4,768,728, and 66.8 % of the variance in log depth is explained by bioproject.
# Without TSS the model sees sequencing depth, which is two thirds cohort
# identity. TSS is applied to both spaces so preprocessing is identical.
# LOG is kept from the original analysis: the coefficient scale is unchanged.
NORM  <- Sys.getenv("MAASLIN_NORM",  "TSS")
TRANS <- Sys.getenv("MAASLIN_TRANS", "LOG")
cat("Normalisation:", NORM, "| transform:", TRANS, "\n\n")

prevalence_bioproject_filter <- function(X, bp, min_prev = 0.10, min_bp = 3) {
  present <- X > 0
  prev <- colMeans(present)
  n_bp <- apply(present, 2, function(p)
                  if (any(p)) length(unique(bp[p])) else 0L)
  prev >= min_prev & n_bp >= min_bp
}

run_one <- function(feat, disease_col, tag) {
  md <- meta
  if (EXCLUDE_DUAL) md <- md %>% filter(!run %in% mixed_ids)
  md <- md %>%
    filter(healthy == 1 | .data[[disease_col]] == 1) %>%
    mutate(target = ifelse(.data[[disease_col]] == 1, 1, 0)) %>%
    select(run, bioproject, target)
  rownames(md) <- md$run

  ids <- intersect(rownames(feat), rownames(md))
  X <- feat[ids, , drop = FALSE]; md <- md[ids, , drop = FALSE]

  keep <- prevalence_bioproject_filter(as.matrix(X), md$bioproject)
  X <- X[, keep, drop = FALSE]

  cat(sprintf("%-16s samples=%d (cases=%d, controls=%d), features after filter=%d\n",
              tag, nrow(md), sum(md$target == 1), sum(md$target == 0), ncol(X)))

  out <- file.path(RAW_DIR, tag)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  suppressMessages(suppressWarnings(
    Maaslin2(input_data = X, input_metadata = md, output = out,
             fixed_effects  = c("target"),
             random_effects = c("bioproject"),
             normalization  = NORM,
             transform      = TRANS,
             min_prevalence = 0.0,
             plot_heatmap = FALSE, plot_scatter = FALSE, cores = 4)))

  res <- read.delim(file.path(out, "all_results.tsv"), stringsAsFactors = FALSE)
  res <- res[res$metadata == "target", ]
  res <- res[order(res$qval, res$pval), ]
  cat(sprintf("%-16s features returned=%d, significant q<0.05=%d\n\n",
              "", nrow(res), sum(res$qval < 0.05, na.rm = TRUE)))
  res
}

# Pathway annotation, as in the original Figure 3 notebook
fmt_id   <- function(x) gsub("\\.", "-", sub("\\.\\..*$", "", x))
fmt_desc <- function(x) {
  p <- strsplit(x, "\\.\\.", perl = TRUE)
  vapply(p, function(v) if (length(v) < 2) ""
         else trimws(gsub("\\s+", " ", gsub("\\.", " ", v[2]))), character(1))
}

annotate_path <- function(res) {
  data.frame(
    pathway_id          = fmt_id(res$feature),
    pathway_description = fmt_desc(res$feature),
    direction           = ifelse(res$coef > 0, "enriched_in_disease", "enriched_in_healthy"),
    metacyc_url         = paste0("https://pubchem.ncbi.nlm.nih.gov/pathway/BioCyc:META_",
                                 fmt_id(res$feature)),
    res, check.names = FALSE, stringsAsFactors = FALSE)
}

r_k02_tax  <- run_one(tax,  "K02_caries",                   "K02_vs_healthy_TAX")
r_k05_tax  <- run_one(tax,  "K05_gingivitis_periodontitis", "K05_vs_healthy_TAX")
r_k02_path <- run_one(path, "K02_caries",                   "K02_vs_healthy_PATH")
r_k05_path <- run_one(path, "K05_gingivitis_periodontitis", "K05_vs_healthy_PATH")

write.csv(r_k02_tax,  file.path(OUT_DIR, "S3_K02_taxa.csv"),      row.names = FALSE)
write.csv(r_k05_tax,  file.path(OUT_DIR, "S3_K05_taxa.csv"),      row.names = FALSE)
write.csv(annotate_path(r_k02_path), file.path(OUT_DIR, "S3_K02_pathways.csv"), row.names = FALSE)
write.csv(annotate_path(r_k05_path), file.path(OUT_DIR, "S3_K05_pathways.csv"), row.names = FALSE)

cat("=== SUMMARY ===\n")
sm <- data.frame(
  sheet   = c("S3. K02 taxa", "S3. K05 taxa", "S3. K02 pathways", "S3.1 K05 pathways"),
  N_old   = c(428, 384, 384, 365),
  N_correct_published = c("no", "no", "yes", "yes"),
  N_new   = c(unique(r_k02_tax$N), unique(r_k05_tax$N),
              unique(r_k02_path$N), unique(r_k05_path$N)),
  rows_old = c(357, 351, 333, 333),
  rows_new = c(nrow(r_k02_tax), nrow(r_k05_tax), nrow(r_k02_path), nrow(r_k05_path)),
  sig_new  = c(sum(r_k02_tax$qval  < 0.05), sum(r_k05_tax$qval  < 0.05),
               sum(r_k02_path$qval < 0.05), sum(r_k05_path$qval < 0.05)))
print(sm, row.names = FALSE)
write.csv(sm, file.path(OUT_DIR, "_summary.csv"), row.names = FALSE)
cat("\nTables:", OUT_DIR, "\n")
