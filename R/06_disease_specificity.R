# =============================================================================
# 06_disease_specificity.R - is a feature specific to one disease or shared?
#
# Several pathways are significant in the same direction in both comparisons, and
# PWY-622 is more strongly associated with periodontal disease than with caries
# although it was read as caries-specific. A direct contrast is needed before any
# directional metabolic claim.
#
# Two things are computed:
#
#   (1) Overlap. Both comparisons share the same 204 controls, so a feature
#       significant in the same direction in both cannot be specific to either.
#       This is well powered and uses results already computed.
#
#   (2) Direct contrast: K02-only cases against K05-only cases, with controls and
#       comorbid participants excluded. Power is limited here: only two cohorts
#       hold both classes (PRJNA1057503 and PRJNA396840, 118 of 279 samples), so
#       both the pooled fit and the two-cohort meta-analysis are reported, along
#       with the disagreement between them.
#
# Output: results/specificity/
# =============================================================================

source("R/00_data.R")
check_versions()

OUTD <- file.path(RES_DIR, "specificity")
dir.create(file.path(OUTD, "maaslin_raw"), recursive = TRUE, showWarnings = FALSE)
SIG <- 0.05

# ------------------------------------------------------------------ (1) overlap
log_msg("Overlap between the two comparisons")
read_pub <- function(task, space) {
  f <- file.path(RES_DIR, "differential_abundance",
                 sprintf("S3_%s_%s.csv", task, if (space == "TAX") "taxa" else "pathways"))
  d <- read.csv(f, stringsAsFactors = FALSE)
  key <- if (space == "TAX") d$feature else trimws(d$pathway_id)
  data.frame(key = key, coef = d$coef, qval = d$qval,
             desc = if (space == "TAX") "" else d$pathway_description,
             stringsAsFactors = FALSE)
}

ov <- list()
for (space in c("TAX", "PATH")) {
  a <- read_pub("K05", space); b <- read_pub("K02", space)
  m <- merge(a, b, by = "key", suffixes = c("_K05", "_K02"))
  m$sig_K05 <- m$qval_K05 < SIG
  m$sig_K02 <- m$qval_K02 < SIG
  m$same_sign <- sign(m$coef_K05) == sign(m$coef_K02)
  m$shared <- m$sig_K05 & m$sig_K02 & m$same_sign
  # which disease the feature leans toward; this is the PWY-622 question
  m$stronger_in <- ifelse(!m$shared, NA,
                   ifelse(abs(m$coef_K05) > abs(m$coef_K02), "K05", "K02"))
  m$space <- space
  ov[[space]] <- m[order(-m$shared, m$qval_K05), ]
  log_msg(sprintf("  %s: significant K05=%d, K02=%d, both same direction=%d",
                  space, sum(m$sig_K05), sum(m$sig_K02), sum(m$shared)))
}
overlap <- do.call(rbind, ov); rownames(overlap) <- NULL
write.csv(overlap, file.path(OUTD, "shared_features.csv"), row.names = FALSE)

# --------------------------------------------------------- (2) direct contrast
log_msg("Direct contrast: K02-only against K05-only")
D <- load_all()
m <- D$meta
is02 <- m$K02_caries == 1 & m$K05_gingivitis_periodontitis == 0
is05 <- m$K05_gingivitis_periodontitis == 1 & m$K02_caries == 0
sel  <- is02 | is05

md <- m[sel, , drop = FALSE]
md$exposure <- factor(ifelse(is02[sel], "caries", "periodontal"),
                      levels = c("periodontal", "caries"))
md$batch <- droplevels(factor(md$bioproject))
tb <- table(md$batch, md$exposure)
ok <- rownames(tb)[tb[, "caries"] >= 3 & tb[, "periodontal"] >= 3]
log_msg(sprintf("  %d samples (%d caries, %d periodontal); cohorts with both classes: %d (%d samples)",
                nrow(md), sum(is02), sum(is05), length(ok), sum(md$batch %in% ok)))

res <- list()
for (space in c("TAX", "PATH")) {
  X0 <- (if (space == "TAX") D$tax else D$path)[sel, , drop = FALSE]
  keep <- filter_features(X0, md$batch)
  X <- X0[, keep, drop = FALSE]
  map <- make_feature_ids(colnames(X), paste0(space, "_"))
  colnames(X) <- map$feature_id

  out <- file.path(OUTD, "maaslin_raw", paste0("pooled_", space))
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  suppressMessages(suppressWarnings(
    Maaslin2(input_data = as.data.frame(X),
             input_metadata = data.frame(exposure = md$exposure, batch = md$batch,
                                         row.names = rownames(X)),
             output = out, fixed_effects = "exposure", random_effects = "batch",
             normalization = NORMALIZATION, transform = TRANSFORM,
             analysis_method = ANALYSIS_METHOD, min_prevalence = 0,
             plot_heatmap = FALSE, plot_scatter = FALSE, cores = 4)))
  r <- read.delim(file.path(out, "all_results.tsv"), stringsAsFactors = FALSE)
  r <- r[r$metadata == "exposure", ]
  r <- merge(map, r, by.x = "feature_id", by.y = "feature")
  r$space <- space; r$model <- "pooled"
  log_msg(sprintf("  %s pooled: features=%d, significant=%d",
                  space, nrow(r), sum(r$qval < SIG, na.rm = TRUE)))

  # meta-analysis over the two cohorts
  sel6 <- md$batch %in% ok
  mf <- NULL
  if (length(ok) >= 2) {
    fit <- suppressMessages(suppressWarnings(
      lm_meta(feature_abd = t(X[sel6, , drop = FALSE]), batch = "batch",
              exposure = "exposure", data = droplevels(md[sel6, , drop = FALSE]),
              control = list(normalization = NORMALIZATION, transform = TRANSFORM,
                             analysis_method = ANALYSIS_METHOD, rma_method = RMA_METHOD,
                             output = file.path(out, "lm_meta"),
                             forest_plot = NULL, verbose = FALSE))))
    mf <- fit$meta_fits
    names(mf)[names(mf) == "qval.fdr"] <- "qval"
    log_msg(sprintf("  %s meta over %d cohorts: significant=%d",
                    space, length(ok), sum(mf$qval < SIG, na.rm = TRUE)))
  }
  res[[space]] <- list(pooled = r, meta = mf)
  write.csv(r, file.path(OUTD, sprintf("direct_contrast_pooled_%s.csv", space)), row.names = FALSE)
  if (!is.null(mf)) write.csv(mf, file.path(OUTD, sprintf("direct_contrast_meta_%s.csv", space)), row.names = FALSE)
}

summ <- do.call(rbind, lapply(names(res), function(s) data.frame(
  space = s, n_samples = nrow(md), n_cohorts_two_class = length(ok),
  n_samples_two_class = sum(md$batch %in% ok),
  n_features = nrow(res[[s]]$pooled),
  n_sig_pooled = sum(res[[s]]$pooled$qval < SIG, na.rm = TRUE),
  n_sig_meta = if (is.null(res[[s]]$meta)) NA else sum(res[[s]]$meta$qval < SIG, na.rm = TRUE))))
write.csv(summ, file.path(OUTD, "direct_contrast_summary.csv"), row.names = FALSE)
cat("\n"); print(summ)
log_msg("Done: ", OUTD)
