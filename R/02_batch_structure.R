# =============================================================================
# 02_batch_structure.R - study structure, with no correction applied to the data
#
# No data-level correction is applied here, neither ComBat nor adjust_batch:
# study structure is modelled rather than removed, so there is no "how much is
# left after correction" to report.
#
# Computed here:
#   1. Marginal models, one term at a time
#   2. Joint model, each term adjusted for the other. Without it, comparing the
#      two marginal R2 values directly is not meaningful
#   3. PERMDISP: a significant PERMANOVA also arises when groups differ in spread
#      rather than in location
#   4. An unfiltered run, which is what Figure 2 shows
# =============================================================================

source("R/00_data.R")
check_versions()

set.seed(42)
NPERM <- 999

log_msg("Loading data")
D <- load_all()

grp <- with(D$meta, ifelse(K02_caries == 1 & K05_gingivitis_periodontitis == 1, "K02+K05",
                    ifelse(K02_caries == 1, "Caries (K02)",
                    ifelse(K05_gingivitis_periodontitis == 1, "Periodontitis (K05)",
                    ifelse(healthy == 1, "Healthy", NA)))))
D$meta$group <- factor(grp)
stopifnot(!any(is.na(D$meta$group)))

# Aitchison: per-sample CLR, then Euclidean distance.
# Pseudocount 1e-6 as in config.py (CLR_PSEUDO).
aitchison <- function(X, pseudo = 1e-6) {
  L <- log(X + pseudo); dist(L - rowMeans(L), method = "euclidean")
}

permanova_set <- function(X, md, space, filtered, sset) {
  out <- list()
  for (metric in c("Bray-Curtis", "Aitchison")) {
    Dm <- if (metric == "Bray-Curtis") vegdist(X, method = "bray") else aitchison(X)

    a_bp <- adonis2(Dm ~ bioproject, data = md, permutations = NPERM, by = "terms")
    a_gr <- adonis2(Dm ~ group,      data = md, permutations = NPERM, by = "terms")
    a_jt <- adonis2(Dm ~ bioproject + group, data = md,
                    permutations = NPERM, by = "margin")
    bd_bp <- anova(betadisper(Dm, md$bioproject))
    bd_gr <- anova(betadisper(Dm, md$group))

    out[[metric]] <- data.frame(
      space = space, filtered = filtered, metric = metric, sample_set = sset,
      n_features = ncol(X), n_samples = nrow(X),
      R2_bioproject_marginal = a_bp$R2[1], p_bioproject_marginal = a_bp$`Pr(>F)`[1],
      R2_group_marginal      = a_gr$R2[1], p_group_marginal      = a_gr$`Pr(>F)`[1],
      R2_bioproject_joint = a_jt$R2[rownames(a_jt) == "bioproject"],
      R2_group_joint      = a_jt$R2[rownames(a_jt) == "group"],
      p_group_joint       = a_jt$`Pr(>F)`[rownames(a_jt) == "group"],
      permdisp_p_bioproject = bd_bp$`Pr(>F)`[1],
      permdisp_p_group      = bd_gr$`Pr(>F)`[1],
      stringsAsFactors = FALSE)
  }
  do.call(rbind, out)
}

# Two sample sets.
#   "pure483" - three mutually exclusive groups. Ordination by clinical group
#               needs unambiguous labels, so the 31 comorbid samples are dropped
#               from it. This is the set reported in Methods 2.3.
#   "all514"  - every sample, comorbid ones as a fourth level. Kept in the table
#               for reference; not quoted in the text.
# Neither set changes the analysis sets: differential abundance and model training
# still use 365 and 384; the restriction applies to ordination only.
SAMPLE_SETS <- list(
  pure483 = which(D$meta$group != "K02+K05"),
  all514  = seq_len(nrow(D$meta)))

results <- list()
for (sset in names(SAMPLE_SETS)) {
  idx <- SAMPLE_SETS[[sset]]
  md  <- droplevels(D$meta[idx, , drop = FALSE])
  log_msg("=== sample set ", sset, ": ", length(idx), " samples, groups ",
          nlevels(md$group), " ===")

  for (space in c("TAX", "PATH")) {
    X0 <- (if (space == "TAX") D$tax else D$path)[idx, , drop = FALSE]

    # (a) unfiltered, as in the published Figure 2 notebook
    log_msg(space, ": no feature filter (", ncol(X0), ")")
    Xr <- X0 / rowSums(X0); Xr[!is.finite(Xr)] <- 0
    results[[paste(sset, space, "raw", sep = "_")]] <-
      permanova_set(Xr, md, space, "no", sset)

    # (b) same filter as the meta-analysis
    keep <- filter_features(X0, md$bioproject)
    X <- X0[, keep, drop = FALSE]; X <- X / rowSums(X); X[!is.finite(X)] <- 0
    log_msg(space, ": filtered (", ncol(X), ")")
    results[[paste(sset, space, "filt", sep = "_")]] <-
      permanova_set(X, md, space, "yes", sset)
  }
}

res <- do.call(rbind, results); rownames(res) <- NULL
write.csv(res, file.path(RES_DIR, "permanova.csv"), row.names = FALSE)
print(res[, c("sample_set", "space", "filtered", "metric", "R2_bioproject_marginal",
              "R2_group_marginal", "R2_bioproject_joint", "R2_group_joint",
              "permdisp_p_bioproject")], digits = 3)
log_msg("Done")
