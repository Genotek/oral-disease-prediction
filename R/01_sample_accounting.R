# =============================================================================
# 01_sample_accounting.R - sample flow from 514 profiled samples to the analysis sets
#
# «Please state one sample count per analysis.»
#
# Single source of every sample count quoted in the manuscript. Produces:
#   1. Samples split into mutually exclusive categories
#   2. Attrition table: 514 down to the final N of each analysis
#   3. Per-bioproject composition (Table S6)
#   4. Primary analysis against the comorbid-excluded sensitivity analysis
# =============================================================================

source("R/00_data.R")
check_versions()

D <- load_all()
md <- D$meta
h  <- md$healthy == 1
k2 <- md$K02_caries == 1
k5 <- md$K05_gingivitis_periodontitis == 1
dl <- k2 & k5

cat("\n=== 1. Sample categories (mutually exclusive) ===\n")
cat_tab <- data.frame(
  category = c("Healthy controls", "Caries only (K02)", "Periodontitis only (K05)",
               "Both diagnoses (K02+K05)", "TOTAL"),
  n = c(sum(h & !k2 & !k5), sum(k2 & !k5), sum(k5 & !k2), sum(dl), nrow(md)))
print(cat_tab, row.names = FALSE)
stopifnot(sum(cat_tab$n[1:4]) == nrow(md))
stopifnot(sum(h & (k2 | k5)) == 0)          # a control cannot also be a case

cat("\n=== 2. Attrition table ===\n")
flow <- list()
for (task in c("K05", "K02")) {
  cc   <- if (task == "K05") k5 else k2
  keep_incl <- cc | h
  keep_excl <- (cc & !dl) | h
  flow[[task]] <- data.frame(
    task = task,
    step = c("Samples with metadata, taxonomic and pathway profiles",
             "Excluded: other disease only",
             "Excluded: both diagnoses (comorbid)",
             "ANALYSED"),
    n_removed = c(NA, sum(!keep_incl), sum(dl), NA),
    n_remaining = c(nrow(md), sum(keep_incl), sum(keep_excl), sum(keep_excl)),
    cases = c(NA, sum(cc), sum(cc & !dl), sum(cc & !dl)),
    controls = c(NA, sum(h), sum(h), sum(h)))
}
flow <- do.call(rbind, flow); rownames(flow) <- NULL
print(flow, row.names = FALSE)

cat("\n=== 3. One N per analysis ===\n")
final <- data.frame(
  analysis = c("K05 taxonomic DA", "K05 pathway DA", "K05 classification",
               "K02 taxonomic DA", "K02 pathway DA", "K02 classification"),
  N = rep(c(sum(((k5 & !dl) | h)), sum(((k2 & !dl) | h))), each = 3),
  cases = rep(c(sum(k5 & !dl), sum(k2 & !dl)), each = 3),
  controls = rep(sum(h), 6))
final$in_manuscript <- c(384, 365, 365, 428, 384, 384)
final$note <- ifelse(final$N == final$in_manuscript, "matches", "MISMATCH")
print(final, row.names = FALSE)

cat("\n=== 4. Per bioproject (source of Table S6) ===\n")
bp <- do.call(rbind, lapply(split(seq_len(nrow(md)), md$bioproject), function(i) {
  data.frame(BioProject = md$bioproject[i][1], n_total = length(i),
             healthy = sum(h[i]), caries_only = sum(k2[i] & !k5[i]),
             perio_only = sum(k5[i] & !k2[i]), both = sum(dl[i]),
             K05_analysed = sum((k5[i] & !dl[i]) | h[i]),
             K02_analysed = sum((k2[i] & !dl[i]) | h[i]))
}))
bp$K05_two_class <- ifelse(bp$healthy >= 3 & (bp$perio_only) >= 3, "yes", "-")
bp$K02_two_class <- ifelse(bp$healthy >= 3 & (bp$caries_only) >= 3, "yes", "-")
rownames(bp) <- NULL
print(bp, row.names = FALSE)

cat("\nTwo-class bioprojects: K05 =", sum(bp$K05_two_class == "yes"),
    " K02 =", sum(bp$K02_two_class == "yes"), "\n")
cat("Bioprojects containing comorbid samples:",
    paste(bp$BioProject[bp$both > 0], collapse = ", "), "\n")

write.csv(cat_tab, file.path(RES_DIR, "sample_categories.csv"), row.names = FALSE)
write.csv(flow,    file.path(RES_DIR, "sample_flow.csv"), row.names = FALSE)
write.csv(final,   file.path(RES_DIR, "final_N_per_analysis.csv"), row.names = FALSE)
write.csv(bp,      file.path(RES_DIR, "cohort_table_S1.csv"), row.names = FALSE)
cat("\nWritten to", RES_DIR, "\n")
