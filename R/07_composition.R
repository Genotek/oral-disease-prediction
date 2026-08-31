# =============================================================================
# 07_composition.R - species-level composition and per-bioproject detail
#
# Produces:
#   (1) a summary species-level composition of the aggregated dataset (Table S8);
#   (2) for each differentially abundant taxon, abundance, prevalence and effect
#       direction within every bioproject (Table S12).
#
# Distinguishing reproducible changes from those carried by one large cohort is
# handled by the meta-analysis (I2, cohort weights, leave-one-cohort-out).
#
# Output: results/composition/
# =============================================================================

source("R/00_data.R")
OUTD <- file.path(RES_DIR, "composition")
dir.create(OUTD, recursive = TRUE, showWarnings = FALSE)
SIG <- 0.05

log_msg("Loading data")
D <- load_all()
X <- D$tax / rowSums(D$tax); X[!is.finite(X)] <- 0   # relative abundance, rows sum to 1
bp <- D$meta$bioproject

# --- (1) summary species composition ------------------------------------------
# Mean relative abundance and prevalence over the whole set, plus the number of
# bioprojects in which the species was detected at all.
present <- X > 0
overall <- data.frame(
  species          = colnames(X),
  mean_abundance   = colMeans(X) * 100,
  median_abundance = apply(X, 2, median) * 100,
  mean_when_present = ifelse(colSums(present) > 0,
                             colSums(X) / pmax(colSums(present), 1) * 100, 0),
  prevalence       = colMeans(present) * 100,
  n_bioprojects    = apply(present, 2, function(p) length(unique(bp[p]))),
  stringsAsFactors = FALSE)
overall <- overall[order(-overall$mean_abundance), ]
write.csv(overall, file.path(OUTD, "species_composition_overall.csv"), row.names = FALSE)
log_msg(sprintf("composition: %d species; top 5 by mean abundance: %s",
                nrow(overall), paste(head(overall$species, 5), collapse = ", ")))

# the same split by clinical group
grp <- with(D$meta, ifelse(K02_caries == 1 & K05_gingivitis_periodontitis == 1, "K02+K05",
                    ifelse(K02_caries == 1, "K02",
                    ifelse(K05_gingivitis_periodontitis == 1, "K05", "Healthy"))))
by_group <- do.call(rbind, lapply(sort(unique(grp)), function(g) {
  i <- grp == g
  data.frame(group = g, n_samples = sum(i), species = colnames(X),
             mean_abundance = colMeans(X[i, , drop = FALSE]) * 100,
             prevalence     = colMeans(X[i, , drop = FALSE] > 0) * 100,
             stringsAsFactors = FALSE)
}))
write.csv(by_group, file.path(OUTD, "species_composition_by_group.csv"), row.names = FALSE)

# --- (2) per bioproject, for differentially abundant taxa ---------------------
da <- unique(unlist(lapply(c("K05", "K02"), function(task) {
  d <- read.csv(file.path(RES_DIR, "differential_abundance", sprintf("S3_%s_taxa.csv", task)),
                stringsAsFactors = FALSE)
  d$feature[d$qval < SIG]
})))
log_msg("differentially abundant taxa (union of both comparisons): ", length(da))

eff <- read.csv(file.path(RES_DIR, "per_study_effects.csv"), stringsAsFactors = FALSE)
eff <- eff[eff$space == "TAX", c("feature", "batch", "coef", "qval", "task")]

rows <- list()
for (task in c("K05", "K02")) {
  S   <- task_subset(D, task)
  Xt  <- S$tax / rowSums(S$tax); Xt[!is.finite(Xt)] <- 0
  sig <- read.csv(file.path(RES_DIR, "differential_abundance", sprintf("S3_%s_taxa.csv", task)),
                  stringsAsFactors = FALSE)
  sig <- sig$feature[sig$qval < SIG]
  Xt  <- Xt[, colnames(Xt) %in% sig, drop = FALSE]

  for (b in sort(unique(S$meta$bioproject))) {
    i <- S$meta$bioproject == b
    for (cls in c("case", "control")) {
      j <- i & S$meta$exposure == cls
      if (!any(j)) next
      rows[[paste(task, b, cls)]] <- data.frame(
        task = task, bioproject = b, group = cls, n = sum(j),
        species = colnames(Xt),
        mean_abundance = colMeans(Xt[j, , drop = FALSE]) * 100,
        prevalence     = colMeans(Xt[j, , drop = FALSE] > 0) * 100,
        stringsAsFactors = FALSE)
    }
  }
}
per_bp <- do.call(rbind, rows); rownames(per_bp) <- NULL

# put cases and controls on one row and attach the within-cohort effect
w <- reshape(per_bp, idvar = c("task", "bioproject", "species"), timevar = "group",
             direction = "wide")
names(w) <- sub("\\.case$", "_cases", sub("\\.control$", "_controls", names(w)))
w <- merge(w, eff, by.x = c("species", "bioproject", "task"),
           by.y = c("feature", "batch", "task"), all.x = TRUE)
names(w)[names(w) == "coef"] <- "beta_within_cohort"
names(w)[names(w) == "qval"] <- "q_within_cohort"
w$direction <- ifelse(is.na(w$beta_within_cohort), NA,
                      ifelse(w$beta_within_cohort > 0, "higher in cases",
                                                       "higher in controls"))
w <- w[order(w$task, w$species, w$bioproject), ]
write.csv(w, file.path(OUTD, "DA_taxa_by_bioproject.csv"), row.names = FALSE)
log_msg(sprintf("per bioproject: %d rows (%d taxa x %d cohorts x 2 comparisons)",
                nrow(w), length(da), length(unique(w$bioproject))))
log_msg("Done: ", OUTD)
