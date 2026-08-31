# =============================================================================
# 04_meta_analysis.R - per-cohort effects combined by random-effects meta-analysis
#
# The confirmatory level of the analysis.
#   «A random-effects meta-analysis reporting a heterogeneity statistic such as I²
#    — for example MMUPHin's lm_meta, from the same bioBakery toolchain already
#    in use, rather than its data-level adjust_batch»
#
# Uses MMUPHin::lm_meta from MMUPHin 1.20.0. The meta-analysis is not assembled
# by hand: the wrapper is called as is, with its own defaults (TSS, AST, LM,
# REML). Internally it fits Maaslin2 within each study and pools the coefficients
# through metafor::rma.uni.
#
# No data-level correction is applied, neither ComBat nor adjust_batch: study
# structure is modelled rather than edited out of the abundances.
#
# Three reference Maaslin2 runs are reported alongside. Preprocessing is identical
# (TSS/AST/LM), so the only difference is how study membership is handled:
#   (1) pooled_all - all 14 bioprojects pooled, study ignored
#   (2) pooled_6   - the same on the six two-class bioprojects, which separates
#                    the change of sample set from the change of model
#   (3) ri_6       - random intercept by study
#   (4) lm_meta    - effects estimated within study, then pooled
# =============================================================================

source("R/00_data.R")
check_versions()

log_msg("Loading data")
D <- load_all()

MIN_PER_CLASS <- 3
SIG <- 0.05

# Reference Maaslin2 run with the same parameters lm_meta uses internally
run_maaslin <- function(X, md, out, random = NULL) {
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  suppressMessages(suppressWarnings(
    Maaslin2(input_data     = as.data.frame(X),
             input_metadata = md,
             output          = out,
             min_abundance   = 0,
             min_prevalence  = 0,          # filtering already done upstream
             normalization   = NORMALIZATION,
             transform       = TRANSFORM,
             analysis_method = ANALYSIS_METHOD,
             fixed_effects   = "exposure",
             random_effects  = random,
             correction      = "BH",
             standardize     = FALSE,
             plot_heatmap    = FALSE,
             plot_scatter    = FALSE,
             save_models     = FALSE)))
  res <- read.delim(file.path(out, "all_results.tsv"), stringsAsFactors = FALSE)
  res[res$value == "case", c("feature", "coef", "stderr", "pval", "qval")]
}

all_meta <- list(); all_per_study <- list(); all_compare <- list(); all_loso <- list()

for (task in c("K05", "K02")) {
  S <- task_subset(D, task)
  ok_batches <- two_class_batches(S$meta, MIN_PER_CLASS)

  for (space in c("TAX", "PATH")) {
    log_msg("=== ", task, " / ", space, " ===")
    X_full <- if (space == "TAX") S$tax else S$path
    keep   <- filter_features(X_full, S$meta$batch)
    X_full <- X_full[, keep, drop = FALSE]

    map <- make_feature_ids(colnames(X_full), paste0(space, "_"))
    colnames(X_full) <- map$feature_id

    md_all <- S$meta
    sel6   <- md_all$batch %in% ok_batches
    X6     <- X_full[sel6, , drop = FALSE]
    md6    <- droplevels(md_all[sel6, , drop = FALSE])

    tag <- paste0(task, "_", space)
    od  <- file.path(RES_DIR, "maaslin_raw", tag)
    dir.create(file.path(od, "lm_meta"), recursive = TRUE, showWarnings = FALSE)

    r_all <- run_maaslin(X_full, md_all, file.path(od, "pooled_all"))
    r_6   <- run_maaslin(X6,     md6,    file.path(od, "pooled_6"))
    r_ri  <- run_maaslin(X6,     md6,    file.path(od, "ri_6"), random = "batch")

    # --- within-study effects pooled by random-effects meta-analysis -----------
    fit <- suppressMessages(suppressWarnings(
      lm_meta(feature_abd = t(X6),            # features x samples
              batch       = "batch",
              exposure    = "exposure",
              data        = md6,
              control     = list(normalization   = NORMALIZATION,
                                 transform       = TRANSFORM,
                                 analysis_method = ANALYSIS_METHOD,
                                 rma_method      = RMA_METHOD,
                                 output          = file.path(od, "lm_meta"),
                                 forest_plot     = NULL,
                                 verbose         = FALSE))))

    # Pooled estimate, I2, tau2 and the weight of each study. The weights show
    # how much of the pooled signal comes from any one cohort.
    mf <- fit$meta_fits
    names(mf)[names(mf) == "qval.fdr"] <- "qval"
    names(mf)[names(mf) == "feature"]  <- "feature_id"
    rownames(mf) <- NULL

    # Within-study estimates, used by Figure S3 and Table S12
    ps <- do.call(rbind, lapply(names(fit$maaslin_fits), function(b) {
      d <- fit$maaslin_fits[[b]]
      d <- d[d$metadata == "exposure" & d$value == "case", , drop = FALSE]
      if (!nrow(d)) return(NULL)
      data.frame(batch = b, feature_id = d$feature, coef = d$coef,
                 stderr = d$stderr, pval = d$pval, qval = d$qval,
                 stringsAsFactors = FALSE)
    }))

    # --- Refit features where REML failed to converge --------------------------
    # metafor estimates tau2 by Fisher scoring; on some features the algorithm
    # does not converge and lm_meta silently returns NA for the whole row. The
    # feature itself can be perfectly well behaved: Treponema_denticola has
    # consistent positive estimates in all six cohorts yet comes back empty.
    # A reduced step size (stepadj = 0.5) fixes it, as documented in help(rma).
    # Such features are refitted here, flagged in the meta_refit column, and the
    # BH correction is then recomputed across all features at once.
    na_ids <- mf$feature_id[is.na(mf$pval)]
    mf$meta_refit <- FALSE
    if (length(na_ids)) {
      wcols <- grep("^weight_", names(mf), value = TRUE)
      for (fid in na_ids) {
        g <- ps[ps$feature_id == fid & is.finite(ps$coef) &
                is.finite(ps$stderr) & ps$stderr > 0, , drop = FALSE]
        if (nrow(g) < 2) next
        f2 <- try(suppressWarnings(rma.uni(yi = g$coef, sei = g$stderr,
                    method = RMA_METHOD,
                    control = list(stepadj = 0.5, maxiter = 1000))), silent = TRUE)
        if (inherits(f2, "try-error")) {
          log_msg("    ", fid, ": no convergence even with stepadj; left unestimated")
          next
        }
        i <- which(mf$feature_id == fid)
        mf$coef[i] <- as.numeric(f2$beta); mf$stderr[i] <- f2$se
        mf$pval[i] <- f2$pval;             mf$k[i]      <- f2$k
        mf$I2[i]   <- f2$I2;               mf$tau2[i]   <- f2$tau2
        w <- 1 / (g$stderr^2 + f2$tau2); w <- 100 * w / sum(w)
        for (j in seq_len(nrow(g))) {
          cn <- paste0("weight_", g$batch[j])
          if (cn %in% wcols) mf[i, cn] <- w[j]
        }
        mf$meta_refit[i] <- TRUE
        log_msg(sprintf("    REML refit (stepadj=0.5): %s | k=%d p=%.3g I2=%.0f%%",
                        fid, f2$k, f2$pval, f2$I2))
      }
      mf$qval <- p.adjust(mf$pval, method = "BH")
    }

    # Leave-one-cohort-out meta-analysis: lm_meta does not offer this, so it is
    # recomputed with the same rma.uni on the estimates the wrapper returned.
    loso <- do.call(rbind, lapply(c(list(NULL), as.list(ok_batches)), function(b) {
      d <- ps[!(ps$batch %in% b), , drop = FALSE]
      d <- d[is.finite(d$coef) & is.finite(d$stderr) & d$stderr > 0, , drop = FALSE]
      rr <- lapply(split(d, d$feature_id), function(g) {
        if (nrow(g) < 2) return(NULL)
        f <- try(suppressWarnings(rma.uni(yi = g$coef, sei = g$stderr,
                                          method = RMA_METHOD,
                                          control = list(maxiter = 1000))), silent = TRUE)
        # Same convergence control as the refit above: without it the features
        # rescued by stepadj = 0.5 drop out of this table and its baseline row
        # disagrees with the main fit.
        if (inherits(f, "try-error"))
          f <- try(suppressWarnings(rma.uni(yi = g$coef, sei = g$stderr,
                     method = RMA_METHOD,
                     control = list(stepadj = 0.5, maxiter = 1000))), silent = TRUE)
        if (inherits(f, "try-error")) return(NULL)
        data.frame(feature_id = g$feature_id[1], pval = f$pval, I2 = f$I2,
                   coef = as.numeric(f$beta), stringsAsFactors = FALSE)
      })
      rr <- do.call(rbind, rr); if (is.null(rr)) return(NULL)
      rr$qval <- p.adjust(rr$pval, method = "BH")
      data.frame(dropped = if (is.null(b)) "(none)" else b,
                 n_sig = sum(rr$qval < SIG, na.rm = TRUE),
                 median_I2 = median(rr$I2, na.rm = TRUE), stringsAsFactors = FALSE)
    }))

    nm <- function(d, s) { names(d)[names(d) != "feature"] <-
        paste0(s, "_", names(d)[names(d) != "feature"]); d }
    cmp <- Reduce(function(a, b) merge(a, b, by = "feature", all = TRUE),
                  list(nm(r_all, "pooledAll"), nm(r_6, "pooled6"), nm(r_ri, "ri6")))
    names(cmp)[1] <- "feature_id"
    mm <- mf[, c("feature_id", "coef", "stderr", "pval", "qval", "k", "I2", "tau2")]
    names(mm)[names(mm) != "feature_id"] <- paste0("meta_", names(mm)[names(mm) != "feature_id"])
    cmp <- merge(merge(map, cmp, by = "feature_id", all.y = TRUE), mm,
                 by = "feature_id", all.x = TRUE)
    cmp$task <- task; cmp$space <- space

    ps    <- merge(map, ps, by = "feature_id", all.y = TRUE); ps$task <- task; ps$space <- space
    m_out <- merge(map, mf, by = "feature_id", all.y = TRUE)
    m_out$task <- task; m_out$space <- space
    if (!is.null(loso)) { loso$task <- task; loso$space <- space }

    all_meta[[tag]] <- m_out; all_per_study[[tag]] <- ps
    all_compare[[tag]] <- cmp; all_loso[[tag]] <- loso

    log_msg(sprintf("  features=%d cohorts=%d | significant: pooled_all=%d pooled_6=%d ri_6=%d lm_meta=%d | median I2=%.1f%%",
                    nrow(map), length(ok_batches),
                    sum(r_all$qval < SIG, na.rm = TRUE), sum(r_6$qval < SIG, na.rm = TRUE),
                    sum(r_ri$qval < SIG, na.rm = TRUE), sum(mf$qval < SIG, na.rm = TRUE),
                    median(mf$I2, na.rm = TRUE)))
  }
}

rbind_fill <- function(l) {
  l <- Filter(Negate(is.null), l)
  cols <- unique(unlist(lapply(l, names)))
  do.call(rbind, lapply(l, function(d) {
    for (cc in setdiff(cols, names(d))) d[[cc]] <- NA
    d[, cols, drop = FALSE]
  }))
}
w <- function(l, f) write.csv(rbind_fill(l), file.path(RES_DIR, f), row.names = FALSE)
w(all_meta,      "meta_analysis_pooled.csv")
w(all_per_study, "per_study_effects.csv")
w(all_compare,   "model_comparison.csv")
w(all_loso,      "leave_one_study_out_meta.csv")
write_session_info(file.path(RES_DIR, "sessionInfo.txt"))
log_msg("Done. Results in ", RES_DIR)
