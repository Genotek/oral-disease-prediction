# Leave-one-cohort-out meta-analysis, using the same convergence control as the
# main fit (stepadj = 0.5). Without it, features rescued by the REML refit in
# 04_meta_analysis.R drop out here and the baseline row disagrees with the number
# of significant features in the meta-analysis (59 against 60 for K05 taxa; the
# difference is Treponema_denticola).
source("R/00_data.R")   # paths, analysis parameters, version check
suppressPackageStartupMessages(library(metafor))
RES <- RES_DIR
ps <- read.csv(file.path(RES, "per_study_effects.csv"), stringsAsFactors = FALSE)

fit_one <- function(g) {
  f <- try(suppressWarnings(rma.uni(yi = g$coef, sei = g$stderr, method = RMA_METHOD,
             control = list(maxiter = 1000))), silent = TRUE)
  if (inherits(f, "try-error")) {                      # second attempt on failure
    f <- try(suppressWarnings(rma.uni(yi = g$coef, sei = g$stderr, method = RMA_METHOD,
               control = list(stepadj = 0.5, maxiter = 1000))), silent = TRUE)
    if (inherits(f, "try-error")) return(NULL)
  }
  data.frame(feature_id = g$feature_id[1], pval = f$pval, I2 = f$I2,
             coef = as.numeric(f$beta), stringsAsFactors = FALSE)
}

out <- list()
for (task in unique(ps$task)) for (space in unique(ps$space)) {
  d0 <- ps[ps$task == task & ps$space == space, , drop = FALSE]
  if (!nrow(d0)) next
  batches <- sort(unique(d0$batch))
  rows <- do.call(rbind, lapply(c(list(NULL), as.list(batches)), function(b) {
    d <- d0[!(d0$batch %in% b), , drop = FALSE]
    d <- d[is.finite(d$coef) & is.finite(d$stderr) & d$stderr > 0, , drop = FALSE]
    rr <- do.call(rbind, lapply(split(d, d$feature_id),
                                function(g) if (nrow(g) < 2) NULL else fit_one(g)))
    if (is.null(rr)) return(NULL)
    rr$qval <- p.adjust(rr$pval, method = "BH")
    data.frame(dropped = if (is.null(b)) "(none)" else b,
               n_features = nrow(rr),
               n_sig = sum(rr$qval < SIG, na.rm = TRUE),
               median_I2 = median(rr$I2, na.rm = TRUE),
               task = task, space = space, stringsAsFactors = FALSE)
  }))
  out[[paste(task, space)]] <- rows
}
res <- do.call(rbind, out); rownames(res) <- NULL
write.csv(res, file.path(RES, "leave_one_study_out_meta_refit.csv"), row.names = FALSE)

old <- read.csv(file.path(RES, "leave_one_study_out_meta.csv"), stringsAsFactors = FALSE)
cmp <- merge(old[, c("dropped","task","space","n_sig")],
             res[, c("dropped","task","space","n_sig","n_features")],
             by = c("dropped","task","space"), suffixes = c("_old","_new"))
cmp <- cmp[order(cmp$task, cmp$space, cmp$dropped), ]
cat("\n=== LOSO: without stepadj -> with the retry ===\n")
print(cmp, row.names = FALSE)
cat("\nrows where n_sig changed:", sum(cmp$n_sig_old != cmp$n_sig_new), "of", nrow(cmp), "\n")
