# =============================================================================
# 08_specificity_figure.R - Figure S4, disease specificity
#
# Axes are the coefficients of the two case-control comparisons. A feature on the
# diagonal is associated equally with both diseases, so it is not specific; one
# near an axis is specific to that disease. Filled points are the features the
# direct contrast separates.
#
# Output: figures/Figure_S4.{png,pdf}
# =============================================================================

source("R/00_data.R")
suppressPackageStartupMessages({ library(ggplot2); library(ggrepel) })
SIG <- 0.05

pooled <- function(task, space) {
  f <- file.path(RES_DIR, "differential_abundance",
                 sprintf("S3_%s_%s.csv", task, if (space == "TAX") "taxa" else "pathways"))
  d <- read.csv(f, stringsAsFactors = FALSE)
  data.frame(key = if (space == "TAX") d$feature else trimws(d$pathway_id),
             coef = d$coef, qval = d$qval, stringsAsFactors = FALSE)
}
contrast <- function(space) {
  d <- read.csv(file.path(RES_DIR, "specificity",
                          sprintf("direct_contrast_pooled_%s.csv", space)),
                stringsAsFactors = FALSE)
  key <- if (space == "TAX") d$feature else trimws(sub(":.*$", "", d$feature))
  data.frame(key = key, qval_dc = d$qval, coef_dc = d$coef, stringsAsFactors = FALSE)
}

dat <- do.call(rbind, lapply(c("TAX", "PATH"), function(space) {
  a <- pooled("K05", space); b <- pooled("K02", space)
  m <- merge(a, b, by = "key", suffixes = c("_K05", "_K02"))
  m <- merge(m, contrast(space), by = "key", all.x = TRUE)
  m <- m[m$qval_K05 < SIG | m$qval_K02 < SIG, ]
  m$separated <- !is.na(m$qval_dc) & m$qval_dc < SIG
  m$shared <- m$qval_K05 < SIG & m$qval_K02 < SIG & sign(m$coef_K05) == sign(m$coef_K02)
  m$class <- ifelse(m$separated, "Separated by direct contrast",
             ifelse(m$shared,    "Significant in both, same direction",
                                 "Significant in one comparison only"))
  m$space <- if (space == "TAX") "Taxa" else "Pathways"
  m
}))
dat$space <- factor(dat$space, levels = c("Taxa", "Pathways"))
dat$class <- factor(dat$class, levels = c("Separated by direct contrast",
                                          "Significant in both, same direction",
                                          "Significant in one comparison only"))

# label what the text discusses, plus everything the direct contrast separates
named <- c("Streptococcus_mutans", "Tannerella_forsythia", "Porphyromonas_gingivalis",
           "Treponema_denticola", "Veillonella_parvula", "Veillonella_rogosae",
           "Streptococcus_australis", "PWY-622", "ARGININE-SYN4-PWY",
           "POLYAMINSYN3-PWY", "PWY-5392", "PWY-2201", "PWY-6507")
dat$lab <- ifelse(dat$key %in% named | dat$separated, gsub("_", " ", dat$key), NA)

lim <- max(abs(c(dat$coef_K05, dat$coef_K02))) * 1.05
p <- ggplot(dat, aes(coef_K05, coef_K02)) +
  geom_hline(yintercept = 0, colour = "grey80", linewidth = .3) +
  geom_vline(xintercept = 0, colour = "grey80", linewidth = .3) +
  geom_abline(slope = 1, intercept = 0, linetype = "22", colour = "grey55", linewidth = .4) +
  geom_point(aes(fill = class, size = class, shape = class),
             colour = "grey25", stroke = .25, alpha = .9) +
  geom_text_repel(aes(label = lab), size = 2.5, family = "Times", colour = "grey15",
                  min.segment.length = 0, segment.size = .2, segment.colour = "grey60",
                  max.overlaps = 25, box.padding = .35, na.rm = TRUE) +
  scale_shape_manual(values = c(21, 21, 21)) +
  scale_fill_manual(values = c("Separated by direct contrast"        = "#1b1b1b",
                               "Significant in both, same direction" = "#9e9e9e",
                               "Significant in one comparison only"  = "#e8e8e8")) +
  scale_size_manual(values = c(2.4, 1.7, 1.3)) +
  coord_equal(xlim = c(-lim, lim), ylim = c(-lim, lim)) +
  facet_wrap(~ space) +
  labs(x = "Effect in periodontal disease status (β vs healthy)",
       y = "Effect in caries (β vs healthy)", fill = NULL, shape = NULL, size = NULL) +
  theme_bw(base_size = 9, base_family = "Times") +
  theme(legend.position = "bottom", legend.box.margin = margin(t = -4),
        panel.grid.minor = element_blank(),
        strip.background = element_rect(fill = "grey92", colour = NA),
        strip.text = element_text(face = "bold"))

# cairo is needed for the Greek beta: the default pdf device cannot render U+03B2
# and drops the character silently.
ggsave(file.path(FIG_DIR, "B6_disease_specificity.png"), p, width = 9, height = 5.4,
       dpi = 300, type = "cairo")
ggsave(file.path(FIG_DIR, "B6_disease_specificity.pdf"), p, width = 9, height = 5.4,
       device = cairo_pdf)

s <- aggregate(key ~ space + class, dat, length); names(s)[3] <- "n"
print(s)
cat("\nWritten: figures/Figure_S4.{png,pdf}\n")
