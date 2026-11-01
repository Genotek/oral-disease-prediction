# Cross-Cohort Prediction of Dental Caries and Periodontal Disease from Salivary Shotgun Metagenomic Data

Analysis code for the manuscript **"Cross-Cohort Prediction of Dental Caries and
Periodontal Disease from Salivary Shotgun Metagenomic Data."** Every number, table
and figure reported in the paper is produced here.

## Layout

| Path | Contents |
|---|---|
| `R/` | composition, differential abundance, meta-analysis, batch structure |
| `ml/` | the configuration grid, validation schemes and nested selection |
| `figures/` | main and supplementary figures |
| `common.py`, `config.py`, `ml_pipeline_article_configs.py` | data loading, feature filtering, transformations and model definitions, imported by the scripts above |
| `paths.py`, `R/00_data.R` | path resolution, shared by the Python and the R side |

Input data are not distributed with the code.

## Configuration

Run every script **from the repository root**. Three environment variables are read
identically by the R and the Python side; each has a default inside the repository, so
setting them is optional.

| Variable | Default | Purpose |
|---|---|---|
| `SALIVARY_WGS_DATA_DIR` | `<repo>/data` | location of `tax.csv`, `path.csv`, `metadata.csv` |
| `SALIVARY_WGS_OUTPUT_DIR` | `<repo>/outputs` | where results, figures and models are written |
| `SALIVARY_WGS_RLIB` | `<repo>/Rlib` | optional library holding MMUPHin 1.20.0; if absent, system libraries are used and the version check reports the mismatch |

Paths are resolved in `paths.py` for Python and in `R/00_data.R` for R. Both read the
same variables, so the two halves of the analysis always look in the same place.
Missing input files produce a named error rather than a traceback.

```
export SALIVARY_WGS_DATA_DIR=/path/to/data
Rscript R/01_sample_accounting.R
```

## Analysis design

Three choices govern every downstream number.

**Study structure is modelled, not removed.** Case-to-control ratios range from 0 % to
100 % across bioprojects, so study membership is not independent of disease status.
Under that imbalance, location–scale correction removes disease-associated variation
together with technical variation. No data-level batch correction is applied; instead
bioproject enters as a random intercept in MaAsLin2 and as the grouping variable for
the folds in the classifiers.

**Pathway abundances are depth-normalised.** HUMAnN reports pathways in reads per
kilobase, so raw values carry sequencing depth, which varies over three orders of
magnitude and is largely determined by bioproject. MaAsLin2 runs with
`normalization = "TSS"` on both feature spaces.

**No cohort informs its own evaluation.** Pooled cross-validation groups folds by
bioproject, and the pipeline configuration is fixed by a nested leave-one-study-out
procedure in which the held-out cohort takes no part in its own selection.

## Order of execution

### R — composition, differential abundance, batch structure

| Step | Script | Produces |
|---|---|---|
| 0 | `R/00_data.R` | shared loading, feature filtering, analysis parameters. Sourced by the others, not run directly |
| 1 | `R/01_sample_accounting.R` | sample flow from 514 profiled samples to the two analysis sets; per-bioproject composition |
| 2 | `R/02_batch_structure.R` | PERMANOVA and PERMDISP: two sample sets × two feature spaces × two filter states × two distance metrics |
| 3 | `R/03_differential_abundance.R` | pooled MaAsLin2 for both diseases and both feature spaces, plus the sensitivity run excluding participants with both diagnoses |
| 4 | `R/04_meta_analysis.R` | per-cohort effects combined by random-effects meta-analysis (MMUPHin `lm_meta`), with I², τ² and per-cohort weights |
| 5 | `R/05_leave_one_cohort_out.R` | meta-analysis repeated with one cohort omitted at a time |
| 6 | `R/06_disease_specificity.R` | overlap between the two case–control comparisons, and the direct caries-versus-periodontal contrast |
| 7 | `R/07_composition.R` | species-level composition of the aggregated dataset; abundance, prevalence and effect direction of each differentially abundant taxon within every bioproject |
| 8 | `R/08_specificity_figure.R` | Figure S4 |

Dependencies between steps: 5, 6 and 7 read the outputs of 3 and 4, and 8 reads the output
of 6. The remaining steps are independent and may be run in any order. A step whose input
is missing stops with the name of the file it expected.

MaAsLin2 is called with `normalization = "TSS"`, `transform = "LOG"`,
`analysis_method = "LM"`, `min_prevalence = 0`, `correction = "BH"`, disease status as a
fixed effect and BioProject as a random effect. Features are retained if non-zero in at
least 10 % of samples and detected in at least three bioprojects; that filter is defined
once, in `R/00_data.R`, and used by every step including the PERMANOVA.

MMUPHin occasionally fails to converge when estimating the between-study variance and
returns an empty row without warning. `R/04_meta_analysis.R` detects those features and
re-estimates them with a reduced step size, then recomputes the Benjamini–Hochberg
correction. Affected features are flagged in the output. `R/05_leave_one_cohort_out.R`
uses the same convergence control, so its baseline agrees with the main fit.

### Python — machine learning

| Step | Script | Produces |
|---|---|---|
| 1 | `ml/grid_and_schemes.py` | all 72 configurations per disease under three validation schemes: study-grouped CV, class-stratified CV, leave-one-study-out. Also the nested selection over the full grid |
| 2 | `ml/nested_selection.py` | nested selection repeated separately within each feature space, with the inner-loop results retained |
| 3 | `ml/compare_schemes.py` | the comparisons quoted in the manuscript: scheme against scheme, weighted against unweighted averaging, nested estimate against the grid maximum |
| 4 | `ml/cohort_sensitivity.py` | sensitivity of the two final configurations to the largest cohort (PRJNA1057503): leave-one-study-out with equal weight per training cohort, with each training cohort capped at the median size (30 repeats), and with PRJNA1057503 or PRJNA741688 removed from training |
| 5 | `ml/cohort_sensitivity_subgroups.py` | AUC in the held-out PRJNA1057503 fold for subgroups of cases (systemic condition recorded in `health_status`, one or both diagnoses), with bootstrap confidence intervals; reads `per_sample_predictions.csv` from step 1 |
| 6 | `ml/make_table_S18.py` | assembles Table S18 from the outputs of steps 4 and 5 |

Step 1 writes incrementally and skips work already on disk, so an interrupted run can be
resumed by re-invoking it. Per-fold predictions for every held-out sample are retained, so
any metric can be recomputed without refitting.

```
python ml/grid_and_schemes.py --n-jobs 12
python ml/nested_selection.py --n-jobs 12
python ml/compare_schemes.py
python ml/cohort_sensitivity.py --n-jobs 12
python ml/cohort_sensitivity_subgroups.py
python ml/make_table_S18.py
```

### Python — figures

| Script | Produces |
|---|---|
| `figures/figure_2_and_S2.py --space both` | Figure 2 (taxonomic profiles) and Figure S2 (pathway profiles) |
| `figures/figure_3.py --mode both` | Figure 3 (pooled) and Figure S3 (meta-analysis) |
| `figures/figure_4_and_5.py --task both` | Figures 4 and 5, the fitted final models, and the SHAP rankings behind Table S19 |
| `figures/palettes.py` | colour scales; imported by `figure_4_and_5.py`, not run directly |

Figure S4 comes from `R/08_specificity_figure.R`. Figure S1 comes from
`figures/figure_S1_technical_qc.ipynb`, which reads the per-bioproject MultiQC and
FastQC summaries rather than the abundance tables. Figure 1 is a schematic and has
no generating script.

All figures are set in Arial with TrueType embedding, and text is kept as text in
the vector output.

Figures 2 and S2 read the PERMANOVA values from the output of `R/02_batch_structure.R`
instead of recomputing them, so the figure and the text cannot disagree; step 2 must
therefore have been run first. Figures 4 and 5 read the nested-selection output of
`ml/nested_selection.py` and fit the final models themselves, writing them to
`<output>/models/`. Panel A of Figures 4 and 5 shows class-stratified
cross-validation, which is the estimate the Abstract reports; the study-grouped
values quoted alongside it in the text come from `ml/grid_and_schemes.py`.

## Environment

Every reported number was produced on one fixed set of package versions, recorded in two
files and enforced at run time.

**R 4.6.0** — versions are listed in `DESCRIPTION` and checked by `R/00_data.R`, which
stops the run if any of MMUPHin, MaAsLin2, metafor, vegan, ggplot2, ggrepel or dplyr
differs. MMUPHin must be 1.x: version 2 renames `lm_meta` to `maaslin_meta` and runs it
on a different engine. Point `SALIVARY_WGS_RLIB` at a library holding the pinned versions
to keep them separate from the system installation.

**Python 3.11.15** — `pip install -r requirements.txt`. The random seed is fixed at 42,
and the versions actually loaded are written to `<output>/ml/session_info.json` on every
machine-learning run, so a result can always be traced to the environment that produced it.

## Citation

Citation details will be added after publication.

## License

MIT License.
