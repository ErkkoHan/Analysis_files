# German cockroach monitoring: transitions, drivers and forecasts

Analysis code for the manuscript by Cheng-Hsuan Han, Sarah Wu and Yu-Syuan Luo,
"Routine longitudinal trap records enable prediction of German cockroach (*Blattella
germanica*) activity and infestation escalation in food-service establishments"
(submitted to *Pest Management Science*).

## Data

The de-identified monitoring records are published with the article as Supporting
Information (Table S1; variable definitions in Table S2). Save the Supporting Information
workbook as `dataset/ESM.xlsx` next to the scripts. Establishment names, addresses and
coordinates are not public, so the study-site map (Fig. 1(A)) cannot be redrawn from the
public data.

## Requirements

- Python 3.11 with the packages in `requirements.txt`
  (`pip install -r requirements.txt`)
- R 4.3 with `glmmTMB` 1.1.14, called from `analysis_pipeline.py`. If `Rscript` is not on
  the PATH, set the environment variable `RSCRIPT_PATH` to its full path.
- Fig. 1(A) only: R packages `sf`, `ggplot2`, `readr`, `rnaturalearth`,
  `rnaturalearthhires` and `ggspatial`

## Running the analysis

Run the scripts in this order from the repository folder:

| Step | Script | Produces | Used in |
| --- | --- | --- | --- |
| 1 | `analysis_pipeline.py` | hyperparameters, GLMMs (`fit_glmm.R`), test-set predictions, permutation importance with shadow-feature test, forecast horizons | Figs. 3–5; Tables S3 and S6 |
| 2 | `refit_predictions.py` | `perf_extended.csv` (adds prediction bias) | Table 1 |
| 3 | `transition_quadrant.py` | risk-state thresholds, transition matrix, escalation rates | Fig. 2 |
| 4 | `classification_report.py` | `classification_metrics.csv` | Table 2 |
| 5 | `make_tables_docx.py` | `Tables_1-2.docx`, `Tables_S5-S6.docx` | Tables 1–2; Tables S5–S6 |
| 6 | `make_figures.py` | `Fig2`–`Fig5` (PNG for screen; JPEG and TIFF at 800 dpi for submission) | Figs. 2–5 |
| – | `export_analysis_sites.py`, `study_site_map.R` | `Fig1a_study_sites` | Fig. 1(A) (needs coordinates) |

All outputs are written to `output/`. `features.py` builds the analytical sample
(pairing of consecutive monitoring periods, lagged predictors and the chronological
train/test split); `journal_style.py` and `docx_tables.py` hold figure and table
formatting. Fig. 1(B) (workflow diagram) was drawn separately and can be converted to
JPEG and TIFF submission files with `python journal_style.py Fig1.png` (170 mm width;
add `--width-mm 82` before the file name for a single-column figure).

## Notes on reproducibility

- `output/hyperparams.csv` holds the hyperparameters selected by expanding-window
  cross-validation (Supporting Information, Table S3) and is reused by default. Delete it
  to repeat the grid search.
- The shadow-feature test (999 refits per outcome) is the slowest step and takes several
  hours. For a quick check, lower `SHADOW_N` in `analysis_pipeline.py`; the *p* values then
  have a coarser resolution.
- Random seeds are listed in Supporting Information, Table S4.
- Running from `ESM.xlsx` reproduces the sample sizes, risk-state transitions, GLM and
  persistence results exactly. Random forest and XGBoost results can differ slightly
  (in our checks, test-set *R*² within 0.002 for next-period predictions and within 0.02
  across forecast horizons) because establishments carry
  de-identified identifiers, which changes the row order that tree ensembles sample
  from. Small differences also arise between operating systems and library builds, and
  multithreaded `glmmTMB` fits can differ in the fifth decimal place between runs.
- Category levels are handled internally with the original (Chinese) labels;
  `features.py` maps the English labels in Table S1 back to them so that the category
  encoding matches the original analysis.

## License

MIT License, © 2026 Hysia Taiwan Ltd. See `LICENSE`.