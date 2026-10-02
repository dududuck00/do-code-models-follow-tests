# Manuscript figures

`build_figures.py` reads `Tables/semantic_families.tex` and `Tables/main_effects.tex` and writes PDF, SVG, and PNG versions of:

- `paired_intervention`: fixed description and inputs, alternative output rules, and paired held-out evaluation.
- `semantic_families_heatmap`: 400 family-level values on a shared 0–6 scale.
- `main_effects_forest`: 45 mean paired accuracy differences, displayed as labeled points without error bars. The historical filename is retained for source compatibility.

`figure_data.json` and `main_effects.csv` contain the displayed values. All differences are in percentage points and come from the retained full-precision analyses before rounding. Install matplotlib and numpy and run `python Figures/build_figures.py` from the manuscript directory to rebuild the figures.
