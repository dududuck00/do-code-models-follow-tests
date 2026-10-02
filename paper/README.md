# FSE 2027 manuscript source

Open `paper.tex` in Overleaf and select XeLaTeX. The project uses the ACM `acmsmall,screen,review,anonymous` class options. Local builds can use `latexmk -xelatex paper.tex` or Tectonic.

## Manuscript organization

1. Introduction: adopting a test-specified rule versus implementing it.
2. Study Design: paired semantic interventions, matched benchmark prompts, and fixed-budget test selection.
3. Experimental Setup: task sets, checkpoints, evaluators, and repeated runs.
4. Behavioral Results: rule adoption, visible adherence and persistent failures, benchmark correctness, and generation utility of test selection.
5. Representation Responses and Behavioral Prediction: equivalent wording controls, associations, and held-out-family prediction.
6. Implications and Scope: evaluation, execution of supplied tests, test selection, and the study settings.
7. Related Work.
8. Conclusion, followed by Data Availability and references.

The presentation reports effect sizes, raw pass rates, paired outcomes, family distributions, and repeated runs. Statistical intervals and associated prose are omitted from the manuscript and rendered figures. Original analysis records in `Results/` are retained as evidence and are not typeset. All table point values remain unchanged. No additional code generations were performed for this revision.

## Files

- `paper.tex`, `paper.bib`: manuscript and bibliography.
- `Tables/`: editable result tables; `main_effects.tex` and `semantic_families.tex` also supply figure data.
- `Figures/`: three figures in PDF, SVG, and PNG, with plotting code and input data. Run `python Figures/build_figures.py` with matplotlib and numpy installed.
- `Results/`: retained experiment records and analysis materials; these are not needed for compilation.
- `data_availability.tex`: the author-provided anonymous repository link.

Anonymous repository: https://anonymous.4open.science/r/do-code-models-follow-tests-C7EA

The paper has no appendix. The FSE page budget is 18 pages of text and figures plus up to four pages of references; Data Availability follows the Conclusion and does not count toward the page limit. See the official call: https://conf.researchr.org/track/fse-2027/fse-2027-papers
