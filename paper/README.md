# Paper Deliverables

This directory contains the course-project paper draft and generated figures.

Generated files:

- `deliverables/`: final course-project submission files, including the report PDF, presentation PDF, notebook, requirements file, and submission zip.
- `main.tex`: LaTeX report source following the assignment-required structure: Introduction, key technical sections, Conclusion, References, and Credit.
- `references.bib`: verified bibliography entries used by `main.tex`.
- `figure_plan.md`: figure plan and data sources.
- `scripts/make_figures.py`: reproducible figure-generation script using `reportlab`.
- `figures/*.pdf`: vector figures for the paper.
- `figures/*.png`: raster copies for notebooks/slides.

Course guideline caveat:

- The assignment PDF says to use a provided LaTeX style file and not the preprint option. That style file was not included in the repository or the mentioned PDF path, so `main.tex` currently uses a conservative `article` format. Replace the preamble/class with the course style file once it is available.

Regenerate figures:

```bash
/Users/xh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 paper/scripts/make_figures.py
```
