# Paper draft

IEEE conference format (`IEEEtran`). Every number comes from `RESULTS.md` and the result files in `results/`.

| File | Contents |
|---|---|
| `main.tex` | The paper |
| `refs.bib` | References |
| `figures/` | Figures 7, 8, 11 and 12 without titles (the captions carry them): `python src/make_contribution_figures.py --paper` |
| `satmon-paper.pdf` | The compiled paper (Overleaf, pdfLaTeX, 6 pages, 0 errors, 0 warnings) |

## Build the PDF

**Overleaf (no install):** zip this folder, then in Overleaf choose New Project → Upload Project and pick the zip. Set the compiler to pdfLaTeX. `IEEEtran` is built in.

**Locally:** with a TeX distribution (MiKTeX, TeX Live or Tectonic):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Before submitting

- Replace the placeholder author names, institution and emails.
- Check the target conference's page limit. The draft is about 6-7 pages with four wide figures and three tables.
- Re-check every reference against its publisher page before submission.
