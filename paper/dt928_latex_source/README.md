# DT 928 — LaTeX source

**Hierarchical Risk Parity Variants for Cryptocurrency Portfolios: Denoising,
Detoning, Tail Dependence, Embeddings, and Statistical Inference, 2020–2026**

Alan Matys · Federico Martin Rodriguez · Emiliano Delfau — Universidad del CEMA

## Contents

| File | Purpose |
|---|---|
| `main.tex` | Full manuscript (self-contained, standard packages only) |
| `references.bib` | Bibliography (natbib / plainnat) |
| `figures/` | The 9 PNG figures referenced by the manuscript |

## Build

Any TeX distribution from 2020 onward works (TeX Live, MiKTeX, MacTeX).
No custom classes or fonts; packages used: amsmath, graphicx, booktabs,
hyperref, geometry, caption, subcaption, float, xcolor, natbib, authblk.

Recommended (runs pdflatex + bibtex the right number of times automatically):

```
latexmk -pdf main.tex
```

Or manually:

```
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

Or upload this whole folder to [Overleaf](https://www.overleaf.com) as a
project (zip it first) — it compiles as-is with the default pdfLaTeX engine.

Expected output: `main.pdf`, 25 pages.
