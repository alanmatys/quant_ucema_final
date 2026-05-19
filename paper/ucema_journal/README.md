# UCEMA Journal Paper — Compile Instructions

Continuation of the MACI 2025 HRP paper. Spec: [`specs/04_paper.md`](../../specs/04_paper.md).

## Compile

```bash
cd paper/ucema_journal
latexmk -pdf main.tex
```

Or one-shot:
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Files
- `main.tex` — single-file paper draft, ~25-30 pages
- `references.bib` — 25 cited works
- Figures: `../figures/*.png` (built by `scripts/generate_paper_figures.py`)

## Result tables
All numbers in §6 (Results) trace to committed CSVs in `data/`:
- `backtest_v2_rebalance_results.csv` — 19-strategy Scenario B table
- `backtest_v2_static_results.csv` — Scenario A
- `backtest_v2_threshold_results.csv` — Scenarios B/C/C′
- `backtest_v2_cost_sensitivity.csv` — cost grid
- `cluster_stability.csv` — 76-snapshot cophenetic + ARI
- `detoned_vs_partialcorr_rho_timeseries.csv` — convergence finding
- `inference_*.csv` — bootstrap CIs, LW Sharpe diff, Hansen SPA
- `hrp_shrunkcov_intensity.csv` — LW α time series

## Reproducing the figures
```bash
PYTHONPATH=. venv/bin/python scripts/generate_paper_figures.py
```

## Reproducing the backtest
```bash
PYTHONPATH=. venv/bin/python -m pytest tests/  # 79/79 must pass
jupyter notebook notebooks/backtest_v2.ipynb
```
