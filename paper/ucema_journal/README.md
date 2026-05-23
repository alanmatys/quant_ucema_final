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
- `main.tex` — single-file paper draft, ~23 pages
- `references.bib` — 31 references
- Figures: `../figures/*.png` (built by `scripts/generate_paper_figures.py`)

## Result tables
All numbers in §6 (Results) trace to committed CSVs in `data/`:
- `backtest_v2_rebalance_results.csv` — 28-row Scenario B table
  (27 constructed strategies including baseline HRP, plus the passive
  `HODL_BTC` benchmark)
- `xgboost_mu_predictions.csv` — walk-forward XGBoost return forecast
  feeding NCOML and HRP-Σμ (76 snapshots × 146 assets)
- `xgboost_best_params.csv` — per-snapshot best hyperparameters from
  the TimeSeriesSplit CV inside each training window
- `backtest_v2_static_results.csv` — Scenario A
- `backtest_v2_threshold_results.csv` — Scenarios B/C/C′
- `backtest_v2_cost_sensitivity.csv` — cost grid
- `backtest_v2_post_covid_results.csv` — post-COVID regime cut
- `backtest_v2_weekly_results.csv` — weekly-cadence robustness check
- `cluster_stability.csv` — 76-snapshot cophenetic + ARI
- `detoned_vs_partialcorr_rho_timeseries.csv` — convergence finding
- `power_analysis.csv` — detectable Sharpe edge at 80% power
- `deflated_sharpe.csv` — Deflated Sharpe Ratio
- `inference_*.csv` — bootstrap CIs, LW Sharpe diff, Hansen SPA
- `hrp_shrunkcov_intensity.csv` — LW α time series

## Reproducing the figures
```bash
uv run python scripts/generate_paper_figures.py
```

## Reproducing the backtest
```bash
uv run pytest -q                                  # 96 tests must pass
uv run python scripts/build_pit_universe_final.py
uv run python scripts/rerun_all.py                # 21-strategy main roster
uv run python scripts/add_denoised.py             # + HRP_Denoised
uv run python scripts/add_nco_rt.py               # + return-tilted NCO
uv run python scripts/add_crisp.py                # + CRISP
uv run python scripts/add_ncocrisp.py             # + NCO-CRISP
uv run python scripts/build_xgboost_signals.py    # walk-forward XGBoost mu (~12 min)
uv run python scripts/add_signal_strategies.py    # + NCOML, HRPSigmaMu
```

Inference is computed on the common aligned daily-return panel after all
compared series are loaded. This is slightly shorter than some
per-strategy daily series once `HODL_BTC` is included.
