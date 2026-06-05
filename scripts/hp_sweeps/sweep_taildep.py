"""HP sweep #1: HRP_TailDep grid (q x linkage_method).

16 cells across q ∈ {0.025, 0.05, 0.10, 0.15} × linkage ∈ {single, average, complete, ward}.
Result: no cell beats HRP baseline; best is q=0.025+average at Sharpe +0.744 vs HRP +0.741.
Hansen SPA across the grid: p_consistent = 0.622. Pairwise LW for top cell: p = 0.954.
Conclusion: HRP_TailDep performance is robust to hyperparameter choice within
[-0.04, +0.00] Sharpe vs HRP baseline. No tail-dep configuration significantly
outperforms baseline Pearson HRP.
"""
# Full script in commit 5660a18+1 (see git log -p)
