"""HP sweep #2: correlation × distance × linkage grid on base HRP.

24 cells = 3 correlations × 4 distance functions × 2 linkages.
Key findings:
- Average linkage consistently > single linkage (+0.01 to +0.02 Sharpe).
- Under SINGLE linkage, all 4 distance functions give IDENTICAL Sharpe per
  correlation type — mathematically expected since single linkage uses only
  the ORDER of distances, which is preserved by any monotonic transform.
- Pearson ≥ Kendall ≥ Spearman (small differences).
- Best cell: pearson + ldp distance + average linkage, Sharpe +0.756.

Hansen SPA on 24-cell grid: p_consistent = 0.416 — cannot reject 'no cell
beats HRP'. Only 2/24 cells have positive studentized scores.
Best-cell LW pairwise p = 0.380 — improvement of +0.015 Sharpe is in noise.
"""
# Full script in commit 5660a18+2
