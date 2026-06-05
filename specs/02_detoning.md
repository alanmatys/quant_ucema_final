# Spec 02 — Correlation Matrix Detoning

**Status:** Draft
**Owner:** Alan Matys, Federico Rodriguez
**Closes:** "Denoting" item from MACI 2025 future-work section (interpreted as **detoning**)
**Depends on:** [01_denoising.md](01_denoising.md)

---

## 1. Motivation

After denoising removes noise eigenvalues, the largest remaining eigenvalue typically
corresponds to a market-wide common factor — every asset loads on it positively. In
the crypto universe this market mode is particularly strong because most assets
co-move with Bitcoin, especially during stress periods.

Removing this market eigenvector ("detoning") exposes the idiosyncratic correlation
structure that HRP's hierarchical clustering actually needs to differentiate clusters.
The expectation is that detoned-HRP allocations are less BTC-dominated and capture
crypto sub-sectors (DeFi, L1s, memes, etc.) more cleanly.

## 2. Requirements

R1. Implement detoning per López de Prado (2020), *Machine Learning for Asset Managers*,
    §2.6, applied to the **denoised** correlation matrix.

R2. Procedure:
   - Eigen-decompose the denoised correlation matrix.
   - Remove the top `n_market_components` eigenvalue/eigenvector pairs.
   - Reconstruct the correlation matrix from the remaining components.
   - Re-normalize so the diagonal is unit (assets remain self-correlated = 1).

R3. Default `n_market_components = 1` (remove the dominant market mode).
    Parameter must be configurable to allow sensitivity analysis.

R4. Expose as a new strategy class `HRPDetoned` that inherits from `HRPDenoised`
    (i.e. denoise → detone → HRP).

## 3. Interface

### 3.1 `src/denoising.py`

```python
def detone_corr(corr: np.ndarray, n_market_components: int = 1) -> np.ndarray:
    """Remove top market eigenvector(s) and renormalize.

    Args:
        corr: NxN denoised correlation matrix.
        n_market_components: Number of top eigenvectors to remove.

    Returns:
        Detoned NxN correlation matrix (unit diagonal, PSD).
    """
```

### 3.2 `src/portfolio_maker.py`

```python
class HRPDetoned(HRPDenoised):
    """HRP using a denoised + detoned correlation matrix."""

    def __init__(self, returns: pd.DataFrame, bandwidth: float = 0.25,
                 n_market_components: int = 1) -> None:
        super().__init__(returns, bandwidth=bandwidth)
        corr_detoned = detone_corr(self.corr.values, n_market_components)
        self.corr = pd.DataFrame(corr_detoned, index=self.corr.index, columns=self.corr.columns)
        std = np.sqrt(np.diag(self.cov.values))
        self.cov = pd.DataFrame(
            self.corr.values * np.outer(std, std),
            index=self.cov.index, columns=self.cov.columns,
        )
```

## 4. Acceptance Criteria

AC1. `detone_corr` output has:
   - shape equal to input,
   - unit diagonal (tolerance 1e-10),
   - symmetric (tolerance 1e-10),
   - positive semi-definite (smallest eigenvalue ≥ -1e-10).

AC2. On the crypto dataset, the mean off-diagonal correlation of the detoned matrix
     is strictly less than that of the denoised matrix (sanity check that the market
     mode was indeed dominant and positive).

AC3. The first removed eigenvector has all-positive loadings (within sign convention),
     confirming it is the market mode and not a sector contrast.

AC4. `HRPDetoned` allocations on the crypto universe produce a **lower maximum
     weight** than `HRPDenoised` on at least 60% of rebalancing dates — evidence
     that removing the market mode reduces concentration.

AC5. `HRPDetoned` weights sum to 1.0 (tolerance 1e-8), non-negative.

AC6. Unit tests in `tests/test_denoising.py` extend the file from Spec 01 to cover
     AC1–AC3.

## 5. Out of Scope

- Multi-factor detoning beyond the dominant mode (parameter is exposed for future use
  but not analyzed in this paper).
- Time-varying market-mode estimation.

## 6. References

- López de Prado, M. (2020). *Machine Learning for Asset Managers*, §2.6.
- Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N., Guhr, T., & Stanley, H. E.
  (2002). "Random matrix approach to cross correlations in financial data."
