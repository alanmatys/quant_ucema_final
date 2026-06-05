# Spec 01 — Correlation Matrix Denoising

**Status:** Draft
**Owner:** Alan Matys, Federico Rodriguez
**Closes:** "Denoising" item from MACI 2025 future-work section

---

## 1. Motivation

Sample correlation matrices estimated from finite samples contain noise eigenvalues that
distort hierarchical clustering and inflate concentration risk in HRP allocations.
Marchenko–Pastur (MP) theory provides a closed-form distribution for the eigenvalues of
random correlation matrices, allowing us to identify and shrink the noise component.

In a crypto universe (high N, moderate T) the noise component is non-trivial. Denoising
is the first piece of the unfinished work left open in the original MACI paper.

## 2. Requirements

R1. Implement Marchenko–Pastur denoising following López de Prado (2020), *Machine
    Learning for Asset Managers*, Chapter 2.

R2. The denoising procedure must:
   - Fit the MP distribution to observed eigenvalues using a KDE-based estimate of the
     maximum eigenvalue under the null hypothesis of pure noise.
   - Classify eigenvalues as **signal** (above the fitted MP upper bound) or **noise**.
   - Replace noise eigenvalues with their mean (constant-residual eigenvalue method).
   - Reconstruct the correlation matrix from the modified eigen-decomposition.

R3. Expose the denoised correlation matrix as a drop-in replacement for the sample
    correlation used by `HRP`, via a new strategy class `HRPDenoised`.

R4. The implementation must not silently change behavior of the existing `HRP` class.

## 3. Interface

### 3.1 `src/denoising.py`

```python
def mp_pdf(var: float, q: float, pts: int) -> pd.Series:
    """Marchenko–Pastur theoretical PDF for given variance and q = T/N."""

def fit_max_eigenvalue(eigenvalues: np.ndarray, q: float,
                       bandwidth: float = 0.25) -> tuple[float, float]:
    """Fit MP distribution to observed eigenvalues via KDE.

    Returns:
        (eMax, var) — upper noise bound and fitted variance.
    """

def denoise_corr_constant_residual(corr: np.ndarray, q: float,
                                   bandwidth: float = 0.25) -> np.ndarray:
    """Apply constant-residual eigenvalue denoising.

    Args:
        corr: NxN sample correlation matrix.
        q:    T/N ratio of observations to assets.
        bandwidth: KDE bandwidth for MP fit.

    Returns:
        Denoised NxN correlation matrix (unit diagonal, PSD).
    """
```

### 3.2 `src/portfolio_maker.py`

```python
class HRPDenoised(HRP):
    """HRP using a Marchenko–Pastur denoised correlation matrix."""

    def __init__(self, returns: pd.DataFrame, bandwidth: float = 0.25) -> None:
        super().__init__(returns)
        T, N = returns.shape
        q = T / N
        corr_denoised = denoise_corr_constant_residual(self.corr.values, q, bandwidth)
        self.corr = pd.DataFrame(corr_denoised, index=self.corr.index, columns=self.corr.columns)
        # Rebuild covariance consistent with denoised corr and original std vector
        std = np.sqrt(np.diag(self.cov.values))
        self.cov = pd.DataFrame(
            self.corr.values * np.outer(std, std),
            index=self.cov.index, columns=self.cov.columns,
        )
```

## 4. Acceptance Criteria

AC1. `denoise_corr_constant_residual` returns a matrix with:
   - shape equal to input,
   - unit diagonal (tolerance 1e-10),
   - symmetric (tolerance 1e-10),
   - positive semi-definite (smallest eigenvalue ≥ -1e-10).

AC2. On a synthetic dataset of T=1000, N=100 with known signal rank=5, the
     procedure identifies a signal subspace with rank in [3, 8] (allowing for
     estimation noise) and reduces total variance attributed to noise eigenvalues
     by at least 50% versus the sample correlation.

AC3. Trace of the denoised correlation matrix equals N (preserved by construction).

AC4. `HRPDenoised` weights sum to 1.0 (tolerance 1e-8) and are non-negative on the
     crypto dataset over the backtest window.

AC5. Existing `HRP` test suite continues to pass without modification.

AC6. Unit tests in `tests/test_denoising.py` cover AC1–AC4 and live alongside
     the implementation.

## 5. Out of Scope

- Targeted shrinkage (e.g. Ledoit–Wolf) — comparison only, not implementation.
- Eigenvalue clipping methods other than constant-residual.
- Denoising of the returns matrix itself.

## 6. References

- López de Prado, M. (2020). *Machine Learning for Asset Managers*, Chapter 2.
- Marchenko, V. A., & Pastur, L. A. (1967). "Distribution of eigenvalues for some sets of random matrices."
- Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). "Noise dressing of financial correlation matrices."
