"""Correlation matrix denoising and detoning.

Implements:
    - Marchenko-Pastur eigenvalue denoising (constant-residual method)
      per Lopez de Prado (2020), *Machine Learning for Asset Managers*, Ch. 2.
      → see specs/01_denoising.md
    - Market-mode detoning by removing the top eigenvector(s) of the
      denoised correlation matrix per Lopez de Prado (2020), §2.6.
      → see specs/02_detoning.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.neighbors import KernelDensity


# ----------------------------------------------------------------------
# Marchenko-Pastur denoising
# ----------------------------------------------------------------------

def mp_pdf(var: float, q: float, pts: int) -> pd.Series:
    """Marchenko-Pastur theoretical PDF on [eMin, eMax] with `pts` grid points.

    Args:
        var: variance scale (sigma^2) — 1 for correlation matrices under H0.
        q:   T / N, ratio of observations to assets (q > 1 required).
        pts: number of grid points.

    Returns:
        pd.Series indexed by eigenvalue grid, values = MP density.
    """
    if q <= 1:
        raise ValueError("Marchenko-Pastur requires q = T/N > 1")
    e_min = var * (1 - (1.0 / q) ** 0.5) ** 2
    e_max = var * (1 + (1.0 / q) ** 0.5) ** 2
    e_vals = np.linspace(e_min, e_max, pts)
    pdf = q / (2 * np.pi * var * e_vals) * np.sqrt((e_max - e_vals) * (e_vals - e_min))
    return pd.Series(pdf, index=e_vals)


def _kde_log_density(observations: np.ndarray, bandwidth: float, grid: np.ndarray) -> np.ndarray:
    """KDE log-density evaluated on a grid; thin wrapper around sklearn."""
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(observations.reshape(-1, 1))
    return kde.score_samples(grid.reshape(-1, 1))


def _err_pdfs(var: float, eigenvalues: np.ndarray, q: float, bandwidth: float, pts: int) -> float:
    """Squared error between empirical KDE and MP theoretical PDF on the noise grid."""
    var = float(var)
    pdf0 = mp_pdf(var, q, pts)
    log_pdf1 = _kde_log_density(eigenvalues, bandwidth=bandwidth, grid=pdf0.index.values)
    pdf1 = np.exp(log_pdf1)
    return float(np.sum((pdf1 - pdf0.values) ** 2))


def fit_max_eigenvalue(
    eigenvalues: np.ndarray,
    q: float,
    bandwidth: float = 0.25,
    pts: int = 1000,
) -> tuple[float, float]:
    """Fit Marchenko-Pastur upper bound by minimizing PDF distance.

    Args:
        eigenvalues: 1D array of observed correlation-matrix eigenvalues.
        q:           T / N.
        bandwidth:   KDE bandwidth for empirical PDF estimation.
        pts:         grid resolution for the comparison.

    Returns:
        (e_max, var) — fitted MP upper bound and the fitted variance.
    """
    result = minimize_scalar(
        _err_pdfs,
        args=(eigenvalues, q, bandwidth, pts),
        bounds=(1e-5, 1 - 1e-5),
        method="bounded",
    )
    var = float(result.x) if result.success else 1.0
    e_max = var * (1 + (1.0 / q) ** 0.5) ** 2
    return e_max, var


def denoise_corr_constant_residual(
    corr: np.ndarray,
    q: float,
    bandwidth: float = 0.25,
) -> np.ndarray:
    """Apply Marchenko-Pastur constant-residual eigenvalue denoising.

    Signal eigenvalues (above the fitted MP upper bound) are preserved;
    noise eigenvalues are replaced with their mean. The reconstructed
    correlation matrix has the same trace as the input.

    Args:
        corr: NxN sample correlation matrix.
        q:    T / N.
        bandwidth: KDE bandwidth used by `fit_max_eigenvalue`.

    Returns:
        NxN denoised correlation matrix (unit diagonal, PSD).
    """
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be square")
    eigvals, eigvecs = np.linalg.eigh(corr)
    # eigh returns ascending; sort descending to follow LdP convention
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    e_max, _ = fit_max_eigenvalue(eigvals, q=q, bandwidth=bandwidth)

    # Replace noise eigenvalues with their mean (constant-residual)
    is_signal = eigvals > e_max
    n_signal = int(is_signal.sum())
    if n_signal == len(eigvals):
        # nothing classed as noise → matrix unchanged
        return corr.copy()
    noise_mean = eigvals[~is_signal].mean()
    new_eigvals = eigvals.copy()
    new_eigvals[~is_signal] = noise_mean

    # Reconstruct, then renormalize to unit diagonal (numerical safeguard)
    denoised = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(denoised))
    denoised = denoised / np.outer(d, d)
    # Symmetrize and snap the diagonal to exactly 1.0 (squareform requires this)
    denoised = (denoised + denoised.T) / 2
    np.fill_diagonal(denoised, 1.0)
    return denoised


# ----------------------------------------------------------------------
# Detoning (market-mode removal)
# ----------------------------------------------------------------------

def detone_corr(corr: np.ndarray, n_market_components: int = 1) -> np.ndarray:
    """Remove the top n_market_components eigenvectors and re-normalize.

    Args:
        corr: NxN denoised (or sample) correlation matrix.
        n_market_components: number of top eigenvalue/eigenvector pairs to remove.

    Returns:
        NxN detoned correlation matrix (unit diagonal, PSD).
    """
    if n_market_components < 1:
        raise ValueError("n_market_components must be >= 1")
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    if n_market_components >= len(eigvals):
        raise ValueError(
            f"n_market_components ({n_market_components}) must be < N ({len(eigvals)})"
        )

    # Zero-out the top market components
    new_eigvals = eigvals.copy()
    new_eigvals[:n_market_components] = 0.0

    detoned = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
    # Renormalize to unit diagonal (the removed components shrink the diagonal)
    d = np.sqrt(np.maximum(np.diag(detoned), 1e-12))
    detoned = detoned / np.outer(d, d)
    detoned = (detoned + detoned.T) / 2
    np.fill_diagonal(detoned, 1.0)
    return detoned
