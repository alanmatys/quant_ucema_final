"""Correlation estimators used by HRP variants in Spec 06.

Provides three "enriched correlation" alternatives to the sample correlation
that all feed the standard HRP pipeline (distance → linkage → quasi-diag →
recursive bisection):

    - partial correlation (via sklearn's graphical lasso)
    - EWMA dynamic correlation
    - empirical lower-tail dependence

The point of separating these from `portfolio_maker.py` is so the math can
be unit-tested in isolation and reused outside HRP.

See specs/06_hrp_variants.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso


# ----------------------------------------------------------------------
# Partial correlation (graphical-lasso precision matrix)
# ----------------------------------------------------------------------

def partial_correlation_from_precision(theta: np.ndarray) -> np.ndarray:
    """Compute partial correlation matrix from a precision matrix Θ.

    ρ_partial[i,j] = -Θ[i,j] / sqrt(Θ[i,i] * Θ[j,j])
    Diagonal is set to 1 by convention.
    """
    if theta.shape[0] != theta.shape[1]:
        raise ValueError("theta must be square")
    d = np.sqrt(np.diag(theta))
    pcorr = -theta / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    return pcorr


def estimate_partial_correlation(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    max_iter: int = 200,
) -> pd.DataFrame:
    """Sparse partial correlation via graphical lasso.

    Args:
        returns: T x N returns DataFrame.
        alpha:   L1 regularization strength on the precision matrix.
        max_iter: graphical-lasso solver iterations.

    Returns:
        N x N partial-correlation DataFrame with the same labels as `returns`.
    """
    cov = returns.cov().values
    # Graphical lasso expects standardized or covariance-like input; sklearn
    # fits to the empirical covariance matrix internally when given samples.
    gl = GraphicalLasso(alpha=alpha, max_iter=max_iter, assume_centered=False)
    gl.fit(returns.values)
    theta = gl.precision_
    pcorr = partial_correlation_from_precision(theta)
    return pd.DataFrame(pcorr, index=returns.columns, columns=returns.columns)


# ----------------------------------------------------------------------
# EWMA correlation
# ----------------------------------------------------------------------

def ewma_correlation(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """EWMA correlation matrix using the latest snapshot.

    Implements the RiskMetrics-style recursion:
        Σ_t = (1 - λ) r_t r_t' + λ Σ_{t-1}

    Args:
        returns: T x N returns DataFrame (most recent rows = most influential).
        lam:     decay parameter in (0, 1); 0.94 is the RiskMetrics default.

    Returns:
        N x N EWMA correlation DataFrame.
    """
    if not (0 < lam < 1):
        raise ValueError("lam must be in (0, 1)")
    r = returns.values - returns.values.mean(axis=0, keepdims=True)
    T, N = r.shape

    # Initialize with the sample covariance over the first window
    cov = np.cov(r, rowvar=False)
    for t in range(T):
        outer = np.outer(r[t], r[t])
        cov = lam * cov + (1 - lam) * outer

    d = np.sqrt(np.diag(cov))
    # Guard against zero-vol assets
    d = np.where(d <= 0, 1e-12, d)
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2
    return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)


# ----------------------------------------------------------------------
# Lower-tail dependence
# ----------------------------------------------------------------------

def lower_tail_dependence(
    returns: pd.DataFrame,
    q: float = 0.05,
) -> pd.DataFrame:
    """Empirical lower-tail dependence coefficient matrix.

        λ_L[i,j] = P( F_j(X_j) ≤ q | F_i(X_i) ≤ q )

    A symmetric version is taken as the average of the two conditional
    probabilities (since the conditional definition is asymmetric).
    Diagonal is set to 1 by convention.

    Args:
        returns: T x N returns DataFrame.
        q: lower-tail quantile threshold in (0, 0.5).

    Returns:
        N x N tail-dependence DataFrame in [0, 1].
    """
    if not (0 < q < 0.5):
        raise ValueError("q must be in (0, 0.5)")

    # Mark which observations are in each asset's lower q-tail
    thresholds = returns.quantile(q)
    tail_mask = returns.le(thresholds, axis=1).astype(int)
    cols = returns.columns

    N = len(cols)
    out = np.zeros((N, N))
    for i in range(N):
        ti = tail_mask.iloc[:, i].values
        denom_i = ti.sum()
        for j in range(i, N):
            tj = tail_mask.iloc[:, j].values
            denom_j = tj.sum()
            both = int((ti & tj).sum())
            cond_i = both / denom_i if denom_i > 0 else 0.0
            cond_j = both / denom_j if denom_j > 0 else 0.0
            lam = 0.5 * (cond_i + cond_j)
            out[i, j] = lam
            out[j, i] = lam
    np.fill_diagonal(out, 1.0)
    return pd.DataFrame(out, index=cols, columns=cols)


def tail_dependence_distance(tail_dep: pd.DataFrame) -> pd.DataFrame:
    """Convert tail-dependence to a distance matrix suitable for clustering.

    d[i,j] = sqrt(1 - λ_L[i,j])    ∈ [0, 1]
    """
    dist = np.sqrt(np.clip(1.0 - tail_dep.values, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=tail_dep.index, columns=tail_dep.columns)
