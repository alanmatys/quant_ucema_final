"""Statistical inference framework for portfolio comparison (Spec 09).

Implements four pieces of machinery:

1. **Stationary block bootstrap** (Politis & Romano 1994) with automatic
   block length selection (Politis & White 2004). Underlies the other two
   tests and produces CIs on arbitrary statistics that respect serial
   dependence.

2. **Ledoit-Wolf Sharpe difference test** (Ledoit & Wolf 2008). Robust
   pairwise test for H0: Sharpe(A) = Sharpe(B). Uses the studentized
   block-bootstrap variant from the paper.

3. **Hansen SPA test** (Hansen 2005). Multiple-strategy test for H0: no
   strategy outperforms the benchmark. Corrects for the multiple-testing
   problem that p-stacking across K candidates creates.

4. **Cluster stability diagnostics**: cophenetic correlation per dendrogram
   plus Adjusted Rand Index between cluster assignments at consecutive
   rebalance dates. Answers "is this dendrogram structure stable or
   estimation noise?", the cluster-level analog of the bootstrap-CI
   question for portfolio metrics.

References:
    Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. JASA.
    Politis, D. N., & White, H. (2004). Automatic block-length selection
        for the dependent bootstrap. Econometric Reviews.
    Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing
        with the Sharpe ratio. Journal of Empirical Finance.
    Hansen, P. R. (2005). A test for superior predictive ability. JBES.
    Hubert, L., & Arabie, P. (1985). Comparing partitions. J. Classification.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score


# ----------------------------------------------------------------------
# 1. Stationary block bootstrap
# ----------------------------------------------------------------------

def optimal_block_length(returns: np.ndarray) -> float:
    """Politis-White (2004) automatic block-length selection.

    Implements the spectral-density approach: estimates the rate of decay
    of the autocovariance function and picks an asymptotically optimal
    expected block length for the stationary bootstrap.

    Args:
        returns: 1D array of returns (length T).

    Returns:
        Recommended expected block length (>= 1).
    """
    r = np.asarray(returns, dtype=float)
    r = r - r.mean()
    n = len(r)
    if n < 10:
        return 1.0

    # Andrews-Bartlett kernel cutoff (heuristic): KN = min(n/2, 2 sqrt(log10 n))
    Kn = int(max(5, min(n // 2, 2 * np.sqrt(np.log10(n)))))

    # Autocorrelations up to Kn
    var = np.var(r)
    if var <= 0:
        return 1.0
    rhos = np.array([float((r[:n - k] @ r[k:]) / ((n - k) * var)) for k in range(1, Kn + 1)])

    # Estimate "significant" lag via 2/sqrt(n) criterion
    sig_threshold = 2.0 / np.sqrt(n)
    significant = np.where(np.abs(rhos) > sig_threshold)[0]
    if len(significant) == 0:
        return 1.0
    m_hat = int(significant.max() + 1)

    # Politis-White estimates:
    #   G_hat = sum_{k=-2m..2m} |k| g(k/m) gamma(k)
    #   D_hat = (2/3) * 2 * (sum_{k=-2m..2m} g(k/m) gamma(k))^2
    # where g is the flat-top kernel and gamma is autocovariance.
    M = min(2 * m_hat, n - 1)
    gammas = np.array([float((r[:n - k] @ r[k:]) / n) for k in range(0, M + 1)])

    def flat_top(x):
        ax = abs(x)
        if ax <= 0.5:
            return 1.0
        elif ax <= 1.0:
            return 2.0 * (1.0 - ax)
        return 0.0

    g_vals = np.array([flat_top(k / m_hat) for k in range(0, M + 1)])
    g_gamma = g_vals * gammas
    # G_hat: include k from -M to M, multiplied by |k|
    G_hat = 2.0 * np.sum(np.arange(1, M + 1) * g_vals[1:] * gammas[1:])
    # D_hat: same kernel but as (sum g*gamma)^2
    D_hat = (4.0 / 3.0) * (gammas[0] + 2.0 * np.sum(g_gamma[1:])) ** 2

    if D_hat <= 0 or G_hat <= 0:
        return float(max(1, m_hat))

    b_opt = (2.0 * G_hat ** 2 / D_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(max(1.0, min(n / 4.0, b_opt)))


def _stationary_block_indices(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Generate one stationary-bootstrap index sequence of length n.

    Each step has probability `p` of starting a new block from a random
    location; otherwise continues the current block (with wrap-around).
    Expected block length = 1/p.
    """
    indices = np.empty(n, dtype=np.int64)
    indices[0] = rng.integers(0, n)
    new_block = rng.random(n - 1) < p
    next_step = (indices[0] + 1) % n
    for t in range(1, n):
        if new_block[t - 1]:
            indices[t] = rng.integers(0, n)
        else:
            indices[t] = next_step
        next_step = (indices[t] + 1) % n
    return indices


def stationary_block_bootstrap(
    returns: pd.Series | np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_iter: int = 10_000,
    block_size: float | None = None,
    seed: int = 42,
) -> dict:
    """Stationary block bootstrap CI for an arbitrary statistic.

    Args:
        returns: 1D series of returns.
        statistic_fn: function (np.ndarray) -> float computed on each
            bootstrap sample.
        n_iter: number of bootstrap iterations.
        block_size: expected block length. If None, picked automatically
            via `optimal_block_length`.
        seed: RNG seed for reproducibility.

    Returns:
        dict with keys: 'point_estimate', 'ci_lower_95', 'ci_upper_95',
        'samples' (length n_iter), 'block_size'.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if block_size is None:
        block_size = optimal_block_length(r)
    p = 1.0 / float(block_size)
    p = float(min(max(p, 1e-6), 1.0))

    rng = np.random.default_rng(seed)
    samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = _stationary_block_indices(n, p, rng)
        samples[i] = float(statistic_fn(r[idx]))

    return {
        "point_estimate": float(statistic_fn(r)),
        "ci_lower_95": float(np.percentile(samples, 2.5)),
        "ci_upper_95": float(np.percentile(samples, 97.5)),
        "samples": samples,
        "block_size": float(block_size),
    }


# ----------------------------------------------------------------------
# 2. Ledoit-Wolf Sharpe difference test (bootstrap variant)
# ----------------------------------------------------------------------

def _sharpe(r: np.ndarray) -> float:
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma <= 0:
        return 0.0
    return mu / sigma


def _sharpe_diff_hac_se(a: np.ndarray, b: np.ndarray, bandwidth: int) -> float:
    """Delta-method HAC standard error of the Sharpe-ratio difference.

    The Sharpe ratio SR = mu / sqrt(gamma - mu^2) with gamma = E[r^2] is a
    smooth function of the moments. For the difference Delta = SR_a - SR_b,
    the delta method gives Var(Delta) = grad' Psi grad / n, where Psi is
    the HAC (Bartlett-kernel) long-run covariance of the influence vector
    y_t = (r_a, r_a^2, r_b, r_b^2). This is the standard error used by the
    Ledoit-Wolf (2008) studentized test.
    """
    n = len(a)
    mu_a, mu_b = float(a.mean()), float(b.mean())
    ga, gb = float((a ** 2).mean()), float((b ** 2).mean())
    va = max(ga - mu_a ** 2, 1e-16)
    vb = max(gb - mu_b ** 2, 1e-16)
    sa, sb = np.sqrt(va), np.sqrt(vb)
    # gradient of Delta wrt (mu_a, gamma_a, mu_b, gamma_b):
    #   d SR/d mu = gamma / sigma^3 ;  d SR/d gamma = -mu / (2 sigma^3)
    grad = np.array([ga / sa ** 3, -mu_a / (2 * sa ** 3),
                     -gb / sb ** 3,  mu_b / (2 * sb ** 3)])
    Y = np.column_stack([a - mu_a, a ** 2 - ga, b - mu_b, b ** 2 - gb])
    psi = Y.T @ Y / n
    L = max(1, int(bandwidth))
    for lag in range(1, L):
        w = 1.0 - lag / L  # Bartlett kernel weight
        g = Y[lag:].T @ Y[:-lag] / n
        psi += w * (g + g.T)
    var = float(grad @ psi @ grad / n)
    return float(np.sqrt(max(var, 1e-20)))


def sharpe_diff_ledoit_wolf(
    r_a: pd.Series | np.ndarray,
    r_b: pd.Series | np.ndarray,
    n_iter: int = 5_000,
    block_size: float | None = None,
    seed: int = 42,
) -> dict:
    """Ledoit-Wolf (2008) robust Sharpe-ratio difference test.

    Tests H0: Sharpe(A) = Sharpe(B) for paired return series. Implements
    the studentized stationary-block bootstrap of Ledoit & Wolf (2008):
    the observed Sharpe difference is studentized by a delta-method HAC
    standard error (`_sharpe_diff_hac_se`); each bootstrap resample is
    studentized by its own HAC standard error; the two-sided p-value is
    the bootstrap tail probability of the studentized statistic. Student
    -ising both the observed and bootstrap statistics is what gives the
    test its asymptotic refinement and robustness to non-iid returns ---
    a plain percentile bootstrap of the raw difference does not have it.

    Args:
        r_a, r_b: paired return series (same length).
        n_iter: bootstrap iterations.
        block_size: expected block length; auto-selected (Politis-White)
            if None. Also used as the HAC Bartlett-kernel bandwidth.
        seed: RNG seed.

    Returns:
        dict: 'sharpe_a', 'sharpe_b', 'sharpe_diff', 'se', 'studentized',
        'p_value', 'ci_95', 'block_size'.
    """
    a = np.asarray(r_a, dtype=float)
    b = np.asarray(r_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("r_a and r_b must have the same length")
    n = len(a)
    sa, sb = _sharpe(a), _sharpe(b)
    obs_diff = sa - sb

    if block_size is None:
        block_size = optimal_block_length((a + b) / 2.0)
    bs = max(2, int(round(float(block_size))))
    p = float(min(max(1.0 / bs, 1e-6), 1.0))

    se = _sharpe_diff_hac_se(a, b, bandwidth=bs)
    obs_stud = obs_diff / se if se > 0 else 0.0

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iter, dtype=float)
    z = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = _stationary_block_indices(n, p, rng)
        aa, bb = a[idx], b[idx]
        d = _sharpe(aa) - _sharpe(bb)
        diffs[i] = d
        se_b = _sharpe_diff_hac_se(aa, bb, bandwidth=bs)
        z[i] = (d - obs_diff) / se_b if se_b > 0 else 0.0

    # Two-sided studentized-bootstrap p-value.
    p_value = float(min(1.0, np.mean(np.abs(z) >= abs(obs_stud))))

    return {
        "sharpe_a": sa,
        "sharpe_b": sb,
        "sharpe_diff": obs_diff,
        "se": se,
        "studentized": obs_stud,
        "p_value": p_value,
        "ci_95": (
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)),
        ),
        "block_size": float(bs),
    }


# ----------------------------------------------------------------------
# 3. Hansen Superior Predictive Ability (SPA) test
# ----------------------------------------------------------------------

def hansen_spa_test(
    returns_matrix: pd.DataFrame,
    benchmark: pd.Series | np.ndarray,
    n_bootstrap: int = 5_000,
    block_size: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Hansen (2005) Superior Predictive Ability test.

    H0: max_k E[d_k] <= 0, where d_k = r_k - r_benchmark (no candidate
    outperforms the benchmark). Reports the lower, consistent and upper
    p-values per Hansen.

    The three p-values are computed with the reference ``arch`` package
    implementation (``arch.bootstrap.SPA``). An earlier hand-rolled
    version mis-scaled the consistent-estimator recentering threshold by
    a factor of sqrt(n), which produced p-values violating the
    p_lower <= p_consistent <= p_upper ordering; delegating to ``arch``
    removes that bug. ``arch`` works in terms of forecast *losses*
    (lower = better), so we pass negated returns: a candidate with a
    lower loss than the benchmark is one that beats it.

    Args:
        returns_matrix: T x K DataFrame of K candidate strategy returns.
        benchmark: T-vector of benchmark returns.
        n_bootstrap: bootstrap iterations.
        block_size: expected block length; auto-selected (Politis-White)
            if None.
        seed: RNG seed.

    Returns:
        DataFrame with one row per strategy. Columns 'mean_diff' and
        'studentized' are per-candidate descriptive scores; 'p_lower',
        'p_consistent', 'p_upper' are the grid-wide SPA p-values and are
        therefore identical across rows by construction (the SPA is a
        single joint test, not a per-candidate test).
    """
    from arch.bootstrap import SPA

    R = returns_matrix.values
    B = np.asarray(benchmark, dtype=float)
    n, K = R.shape
    if len(B) != n:
        raise ValueError("benchmark must have same length as returns_matrix")

    D = R - B[:, None]  # n x K performance differentials
    d_bar = D.mean(axis=0)

    if block_size is None:
        block_size = optimal_block_length(B)
    bs = int(max(2, round(float(block_size))))

    # Per-candidate descriptive studentized scores: long-run sd of d_kt
    # via the stationary bootstrap (sqrt(n) * sd-of-the-mean).
    rng = np.random.default_rng(seed)
    p_geom = float(min(max(1.0 / bs, 1e-6), 1.0))
    boot_means = np.empty((n_bootstrap, K), dtype=float)
    for i in range(n_bootstrap):
        idx = _stationary_block_indices(n, p_geom, rng)
        boot_means[i, :] = D[idx, :].mean(axis=0)
    omega = boot_means.std(axis=0, ddof=1) * np.sqrt(n)
    omega = np.where(omega <= 0, 1e-12, omega)
    studentized = np.sqrt(n) * d_bar / omega

    # SPA p-values via arch. Negate returns -> losses (lower = better).
    spa = SPA(-B, -R, block_size=bs, reps=n_bootstrap,
              bootstrap="stationary", studentize=True, seed=seed)
    spa.compute()
    pv = spa.pvalues  # Series indexed by 'lower', 'consistent', 'upper'

    return pd.DataFrame(
        {
            "mean_diff": d_bar,
            "studentized": studentized,
            "p_lower": float(pv["lower"]),
            "p_consistent": float(pv["consistent"]),
            "p_upper": float(pv["upper"]),
        },
        index=returns_matrix.columns,
    )


# ----------------------------------------------------------------------
# 4. Cluster stability diagnostics
# ----------------------------------------------------------------------

def cophenetic_correlation(
    distance_matrix: np.ndarray | pd.DataFrame,
    linkage_matrix: np.ndarray,
) -> float:
    """Pearson correlation between cophenetic distances and original distances.

    Args:
        distance_matrix: NxN pairwise distance matrix (zero diagonal).
        linkage_matrix: scipy linkage matrix.

    Returns:
        Cophenetic correlation in [-1, 1]. >= 0.7 conventionally "good".
    """
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values
    # scipy requires square form with zero diagonal; convert to condensed
    d = distance_matrix.copy()
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    condensed = squareform(d, checks=False)
    coph, _ = cophenet(linkage_matrix, condensed)
    return float(coph)


def adjusted_rand_between_snapshots(
    linkage_a: np.ndarray,
    linkage_b: np.ndarray,
    labels_a: list[str],
    labels_b: list[str],
    k: int = 5,
) -> float:
    """ARI between two cluster assignments cut at K clusters.

    Args:
        linkage_a, linkage_b: scipy linkage matrices for two snapshots.
        labels_a, labels_b: asset name lists for each snapshot (used to
            align on the intersection of common assets).
        k: number of clusters to cut at.

    Returns:
        ARI in [-1, 1]. Returns NaN if the intersection has fewer than
        K+1 assets (insufficient overlap for a meaningful comparison).
    """
    common = sorted(set(labels_a) & set(labels_b))
    if len(common) <= k:
        return float("nan")

    idx_a = [labels_a.index(s) for s in common]
    idx_b = [labels_b.index(s) for s in common]
    cl_a = fcluster(linkage_a, t=k, criterion="maxclust")
    cl_b = fcluster(linkage_b, t=k, criterion="maxclust")
    return float(adjusted_rand_score(cl_a[idx_a], cl_b[idx_b]))
