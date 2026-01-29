"""
Portfolio Allocation Strategies Module

This module implements three portfolio allocation strategies:
1. HRP (Hierarchical Risk Parity) - Based on Lopez de Prado's methodology
2. IVP (Inverse Variance Portfolio) - Simple inverse variance weighting
3. MVP (Minimum Variance Portfolio) - Optimization-based minimum variance

Reference:
    Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


class PortfolioStrategy(ABC):
    """
    Abstract base class for portfolio allocation strategies.

    All portfolio strategies inherit from this class and must implement
    the `get_weights()` method to compute portfolio weights.

    Attributes:
        returns (pd.DataFrame): Historical returns matrix (T x N) where T is the
            number of time periods and N is the number of assets.
        cov (pd.DataFrame): Covariance matrix of asset returns (N x N).
        corr (pd.DataFrame): Correlation matrix of asset returns (N x N).
        weights (pd.Series): Portfolio weights after calling `get_weights()`.

    Example:
        >>> returns = pd.DataFrame(...)  # Your returns data
        >>> strategy = HRP(returns)
        >>> weights = strategy.get_weights()
        >>> print(weights.sum())  # Should be 1.0
    """

    def __init__(self, returns: pd.DataFrame) -> None:
        """
        Initialize the portfolio strategy with historical returns.

        Args:
            returns: DataFrame of asset returns with shape (T, N) where
                T is the number of time periods and N is the number of assets.
                Columns should be asset names/symbols.
        """
        self.returns = returns
        self.cov = returns.cov()
        self.corr = returns.corr()
        self.weights: Optional[pd.Series] = None

    @abstractmethod
    def get_weights(self) -> pd.Series:
        """
        Compute and return portfolio weights.

        Returns:
            pd.Series: Portfolio weights indexed by asset names.
                Weights sum to 1.0 and are non-negative.
        """
        raise NotImplementedError("Subclasses must implement get_weights()")


class HRP(PortfolioStrategy):
    """
    Hierarchical Risk Parity (HRP) Portfolio Strategy.

    HRP is a portfolio allocation method developed by Marcos Lopez de Prado
    that uses hierarchical clustering to build diversified portfolios. Unlike
    traditional mean-variance optimization, HRP does not require matrix inversion,
    making it more stable and robust to estimation errors.

    Algorithm Overview:
        1. Tree Clustering: Build a hierarchical tree based on correlation distances
        2. Quasi-Diagonalization: Reorganize covariance matrix to group similar assets
        3. Recursive Bisection: Allocate weights top-down using inverse variance

    Advantages over traditional methods:
        - No need for covariance matrix inversion (more stable)
        - Naturally incorporates hierarchical structure of assets
        - More robust to estimation errors in correlation/covariance
        - Better out-of-sample performance in many empirical studies

    Reference:
        Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform
        Out-of-Sample. Journal of Portfolio Management, 42(4), 59-69.

    Example:
        >>> returns = pd.DataFrame(...)
        >>> hrp = HRP(returns)
        >>> weights = hrp.get_weights()
    """

    @staticmethod
    def correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
        """
        Convert correlation matrix to distance matrix.

        Uses the formula: d_ij = sqrt((1 - rho_ij) / 2)

        This transformation ensures:
            - d_ij = 0 when rho_ij = 1 (perfectly correlated)
            - d_ij = 1 when rho_ij = -1 (perfectly anti-correlated)
            - d_ij = 0.707 when rho_ij = 0 (uncorrelated)

        Args:
            corr: Correlation matrix (N x N)

        Returns:
            Distance matrix suitable for hierarchical clustering
        """
        dist = ((1 - corr) / 2.0) ** 0.5
        dist[~np.isfinite(dist)] = 0
        return dist

    @staticmethod
    def get_quasi_diag(link: np.ndarray) -> list:
        """
        Sort assets by hierarchical clustering order (quasi-diagonalization).

        This function traverses the dendrogram and returns the order of assets
        that places similar assets adjacent to each other, creating a
        quasi-diagonal covariance matrix structure.

        Args:
            link: Linkage matrix from scipy.cluster.hierarchy.linkage

        Returns:
            List of asset indices in quasi-diagonal order
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    @staticmethod
    def get_cluster_var(cov: pd.DataFrame, c_items: list) -> float:
        """
        Compute the variance of a cluster using inverse-variance weights.

        Within each cluster, assets are weighted by their inverse variance,
        then the cluster's total variance is computed.

        Formula:
            w_i = (1/sigma_i^2) / sum(1/sigma_j^2)  for j in cluster
            cluster_var = w' * Cov * w

        Args:
            cov: Full covariance matrix
            c_items: List of asset names/indices in the cluster

        Returns:
            Cluster variance (scalar)
        """
        cov_ = cov.loc[c_items, c_items]
        w_ = 1.0 / np.diag(cov_)
        w_ /= w_.sum()
        w_ = w_.reshape(-1, 1)
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    @staticmethod
    def get_rec_bipart(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
        """
        Compute HRP weights through recursive bisection.

        Starting from equal weights, recursively split the portfolio into
        two clusters and allocate weights inversely proportional to each
        cluster's variance.

        At each split:
            alpha = 1 - var_left / (var_left + var_right)
            weights_left *= alpha
            weights_right *= (1 - alpha)

        This ensures more weight goes to lower-variance clusters.

        Args:
            cov: Covariance matrix
            sort_ix: Asset names in quasi-diagonal order

        Returns:
            pd.Series of portfolio weights indexed by asset names
        """
        w = pd.Series(1.0, index=sort_ix, dtype="float64")
        c_items = [sort_ix]

        while len(c_items) > 0:
            # Split each cluster into two halves
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]

            # Allocate weights between each pair of sub-clusters
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                c_var0 = HRP.get_cluster_var(cov, c_items0)
                c_var1 = HRP.get_cluster_var(cov, c_items1)

                # Inverse variance allocation between clusters
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha

        return w

    def get_weights(self) -> pd.Series:
        """
        Compute HRP portfolio weights.

        Steps:
            1. Convert correlation to distance matrix
            2. Perform hierarchical clustering (single linkage)
            3. Quasi-diagonalize the covariance matrix
            4. Apply recursive bisection to get final weights

        Returns:
            pd.Series of portfolio weights (sums to 1.0)
        """
        # Step 1: Correlation to distance
        dist = self.correl_dist(self.corr)
        dist = pd.DataFrame(dist, index=self.corr.index, columns=self.corr.index)
        dist = dist.fillna(0)
        dist = (dist + dist.T) / 2  # Ensure symmetry

        # Step 2: Hierarchical clustering
        condensed_dist = squareform(dist.values)
        link = linkage(condensed_dist, "single")

        # Step 3: Quasi-diagonalization
        sort_ix = self.get_quasi_diag(link)
        sort_ix = self.corr.index[sort_ix].tolist()

        # Step 4: Recursive bisection
        self.weights = self.get_rec_bipart(self.cov, sort_ix)

        return self.weights


class IVP(PortfolioStrategy):
    """
    Inverse Variance Portfolio (IVP) Strategy.

    The simplest risk-based allocation strategy that weights assets
    inversely proportional to their individual variances.

    Formula:
        w_i = (1/sigma_i^2) / sum(1/sigma_j^2)

    Properties:
        - Simple and fast to compute
        - No optimization required
        - Ignores correlations between assets
        - Allocates more to lower-volatility assets

    Use case:
        - Baseline comparison for more sophisticated methods
        - When correlation estimates are unreliable
        - Quick portfolio construction

    Example:
        >>> returns = pd.DataFrame(...)
        >>> ivp = IVP(returns)
        >>> weights = ivp.get_weights()
    """

    def get_weights(self) -> pd.Series:
        """
        Compute IVP portfolio weights.

        Returns:
            pd.Series of portfolio weights inversely proportional to variance
        """
        ivp = 1.0 / np.diag(self.cov)
        ivp /= ivp.sum()
        self.weights = pd.Series(ivp, index=self.cov.index)
        return self.weights


class MVP(PortfolioStrategy):
    """
    Minimum Variance Portfolio (MVP) Strategy.

    Classic mean-variance optimization that minimizes portfolio variance
    subject to full investment constraint.

    Optimization Problem:
        minimize    w' * Cov * w
        subject to  sum(w) = 1
                    0 <= w_i <= 1  (long-only constraint)

    Properties:
        - Requires numerical optimization
        - Considers full covariance structure
        - Can be sensitive to estimation errors
        - May concentrate in few low-variance assets

    Note:
        Uses SLSQP (Sequential Least Squares Programming) optimizer
        from scipy with long-only constraints.

    Example:
        >>> returns = pd.DataFrame(...)
        >>> mvp = MVP(returns)
        >>> weights = mvp.get_weights()
    """

    def get_weights(self) -> pd.Series:
        """
        Compute MVP portfolio weights via optimization.

        Returns:
            pd.Series of portfolio weights that minimize variance
        """
        n = len(self.cov)
        initial_weights = np.ones(n) / n

        # Full investment constraint
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Long-only bounds
        bounds = [(0, 1) for _ in range(n)]

        def portfolio_variance(weights: np.ndarray) -> float:
            """Objective function: portfolio variance."""
            return np.dot(weights.T, np.dot(self.cov, weights))

        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        self.weights = pd.Series(result.x, index=self.cov.index)
        return self.weights
