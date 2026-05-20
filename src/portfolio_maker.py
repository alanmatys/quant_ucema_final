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
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform

from sklearn.covariance import LedoitWolf

from src.denoising import denoise_corr_constant_residual, detone_corr
from src.hrp_variants import (
    estimate_partial_correlation,
    ewma_correlation,
    lower_tail_dependence,
    tail_dependence_distance,
)


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

    def __init__(self, returns: pd.DataFrame, linkage_method: str = "single") -> None:
        """Initialize HRP with optional linkage method override.

        Args:
            returns: T x N returns DataFrame.
            linkage_method: scipy linkage method ('single', 'average', 'complete',
                'ward', etc.). Default 'single' preserves backward compatibility
                with the original Lopez de Prado specification. Subclasses inherit
                this attribute; override via `self.linkage_method = 'average'`
                after construction, or pass through constructors that support it.
        """
        super().__init__(returns)
        self.linkage_method = linkage_method

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
        link = linkage(condensed_dist, self.linkage_method)

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


# =============================================================================
# MOMENTUM STRATEGIES
# =============================================================================

class CrossSectionalMomentum(PortfolioStrategy):
    """
    Cross-Sectional Momentum Portfolio Strategy.

    Ranks assets by cumulative returns over a formation period and allocates
    to top performers while excluding bottom performers.

    Crypto-optimized defaults: 21-day formation, 7-day holding, top 30%.

    Reference:
        Liu, Y., Tsyvinski, A., & Wu, X. (2022). Common Risk Factors in Cryptocurrency.
    """

    def __init__(self, returns, formation_period=21, holding_period=7,
                 top_percentile=0.3, bottom_percentile=0.2, weighting_scheme='equal'):
        """
        Args:
            returns: Historical returns DataFrame (T x N)
            formation_period: Lookback days for momentum calculation (default: 21)
            holding_period: Days to hold positions (default: 7)
            top_percentile: Include assets in top X% (default: 0.3)
            bottom_percentile: Exclude assets in bottom X% (default: 0.2)
            weighting_scheme: 'equal', 'momentum', or 'inverse_vol'
        """
        super().__init__(returns)
        self.formation_period = formation_period
        self.holding_period = holding_period
        self.top_percentile = top_percentile
        self.bottom_percentile = bottom_percentile
        self.weighting_scheme = weighting_scheme
        self.momentum_scores = None
        self.selected_assets = None

    def calculate_momentum_scores(self):
        """Calculate momentum score (cumulative return) for each asset."""
        # Use formation_period, skip last day to avoid short-term reversal
        if len(self.returns) < self.formation_period:
            formation_returns = self.returns
        else:
            formation_returns = self.returns.iloc[-(self.formation_period+1):-1]

        # Cumulative return over formation period
        cumulative_returns = (1 + formation_returns).prod() - 1
        self.momentum_scores = cumulative_returns
        return cumulative_returns

    def select_assets(self):
        """Select assets based on momentum ranking."""
        if self.momentum_scores is None:
            self.calculate_momentum_scores()

        n_assets = len(self.momentum_scores)
        n_top = max(1, int(n_assets * self.top_percentile))
        n_exclude = int(n_assets * self.bottom_percentile)

        # Rank and select
        ranked = self.momentum_scores.sort_values(ascending=False)

        # Exclude bottom losers, take top winners
        if n_exclude > 0 and n_exclude < len(ranked):
            eligible = ranked.iloc[:-n_exclude]
        else:
            eligible = ranked

        selected = eligible.head(n_top).index.tolist()
        self.selected_assets = selected
        return selected

    def get_weights(self):
        """Compute portfolio weights based on momentum selection."""
        selected = self.select_assets()

        # Initialize weights to zero
        weights = pd.Series(0.0, index=self.returns.columns)

        if len(selected) == 0:
            # Fallback: equal weight all
            weights = pd.Series(1.0 / len(self.returns.columns), index=self.returns.columns)
            self.weights = weights
            return weights

        if self.weighting_scheme == 'equal':
            weights[selected] = 1.0 / len(selected)

        elif self.weighting_scheme == 'momentum':
            # Weight by momentum score (shifted to positive)
            mom_scores = self.momentum_scores[selected]
            shifted = mom_scores - mom_scores.min() + 0.001
            weights[selected] = shifted / shifted.sum()

        elif self.weighting_scheme == 'inverse_vol':
            # Inverse volatility within selected assets
            vols = self.returns[selected].std()
            inv_vol = 1.0 / vols
            inv_vol = inv_vol.replace([np.inf, -np.inf], 0).fillna(0)
            if inv_vol.sum() > 0:
                weights[selected] = inv_vol / inv_vol.sum()
            else:
                weights[selected] = 1.0 / len(selected)

        self.weights = weights
        return weights


class TimeSeriesMomentum(PortfolioStrategy):
    """
    Time-Series Momentum (Trend Following) Portfolio Strategy.

    Each asset receives weight based on its own trend signal using
    moving average crossovers or absolute momentum.

    Reference:
        Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012).
        Time Series Momentum. Journal of Financial Economics.
    """

    def __init__(self, returns, signal_type='ma_crossover',
                 fast_period=7, slow_period=28, abs_lookback=21,
                 position_sizing='equal', volatility_target=0.15):
        """
        Args:
            returns: Historical returns DataFrame (T x N)
            signal_type: 'ma_crossover' or 'absolute'
            fast_period: Fast MA lookback (default: 7)
            slow_period: Slow MA lookback (default: 28)
            abs_lookback: Lookback for absolute momentum (default: 21)
            position_sizing: 'equal' or 'volatility_target'
            volatility_target: Target annualized vol (default: 0.15)
        """
        super().__init__(returns)
        self.signal_type = signal_type
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.abs_lookback = abs_lookback
        self.position_sizing = position_sizing
        self.volatility_target = volatility_target
        self.trend_signals = None

    def _calculate_cumulative_prices(self):
        """Convert returns to price index for MA calculation."""
        return (1 + self.returns).cumprod()

    def calculate_ma_crossover_signals(self):
        """Calculate MA crossover signals (1 = long, 0 = no position)."""
        prices = self._calculate_cumulative_prices()

        fast_ma = prices.rolling(window=self.fast_period).mean().iloc[-1]
        slow_ma = prices.rolling(window=self.slow_period).mean().iloc[-1]

        # Long when fast > slow (uptrend)
        signals = (fast_ma > slow_ma).astype(float)
        return signals

    def calculate_absolute_momentum_signals(self):
        """Calculate absolute momentum signals (1 if return > 0, else 0)."""
        if len(self.returns) < self.abs_lookback:
            lookback_returns = self.returns
        else:
            lookback_returns = self.returns.iloc[-self.abs_lookback:]

        cumulative = (1 + lookback_returns).prod() - 1
        signals = (cumulative > 0).astype(float)
        return signals

    def calculate_trend_signals(self):
        """Calculate trend signals based on selected method."""
        if self.signal_type == 'ma_crossover':
            signals = self.calculate_ma_crossover_signals()
        elif self.signal_type == 'absolute':
            signals = self.calculate_absolute_momentum_signals()
        else:
            raise ValueError(f"Unknown signal_type: {self.signal_type}")

        self.trend_signals = signals
        return signals

    def get_weights(self):
        """Compute portfolio weights based on trend signals."""
        signals = self.calculate_trend_signals()
        n_trending = signals.sum()

        if n_trending == 0:
            # No trends: equal weight all (defensive)
            weights = pd.Series(1.0 / len(self.returns.columns), index=self.returns.columns)
            self.weights = weights
            return weights

        if self.position_sizing == 'equal':
            weights = signals / n_trending

        elif self.position_sizing == 'volatility_target':
            # Scale by inverse volatility
            asset_vols = self.returns.std() * np.sqrt(365)
            asset_vols = asset_vols.replace(0, np.inf)
            target_weights = signals / asset_vols
            target_weights = target_weights.replace([np.inf, -np.inf], 0).fillna(0)

            if target_weights.sum() > 0:
                weights = target_weights / target_weights.sum()
            else:
                weights = signals / n_trending
        else:
            weights = signals / n_trending

        self.weights = weights
        return weights


class RiskManagedMomentum(PortfolioStrategy):
    """
    Risk-Managed Momentum Portfolio (Barroso & Santa-Clara Method).

    Combines cross-sectional momentum with volatility scaling to avoid
    momentum crashes and improve risk-adjusted returns.

    Key insight: Scale exposure inversely to recent portfolio volatility.

    Reference:
        Barroso, P., & Santa-Clara, P. (2015). Momentum Has Its Moments.
        Journal of Financial Economics.
    """

    def __init__(self, returns, formation_period=21, vol_lookback=63,
                 target_volatility=0.12, max_leverage=2.0, min_leverage=0.25,
                 top_percentile=0.3):
        """
        Args:
            returns: Historical returns DataFrame (T x N)
            formation_period: Momentum lookback (default: 21)
            vol_lookback: Volatility estimation window (default: 63)
            target_volatility: Target portfolio vol (default: 0.12)
            max_leverage: Cap on leverage (default: 2.0)
            min_leverage: Floor on exposure (default: 0.25)
            top_percentile: Momentum selection threshold (default: 0.3)
        """
        super().__init__(returns)
        self.formation_period = formation_period
        self.vol_lookback = vol_lookback
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.top_percentile = top_percentile
        self.raw_weights = None
        self.realized_volatility = None
        self.vol_scaling_factor = None

    def calculate_momentum_scores(self):
        """Calculate momentum scores (cumulative returns)."""
        if len(self.returns) < self.formation_period:
            formation_returns = self.returns
        else:
            formation_returns = self.returns.iloc[-(self.formation_period+1):-1]

        return (1 + formation_returns).prod() - 1

    def select_momentum_assets(self):
        """Select top momentum assets."""
        scores = self.calculate_momentum_scores()
        n_select = max(1, int(len(scores) * self.top_percentile))
        return scores.sort_values(ascending=False).head(n_select).index.tolist()

    def calculate_raw_momentum_weights(self):
        """Calculate raw (unscaled) momentum portfolio weights."""
        selected = self.select_momentum_assets()
        weights = pd.Series(0.0, index=self.returns.columns)
        weights[selected] = 1.0 / len(selected)
        self.raw_weights = weights
        return weights

    def calculate_realized_volatility(self):
        """Estimate recent realized volatility of the momentum portfolio."""
        raw_weights = self.calculate_raw_momentum_weights()

        if len(self.returns) < self.vol_lookback:
            vol_returns = self.returns
        else:
            vol_returns = self.returns.iloc[-self.vol_lookback:]

        # Portfolio returns with raw weights
        portfolio_returns = (vol_returns * raw_weights).sum(axis=1)

        # Annualized volatility
        realized_vol = portfolio_returns.std() * np.sqrt(365)
        self.realized_volatility = realized_vol
        return realized_vol

    def calculate_vol_scaling_factor(self):
        """Calculate volatility scaling factor with leverage bounds."""
        realized_vol = self.calculate_realized_volatility()

        if realized_vol == 0 or np.isnan(realized_vol):
            scaling = 1.0
        else:
            scaling = self.target_volatility / realized_vol

        # Apply leverage bounds
        scaling = np.clip(scaling, self.min_leverage, self.max_leverage)
        self.vol_scaling_factor = scaling
        return scaling

    def get_weights(self):
        """Compute risk-managed momentum weights with volatility scaling.

        Per Barroso & Santa-Clara (2015): scale raw_weights by
        sigma_target / sigma_realized, capped at [min_leverage, max_leverage].

        Previously this method renormalised the scaled weights to sum=1 —
        which CANCELLED the scaling entirely (because raw_weights already
        sums to 1, scaling by a scalar then renormalising gives back the
        original). The bug was caught in the Phase 5d code review of
        2026-05-19, evidenced by RM_MOM producing Sharpe = 0.216 identical
        to CS_MOM_eq21.

        Correct behaviour: hold the residual as cash (zero return). For
        long-only no-leverage we clip scaling at 1.0 maximum, so the
        weights sum to scaling (≤ 1.0) and the remainder (1 - scaling) is
        implicit cash. With scaling < 1, this de-risks during high-vol
        regimes — exactly what the Barroso & Santa-Clara prescription does.
        """
        raw_weights = self.calculate_raw_momentum_weights()
        scaling = self.calculate_vol_scaling_factor()

        # Cap at 1.0 to enforce long-only no-leverage
        effective_scaling = float(min(scaling, 1.0))

        # Apply scaling — residual (1 - effective_scaling) is implicit cash
        scaled_weights = raw_weights * effective_scaling
        # No renormalisation. Sum can be < 1.0; backtest treats unallocated
        # weight as a zero-return cash position.

        self.weights = scaled_weights
        return scaled_weights


class MomentumHRP(PortfolioStrategy):
    """
    Combined Momentum Selection + HRP Weighting Strategy.

    Two-stage approach:
        1. Use cross-sectional momentum to select top-performing assets
        2. Apply HRP to allocate weights among selected assets

    Combines momentum's alpha generation with HRP's diversification benefits.
    """

    def __init__(self, returns, formation_period=21, top_percentile=0.4,
                 bottom_exclude=0.2, min_assets=5):
        """
        Args:
            returns: Historical returns DataFrame (T x N)
            formation_period: Momentum lookback (default: 21)
            top_percentile: Assets to include (default: 0.4)
            bottom_exclude: Assets to exclude (default: 0.2)
            min_assets: Minimum assets for HRP (default: 5)
        """
        super().__init__(returns)
        self.formation_period = formation_period
        self.top_percentile = top_percentile
        self.bottom_exclude = bottom_exclude
        self.min_assets = min_assets
        self.momentum_scores = None
        self.selected_assets = None
        self.hrp_weights_subset = None

    def calculate_momentum_scores(self):
        """Calculate momentum scores for asset selection."""
        if len(self.returns) < self.formation_period:
            formation_returns = self.returns
        else:
            formation_returns = self.returns.iloc[-(self.formation_period+1):-1]

        self.momentum_scores = (1 + formation_returns).prod() - 1
        return self.momentum_scores

    def select_assets_by_momentum(self):
        """Select assets using momentum ranking with minimum diversification."""
        scores = self.calculate_momentum_scores()
        n_assets = len(scores)

        # Calculate selection bounds
        n_top = max(self.min_assets, int(n_assets * self.top_percentile))
        n_exclude = int(n_assets * self.bottom_exclude)

        # Rank by momentum
        ranked = scores.sort_values(ascending=False)

        # Exclude bottom, take from remaining
        if n_exclude > 0 and n_exclude < len(ranked):
            eligible = ranked.iloc[:-n_exclude]
        else:
            eligible = ranked

        selected = eligible.head(n_top).index.tolist()

        # Ensure minimum assets
        if len(selected) < self.min_assets:
            selected = ranked.head(self.min_assets).index.tolist()

        self.selected_assets = selected
        return selected

    def get_weights(self):
        """Compute Momentum+HRP weights."""
        # Stage 1: Momentum selection
        selected = self.select_assets_by_momentum()

        # Stage 2: HRP on selected assets
        subset_returns = self.returns[selected]

        # Apply HRP to the subset
        hrp = HRP(subset_returns)
        hrp_weights = hrp.get_weights()
        self.hrp_weights_subset = hrp_weights

        # Map back to full universe
        weights = pd.Series(0.0, index=self.returns.columns)
        weights[selected] = hrp_weights

        self.weights = weights
        return weights


# =============================================================================
# HRP VARIANTS — correlation-matrix modifications (Specs 01, 02, 06)
# =============================================================================
#
# All variants below follow the same pattern: override __init__ to replace
# self.corr (and rebuild self.cov consistently) before the inherited HRP
# pipeline runs unchanged. HRPTailDep is the exception — it uses a tail-based
# distance matrix, so it overrides get_weights too.


def _rebuild_cov_from_corr(corr: pd.DataFrame, std: np.ndarray) -> pd.DataFrame:
    """Reconstruct a covariance matrix from a corr matrix and a std vector."""
    cov = corr.values * np.outer(std, std)
    return pd.DataFrame(cov, index=corr.index, columns=corr.columns)


class HRPDenoised(HRP):
    """HRP using a Marchenko-Pastur denoised correlation matrix (Spec 01)."""

    def __init__(self, returns: pd.DataFrame, bandwidth: float = 0.25) -> None:
        super().__init__(returns)
        T, N = returns.shape
        if T <= N:
            # MP requires q = T/N > 1; fall back to sample correlation
            return
        q = T / N
        corr_denoised = denoise_corr_constant_residual(self.corr.values, q=q, bandwidth=bandwidth)
        self.corr = pd.DataFrame(corr_denoised, index=self.corr.index, columns=self.corr.columns)
        std = np.sqrt(np.diag(self.cov.values))
        self.cov = _rebuild_cov_from_corr(self.corr, std)


class HRPDetoned(HRPDenoised):
    """HRP using a denoised + detoned correlation matrix (Spec 02)."""

    def __init__(
        self,
        returns: pd.DataFrame,
        bandwidth: float = 0.25,
        n_market_components: int = 1,
    ) -> None:
        super().__init__(returns, bandwidth=bandwidth)
        if returns.shape[0] <= returns.shape[1]:
            return  # already fell back to sample; skip detoning too
        corr_detoned = detone_corr(self.corr.values, n_market_components=n_market_components)
        self.corr = pd.DataFrame(corr_detoned, index=self.corr.index, columns=self.corr.columns)
        std = np.sqrt(np.diag(self.cov.values))
        self.cov = _rebuild_cov_from_corr(self.corr, std)


class HRPPartialCorr(HRP):
    """HRP using a sparse partial-correlation matrix (Spec 06).

    Default `alpha = 1e-3` is calibrated for crypto returns (variance ~0.004).
    Previous default of 0.05 was inappropriate for the scale — it shrunk
    all off-diagonals to zero, making this variant silently degenerate to
    IVP (bug caught in Phase 8h HP sweep, when all 16 alpha cells gave
    identical Sharpe = 0.697 = IVP's exact value). For other-asset-classes
    with different return variance, alpha should be retuned or use_cv=True.
    """

    def __init__(self, returns: pd.DataFrame, alpha: float = 1e-3,
                 use_cv: bool = False) -> None:
        super().__init__(returns)
        pcorr = estimate_partial_correlation(returns, alpha=alpha, use_cv=use_cv)
        self.corr = pcorr
        std = np.sqrt(np.diag(self.cov.values))
        self.cov = _rebuild_cov_from_corr(self.corr, std)


class HRPDynamic(HRP):
    """HRP using an EWMA correlation snapshot (Spec 06)."""

    def __init__(self, returns: pd.DataFrame, lam: float = 0.94) -> None:
        super().__init__(returns)
        self.corr = ewma_correlation(returns, lam=lam)
        std = np.sqrt(np.diag(self.cov.values))
        self.cov = _rebuild_cov_from_corr(self.corr, std)


class HRPShrunkCov(HRP):
    """HRP using a Ledoit-Wolf shrunk covariance matrix (Spec 06 §2.4).

    Stabilizes the covariance estimate by analytically blending the sample
    covariance with a scaled-identity shrinkage target. The shrinkage
    intensity α ∈ [0, 1] is chosen by the Ledoit-Wolf formula and saved as
    `self.shrinkage_intensity` for paper diagnostics.
    """

    def __init__(self, returns: pd.DataFrame) -> None:
        super().__init__(returns)
        lw = LedoitWolf(assume_centered=False)
        lw.fit(returns.values)
        cov_shrunk = lw.covariance_
        self.cov = pd.DataFrame(cov_shrunk, index=self.cov.index, columns=self.cov.columns)
        std = np.sqrt(np.diag(cov_shrunk))
        std = np.where(std <= 0, 1e-12, std)
        corr_shrunk = cov_shrunk / np.outer(std, std)
        np.fill_diagonal(corr_shrunk, 1.0)
        corr_shrunk = (corr_shrunk + corr_shrunk.T) / 2
        self.corr = pd.DataFrame(corr_shrunk, index=self.corr.index, columns=self.corr.columns)
        self.shrinkage_intensity = float(lw.shrinkage_)


class HRPTailDep(HRP):
    """HRP using empirical lower-tail dependence as the distance (Spec 06).

    Overrides `get_weights` because the distance matrix is built directly from
    tail dependence rather than via the correlation→distance transform. Falls
    back to base HRP on failure (insufficient tail observations, etc.),
    logging via `self.fallback_used`.
    """

    def __init__(self, returns: pd.DataFrame, q: float = 0.05) -> None:
        super().__init__(returns)
        self.q = q
        self.fallback_used = False
        self.tail_dep: Optional[pd.DataFrame] = None

    def get_weights(self) -> pd.Series:
        try:
            self.tail_dep = lower_tail_dependence(self.returns, q=self.q)
            dist = tail_dependence_distance(self.tail_dep)
            dist = (dist + dist.T) / 2
            condensed = squareform(dist.values, checks=False)
            link = linkage(condensed, self.linkage_method)
            sort_ix = HRP.get_quasi_diag(link)
            sort_ix = self.corr.index[sort_ix].tolist()
            self.weights = HRP.get_rec_bipart(self.cov, sort_ix)
            return self.weights
        except Exception:
            # Fall back to base HRP on any failure (e.g. degenerate tail data)
            self.fallback_used = True
            return super().get_weights()


class HRPVolStd(HRP):
    """HRP using returns standardized by their rolling volatility (Spec 06 §2.5).

    Standardizing returns by per-asset rolling volatility prevents the
    clustering from being dominated by raw volatility scale differences
    (e.g. BTC vs SHIB), surfacing genuine co-movement structure instead.
    Closer in spirit to GARCH-standardized residuals.

    Implementation: divide each daily return by its rolling-window std
    (shifted by one day to avoid look-ahead), then compute Pearson
    correlation on the standardized series. Covariance is kept as the
    sample covariance of the raw returns (the standardization is for the
    cluster topology, not the risk-allocation step).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        vol_window: int = 30,
        linkage_method: str = "single",
    ) -> None:
        super().__init__(returns, linkage_method=linkage_method)
        self.vol_window = vol_window
        rolling_vol = returns.rolling(window=vol_window, min_periods=10).std()
        rolling_vol = rolling_vol.shift(1)
        standardized = returns.div(rolling_vol).dropna(how="any")
        if standardized.empty or standardized.shape[0] < 3:
            # Insufficient data — fall back to vanilla correlation
            return
        new_corr = standardized.corr()
        # Snap diagonal to unit and symmetrize
        new_corr_vals = new_corr.values.copy()
        np.fill_diagonal(new_corr_vals, 1.0)
        new_corr_vals = (new_corr_vals + new_corr_vals.T) / 2
        self.corr = pd.DataFrame(new_corr_vals, index=new_corr.index, columns=new_corr.columns)


class HRPTailDepShrunk(HRPTailDep):
    """Hybrid: lower-tail-dependence distance + Ledoit-Wolf shrunk covariance.

    Combines Report 3's #1 recommended HRP extension for crypto:
    *tail-dependence distance* for the dendrogram topology (so the clusters
    reflect joint-crash behaviour, which is what long-only crypto risk
    actually looks like) with *Ledoit-Wolf shrunk covariance* in the
    recursive-bisection step (so the within-cluster weight allocation is
    not dominated by sample-covariance noise).

    Inherits the tail-dep `get_weights` path from `HRPTailDep` (which already
    uses `self.cov` for bisection), so we only need to substitute `self.cov`
    with the LW-shrunk version in `__init__`. Inherits the same fallback
    behaviour: on degenerate tail data the parent class falls back to base
    HRP, which then uses the shrunk covariance.
    """

    def __init__(self, returns: pd.DataFrame, q: float = 0.05) -> None:
        super().__init__(returns, q=q)
        lw = LedoitWolf(assume_centered=False)
        lw.fit(returns.values)
        cov_shrunk = lw.covariance_
        self.cov = pd.DataFrame(cov_shrunk, index=self.cov.index, columns=self.cov.columns)
        self.shrinkage_intensity = float(lw.shrinkage_)


# =============================================================================
# COMPARATOR STRATEGIES — ERC, MaxDiv, Network Risk Parity (Spec 07)
# =============================================================================

class ERC(PortfolioStrategy):
    """Equal Risk Contribution portfolio (Maillard, Roncalli & Teïletche 2010).

    Solves for long-only, fully-invested weights such that each asset
    contributes equally to portfolio risk:

        RC_i = w_i * (Σw)_i ≈ constant for all i

    Implementation: SLSQP minimization of the sum of squared deviations
    between actual and target risk contributions (uniform 1/N target).
    """

    def get_weights(self) -> pd.Series:
        cov = self.cov.values
        n = cov.shape[0]
        target = 1.0 / n
        x0 = np.full(n, 1.0 / n)

        def objective(w: np.ndarray) -> float:
            portfolio_var = float(w @ cov @ w)
            if portfolio_var <= 0:
                return 1e8
            marginal_risk = cov @ w
            rc = w * marginal_risk / portfolio_var  # normalized RC, sums to 1
            return float(np.sum((rc - target) ** 2))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(1e-8, 1.0) for _ in range(n)]
        result = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        w = np.clip(result.x, 0, None)
        w = w / w.sum()
        self.weights = pd.Series(w, index=self.cov.index)
        return self.weights


class MaxDiv(PortfolioStrategy):
    """Maximum Diversification portfolio (Choueifaty & Coignard 2008).

    Maximizes the diversification ratio:

        DR(w) = (w' · σ) / sqrt(w' · Σ · w)

    Long-only, fully invested. Implementation: SLSQP minimization of -DR(w).
    """

    def get_weights(self) -> pd.Series:
        cov = self.cov.values
        vols = np.sqrt(np.diag(cov))
        n = cov.shape[0]
        x0 = np.full(n, 1.0 / n)

        def neg_div_ratio(w: np.ndarray) -> float:
            numer = float(w @ vols)
            denom = float(np.sqrt(max(w @ cov @ w, 1e-16)))
            return -numer / denom

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0) for _ in range(n)]
        result = minimize(
            neg_div_ratio, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        w = np.clip(result.x, 0, None)
        w = w / w.sum()
        self.weights = pd.Series(w, index=self.cov.index)
        return self.weights


class NetworkRiskParity(PortfolioStrategy):
    """Network Risk Parity (Ciciretti & Pallotta 2024).

    Builds a Minimum Spanning Tree (MST) from the correlation-derived distance
    matrix, then allocates weights inverse to a combination of asset volatility
    and MST node degree. Highly connected (hub) assets receive less weight,
    diversifying away from common-factor exposure.

    Weighting (default 'inverse_degree'):
        w_i ∝ 1 / (σ_i × (degree_i + 1))     # +1 prevents leaf-only concentration
        then normalized to sum to 1.

    Args:
        returns: T x N returns DataFrame.
        network_type: 'mst' (default) — PMFG reserved as future work.
        centrality: 'inverse_degree' (default) or 'inverse_eigenvector'.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        network_type: str = "mst",
        centrality: str = "inverse_degree",
    ) -> None:
        super().__init__(returns)
        if network_type != "mst":
            raise NotImplementedError("Only MST is implemented; PMFG is future work.")
        self.network_type = network_type
        self.centrality = centrality
        self.mst_edges: Optional[list[tuple[int, int]]] = None
        self.degrees: Optional[pd.Series] = None

    def _build_mst(self) -> np.ndarray:
        """Build MST adjacency from correlation distance d_ij = sqrt(0.5 (1 - ρ))."""
        corr = self.corr.values
        dist = np.sqrt(np.maximum(0.5 * (1 - corr), 0))
        np.fill_diagonal(dist, 0.0)
        mst_sparse = minimum_spanning_tree(dist)
        # scipy MST returns upper-triangular; symmetrize
        mst_dense = mst_sparse.toarray()
        mst_adj = (mst_dense + mst_dense.T) > 0
        return mst_adj.astype(int)

    def get_weights(self) -> pd.Series:
        mst_adj = self._build_mst()
        n = mst_adj.shape[0]

        # Record edges and degree for diagnostics
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if mst_adj[i, j]:
                    edges.append((i, j))
        self.mst_edges = edges
        degrees = mst_adj.sum(axis=1)
        self.degrees = pd.Series(degrees, index=self.corr.index)

        vols = np.sqrt(np.diag(self.cov.values))
        vols = np.where(vols <= 0, 1e-12, vols)

        if self.centrality == "inverse_degree":
            inverse_score = 1.0 / (vols * (degrees + 1))
        elif self.centrality == "inverse_eigenvector":
            # Power iteration for the dominant eigenvector of the adjacency
            v = np.ones(n) / np.sqrt(n)
            for _ in range(200):
                v_new = mst_adj @ v
                norm = np.linalg.norm(v_new)
                if norm < 1e-12:
                    break
                v_new = v_new / norm
                if np.max(np.abs(v_new - v)) < 1e-10:
                    break
                v = v_new
            ev_centrality = np.abs(v)
            inverse_score = 1.0 / (vols * (ev_centrality + 1e-6))
        else:
            raise ValueError(f"unknown centrality: {self.centrality}")

        w = inverse_score / inverse_score.sum()
        self.weights = pd.Series(w, index=self.cov.index)
        return self.weights


# =============================================================================
# EMBEDDING-DISTANCE HRP VARIANTS (feature/embedding-hrp-variants)
# =============================================================================
# These four variants replace HRP's correlation-derived distance with an
# embedding-derived distance. All inherit base HRP's bisection logic and use
# the sample covariance for within-cluster variance allocation; only the
# dendrogram topology changes.
#
# Same fallback pattern as HRPTailDep: on any failure, log via
# `self.fallback_used` and fall back to base HRP.

def _embedding_to_hrp_weights(
    obj: "HRP",
    distance_matrix: pd.DataFrame,
    cov: pd.DataFrame,
) -> pd.Series:
    """Shared logic for embedding-based HRP variants: linkage on distance,
    quasi-diagonalisation, recursive bisection using sample covariance."""
    d = distance_matrix.reindex(index=cov.index, columns=cov.index).fillna(1.0)
    d_arr = d.values.copy()
    d_arr = (d_arr + d_arr.T) / 2
    np.fill_diagonal(d_arr, 0.0)
    condensed = squareform(d_arr, checks=False)
    link = linkage(condensed, obj.linkage_method)
    sort_ix = HRP.get_quasi_diag(link)
    sort_ix = cov.index[sort_ix].tolist()
    return HRP.get_rec_bipart(cov, sort_ix)


class HRPPathSig(HRP):
    """HRP with path-signature embedding distance.

    Per-asset path signatures (level-3 truncation by default) are computed
    on a rolling log-price window, then converted to a cosine-distance
    matrix that replaces HRP's correlation distance.

    Optionally `extra_channels` (e.g. quote volume, trade count) are
    appended to each asset's path — orthogonal information the return
    series does not contain, letting the signature capture cross-channel
    structure (price/volume lead-lag) that correlation cannot. See
    `src/embeddings/path_signatures.py`.

    Reference: Chen (1957), Lyons (1998); crypto application Lyons & Akyildirim (2024).
    """

    def __init__(self, returns: pd.DataFrame, window: int = 60, level: int = 3,
                 linkage_method: str = "single",
                 extra_channels: Optional[list] = None) -> None:
        super().__init__(returns, linkage_method=linkage_method)
        self.window = window
        self.level = level
        self.extra_channels = extra_channels
        self.fallback_used = False
        self.sig_df: Optional[pd.DataFrame] = None

    def get_weights(self) -> pd.Series:
        try:
            from src.embeddings.path_signatures import (
                asset_path_signatures, signatures_to_distance,
            )
            self.sig_df = asset_path_signatures(
                self.returns, window=self.window, level=self.level,
                extra_channels=self.extra_channels,
            )
            dist = signatures_to_distance(self.sig_df)
            self.weights = _embedding_to_hrp_weights(self, dist, self.cov)
            return self.weights
        except Exception:
            self.fallback_used = True
            return super().get_weights()


class HRPNodeEmbed(HRP):
    """HRP with node2vec embedding distance.

    Builds a kNN graph from sample correlations, runs node2vec random walks +
    skip-gram, and uses cosine distance on the resulting embeddings.

    Reference: Grover & Leskovec (2016).
    """

    def __init__(self, returns: pd.DataFrame, dimensions: int = 32, k: int = 10,
                 walk_length: int = 20, num_walks: int = 40, p: float = 1.0,
                 q: float = 1.0, seed: int = 42, linkage_method: str = "single",
                 weight_mode: str = "absolute_corr") -> None:
        super().__init__(returns, linkage_method=linkage_method)
        self.dimensions = dimensions
        self.k = k
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.seed = seed
        self.weight_mode = weight_mode
        self.fallback_used = False
        self.emb_df: Optional[pd.DataFrame] = None

    def get_weights(self) -> pd.Series:
        try:
            from src.embeddings.graph_emb import (
                build_corr_knn_graph, node2vec_embeddings, embeddings_to_distance,
            )
            g = build_corr_knn_graph(self.corr, k=self.k, weight_mode=self.weight_mode)
            self.emb_df = node2vec_embeddings(
                g, dimensions=self.dimensions, walk_length=self.walk_length,
                num_walks=self.num_walks, p=self.p, q=self.q, seed=self.seed,
            )
            dist = embeddings_to_distance(self.emb_df)
            self.weights = _embedding_to_hrp_weights(self, dist, self.cov)
            return self.weights
        except Exception:
            self.fallback_used = True
            return super().get_weights()


class HRPContrastive(HRP):
    """HRP with contrastive-SSL embedding distance.

    Trains a small 1D-CNN encoder via NT-Xent contrastive loss on
    Gaussian-jittered window pairs, then embeds each asset as the mean
    of the encoder outputs over its windows.
    """

    def __init__(self, returns: pd.DataFrame, window: int = 40, stride: int = 5,
                 emb_dim: int = 32, hidden: int = 16, batch_size: int = 64,
                 epochs: int = 10, lr: float = 1e-3, seed: int = 42,
                 linkage_method: str = "single",
                 extra_channels: Optional[list] = None) -> None:
        super().__init__(returns, linkage_method=linkage_method)
        self.window = window
        self.stride = stride
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.extra_channels = extra_channels
        self.fallback_used = False
        self.emb_df: Optional[pd.DataFrame] = None

    def get_weights(self) -> pd.Series:
        try:
            from src.embeddings.contrastive import (
                train_contrastive_encoder, asset_embeddings_from_encoder,
                embeddings_to_distance,
            )
            encoder, _ = train_contrastive_encoder(
                self.returns, window=self.window, stride=self.stride,
                emb_dim=self.emb_dim, hidden=self.hidden,
                batch_size=self.batch_size, epochs=self.epochs,
                lr=self.lr, seed=self.seed, extra_channels=self.extra_channels,
            )
            self.emb_df = asset_embeddings_from_encoder(
                encoder, self.returns, window=self.window, stride=self.stride,
                extra_channels=self.extra_channels,
            )
            dist = embeddings_to_distance(self.emb_df)
            self.weights = _embedding_to_hrp_weights(self, dist, self.cov)
            return self.weights
        except Exception:
            self.fallback_used = True
            return super().get_weights()


class HRPTS2Vec(HRP):
    """HRP with TS2Vec-lite (timestamp-mask contrastive) embedding distance.

    Same encoder family as `HRPContrastive` but with random-mask
    augmentation in the style of Yue et al. (2022) instead of Gaussian
    jitter — closer to the official TS2Vec recipe.
    """

    def __init__(self, returns: pd.DataFrame, window: int = 40, stride: int = 5,
                 emb_dim: int = 32, hidden: int = 16, mask_ratio: float = 0.3,
                 batch_size: int = 64, epochs: int = 10, lr: float = 1e-3,
                 seed: int = 42, linkage_method: str = "single",
                 extra_channels: Optional[list] = None) -> None:
        super().__init__(returns, linkage_method=linkage_method)
        self.window = window
        self.stride = stride
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.mask_ratio = mask_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.extra_channels = extra_channels
        self.fallback_used = False
        self.emb_df: Optional[pd.DataFrame] = None

    def get_weights(self) -> pd.Series:
        try:
            from src.embeddings.ts2vec_lite import (
                train_ts2vec_lite_encoder, asset_embeddings_from_encoder,
                embeddings_to_distance,
            )
            encoder = train_ts2vec_lite_encoder(
                self.returns, window=self.window, stride=self.stride,
                emb_dim=self.emb_dim, hidden=self.hidden,
                mask_ratio=self.mask_ratio, batch_size=self.batch_size,
                epochs=self.epochs, lr=self.lr, seed=self.seed,
                extra_channels=self.extra_channels,
            )
            self.emb_df = asset_embeddings_from_encoder(
                encoder, self.returns, window=self.window, stride=self.stride,
                extra_channels=self.extra_channels,
            )
            dist = embeddings_to_distance(self.emb_df)
            self.weights = _embedding_to_hrp_weights(self, dist, self.cov)
            return self.weights
        except Exception:
            self.fallback_used = True
            return super().get_weights()
