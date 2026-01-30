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
        """Compute risk-managed momentum weights with volatility scaling."""
        raw_weights = self.calculate_raw_momentum_weights()
        scaling = self.calculate_vol_scaling_factor()

        # Scale weights
        scaled_weights = raw_weights * scaling

        # Normalize to sum to 1 (long-only, no actual leverage)
        if scaled_weights.sum() > 0:
            weights = scaled_weights / scaled_weights.sum()
        else:
            weights = pd.Series(1.0 / len(self.returns.columns), index=self.returns.columns)

        self.weights = weights
        return weights


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
