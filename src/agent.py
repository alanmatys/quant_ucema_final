"""
Portfolio Transition Analysis Module

This module provides tools to analyze whether rebalancing a portfolio
is worthwhile after accounting for transaction costs.

The main class `PortfolioChangeAnalyzerUSD` helps determine if the expected
improvement in risk-adjusted returns justifies the trading costs.
"""

import numpy as np
import pandas as pd


class PortfolioChangeAnalyzerUSD:
    """
    Analyzer for portfolio rebalancing decisions with transaction costs.

    This class evaluates whether transitioning from a current portfolio
    to a proposed portfolio is beneficial after accounting for transaction
    costs (e.g., Binance trading fees).

    The analyzer computes:
        - Portfolio turnover (sum of absolute weight changes)
        - Transaction costs based on turnover
        - Risk-adjusted returns (Sharpe ratio) for both portfolios
        - Net benefit after costs

    Attributes:
        current_weights (pd.Series): Current portfolio weights
        proposed_weights (pd.Series): Target portfolio weights
        returns (pd.DataFrame): Historical returns for assets
        transaction_cost_rate (float): One-way trading cost (default: 0.1%)

    Example:
        >>> analyzer = PortfolioChangeAnalyzerUSD(
        ...     current_weights=old_weights,
        ...     proposed_weights=new_weights,
        ...     returns=returns_df,
        ...     transaction_cost_rate=0.001
        ... )
        >>> result = analyzer.analyze_transition()
        >>> if result['recommendation']:
        ...     print("Rebalance recommended")
    """

    def __init__(
        self,
        current_weights: pd.Series,
        proposed_weights: pd.Series,
        returns: pd.DataFrame,
        transaction_cost_rate: float = 0.001
    ) -> None:
        """
        Initialize the portfolio transition analyzer.

        Args:
            current_weights: Current portfolio weights indexed by asset names
            proposed_weights: Proposed portfolio weights indexed by asset names
            returns: Historical returns DataFrame (T x N)
            transaction_cost_rate: One-way transaction cost rate.
                Default is 0.001 (0.1%), based on Binance spot trading fees.
                Reference: https://www.binance.com/en/fee/trading
        """
        self.current_weights = current_weights
        self.proposed_weights = proposed_weights
        self.returns = returns
        self.transaction_cost_rate = transaction_cost_rate

        # Align weights with returns columns
        self.current_weights = self.current_weights.reindex(returns.columns).fillna(0)
        self.proposed_weights = self.proposed_weights.reindex(returns.columns).fillna(0)

    def calculate_turnover(self) -> float:
        """
        Calculate portfolio turnover.

        Turnover is the sum of absolute weight changes, representing
        the total trading volume needed to rebalance.

        Returns:
            Total turnover as a fraction (0 to 2, where 2 means
            complete portfolio replacement)
        """
        weight_diff = self.proposed_weights - self.current_weights
        turnover = np.abs(weight_diff).sum()
        return turnover

    def calculate_transaction_costs(self) -> float:
        """
        Calculate total transaction costs for rebalancing.

        Cost = turnover * transaction_cost_rate

        Returns:
            Total cost as a fraction of portfolio value
        """
        turnover = self.calculate_turnover()
        total_cost = turnover * self.transaction_cost_rate
        return total_cost

    def calculate_portfolio_metrics(self, weights: pd.Series) -> dict:
        """
        Calculate key portfolio performance metrics.

        Computes annualized expected return, volatility, and Sharpe ratio
        for a given set of weights.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary containing:
                - expected_return: Annualized expected return
                - volatility: Annualized portfolio volatility
                - sharpe_ratio: Return / Volatility (no risk-free rate for crypto)
        """
        # Annualized expected return (assuming daily returns)
        expected_return = np.sum(self.returns.mean() * weights) * 365

        # Annualized volatility
        portfolio_vol = np.sqrt(
            np.dot(weights.T, np.dot(self.returns.cov() * 365, weights))
        )

        # Sharpe ratio (no risk-free rate assumption for crypto)
        sharpe_ratio = expected_return / portfolio_vol if portfolio_vol > 0 else 0

        return {
            'expected_return': expected_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }

    def analyze_transition(self, min_improvement_threshold: float = 0.001) -> dict:
        """
        Analyze whether portfolio rebalancing is worthwhile.

        Compares the current and proposed portfolios, accounting for
        transaction costs, to determine if rebalancing improves
        risk-adjusted returns.

        Args:
            min_improvement_threshold: Minimum required Sharpe ratio
                improvement to recommend transition. Default is 0.001 (0.1%).
                Set to None to always recommend transition.

        Returns:
            Dictionary containing:
                - recommendation (bool): Whether to rebalance
                - current_portfolio (dict): Current portfolio metrics
                - proposed_portfolio (dict): Proposed portfolio metrics
                - transaction_costs (float): Cost of rebalancing
                - adjusted_sharpe_ratio (float): Proposed Sharpe after costs
                - sharpe_improvement (float): Net improvement in Sharpe ratio
                - turnover (float): Portfolio turnover
        """
        # Calculate transaction costs
        transaction_costs = self.calculate_transaction_costs()

        # Calculate metrics for both portfolios
        current_metrics = self.calculate_portfolio_metrics(self.current_weights)
        proposed_metrics = self.calculate_portfolio_metrics(self.proposed_weights)

        # Adjust proposed return for transaction costs
        adjusted_return = proposed_metrics['expected_return'] - transaction_costs
        adjusted_sharpe = (
            adjusted_return / proposed_metrics['volatility']
            if proposed_metrics['volatility'] > 0 else 0
        )

        # Calculate improvement
        sharpe_improvement = adjusted_sharpe - current_metrics['sharpe_ratio']

        # Decision logic
        if min_improvement_threshold is None:
            should_transition = True
        else:
            should_transition = sharpe_improvement > min_improvement_threshold

        return {
            'recommendation': should_transition,
            'current_portfolio': current_metrics,
            'proposed_portfolio': proposed_metrics,
            'transaction_costs': transaction_costs,
            'adjusted_sharpe_ratio': adjusted_sharpe,
            'sharpe_improvement': sharpe_improvement,
            'turnover': self.calculate_turnover()
        }
