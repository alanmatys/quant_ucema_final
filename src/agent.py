import numpy as np

class PortfolioChangeAnalyzerUSD:
    def __init__(self, current_weights, proposed_weights, returns, transaction_cost_rate=0.001): # 0.1% transaction costs (Binance: https://www.binance.com/en/fee/trading)
        """
        Initialize the analyzer with current and proposed portfolio weights
        
        Parameters:
        - current_weights: pd.Series of current portfolio weights
        - proposed_weights: pd.Series of proposed portfolio weights
        - returns: pd.DataFrame of historical returns
        - transaction_cost_rate: One-way transaction cost (default 0.1%)
        """
        self.current_weights = current_weights
        self.proposed_weights = proposed_weights
        self.returns = returns
        self.transaction_cost_rate = transaction_cost_rate
        
        # Ensure weights are aligned
        self.current_weights = self.current_weights.reindex(returns.columns).fillna(0)
        self.proposed_weights = self.proposed_weights.reindex(returns.columns).fillna(0)
        
    def calculate_turnover(self):
        """Calculate the portfolio turnover (sum of absolute weight changes)"""
        weight_diff = self.proposed_weights - self.current_weights
        turnover = np.abs(weight_diff).sum()
        return turnover
    
    def calculate_transaction_costs(self):
        """Calculate the total transaction costs (both buying and selling)"""
        turnover = self.calculate_turnover()
        total_cost = turnover * self.transaction_cost_rate
        return total_cost
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate expected return and risk for a given portfolio"""
        expected_return = np.sum(self.returns.mean() * weights) * 365  # Annualized
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 365, weights)))
        sharpe_ratio = expected_return / portfolio_vol # No risk-free rate in crypto
        
        return {
            'expected_return': expected_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    def analyze_transition(self, min_improvement_threshold=0.001):
        """
        Analyze whether transitioning to the new portfolio is worth it
        
        Parameters:
        - min_improvement_threshold: Minimum improvement in Sharpe ratio required
        
        Returns:
        - dict with recommendation and analysis details
        """
        # Calculate transaction costs
        transaction_costs = self.calculate_transaction_costs()
        
        # Calculate metrics for both portfolios
        current_metrics = self.calculate_portfolio_metrics(self.current_weights)
        proposed_metrics = self.calculate_portfolio_metrics(self.proposed_weights)
        
        # Adjust proposed return for transaction costs
        adjusted_return = proposed_metrics['expected_return'] - transaction_costs
        adjusted_sharpe = adjusted_return / proposed_metrics['volatility']
        
        # Calculate improvement
        sharpe_improvement = adjusted_sharpe - current_metrics['sharpe_ratio']
        
        # Decision logic
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


class PortfolioChangeAnalyzerPairs:
    def __init__(self, current_weights, proposed_weights, returns, transaction_cost_rate=0.001): # 0.1% transaction costs (Binance: https://www.binance.com/en/fee/trading)
        """
        Initialize the analyzer with current and proposed portfolio weights for pairs trading
        
        Parameters:
        - current_weights: pd.Series of current portfolio weights
        - proposed_weights: pd.Series of proposed portfolio weights
        - returns: pd.DataFrame of historical returns
        - transaction_cost_rate: One-way transaction cost (default 0.1%)
        """
        self.current_weights = current_weights
        self.proposed_weights = proposed_weights
        self.returns = returns
        self.transaction_cost_rate = transaction_cost_rate
        
        # Ensure weights are aligned
        self.current_weights = self.current_weights.reindex(returns.columns).fillna(0)
        self.proposed_weights = self.proposed_weights.reindex(returns.columns).fillna(0)
        
    def calculate_turnover(self):
        """Calculate the portfolio turnover considering direct pairs trading"""
        weight_diff = self.proposed_weights - self.current_weights
        # Only count the absolute changes for one side of each pair trade
        # since we're directly converting one asset to another
        turnover = np.abs(weight_diff).sum() / 2
        return turnover
    
    def calculate_transaction_costs(self):
        """Calculate the total transaction costs for pairs trading (single conversion)"""
        turnover = self.calculate_turnover()
        # Only one transaction cost since we're directly converting between pairs
        total_cost = turnover * self.transaction_cost_rate
        return total_cost
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate expected return and risk for a given portfolio"""
        expected_return = np.sum(self.returns.mean() * weights) * 365  # Annualized
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 365, weights)))
        sharpe_ratio = expected_return / portfolio_vol # No risk-free rate in crypto
        
        return {
            'expected_return': expected_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    def analyze_transition(self, min_improvement_threshold=0.001):
        """
        Analyze whether transitioning to the new portfolio via pairs trading is worth it
        
        Parameters:
        - min_improvement_threshold: Minimum improvement in Sharpe ratio required
        
        Returns:
        - dict with recommendation and analysis details
        """
        # Calculate transaction costs (reduced due to pairs trading)
        transaction_costs = self.calculate_transaction_costs()
        
        # Calculate metrics for both portfolios
        current_metrics = self.calculate_portfolio_metrics(self.current_weights)
        proposed_metrics = self.calculate_portfolio_metrics(self.proposed_weights)
        
        # Adjust proposed return for transaction costs
        adjusted_return = proposed_metrics['expected_return'] - transaction_costs
        adjusted_sharpe = adjusted_return / proposed_metrics['volatility']
        
        # Calculate improvement
        sharpe_improvement = adjusted_sharpe - current_metrics['sharpe_ratio']
        
        # Decision logic
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