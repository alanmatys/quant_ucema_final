"""
Quantitative Portfolio Allocation Package

This package provides portfolio allocation strategies for cryptocurrency assets,
with a focus on Hierarchical Risk Parity (HRP) and Momentum strategies.

Modules:
    portfolio_maker: Portfolio allocation strategies (HRP, IVP, MVP, Momentum)
    binance_data: Data collection utilities for Binance API
    coingecko_data: Data collection utilities for CoinGecko API
    agent: Portfolio transition analysis tools
"""

from src.portfolio_maker import (
    HRP, IVP, MVP, PortfolioStrategy,
    CrossSectionalMomentum, TimeSeriesMomentum,
    RiskManagedMomentum, MomentumHRP
)
from src.binance_data import get_historical_klines, get_usdt_symbols, get_symbols_from_list
from src.coingecko_data import (
    get_historical_prices, get_multiple_coins_data,
    create_returns_matrix, get_top_coins_by_market_cap,
    fetch_coingecko_dataset, get_coingecko_id
)
from src.agent import PortfolioChangeAnalyzerUSD

__all__ = [
    # Base class
    "PortfolioStrategy",
    # Risk-based strategies
    "HRP",
    "IVP",
    "MVP",
    # Momentum strategies
    "CrossSectionalMomentum",
    "TimeSeriesMomentum",
    "RiskManagedMomentum",
    "MomentumHRP",
    # Binance data utilities
    "get_historical_klines",
    "get_usdt_symbols",
    "get_symbols_from_list",
    # CoinGecko data utilities
    "get_historical_prices",
    "get_multiple_coins_data",
    "create_returns_matrix",
    "get_top_coins_by_market_cap",
    "fetch_coingecko_dataset",
    "get_coingecko_id",
    # Analysis tools
    "PortfolioChangeAnalyzerUSD",
]
