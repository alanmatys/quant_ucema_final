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
    RiskManagedMomentum, MomentumHRP,
    HRPDenoised, HRPDetoned,
    HRPPartialCorr, HRPDynamic, HRPTailDep, HRPShrunkCov,
    HRPVolStd, HRPTailDepShrunk,
    ERC, MaxDiv, NetworkRiskParity,
)
from src.denoising import (
    mp_pdf, fit_max_eigenvalue,
    denoise_corr_constant_residual, detone_corr,
)
from src.hrp_variants import (
    partial_correlation_from_precision, estimate_partial_correlation,
    ewma_correlation, lower_tail_dependence, tail_dependence_distance,
)
from src.inference import (
    optimal_block_length, stationary_block_bootstrap,
    sharpe_diff_ledoit_wolf, hansen_spa_test,
    cophenetic_correlation, adjusted_rand_between_snapshots,
)
from src.binance_data import get_historical_klines, get_usdt_symbols, get_symbols_from_list
from src.coingecko_data import (
    get_historical_prices, get_multiple_coins_data,
    create_returns_matrix, get_top_coins_by_market_cap,
    fetch_coingecko_dataset, get_coingecko_id
)
from src.agent import PortfolioChangeAnalyzerUSD
from src.universe import (
    fetch_binance_extended_prices,
    to_monthly_snapshots,
    build_pit_universe,
    summarize_pit_universe,
    load_binance_listings,
    load_candidates,
    load_pit_universe,
    DEFAULT_EXCLUDED_SYMBOLS,
)

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
    # HRP variants (Specs 01, 02, 06)
    "HRPDenoised",
    "HRPDetoned",
    "HRPPartialCorr",
    "HRPDynamic",
    "HRPTailDep",
    "HRPShrunkCov",
    "HRPVolStd",
    "HRPTailDepShrunk",
    # Comparator strategies (Spec 07)
    "ERC",
    "MaxDiv",
    "NetworkRiskParity",
    "mp_pdf",
    "fit_max_eigenvalue",
    "denoise_corr_constant_residual",
    "detone_corr",
    "partial_correlation_from_precision",
    "estimate_partial_correlation",
    "ewma_correlation",
    "lower_tail_dependence",
    "tail_dependence_distance",
    # Statistical inference (Spec 09)
    "optimal_block_length",
    "stationary_block_bootstrap",
    "sharpe_diff_ledoit_wolf",
    "hansen_spa_test",
    "cophenetic_correlation",
    "adjusted_rand_between_snapshots",
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
    # Point-in-time universe
    "fetch_binance_extended_prices",
    "to_monthly_snapshots",
    "build_pit_universe",
    "summarize_pit_universe",
    "load_binance_listings",
    "load_candidates",
    "load_pit_universe",
    "DEFAULT_EXCLUDED_SYMBOLS",
]
