# HRP and Momentum Strategies for Cryptocurrency Portfolios

A quantitative finance research project implementing **Hierarchical Risk Parity (HRP)** and **Momentum-based strategies** on cryptocurrency portfolios.

## Authors

- Federico Rodriguez
- Alan Matys

*UCEMA - Quantitative Finance Course Final Project*

---

## Objective

This project evaluates portfolio allocation methods for cryptocurrency assets, combining:
1. **Risk-based allocation** (HRP, IVP, MVP) for diversification and stability
2. **Momentum strategies** for return enhancement

The goal is to find strategies that deliver better risk-adjusted returns than simple alternatives.

### Reference Papers
- [Building Diversified Portfolios that Outperform Out-of-Sample](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678) - Lopez de Prado, M. (2016)
- [Common Risk Factors in Cryptocurrency](https://www.sciencedirect.com/science/article/abs/pii/S1544612325011377) - Liu et al.
- [Momentum Has Its Moments](https://www.sciencedirect.com/science/article/abs/pii/S154461232030177X) - Barroso & Santa-Clara

---

## Project Structure

```
quant_ucema_final/
├── src/                           # Core Python modules
│   ├── __init__.py                # Package exports
│   ├── portfolio_maker.py         # Portfolio allocation strategies (HRP, IVP, MVP, Momentum)
│   ├── binance_data.py            # Binance API data collection
│   ├── coingecko_data.py          # CoinGecko API data collection (alternative source)
│   └── agent.py                   # Portfolio transition analysis
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── data_collection.ipynb      # Historical data fetching
│   ├── corr_matrix_dendrogam.ipynb # Correlation analysis & dendrogram
│   ├── hrp.ipynb                  # Static HRP allocation analysis
│   ├── backtesting.ipynb          # Dynamic rebalancing backtest
│   ├── momentum_backtest.ipynb    # Momentum strategies backtesting
│   └── slides_examples.ipynb      # Educational examples
├── data/                          # Historical price data (CSV)
├── docs/                          # Documentation & formulas
├── paper/                         # Academic paper (LaTeX)
├── requirements.txt               # Python dependencies
└── environment.yml                # Conda environment
```

---

## Portfolio Strategies

### 1. Hierarchical Risk Parity (HRP)

HRP uses hierarchical clustering to build diversified portfolios without requiring covariance matrix inversion, making it more stable and robust to estimation errors.

**Algorithm Steps:**
1. **Distance Matrix**: Convert correlation to distance using `d_ij = sqrt((1 - rho_ij) / 2)`
2. **Hierarchical Clustering**: Build a dendrogram using single-linkage clustering
3. **Quasi-Diagonalization**: Reorder assets to place similar ones adjacent
4. **Recursive Bisection**: Allocate weights inversely proportional to cluster variance

**Advantages:**
- No matrix inversion required (numerically stable)
- Naturally incorporates hierarchical asset structure
- More robust to estimation errors
- Better out-of-sample performance

### 2. Inverse Variance Portfolio (IVP)

Simple risk-based allocation that weights assets inversely to their variance:

```
w_i = (1/sigma_i^2) / sum(1/sigma_j^2)
```

### 3. Minimum Variance Portfolio (MVP)

Classic optimization-based approach that minimizes portfolio variance:

```
minimize    w' * Cov * w
subject to  sum(w) = 1, w >= 0
```

### 4. Cross-Sectional Momentum (CS_MOM)

Cross-sectional momentum exploits the empirical observation that assets with strong recent performance tend to continue outperforming in the near future. This strategy ranks all assets by their cumulative returns and goes long on winners while avoiding losers.

**Methodology:**
1. Calculate cumulative return over formation period for each asset:
   ```
   MOM_i = ∏(1 + r_i,t) - 1    for t ∈ [T-formation, T-1]
   ```
2. Rank assets by momentum score (skip last day to avoid reversal effect)
3. Select top percentile, exclude bottom percentile
4. Allocate weights using chosen scheme (equal, momentum-weighted, or inverse-vol)

**Parameters (Crypto-optimized):**
- `formation_period`: 21 days — Crypto momentum decays faster than equities (optimal: 1-4 weeks)
- `top_percentile`: 30% — Select top performers
- `bottom_percentile`: 20% — Exclude worst losers (reduces drawdowns)
- `weighting_scheme`: 'equal', 'momentum', or 'inverse_vol'

**Why these parameters?** Liu et al. (2022) found that cryptocurrency momentum works best with shorter formation periods (1-4 weeks) compared to equities (3-12 months), due to faster information diffusion in crypto markets.

### 5. Time-Series Momentum (TS_MOM)

Time-series momentum (trend following) makes allocation decisions for each asset independently based on its own past returns. Unlike cross-sectional momentum, it doesn't compare assets against each other.

**Methodology (MA Crossover):**
1. Calculate fast and slow moving averages of prices:
   ```
   MA_fast = (1/fast_period) × Σ P_t
   MA_slow = (1/slow_period) × Σ P_t
   ```
2. Generate signal: Long when MA_fast > MA_slow (uptrend)
3. Weight trending assets equally or by inverse volatility

**Methodology (Absolute Momentum):**
1. Calculate cumulative return over lookback period
2. Long if return > 0, otherwise no position (cash equivalent)

**Parameters:**
- `fast_period`: 7 days — Captures short-term trends
- `slow_period`: 28 days — Filters noise, confirms trend
- `signal_type`: 'ma_crossover' or 'absolute'
- `position_sizing`: 'equal' or 'volatility_target'

**Academic Basis:** Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum" documented trend-following returns across 58 markets over 25 years.

### 6. Risk-Managed Momentum (RM_MOM)

Risk-managed momentum addresses the key weakness of momentum strategies: occasional severe crashes (e.g., 2009 momentum crash in equities). The insight is that momentum crashes occur during high volatility periods, so scaling exposure inversely to volatility improves risk-adjusted returns.

**Methodology (Barroso & Santa-Clara, 2015):**
1. Calculate raw momentum portfolio weights (cross-sectional selection)
2. Estimate realized volatility of the momentum portfolio:
   ```
   σ_realized = std(r_portfolio) × √365    (annualized)
   ```
3. Compute scaling factor:
   ```
   λ = σ_target / σ_realized
   ```
4. Apply leverage bounds and normalize:
   ```
   λ_bounded = clip(λ, min_leverage, max_leverage)
   w_final = normalize(w_raw × λ_bounded)
   ```

**Parameters:**
- `target_volatility`: 12% annualized — Standard institutional target
- `vol_lookback`: 63 days (~3 months) — Balances responsiveness and stability
- `max_leverage`: 2.0 — Cap during low-vol periods
- `min_leverage`: 0.25 — Floor during high-vol periods

**Key Result:** Barroso & Santa-Clara showed this approach improves momentum Sharpe from ~0.53 to ~0.97, nearly doubling risk-adjusted returns by avoiding crash periods.

### 7. Momentum + HRP (MOM_HRP)

This hybrid strategy combines momentum's alpha-generation capability with HRP's superior diversification. The intuition is to use momentum for *what* to hold and HRP for *how much* to hold.

**Two-Stage Methodology:**

*Stage 1 - Momentum Selection:*
1. Calculate momentum scores for all assets
2. Rank and select top performers (40% threshold)
3. Exclude bottom losers (20%)
4. Ensure minimum assets for clustering (default: 5)

*Stage 2 - HRP Allocation:*
1. Extract correlation/covariance matrix for selected assets only
2. Apply full HRP algorithm:
   - Distance matrix → Hierarchical clustering → Quasi-diagonalization → Recursive bisection
3. Map weights back to full universe (non-selected assets get 0 weight)

**Parameters:**
- `formation_period`: 21 days — Momentum lookback
- `top_percentile`: 40% — Higher than pure momentum to ensure enough assets for HRP clustering
- `min_assets`: 5 — Minimum for meaningful diversification

**Advantages:**
- Captures momentum premium while maintaining diversification
- Avoids momentum concentration risk
- Weights are stable relative to pure momentum (lower turnover)
- HRP handles correlated momentum picks gracefully

---

## Strategy Comparison

| Strategy | Selection | Weighting | Turnover | Diversification | Crash Risk |
|----------|-----------|-----------|----------|-----------------|------------|
| HRP | All assets | Risk-based | Low | High | Low |
| CS_MOM | Momentum rank | Equal/Vol | High | Low-Medium | Medium-High |
| TS_MOM | Trend signal | Equal/Vol | Medium | Medium | Medium |
| RM_MOM | Momentum + Vol | Vol-scaled | Medium | Low-Medium | Low |
| MOM_HRP | Momentum | HRP-based | Medium | Medium-High | Low-Medium |

---

## Data Sources

### Primary: Binance API
- **Type**: Spot market OHLCV data
- **Period**: December 31, 2018 - January 1, 2024
- **Interval**: Daily candlesticks
- **Universe**: All USDT trading pairs (excluding stablecoins)

### Alternative: CoinGecko API
- **Type**: Market prices, volumes, and market caps
- **Access**: Free tier (rate limited ~10-30 calls/minute)
- **Use case**: Cross-validation and additional market cap data

**Stablecoins excluded**: USDC, BUSD, DAI, TUSD, USDP, GUSD, FRAX, USDD, etc.

---

## Key Results

### Backtesting Configuration
- **Training Window**: 2 years (rolling)
- **Rebalancing Frequency**: Monthly (30 days)
- **Transaction Costs**: 0.1% per trade (Binance spot fees)
- **Test Period**: January 2022 - September 2023

### Performance Metrics

| Strategy | Annualized Return | Volatility | Sharpe Ratio | Max Drawdown |
|----------|-------------------|------------|--------------|--------------|
| HRP      | Varies by period  | Lower      | Higher       | Lower        |
| IVP      | Varies by period  | Medium     | Medium       | Medium       |
| MVP      | Varies by period  | Lowest     | Medium       | Medium       |
| BTC B&H  | Varies by period  | Highest    | Lower        | Highest      |

### Key Findings

1. **HRP provides better diversification** compared to MVP which tends to concentrate in few low-variance assets
2. **Transaction costs matter**: Monthly rebalancing costs ~0.1-0.2% per period, impacting net returns
3. **Stability**: HRP weights are more stable over time, resulting in lower turnover
4. **Crypto-specific considerations**: High correlation during market stress reduces diversification benefits

---

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate quant_ucema
```

### Using pip

```bash
pip install -r requirements.txt
```

---

## Usage

### Quick Start

```python
import pandas as pd
from src.portfolio_maker import (
    HRP, IVP, MVP,
    CrossSectionalMomentum, TimeSeriesMomentum,
    RiskManagedMomentum, MomentumHRP
)

# Load your returns data
returns = pd.read_csv('data/returns.csv', index_col=0, parse_dates=True)

# Create HRP portfolio
hrp = HRP(returns)
hrp_weights = hrp.get_weights()

# Create momentum portfolio
cs_mom = CrossSectionalMomentum(
    returns,
    formation_period=21,
    top_percentile=0.3,
    weighting_scheme='equal'
)
mom_weights = cs_mom.get_weights()

# Create combined Momentum + HRP portfolio
mom_hrp = MomentumHRP(returns, formation_period=21, top_percentile=0.4)
combined_weights = mom_hrp.get_weights()

print(f"HRP weights sum: {hrp_weights.sum():.4f}")
print(f"Momentum weights sum: {mom_weights.sum():.4f}")
print(f"MomentumHRP weights sum: {combined_weights.sum():.4f}")
```

### Fetching Data from Binance

```python
from src.binance_data import get_historical_klines, get_usdt_symbols

# Get all USDT pairs
symbols = get_usdt_symbols()

# Fetch historical data for BTC
btc_data = get_historical_klines(
    symbol='BTCUSDT',
    interval='1d',
    start_str='2023-01-01',
    end_str='2024-01-01'
)
```

### Fetching Data from CoinGecko (Alternative)

```python
from src.coingecko_data import (
    get_historical_prices,
    fetch_coingecko_dataset,
    get_top_coins_by_market_cap
)

# Get top 50 coins by market cap
top_coins = get_top_coins_by_market_cap(limit=50)

# Fetch historical prices for Bitcoin
btc_prices = get_historical_prices('bitcoin', days=365)

# Convenience: Fetch complete returns matrix for top N coins
returns = fetch_coingecko_dataset(n_coins=30, days=365, exclude_stablecoins=True)
```

### Analyzing Portfolio Transitions

```python
from src.agent import PortfolioChangeAnalyzerUSD

analyzer = PortfolioChangeAnalyzerUSD(
    current_weights=old_weights,
    proposed_weights=new_weights,
    returns=returns_df,
    transaction_cost_rate=0.001  # 0.1%
)

result = analyzer.analyze_transition()
if result['recommendation']:
    print(f"Rebalance recommended. Sharpe improvement: {result['sharpe_improvement']:.4f}")
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `data_collection.ipynb` | Fetch and preprocess historical data from Binance |
| `corr_matrix_dendrogam.ipynb` | Visualize asset correlations and hierarchical clustering |
| `hrp.ipynb` | Static portfolio allocation analysis and comparison |
| `backtesting.ipynb` | Full backtest with monthly rebalancing and transaction costs |
| `momentum_backtest.ipynb` | **Momentum strategies backtesting** with turnover analysis |
| `slides_examples.ipynb` | Educational examples for presentations |

---

## Dependencies

Core libraries:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scipy` - Optimization and clustering
- `matplotlib` / `seaborn` - Visualization
- `requests` - API calls

See `requirements.txt` for complete list.

---

## License

This project is for educational purposes as part of UCEMA's Quantitative Finance course.
