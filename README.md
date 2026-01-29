# Hierarchical Risk Parity for Cryptocurrency Portfolios

A quantitative finance research project implementing **Hierarchical Risk Parity (HRP)** methodology on cryptocurrency portfolios, comparing it against traditional allocation strategies.

## Authors

- Federico Rodriguez
- Alan Matys

*UCEMA - Quantitative Finance Course Final Project*

---

## Objective

This project evaluates the **Hierarchical Risk Parity (HRP)** portfolio allocation method developed by Marcos Lopez de Prado on cryptocurrency assets. We compare HRP against traditional approaches (Inverse Variance and Minimum Variance portfolios) using historical data from Binance.

### Reference Paper
- [Building Diversified Portfolios that Outperform Out-of-Sample](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678) - Lopez de Prado, M. (2016)

---

## Project Structure

```
quant_ucema_final/
├── src/                           # Core Python modules
│   ├── __init__.py                # Package exports
│   ├── portfolio_maker.py         # Portfolio allocation strategies
│   ├── binance_data.py            # Binance API data collection
│   └── agent.py                   # Portfolio transition analysis
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── data_collection.ipynb      # Historical data fetching
│   ├── corr_matrix_dendrogam.ipynb # Correlation analysis & dendrogram
│   ├── hrp.ipynb                  # Static HRP allocation analysis
│   ├── backtesting.ipynb          # Dynamic rebalancing backtest
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

---

## Data

- **Source**: Binance API (spot market)
- **Period**: December 31, 2018 - January 1, 2024
- **Interval**: Daily OHLCV candlesticks
- **Universe**: All USDT trading pairs (excluding stablecoins)
- **Stablecoins excluded**: USDC, BUSD, DAI, TUSD, USDP, GUSD, FRAX, etc.

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
from src.portfolio_maker import HRP, IVP, MVP

# Load your returns data
returns = pd.read_csv('data/returns.csv', index_col=0, parse_dates=True)

# Create HRP portfolio
hrp = HRP(returns)
weights = hrp.get_weights()

print(f"Portfolio weights sum: {weights.sum():.4f}")
print(weights.head(10))
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
