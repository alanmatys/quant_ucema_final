# Spec 08 — Point-in-Time Universe Reconstruction

**Status:** Draft
**Owner:** Alan Matys, Federico Rodriguez
**Blocks:** [03_backtest_v2.md](03_backtest_v2.md) (universe must be PIT before backtest is run)

---

## 1. Motivation

The current dataset
[data/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv](../data/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv)
contains only pairs that **survived** to the data-collection date in January
2024. This embeds two biases that any reviewer will flag:

- **Survivorship bias** — failed/delisted tokens (Terra/LUNA, FTT, dozens of
  smaller dead pairs) are absent, inflating measured returns.
- **Look-ahead in universe selection** — using today's market-cap rankings to
  pick assets implicitly uses information not available at the rebalance date.

A point-in-time (PIT) reconstruction rebuilds the universe **as it existed at
each monthly rebalance date**, including assets that subsequently failed.

**Methodology note (revised after R4 dry-run):** the original spec required
CoinGecko historical market cap as the ranking signal. CoinGecko's free demo
tier caps history at 365 days, and paid plans start at $129/mo — out of scope
for this paper. We instead rank by **rolling 30-day Binance USDT quote volume**,
a *tradable-liquidity* proxy that is (a) free, (b) available for the full
2019-2024 window, (c) more directly aligned with the paper's contribution
(constructing portfolios that could actually have been traded), and (d) a
standard liquidity proxy in academic crypto research. The methodological shift
is documented in the paper and cited as a deliberate choice, not a workaround.
A market-cap robustness check using Pro CoinGecko is reserved as future work.

## 2. Requirements

R1. **Monthly universe snapshots** spanning Jan 2019 → Dec 2023, one per
    rebalance date.

R2. **Selection criteria** (applied at each snapshot date):
   - Top 30 by **rolling 30-day Binance USDT quote volume** on the snapshot date.
   - Minimum age: 180 days of price history at that date.
   - Minimum median daily USD volume over the prior 30 days: $1M.
   - Excluded: stablecoins, wrapped tokens (WBTC, stETH, etc.), leveraged/inverse
     tokens, exchange tokens of the venue used for execution (avoid endogeneity).
   - Tradable on Binance USDT spot as of the snapshot date.

R3. **Entry/exit buffer** to avoid churn:
   - An asset entering top 30 enters the universe only after 2 consecutive
     monthly snapshots above the threshold.
   - An asset leaving top 30 stays in the universe for 1 additional snapshot
     before being removed.

R4. **Data sources:**
   - Historical market-cap rankings: **CoinGecko historical data**
     (free API; demo tier rate-limited to ~30 calls/min — accept the slower
     ingestion).
   - Historical price/volume: extend existing Binance ingestion to fetch all
     pairs that were ever in the universe, including currently-delisted ones
     (Binance historical data endpoint provides this).
   - Cross-reference Binance USDT listing/delisting dates from Binance
     announcements (manual JSON file, committed to repo) for the small number
     of high-impact delistings (LUNA, FTT, etc.).

R5. **Failed-token handling:**
   - When a token is delisted, the position is liquidated at the last available
     price.
   - Liquidation slippage modeled as **2× normal slippage** in the cost model
     (per Spec 03 amended).

R6. **Reproducibility:** the PIT universe is built by a deterministic notebook
    that produces a single artifact `data/pit_universe.csv` with columns
    `[date, symbol, market_cap, volume_30d, included]`. (Originally specified as
    parquet; downgraded to CSV because the artifact is small — ≪1MB — and
    avoiding the `pyarrow` dependency keeps the bootstrap simple.)

## 3. Interface

### 3.1 `src/universe.py` (new module)

```python
def fetch_historical_market_caps(start: str, end: str,
                                  top_n: int = 100) -> pd.DataFrame:
    """Fetch monthly market-cap rankings from CoinGecko historical API."""

def build_pit_universe(market_caps: pd.DataFrame,
                       binance_listings: pd.DataFrame,
                       price_history: pd.DataFrame,
                       top_n: int = 30,
                       min_age_days: int = 180,
                       min_median_volume_usd: float = 1e6,
                       exclude_patterns: list[str] = None,
                       entry_buffer_months: int = 2,
                       exit_buffer_months: int = 1) -> pd.DataFrame:
    """Build the monthly point-in-time universe DataFrame."""

def load_pit_universe(path: str = "data/pit_universe.csv") -> pd.DataFrame:
    """Load the built PIT universe artifact."""
```

### 3.2 Notebook

`notebooks/build_pit_universe.ipynb` — calls the three functions above,
saves `data/pit_universe.csv` and a human-readable
`data/pit_universe_summary.csv` for review.

### 3.3 Data artifacts

```
data/coingecko_market_caps_monthly.csv           # Raw market-cap ingestion (cached)
data/coingecko_candidates.json                   # Candidate coin list (hand-curated)
data/binance_listings_manual.json                # Hand-curated listing/delisting
data/pit_universe.csv                            # The PIT universe (committed)
data/pit_universe_summary.csv                    # Human-readable summary (committed)
data/binance_usdt_pairs_pit_2019-2024_1d.csv     # Extended price CSV including
                                                  # delisted pairs (committed)
```

## 4. Acceptance Criteria

AC1. PIT universe contains at least one asset that was subsequently delisted
     (e.g. LUNA in 2022) — confirms the bias fix is real, not cosmetic.

AC2. Median universe size per snapshot is 30 ± 3 (entry/exit buffers prevent
     exact-30 lock).

AC3. No stablecoins (verified against an explicit exclusion list including
     USDT, USDC, BUSD, DAI, TUSD, USDP, GUSD, FRAX, USDD, FDUSD, PYUSD, USDe).

AC4. Each asset's first appearance is at least 180 days after its earliest
     observed price (age constraint enforced).

AC5. Notebook runs end-to-end on a clean kernel; PIT universe regenerable from
     committed inputs.

AC6. Aggregate summary: ≥10 distinct assets present in the universe at some
     point that are NOT present at the end date (proves survivorship is
     actually being addressed).

## 5. Out of Scope

- Tick-level or intraday data ingestion.
- Cross-venue listing arbitrage (single venue = Binance for execution).
- Manual reconstruction of pre-2019 universe (LUNA-1, BCH fork details, etc.).
- Glassnode/Kaiko paid-tier integration (rejected per deep-research review
  scope discussion).

## 6. References

- CoinGecko API documentation: https://www.coingecko.com/en/api/documentation
- Binance public delisting announcements (manual collection).
- Han, Y., et al. (2024). Realistic-assumption momentum in cryptocurrency — flags
  survivorship as a recurring bias source.

## 7. Open Questions

OQ1. **Top-30 vs Top-20 vs Top-40** — pick one for headline results; report
     others as robustness. **Recommended: Top-30** (matches typical academic
     crypto papers).

OQ2. Should CoinGecko historical data be cached locally given rate limits?
     **Recommended: yes**, commit a frozen snapshot to the repo so the build is
     fully reproducible offline.
