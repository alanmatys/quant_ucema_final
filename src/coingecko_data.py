"""
CoinGecko Data Collection Module

This module provides utilities for fetching historical cryptocurrency
price data from the CoinGecko API (free tier).

Note: CoinGecko free API has rate limits (~10-30 calls/minute).
"""

import json
import time
from typing import List, Optional, Dict
from datetime import datetime, timedelta

import pandas as pd
import requests


# Common crypto symbols mapping (CoinGecko uses different IDs)
COINGECKO_ID_MAP = {
    'BTCUSDT': 'bitcoin',
    'ETHUSDT': 'ethereum',
    'BNBUSDT': 'binancecoin',
    'XRPUSDT': 'ripple',
    'ADAUSDT': 'cardano',
    'SOLUSDT': 'solana',
    'DOTUSDT': 'polkadot',
    'DOGEUSDT': 'dogecoin',
    'AVAXUSDT': 'avalanche-2',
    'SHIBUSDT': 'shiba-inu',
    'MATICUSDT': 'matic-network',
    'LTCUSDT': 'litecoin',
    'LINKUSDT': 'chainlink',
    'ATOMUSDT': 'cosmos',
    'UNIUSDT': 'uniswap',
    'ETCUSDT': 'ethereum-classic',
    'XLMUSDT': 'stellar',
    'NEARUSDT': 'near',
    'ALGOUSDT': 'algorand',
    'VETUSDT': 'vechain',
    'ICPUSDT': 'internet-computer',
    'FILUSDT': 'filecoin',
    'TRXUSDT': 'tron',
    'AAVEUSDT': 'aave',
    'EOSUSDT': 'eos',
    'XTZUSDT': 'tezos',
    'THETAUSDT': 'theta-token',
    'AXSUSDT': 'axie-infinity',
    'SANDUSDT': 'the-sandbox',
    'MANAUSDT': 'decentraland',
    'FTMUSDT': 'fantom',
    'RUNEUSDT': 'thorchain',
    'ZILUSDT': 'zilliqa',
    'ENJUSDT': 'enjincoin',
    'CHZUSDT': 'chiliz',
    'BATUSDT': 'basic-attention-token',
    'ZECUSDT': 'zcash',
    'DASHUSDT': 'dash',
    'NEOUSDT': 'neo',
    'WAVESUSDT': 'waves',
    'QTUMUSDT': 'qtum',
    'ONTUSDT': 'ontology',
    'IOTAUSDT': 'iota',
    'ICXUSDT': 'icon',
    'NULSUSDT': 'nuls',
}


def get_coingecko_id(binance_symbol: str) -> Optional[str]:
    """
    Convert Binance symbol to CoinGecko ID.

    Args:
        binance_symbol: Binance trading pair (e.g., 'BTCUSDT')

    Returns:
        CoinGecko ID or None if not found
    """
    return COINGECKO_ID_MAP.get(binance_symbol)


def get_historical_prices(
    coin_id: str,
    vs_currency: str = 'usd',
    days: int = 365,
    interval: str = 'daily'
) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data from CoinGecko API.

    Args:
        coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
        vs_currency: Quote currency (default: 'usd')
        days: Number of days of history (max: 365 for free tier with daily)
        interval: 'daily' or 'hourly' (hourly only for < 90 days)

    Returns:
        DataFrame with columns: timestamp, price, market_cap, volume
        Returns None if request fails.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': interval
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'prices' not in data:
            return None

        # Convert to DataFrame
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')

        if 'market_caps' in data:
            market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
            market_caps['timestamp'] = pd.to_datetime(market_caps['timestamp'], unit='ms')
            prices_df = prices_df.merge(market_caps, on='timestamp', how='left')

        if 'total_volumes' in data:
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
            prices_df = prices_df.merge(volumes, on='timestamp', how='left')

        prices_df.set_index('timestamp', inplace=True)
        return prices_df

    except requests.RequestException as e:
        print(f"Error fetching {coin_id}: {e}")
        return None


def get_multiple_coins_data(
    coin_ids: List[str],
    days: int = 365,
    delay: float = 2.0
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple coins.

    Args:
        coin_ids: List of CoinGecko coin IDs
        days: Number of days of history
        delay: Delay between requests (seconds) to avoid rate limits

    Returns:
        Dictionary mapping coin_id to DataFrame
    """
    results = {}

    for i, coin_id in enumerate(coin_ids):
        print(f"Fetching {coin_id} ({i+1}/{len(coin_ids)})...")

        df = get_historical_prices(coin_id, days=days)
        if df is not None:
            results[coin_id] = df

        # Rate limiting
        if i < len(coin_ids) - 1:
            time.sleep(delay)

    return results


def create_returns_matrix(
    coin_data: Dict[str, pd.DataFrame],
    price_column: str = 'price'
) -> pd.DataFrame:
    """
    Create returns matrix from multiple coin price data.

    Args:
        coin_data: Dictionary mapping coin_id to price DataFrame
        price_column: Column name containing prices

    Returns:
        DataFrame with coins as columns and dates as index, values are daily returns
    """
    # Combine all prices into single DataFrame
    prices = {}
    for coin_id, df in coin_data.items():
        if price_column in df.columns:
            prices[coin_id] = df[price_column]

    prices_df = pd.DataFrame(prices)

    # Calculate daily returns
    returns = prices_df.pct_change().dropna()

    return returns


def get_top_coins_by_market_cap(limit: int = 100) -> List[Dict]:
    """
    Get top cryptocurrencies by market cap from CoinGecko.

    Args:
        limit: Number of coins to fetch (max 250 per page)

    Returns:
        List of coin dictionaries with id, symbol, name, market_cap
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"

    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': min(limit, 250),
        'page': 1,
        'sparkline': False
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        return [
            {
                'id': coin['id'],
                'symbol': coin['symbol'].upper(),
                'name': coin['name'],
                'market_cap': coin.get('market_cap', 0)
            }
            for coin in data
        ]

    except requests.RequestException as e:
        print(f"Error fetching top coins: {e}")
        return []


def fetch_coingecko_dataset(
    n_coins: int = 50,
    days: int = 365,
    exclude_stablecoins: bool = True
) -> pd.DataFrame:
    """
    Convenience function to fetch a complete dataset from CoinGecko.

    Args:
        n_coins: Number of top coins by market cap
        days: Days of history
        exclude_stablecoins: Whether to exclude stablecoins

    Returns:
        Returns matrix DataFrame
    """
    # Get top coins
    print(f"Fetching top {n_coins} coins by market cap...")
    top_coins = get_top_coins_by_market_cap(limit=n_coins + 20)  # Buffer for stablecoins

    # Stablecoins to exclude
    stablecoins = {'usdt', 'usdc', 'busd', 'dai', 'tusd', 'usdp', 'frax', 'usdd', 'gusd'}

    # Filter
    if exclude_stablecoins:
        top_coins = [c for c in top_coins if c['symbol'].lower() not in stablecoins]

    # Take top n
    top_coins = top_coins[:n_coins]
    coin_ids = [c['id'] for c in top_coins]

    print(f"Fetching historical data for {len(coin_ids)} coins...")
    coin_data = get_multiple_coins_data(coin_ids, days=days, delay=2.5)

    print(f"Creating returns matrix...")
    returns = create_returns_matrix(coin_data)

    # Rename columns to include USDT suffix for consistency
    returns.columns = [col.upper() + 'USDT' if not col.upper().endswith('USDT') else col.upper()
                       for col in returns.columns]

    return returns
