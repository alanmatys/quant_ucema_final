"""
Binance Data Collection Module

This module provides utilities for fetching historical cryptocurrency
price data from the Binance API.

Functions:
    get_historical_klines: Fetch OHLCV candlestick data for a symbol
    get_usdt_symbols: Get all USDT trading pairs available on Binance
    get_symbols_from_list: Filter symbols that exist on Binance
"""

import json
import time
from typing import List, Optional

import pandas as pd
import requests
from datetime import datetime


def get_historical_klines(
    symbol: str,
    interval: str,
    start_str: str,
    end_str: str,
    last_timestamp: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Fetch historical candlestick (OHLCV) data from Binance API.

    Retrieves price data with automatic pagination to handle Binance's
    1000-record limit per request.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
        interval: Candlestick interval. Options include:
            '1m', '3m', '5m', '15m', '30m' (minutes)
            '1h', '2h', '4h', '6h', '8h', '12h' (hours)
            '1d', '3d' (days)
            '1w', '1M' (week, month)
        start_str: Start date in 'YYYY-MM-DD' format
        end_str: End date in 'YYYY-MM-DD' format
        last_timestamp: Optional timestamp (ms) to continue from previous data

    Returns:
        DataFrame with columns:
            - open_time (index): Candlestick open datetime
            - open, high, low, close: Price data (float)
            - volume: Trading volume in base asset
            - close_time: Candlestick close datetime
            - quote_volume: Trading volume in quote asset (USDT)
            - num_trades: Number of trades in the period
            - taker_base_vol: Taker buy base asset volume
            - taker_quote_vol: Taker buy quote asset volume
        Returns None if no data is available.

    Example:
        >>> df = get_historical_klines(
        ...     symbol='BTCUSDT',
        ...     interval='1d',
        ...     start_str='2023-01-01',
        ...     end_str='2023-12-31'
        ... )
        >>> print(df['close'].head())
    """
    root_url = 'https://api.binance.com/api/v3/klines'

    # Convert date strings to timestamps in milliseconds
    start_ts = (
        last_timestamp
        if last_timestamp
        else int(datetime.strptime(start_str, '%Y-%m-%d').timestamp() * 1000)
    )
    end_ts = int(datetime.strptime(end_str, '%Y-%m-%d').timestamp() * 1000)

    all_data = []

    # Paginate through results (Binance limits to 1000 records per request)
    while start_ts < end_ts:
        url = (
            f"{root_url}?symbol={symbol}&interval={interval}"
            f"&startTime={start_ts}&endTime={end_ts}&limit=1000"
        )

        data = json.loads(requests.get(url).text)
        if not data:
            break

        all_data.extend(data)

        # Move to next timestamp after the last received candle
        start_ts = int(data[-1][0]) + 1

        # Rate limiting to avoid API throttling
        time.sleep(0.1)

    if all_data:
        df = pd.DataFrame(all_data)
        df.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'num_trades',
            'taker_base_vol', 'taker_quote_vol', 'ignore'
        ]

        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # Set open_time as index
        df.set_index('open_time', inplace=True)

        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    return None


def get_usdt_symbols() -> List[str]:
    """
    Get all USDT trading pairs available on Binance.

    Fetches the current exchange information and filters for
    symbols that trade against USDT.

    Returns:
        List of trading pair symbols ending in 'USDT'
        (e.g., ['BTCUSDT', 'ETHUSDT', ...])

    Example:
        >>> symbols = get_usdt_symbols()
        >>> print(f"Found {len(symbols)} USDT pairs")
        >>> print(symbols[:5])
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = json.loads(requests.get(url).text)
    symbols = [
        x['symbol']
        for x in response['symbols']
        if x['symbol'].endswith('USDT')
    ]
    return symbols


def get_symbols_from_list(symbol_list: List[str]) -> List[str]:
    """
    Filter a list of symbols to only those available on Binance.

    Useful for validating a predefined list of symbols before
    attempting to fetch historical data.

    Args:
        symbol_list: List of symbol names to validate

    Returns:
        List of symbols that exist on Binance

    Example:
        >>> wanted = ['BTCUSDT', 'ETHUSDT', 'FAKECOINUSDT']
        >>> available = get_symbols_from_list(wanted)
        >>> print(available)  # ['BTCUSDT', 'ETHUSDT']
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = json.loads(requests.get(url).text)
    available_symbols = [x['symbol'] for x in response['symbols']]
    return [symbol for symbol in symbol_list if symbol in available_symbols]
